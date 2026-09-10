"""NetworkController: the shared simulation pipeline. See PLAN.md Section 8.

`simulate_batch()` is the stateless Monte-Carlo pipeline used by Experiments
1-7 and the dashboard's Live tab: for n_trials, regenerate channel fading (at
a topology fixed for the whole batch) -> run the configured RIS algorithm ->
evaluate (pairing -> power allocation -> SIC -> rates) -> average metrics.

`step()`/`maybe_reconfigure()` is the stateful per-time-step pipeline used by
Experiment 8: advance user mobility -> regenerate channel at the new
positions -> decide whether to reconfigure the RIS -> evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ris_noma_sim.core.channel import effective_channel, generate_channels
from ris_noma_sim.core.metrics import ber, energy_efficiency_bit_per_j, jain_fairness, oma_sum_rate_bps_hz, outage_indicator
from ris_noma_sim.core.sic import noise_power_w
from ris_noma_sim.network import mobility
from ris_noma_sim.network.topology import Topology, generate_topology
from ris_noma_sim.optimization import fairness_aware, fixed_phase, max_snr, max_sumrate, random_phase
from ris_noma_sim.optimization.objective import EvalResult, evaluate

_RIS_ALGORITHMS = {
    "random": random_phase.optimize,
    "fixed": fixed_phase.optimize,
    "max_snr": max_snr.optimize,
    "max_sumrate": max_sumrate.optimize,
    "fairness_aware": fairness_aware.optimize,
}


def rates_to_sinr(rates_bps_hz: np.ndarray) -> np.ndarray:
    """Invert R = log2(1+SINR): SINR = 2**R - 1. Exact and avoids threading
    a second SINR array through EvalResult."""
    return 2.0 ** rates_bps_hz - 1.0


@dataclass
class BatchResult:
    sum_rate_bps_hz: float
    avg_user_rate_bps_hz: float
    jain_fairness_index: float
    outage_probability: float
    energy_efficiency_bit_per_j: float
    avg_ber: float
    per_trial_sum_rate_bps_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_per_user_rate_bps_hz: np.ndarray = field(default_factory=lambda: np.array([]))  # length n_users, trial-averaged


@dataclass
class ControllerState:
    time_s: float
    topology: Topology
    theta: np.ndarray
    user_rates_bps_hz: np.ndarray
    sum_rate_bps_hz: float
    reconfigured_this_step: bool
    reconfigure_cause: str | None = None


class NetworkController:
    def __init__(self, config, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._ris_algo = _RIS_ALGORITHMS[config.ris_algo]
        self.topology: Topology = generate_topology(config, rng)

        # Dynamic-mode (step()) state; populated by init_dynamic().
        self._mobility_state: mobility.MobilityState | None = None
        self._last_reconfig_time_s: float = 0.0
        self._last_reconfig_sinr: np.ndarray | None = None
        self._current_state: ControllerState | None = None

    def _generate_channels(self, topology: Topology):
        cfg = self.config
        return generate_channels(
            n_ris=cfg.n_ris,
            n_users=cfg.n_users,
            m_antennas=1,
            channel_type=cfg.channel_type,
            rician_k_factor_db=cfg.rician_k_factor,
            path_loss_exponent=cfg.path_loss_exponent,
            bs_to_ris_m=topology.bs_to_ris_m,
            ris_to_user_m=topology.ris_to_user_m,
            bs_to_user_m=topology.bs_to_user_m,
            direct_link_blocked=cfg.direct_link_blocked,
            rng=self.rng,
        )

    # ---- Stateless Monte-Carlo batch (Experiments 1-7, dashboard Live tab) ----

    def simulate_batch(self) -> BatchResult:
        cfg = self.config
        sum_rates, avg_user_rates, fairness_vals = [], [], []
        outages, ee_vals, ber_vals = [], [], []
        per_user_rates = np.zeros(cfg.n_users)

        for _ in range(cfg.n_trials):
            channels = self._generate_channels(self.topology)
            theta = self._ris_algo(channels, cfg, self.rng)
            result: EvalResult = evaluate(channels, theta, cfg)

            sum_rates.append(result.sum_rate_bps_hz)
            avg_user_rates.append(float(np.mean(result.user_rates_bps_hz)))
            per_user_rates += result.user_rates_bps_hz
            fairness_vals.append(jain_fairness(result.user_rates_bps_hz))
            outages.append(float(np.mean(outage_indicator(result.user_rates_bps_hz, cfg.outage_rate_threshold_bps_hz))))
            ee_vals.append(
                energy_efficiency_bit_per_j(result.sum_rate_bps_hz, cfg.tx_power_w, cfg.p_bs_static_w, cfg.n_ris, cfg.p_element_w)
            )
            sinr = rates_to_sinr(result.user_rates_bps_hz)
            ber_vals.append(float(np.mean(ber(sinr, cfg.modulation))))

        return BatchResult(
            sum_rate_bps_hz=float(np.mean(sum_rates)),
            avg_user_rate_bps_hz=float(np.mean(avg_user_rates)),
            jain_fairness_index=float(np.mean(fairness_vals)),
            outage_probability=float(np.mean(outages)),
            energy_efficiency_bit_per_j=float(np.mean(ee_vals)),
            avg_ber=float(np.mean(ber_vals)),
            per_trial_sum_rate_bps_hz=np.array(sum_rates),
            avg_per_user_rate_bps_hz=per_user_rates / cfg.n_trials,
        )

    def simulate_noma_vs_oma_batch(self) -> tuple[float, float]:
        """Experiment 5 only: average (NOMA sum-rate, OMA sum-rate) computed
        under IDENTICAL per-trial channel/RIS realizations, for a fair
        paired comparison (PLAN.md Sec 11 exp5). NOMA uses the normal
        pipeline (evaluate()); OMA uses metrics.oma_sum_rate_bps_hz on the
        same trial's post-RIS effective-channel gains."""
        cfg = self.config
        noma_rates, oma_rates = [], []
        noise_w = noise_power_w(cfg.tx_power_w, cfg.snr_db)

        for _ in range(cfg.n_trials):
            channels = self._generate_channels(self.topology)
            theta = self._ris_algo(channels, cfg, self.rng)
            noma_result = evaluate(channels, theta, cfg)
            noma_rates.append(noma_result.sum_rate_bps_hz)

            gains = np.abs(effective_channel(channels, theta)[:, 0]) ** 2
            oma_rates.append(oma_sum_rate_bps_hz(gains, cfg.tx_power_w, noise_w))

        return float(np.mean(noma_rates)), float(np.mean(oma_rates))

    # ---- Stateful per-time-step pipeline (Experiment 8) ----

    def init_dynamic(self) -> ControllerState:
        """Initialize mobility + an initial RIS configuration at t=0. Always
        reconfigures at t=0 (there is no prior configuration)."""
        self._mobility_state = mobility.init_mobility(self.topology, self.config, self.rng)
        channels = self._generate_channels(self._mobility_state.topology)
        theta = self._ris_algo(channels, self.config, self.rng)
        result = evaluate(channels, theta, self.config)

        self._last_reconfig_time_s = 0.0
        self._last_reconfig_sinr = rates_to_sinr(result.user_rates_bps_hz)
        self._current_state = ControllerState(
            time_s=0.0,
            topology=self._mobility_state.topology,
            theta=theta,
            user_rates_bps_hz=result.user_rates_bps_hz,
            sum_rate_bps_hz=result.sum_rate_bps_hz,
            reconfigured_this_step=True,
            reconfigure_cause="init",
        )
        return self._current_state

    def maybe_reconfigure(self, current_sinr: np.ndarray, current_time_s: float) -> tuple[bool, str | None]:
        """Sec 8 ReconfigTrigger: 'periodic' fires every `reconfig_period_s`
        seconds; 'sinr_drop' (default) fires when any user's instantaneous
        SINR has dropped more than `reconfig_sinr_drop_db` dB relative to the
        SINR measured at the last reconfiguration."""
        cfg = self.config
        if cfg.reconfig_mode == "periodic":
            # Epsilon guards against float accumulation in repeated
            # `time_s += mobility_dt_s` additions landing just under the
            # threshold (e.g. 0.1+0.1+0.1 != 0.3 exactly in binary floating
            # point), which would otherwise make the trigger intermittently
            # fire one step late.
            if current_time_s - self._last_reconfig_time_s >= cfg.reconfig_period_s - 1e-9:
                return True, "periodic"
            return False, None

        # "sinr_drop"
        sinr_db_now = 10.0 * np.log10(np.maximum(current_sinr, 1e-15))
        sinr_db_last = 10.0 * np.log10(np.maximum(self._last_reconfig_sinr, 1e-15))
        drop_db = sinr_db_last - sinr_db_now
        if np.any(drop_db > cfg.reconfig_sinr_drop_db):
            return True, "sinr_drop"
        return False, None

    def step(self) -> ControllerState:
        """Advance one `config.mobility_dt_s` time step: move users, refresh
        the channel, decide whether to reconfigure the RIS, evaluate."""
        if self._current_state is None:
            raise RuntimeError("call init_dynamic() before step()")

        self._mobility_state = mobility.step(self._mobility_state, self.config, self.rng)
        new_time_s = self._current_state.time_s + self.config.mobility_dt_s
        channels = self._generate_channels(self._mobility_state.topology)

        # Evaluate performance of the *current* theta at the new positions
        # first, since reconfiguration is a decision, not a foregone conclusion.
        result_current_theta = evaluate(channels, self._current_state.theta, self.config)
        sinr_current = rates_to_sinr(result_current_theta.user_rates_bps_hz)

        should_reconfig, cause = self.maybe_reconfigure(sinr_current, new_time_s)

        if should_reconfig:
            theta = self._ris_algo(channels, self.config, self.rng)
            result = evaluate(channels, theta, self.config)
            self._last_reconfig_time_s = new_time_s
            self._last_reconfig_sinr = rates_to_sinr(result.user_rates_bps_hz)
        else:
            theta = self._current_state.theta
            result = result_current_theta

        self._current_state = ControllerState(
            time_s=new_time_s,
            topology=self._mobility_state.topology,
            theta=theta,
            user_rates_bps_hz=result.user_rates_bps_hz,
            sum_rate_bps_hz=result.sum_rate_bps_hz,
            reconfigured_this_step=should_reconfig,
            reconfigure_cause=cause,
        )
        return self._current_state
