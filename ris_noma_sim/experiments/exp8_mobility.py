"""Experiment 8: user mobility -> RIS reconfiguration. See PLAN.md Sections 8, 11.

Runs the NetworkController's stateful mobility/reconfiguration loop and logs
per-user SINR over time with reconfiguration-event markers, compared against
a "never reconfigure" baseline that keeps the initial RIS configuration fixed
for the whole run under the same mobility trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.channel import generate_channels
from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.experiments.plotting import ensure_results_dir, save_csv
from ris_noma_sim.network import mobility
from ris_noma_sim.network.controller import NetworkController, rates_to_sinr
from ris_noma_sim.optimization.objective import evaluate

N_STEPS = 200  # at mobility_dt_s=0.1s => 20s of simulated time


def run(seed: int = 42) -> pd.DataFrame:
    cfg = SimConfig(
        n_ris=64, n_users=2, cluster_size=2, snr_db=15.0, ris_bits=2,
        ris_algo="max_sumrate", power_algo="fair_constrained",
        reconfig_mode="sinr_drop", reconfig_sinr_drop_db=3.0,
        user_speed_mps=1.0, mobility_dt_s=0.1,
    )

    # --- Reconfiguring run ---
    rng = np.random.default_rng(seed)
    controller = NetworkController(cfg, rng)
    state0 = controller.init_dynamic()
    rows = [dict(time_s=state0.time_s, avg_sinr_db=_avg_sinr_db(state0.user_rates_bps_hz), reconfigured=True)]
    for _ in range(N_STEPS):
        s = controller.step()
        rows.append(dict(time_s=s.time_s, avg_sinr_db=_avg_sinr_db(s.user_rates_bps_hz), reconfigured=bool(s.reconfigured_this_step)))
    df_reconfig = pd.DataFrame(rows)

    # --- "Never reconfigure" baseline: same mobility trajectory (same seed),
    # RIS configuration fixed at its t=0 value for the whole run. ---
    rng_baseline = np.random.default_rng(seed)
    controller_baseline = NetworkController(cfg, rng_baseline)
    state0_b = controller_baseline.init_dynamic()
    fixed_theta = state0_b.theta
    mobility_state = controller_baseline._mobility_state
    rows_b = [dict(time_s=0.0, avg_sinr_db=_avg_sinr_db(state0_b.user_rates_bps_hz))]
    for i in range(N_STEPS):
        mobility_state = mobility.step(mobility_state, cfg, rng_baseline)
        channels = generate_channels(
            n_ris=cfg.n_ris, n_users=cfg.n_users, m_antennas=1, channel_type=cfg.channel_type,
            rician_k_factor_db=cfg.rician_k_factor, path_loss_exponent=cfg.path_loss_exponent,
            bs_to_ris_m=mobility_state.topology.bs_to_ris_m, ris_to_user_m=mobility_state.topology.ris_to_user_m,
            bs_to_user_m=mobility_state.topology.bs_to_user_m, direct_link_blocked=cfg.direct_link_blocked, rng=rng_baseline,
        )
        result = evaluate(channels, fixed_theta, cfg)
        rows_b.append(dict(time_s=(i + 1) * cfg.mobility_dt_s, avg_sinr_db=_avg_sinr_db(result.user_rates_bps_hz)))
    df_baseline = pd.DataFrame(rows_b)

    df = df_reconfig.merge(df_baseline, on="time_s", suffixes=("_reconfig", "_baseline"))
    save_csv(df, "exp8_mobility.csv")
    _plot(df)
    return df


def _avg_sinr_db(rates_bps_hz: np.ndarray) -> float:
    sinr = rates_to_sinr(rates_bps_hz)
    return float(10 * np.log10(np.mean(np.maximum(sinr, 1e-15))))


def _plot(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["time_s"], df["avg_sinr_db_reconfig"], label="Reconfiguring RIS", color="#1f77b4", linewidth=1.8)
    ax.plot(df["time_s"], df["avg_sinr_db_baseline"], label="Never reconfigure (baseline)", color="#d62728", linewidth=1.5, linestyle="--")
    reconfig_times = df.loc[df["reconfigured"] & (df["time_s"] > 0), "time_s"]
    for t in reconfig_times:
        ax.axvline(t, color="green", alpha=0.2, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average SINR (dB)")
    ax.set_title("Experiment 8: Mobility-Triggered RIS Reconfiguration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    from ris_noma_sim.experiments.plotting import RESULTS_DIR

    fig.savefig(RESULTS_DIR / "exp8_mobility.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
