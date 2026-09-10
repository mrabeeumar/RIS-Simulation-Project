"""Shared "evaluate a candidate RIS configuration" pipeline.

Used both by the search loop inside `max_sumrate.py` / `fairness_aware.py`
and, unchanged, by `network/controller.py`'s final metrics pass -- so the
optimization search and the controller's reported results are guaranteed to
agree (never two separate implementations of channel -> pairing -> power
allocation -> SIC -> rate). See PLAN.md Section 2 (repo layout note) and the
pipeline described in Section 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ris_noma_sim.core.channel import ChannelSet, effective_channel
from ris_noma_sim.core.metrics import jain_fairness, system_sum_rate_bps_hz
from ris_noma_sim.core.pairing import pair_users
from ris_noma_sim.core.ris import quantize_phase, quantized_phase_levels
from ris_noma_sim.core.sic import cluster_rate_bps_hz, cluster_sinr, decoding_order, noise_power_w
from ris_noma_sim.optimization.power_allocation import allocate


@dataclass
class EvalResult:
    sum_rate_bps_hz: float
    user_rates_bps_hz: np.ndarray  # length n_users, indexed by original user index
    cluster_rates: list = field(default_factory=list)  # per-cluster per-user rate arrays


def evaluate(channels: ChannelSet, theta: np.ndarray, config) -> EvalResult:
    """Effective channel -> pairing -> power allocation -> SIC SINR/rate ->
    normalized system sum-rate, for one RIS configuration `theta`."""
    h_eff = effective_channel(channels, theta)
    gains = np.abs(h_eff[:, 0]) ** 2  # M=1 (SISO) assumption per PLAN.md Sec 4
    n_users = len(gains)

    clusters = pair_users(gains, config.cluster_size)
    noise_w = noise_power_w(config.tx_power_w, config.snr_db)

    cluster_rates = []
    user_rates = np.zeros(n_users)
    for cluster in clusters:
        cluster_arr = np.array(cluster)
        local_order = decoding_order(gains[cluster_arr])
        ordered_user_idx = cluster_arr[local_order]
        gains_desc = gains[ordered_user_idx]

        a = allocate(gains_desc, config)
        sinr = cluster_sinr(gains_desc, a, config.tx_power_w, noise_w, config.sic_epsilon)
        rates = cluster_rate_bps_hz(sinr)

        cluster_rates.append(rates)
        user_rates[ordered_user_idx] = rates

    sum_rate = system_sum_rate_bps_hz(cluster_rates)
    return EvalResult(sum_rate_bps_hz=sum_rate, user_rates_bps_hz=user_rates, cluster_rates=cluster_rates)


def sum_rate_score(result: EvalResult, config) -> float:
    return result.sum_rate_bps_hz


def fairness_aware_score(result: EvalResult, config) -> float:
    """alpha*Rsum + beta*JainIndex(rates), see PLAN.md Sec 7 algorithm 5."""
    fairness = jain_fairness(result.user_rates_bps_hz)
    return config.fairness_alpha * result.sum_rate_bps_hz + config.fairness_beta * fairness


def _candidate_grid(ris_bits) -> np.ndarray:
    if ris_bits == "continuous":
        return np.arange(8) * (2 * np.pi / 8)
    return quantized_phase_levels(ris_bits)


def coordinate_ascent(
    channels: ChannelSet,
    config,
    theta_init: np.ndarray,
    score_fn,
    max_iters: int = 5,
    tol: float = 1e-3,
) -> np.ndarray:
    """Element-by-element coordinate ascent maximizing `score_fn(evaluate(...), config)`.

    Candidate grid depends on `config.ris_bits` (PLAN.md Sec 7 algorithm 4):
    for a finite bit depth, candidates are exactly the quantized phase
    alphabet (search and final output already agree, no separate
    quantization pass needed); for "continuous", candidates are 8 coarse
    phases refined by one `scipy.optimize.minimize_scalar` polish per
    element per sweep.
    """
    from scipy.optimize import minimize_scalar

    n_ris = channels.H_BR.shape[0]
    if n_ris == 0:
        return np.array([])

    theta = np.array(theta_init, dtype=float, copy=True)
    candidates = _candidate_grid(config.ris_bits)
    continuous = config.ris_bits == "continuous"

    current_score = score_fn(evaluate(channels, theta, config), config)
    for _ in range(max_iters):
        sweep_start_score = current_score
        for n in range(n_ris):
            best_theta_n = theta[n]
            best_score = current_score
            for cand in candidates:
                trial = theta.copy()
                trial[n] = cand
                score = score_fn(evaluate(channels, trial, config), config)
                if score > best_score:
                    best_score = score
                    best_theta_n = cand
            theta[n] = best_theta_n
            current_score = best_score

            if continuous:
                def neg_score(phase, _n=n):
                    trial = theta.copy()
                    trial[_n] = phase
                    return -score_fn(evaluate(channels, trial, config), config)

                lo, hi = theta[n] - np.pi / 8, theta[n] + np.pi / 8
                res = minimize_scalar(neg_score, bounds=(lo, hi), method="bounded")
                if -res.fun > current_score:
                    theta[n] = res.x
                    current_score = -res.fun

        if current_score - sweep_start_score < tol:
            break

    return quantize_phase(theta, config.ris_bits)


def multi_start_coordinate_ascent(
    channels: ChannelSet,
    config,
    theta_max_snr: np.ndarray,
    score_fn,
    rng: np.random.Generator,
    n_random_restarts: int = 3,
) -> np.ndarray:
    """Run `coordinate_ascent` from the Max-SNR seed plus `n_random_restarts`
    random-phase seeds, return the best-scoring result. See PLAN.md Sec 7
    algorithms 4-5: a single coordinate-ascent run can get stuck in a local
    optimum that only a simultaneous multi-element flip would escape.
    """
    n_ris = channels.H_BR.shape[0]
    if n_ris == 0:
        return np.array([])

    seeds = [theta_max_snr]
    for _ in range(n_random_restarts):
        seeds.append(quantize_phase(rng.uniform(0, 2 * np.pi, size=n_ris), config.ris_bits))

    best_theta = None
    best_score = -np.inf
    for seed in seeds:
        theta = coordinate_ascent(channels, config, seed, score_fn)
        score = score_fn(evaluate(channels, theta, config), config)
        if score > best_score:
            best_score = score
            best_theta = theta
    return best_theta
