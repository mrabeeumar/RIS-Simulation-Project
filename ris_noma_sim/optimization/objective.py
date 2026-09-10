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


@dataclass
class AllocationPlan:
    """Cluster membership + decoding order + power fractions, frozen from one
    reference theta. Re-scoring a nearby candidate theta against a frozen
    plan (`evaluate_with_plan`) skips pairing/power-allocation entirely --
    the expensive parts (`power_allocation.allocate`'s CVXPY path for K>2) --
    and only recomputes the cheap effective-channel/SINR/rate steps."""

    cluster_user_idx: list  # list[np.ndarray], descending-gain user indices per cluster (at the reference theta)
    cluster_power_fractions: list  # list[np.ndarray], power fractions per cluster, same order


def plan_allocation(channels: ChannelSet, theta: np.ndarray, config) -> AllocationPlan:
    h_eff = effective_channel(channels, theta)
    gains = np.abs(h_eff[:, 0]) ** 2  # M=1 (SISO) assumption per PLAN.md Sec 4
    clusters = pair_users(gains, config.cluster_size)

    cluster_user_idx, cluster_power_fractions = [], []
    for cluster in clusters:
        cluster_arr = np.array(cluster)
        order = decoding_order(gains[cluster_arr])
        ordered_user_idx = cluster_arr[order]
        gains_desc = gains[ordered_user_idx]
        a = allocate(gains_desc, config)
        cluster_user_idx.append(ordered_user_idx)
        cluster_power_fractions.append(a)

    return AllocationPlan(cluster_user_idx=cluster_user_idx, cluster_power_fractions=cluster_power_fractions)


def evaluate_with_plan(channels: ChannelSet, theta: np.ndarray, plan: AllocationPlan, config) -> EvalResult:
    """Score `theta` against a frozen `AllocationPlan` (see `plan_allocation`):
    recomputes only effective-channel gains -> SIC SINR/rate for the plan's
    fixed cluster membership and power fractions -- no pairing or power
    allocation solve. Used by `coordinate_ascent`'s inner per-candidate loop;
    `evaluate()` (fresh plan every call) remains the source of truth."""
    h_eff = effective_channel(channels, theta)
    gains = np.abs(h_eff[:, 0]) ** 2
    n_users = len(gains)
    noise_w = noise_power_w(config.tx_power_w, config.snr_db)

    cluster_rates = []
    user_rates = np.zeros(n_users)
    for ordered_user_idx, a in zip(plan.cluster_user_idx, plan.cluster_power_fractions):
        gains_desc = gains[ordered_user_idx]
        sinr = cluster_sinr(gains_desc, a, config.tx_power_w, noise_w, config.sic_epsilon)
        rates = cluster_rate_bps_hz(sinr)
        cluster_rates.append(rates)
        user_rates[ordered_user_idx] = rates

    sum_rate = system_sum_rate_bps_hz(cluster_rates)
    return EvalResult(sum_rate_bps_hz=sum_rate, user_rates_bps_hz=user_rates, cluster_rates=cluster_rates)


def evaluate(channels: ChannelSet, theta: np.ndarray, config) -> EvalResult:
    """Effective channel -> pairing -> power allocation -> SIC SINR/rate ->
    normalized system sum-rate, for one RIS configuration `theta`. Always
    solves pairing/power allocation fresh -- this is the source of truth
    used by NetworkController's final metrics pass; see `evaluate_with_plan`
    for the cheaper per-candidate variant used inside coordinate ascent."""
    plan = plan_allocation(channels, theta, config)
    return evaluate_with_plan(channels, theta, plan, config)


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

    Power allocation is recomputed once per full sweep, not per candidate
    (PLAN.md Sec 7: "using current power allocation, recomputed via the
    configured power_algo each full sweep") -- `plan_allocation` is solved
    once at the start of each sweep and every candidate phase in that sweep
    is scored against the frozen plan via `evaluate_with_plan`, which skips
    pairing/power-allocation (including any CVXPY solve) entirely. This is
    what makes coordinate ascent tractable for N up to 256: scoring against a
    fresh `evaluate()` for every one of the ~n_ris*n_candidates trial phases
    per sweep was measured to make a single N=256 optimization take minutes
    during implementation.
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
        plan = plan_allocation(channels, theta, config)

        for n in range(n_ris):
            best_theta_n = theta[n]
            best_score = current_score
            for cand in candidates:
                trial = theta.copy()
                trial[n] = cand
                score = score_fn(evaluate_with_plan(channels, trial, plan, config), config)
                if score > best_score:
                    best_score = score
                    best_theta_n = cand
            theta[n] = best_theta_n
            current_score = best_score

            if continuous:
                def neg_score(phase, _n=n):
                    trial = theta.copy()
                    trial[_n] = phase
                    return -score_fn(evaluate_with_plan(channels, trial, plan, config), config)

                lo, hi = theta[n] - np.pi / 8, theta[n] + np.pi / 8
                res = minimize_scalar(neg_score, bounds=(lo, hi), method="bounded")
                if -res.fun > current_score:
                    theta[n] = res.x
                    current_score = -res.fun

        # Re-check against the TRUE score (fresh plan) at the end of the
        # sweep, both to decide convergence honestly and so the next sweep's
        # frozen plan reflects this sweep's actual improvement.
        current_score = score_fn(evaluate(channels, theta, config), config)
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
