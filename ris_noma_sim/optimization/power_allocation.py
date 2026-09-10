"""NOMA power allocation strategies. See PLAN.md Section 6.4.

Interface: `allocate(gains_desc, config) -> np.ndarray` -- `gains_desc` are a
single cluster's channel gains |h_k|^2, already ordered by descending gain
(position 0 = strongest, matching core/sic.py's decoding-order convention).
Returns power fractions `a` in the same order, summing to 1, satisfying
a_0 <= a_1 <= ... (weaker channel gets more power).
"""

from __future__ import annotations

import numpy as np

from ris_noma_sim.core.sic import noise_power_w


def _fixed(gains_desc: np.ndarray, config) -> np.ndarray:
    k = len(gains_desc)
    a = np.empty(k)
    a[-1] = config.power_alloc_fixed_weak
    remainder = 1.0 - a[-1]
    if k > 1:
        a[:-1] = remainder / (k - 1)
    return a


def _inverse_gain(gains_desc: np.ndarray, config) -> np.ndarray:
    inv = 1.0 / np.asarray(gains_desc, dtype=float)
    return inv / np.sum(inv)


def _fair_constrained_two_user(gains_desc: np.ndarray, config) -> np.ndarray:
    """Closed-form QoS-floor allocation for a 2-user cluster. See PLAN.md
    Sec 6.4 for the derivation: SINR_1 = a_1*P*g_1/(P*g_1*a_0 + N0) with
    a_0 = 1-a_1, solved for the QoS floor SINR_1 >= 2^R_th - 1."""
    P = config.tx_power_w
    N0 = noise_power_w(P, config.snr_db)
    g1 = gains_desc[1]
    target = 2.0 ** config.outage_rate_threshold_bps_hz - 1.0
    a1_min = target * (P * g1 + N0) / (P * g1 * (1.0 + target))
    a1 = min(max(a1_min, 0.5), 1.0)
    return np.array([1.0 - a1, a1])


def _sca_power_allocation(gains_desc: np.ndarray, config, enforce_qos: bool) -> np.ndarray:
    """Successive Convex Approximation via CVXPY for K>2 clusters. See
    PLAN.md Sec 6.4. Falls back to inverse-gain allocation if any outer
    iteration is infeasible or the solver fails to converge."""
    import cvxpy as cp

    k = len(gains_desc)
    P = config.tx_power_w
    N0 = noise_power_w(P, config.snr_db)
    g = np.asarray(gains_desc, dtype=float)
    target = 2.0 ** config.outage_rate_threshold_bps_hz - 1.0
    eps = config.sic_epsilon

    a_prev = np.full(k, 1.0 / k)
    for _ in range(10):
        a = cp.Variable(k)
        constraints = [a >= 1e-9, cp.sum(a) == 1]
        constraints += [a[m] >= a[m - 1] for m in range(1, k)]

        denom_consts = []
        for m in range(k):
            cum_stronger = float(np.sum(a_prev[:m]))
            cum_weaker = float(np.sum(a_prev[m + 1 :]))
            denom = N0 + P * g[m] * (cum_stronger + eps * cum_weaker)
            denom_consts.append(denom)
            if enforce_qos:
                constraints.append(P * g[m] * a[m] >= target * denom)

        objective = cp.Maximize(
            cp.sum([cp.log(1 + P * g[m] * a[m] / denom_consts[m]) for m in range(k)])
        )
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS)
        except cp.error.SolverError:
            return _inverse_gain(gains_desc, config)

        if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return _inverse_gain(gains_desc, config)

        a_new = np.clip(np.asarray(a.value).flatten(), 1e-9, None)
        a_new = a_new / np.sum(a_new)
        converged = np.max(np.abs(a_new - a_prev)) < 1e-4
        a_prev = a_new
        if converged:
            break
    return a_prev


def allocate(gains_desc: np.ndarray, config) -> np.ndarray:
    gains_desc = np.asarray(gains_desc, dtype=float)
    k = len(gains_desc)
    if k == 1:
        return np.array([1.0])  # singleton cluster: no superposition, full power

    algo = config.power_algo
    if algo == "fixed":
        return _fixed(gains_desc, config)
    if algo == "inverse_gain":
        return _inverse_gain(gains_desc, config)
    if algo == "fair_constrained":
        if k == 2:
            return _fair_constrained_two_user(gains_desc, config)
        return _sca_power_allocation(gains_desc, config, enforce_qos=True)
    if algo == "max_sumrate_qos":
        return _sca_power_allocation(gains_desc, config, enforce_qos=False)
    raise ValueError(f"unknown power_algo: {algo!r}")
