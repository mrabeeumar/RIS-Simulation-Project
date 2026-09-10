"""Sum-rate, EE, Jain fairness, outage, BER. See PLAN.md Sections 9-10."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import erfc


def system_sum_rate_bps_hz(cluster_rates: Sequence[np.ndarray]) -> float:
    """R_sum = sum_over_clusters( (1/n_clusters) * sum_{m in cluster} R_m ).

    Each cluster occupies an orthogonal resource share of size
    1/n_clusters -- see PLAN.md Sec 6.1. `cluster_rates` is a list of
    per-user-rate arrays, one array per cluster.
    """
    n_clusters = len(cluster_rates)
    if n_clusters == 0:
        return 0.0
    return sum(float(np.sum(r)) for r in cluster_rates) / n_clusters


def oma_sum_rate_bps_hz(gains: np.ndarray, tx_power_w: float, noise_w: float) -> float:
    """R_k^OMA = (1/n_users) * log2(1 + P*g_k/N0); R_sum^OMA = sum_k R_k^OMA."""
    gains = np.asarray(gains, dtype=float)
    n_users = len(gains)
    if n_users == 0:
        return 0.0
    per_user = (1.0 / n_users) * np.log2(1.0 + tx_power_w * gains / noise_w)
    return float(np.sum(per_user))


def jain_fairness(rates: np.ndarray) -> float:
    """J = (sum_k R_k)^2 / (n_users * sum_k R_k^2)."""
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    if n == 0:
        return 0.0
    denom = n * np.sum(rates**2)
    if denom == 0.0:
        return 1.0  # all-zero rates are trivially "fair" (no one is worse off than anyone else)
    return float(np.sum(rates) ** 2 / denom)


def outage_indicator(rates: np.ndarray, threshold_bps_hz: float) -> np.ndarray:
    """Boolean array: True where a user's rate is below the outage threshold."""
    return np.asarray(rates, dtype=float) < threshold_bps_hz


def energy_efficiency_bit_per_j(
    r_sum_bps_hz: float,
    tx_power_w: float,
    p_bs_static_w: float,
    n_ris: int,
    p_element_w: float,
) -> float:
    """EE = R_sum / (P_tx + P_bs_static + n_ris * p_element)."""
    total_power_w = tx_power_w + p_bs_static_w + n_ris * p_element_w
    if total_power_w <= 0:
        return 0.0
    return r_sum_bps_hz / total_power_w


def ber(sinr: np.ndarray, modulation: str) -> np.ndarray:
    """BPSK: BER = 0.5*erfc(sqrt(SINR)). QPSK: BER = 0.5*erfc(sqrt(SINR/2))."""
    sinr = np.asarray(sinr, dtype=float)
    sinr = np.clip(sinr, 0.0, None)
    if modulation == "bpsk":
        return 0.5 * erfc(np.sqrt(sinr))
    if modulation == "qpsk":
        return 0.5 * erfc(np.sqrt(sinr / 2.0))
    raise ValueError(f"unknown modulation: {modulation!r}")
