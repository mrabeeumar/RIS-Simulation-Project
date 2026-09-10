"""SIC decoding order and K-user SINR. See PLAN.md Section 6.2-6.3.

Direction, precisely (this bit backwards silently corrupts every downstream
metric -- see .claude/CLAUDE.md): within a cluster sorted by descending
channel gain (position 0 = strongest), power is allocated *inversely* to gain
(weakest position gets the most power, PLAN.md Sec 6.2). The receiver at
position m cancels the higher-power signals of *weaker* users (j>m) -- those
are trivially decodable at m's better channel quality -- subject to residual
`epsilon`; it suffers *full, uncancelled* interference from stronger users'
lower-power signals (j<m), since it has no way to decode those. Position 0
(strongest) benefits most from SIC (only residual interference); the last
position (weakest) never cancels anything and its SINR is independent of
`epsilon`.
"""

from __future__ import annotations

import numpy as np


def noise_power_w(tx_power_w: float, snr_db: float) -> float:
    """N0 = P / 10**(snr_db/10). P is the single fixed quantity (Watts);
    N0 is always derived from it -- see PLAN.md Sec 10."""
    return tx_power_w / (10.0 ** (snr_db / 10.0))


def decoding_order(gains: np.ndarray) -> np.ndarray:
    """Indices that sort `gains` in descending order (position 0 = strongest,
    decodes/cancels first)."""
    return np.argsort(-np.asarray(gains))


def cluster_sinr(
    gains_desc: np.ndarray,
    power_fractions_desc: np.ndarray,
    tx_power_w: float,
    noise_w: float,
    sic_epsilon: float,
) -> np.ndarray:
    """SINR per user, both arrays already ordered by descending gain
    (position 0 = strongest / lowest power fraction).

    SINR_m = (a_m * P * g_m) /
             ( P*g_m * sum_{j<m} a_j              -- stronger users, uncancellable, full interference
               + epsilon * P*g_m * sum_{j>m} a_j    -- weaker users, cancelled with residual epsilon
               + N0 )
    """
    gains = np.asarray(gains_desc, dtype=float)
    a = np.asarray(power_fractions_desc, dtype=float)
    k = len(gains)

    cum_stronger = np.concatenate(([0.0], np.cumsum(a)[:-1])) if k > 0 else np.array([])  # sum_{j<m} a_j
    total = np.sum(a)
    cum_weaker = (total - np.cumsum(a)) if k > 0 else np.array([])  # sum_{j>m} a_j

    interference = tx_power_w * gains * (cum_stronger + sic_epsilon * cum_weaker)
    signal = a * tx_power_w * gains
    return signal / (interference + noise_w)


def cluster_rate_bps_hz(sinr: np.ndarray) -> np.ndarray:
    """R_m = log2(1 + SINR_m), per user."""
    return np.log2(1.0 + np.asarray(sinr))
