"""Algorithm 4: Max-Sum-Rate via coordinate ascent, initialized from Max-SNR.
See PLAN.md Section 7."""

from __future__ import annotations

import numpy as np

from ris_noma_sim.core.channel import ChannelSet
from ris_noma_sim.optimization import max_snr
from ris_noma_sim.optimization.objective import multi_start_coordinate_ascent, sum_rate_score


def optimize(channels: ChannelSet, config, rng: np.random.Generator) -> np.ndarray:
    """Multi-start coordinate ascent: Max-SNR seed + 3 random restarts,
    best-of-4 (PLAN.md Sec 7 algorithm 4). `rng` is required (used for the
    random restarts), unlike the other 4 RIS algorithms."""
    theta_max_snr = max_snr.optimize(channels, config)
    if len(theta_max_snr) == 0:
        return theta_max_snr
    return multi_start_coordinate_ascent(channels, config, theta_max_snr, sum_rate_score, rng)
