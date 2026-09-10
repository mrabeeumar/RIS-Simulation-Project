"""Algorithm 1: Random RIS phases. See PLAN.md Section 7."""

from __future__ import annotations

import numpy as np

from ris_noma_sim.core.channel import ChannelSet
from ris_noma_sim.core.ris import quantize_phase


def optimize(channels: ChannelSet, config, rng: np.random.Generator) -> np.ndarray:
    n_ris = channels.H_BR.shape[0]
    if n_ris == 0:
        return np.array([])
    theta = rng.uniform(0, 2 * np.pi, size=n_ris)
    return quantize_phase(theta, config.ris_bits)
