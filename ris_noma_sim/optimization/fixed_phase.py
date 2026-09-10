"""Algorithm 2: Fixed RIS phases (baseline, no intelligence). See PLAN.md Section 7."""

from __future__ import annotations

import numpy as np

from ris_noma_sim.core.channel import ChannelSet
from ris_noma_sim.core.ris import quantize_phase


def optimize(channels: ChannelSet, config, rng: np.random.Generator) -> np.ndarray:
    """`rng` is unused (Fixed is deterministic) but kept for interface
    consistency with the other 4 RIS algorithms -- see PLAN.md Sec 7."""
    n_ris = channels.H_BR.shape[0]
    theta = np.zeros(n_ris)
    return quantize_phase(theta, config.ris_bits)
