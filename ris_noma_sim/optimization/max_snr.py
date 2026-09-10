"""Algorithm 3: Max-SNR RIS phase alignment. See PLAN.md Section 7.

theta_n* = -angle( sum_k conj(h_RU,k,n) * H_BR,n ) -- phase-aligns element n
to the composite (summed) user channel. The conjugate on h_RU is required
because the effective-channel formula (PLAN.md Sec 4) uses h_RU,k^H.
M=1 (SISO) is assumed, matching the rest of this milestone.
"""

from __future__ import annotations

import numpy as np

from ris_noma_sim.core.channel import ChannelSet
from ris_noma_sim.core.ris import quantize_phase


def optimize(channels: ChannelSet, config, rng: np.random.Generator | None = None) -> np.ndarray:
    """`rng` is unused (Max-SNR is deterministic) but accepted for interface
    consistency with the other 4 RIS algorithms -- see PLAN.md Sec 7."""
    n_ris = channels.H_BR.shape[0]
    if n_ris == 0:
        return np.array([])
    composite = channels.H_BR[:, 0] * np.sum(channels.h_RU.conj(), axis=1)  # (n_ris,)
    theta = -np.angle(composite)
    return quantize_phase(theta, config.ris_bits)
