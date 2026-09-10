"""RIS phase quantization. See PLAN.md Section 5."""

from __future__ import annotations

from typing import Literal

import numpy as np


def quantized_phase_levels(bits: int) -> np.ndarray:
    """The 2**bits discrete phase levels in [0, 2*pi) for a given resolution."""
    n_levels = 2**bits
    return np.arange(n_levels) * (2 * np.pi / n_levels)


def quantize_phase(theta: np.ndarray, bits: int | Literal["continuous"]) -> np.ndarray:
    """theta_q = round(theta / (2*pi/2**bits)) * (2*pi/2**bits).

    `bits == "continuous"` returns theta unchanged. Output is wrapped to
    [0, 2*pi).
    """
    if bits == "continuous":
        return np.mod(theta, 2 * np.pi)
    step = 2 * np.pi / (2**bits)
    return np.mod(np.round(theta / step) * step, 2 * np.pi)
