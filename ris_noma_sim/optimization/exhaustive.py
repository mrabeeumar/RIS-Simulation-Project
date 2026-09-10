"""Brute-force exhaustive search over the quantized phase alphabet.

Used only as a validation baseline for small N (<=8) in unit tests -- not one
of the 5 production RIS algorithms. See PLAN.md Section 7.
"""

from __future__ import annotations

import itertools

import numpy as np

from ris_noma_sim.core.channel import ChannelSet
from ris_noma_sim.core.ris import quantized_phase_levels
from ris_noma_sim.optimization.objective import evaluate, sum_rate_score


def optimize(channels: ChannelSet, config, rng: np.random.Generator | None = None) -> np.ndarray:
    n_ris = channels.H_BR.shape[0]
    if n_ris == 0:
        return np.array([])
    if n_ris > 8:
        raise ValueError("exhaustive search is only intended for n_ris <= 8 (validation baseline)")
    if config.ris_bits == "continuous":
        raise ValueError("exhaustive search requires a finite ris_bits resolution")

    levels = quantized_phase_levels(config.ris_bits)
    best_theta = None
    best_score = -np.inf
    for combo in itertools.product(levels, repeat=n_ris):
        theta = np.array(combo)
        score = sum_rate_score(evaluate(channels, theta, config), config)
        if score > best_score:
            best_score = score
            best_theta = theta
    return best_theta
