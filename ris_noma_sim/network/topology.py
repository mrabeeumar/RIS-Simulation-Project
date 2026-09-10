"""BS/RIS/user placement. See PLAN.md Section 4 ("Default topology")."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Topology:
    bs_pos: np.ndarray  # (2,)
    ris_pos: np.ndarray  # (2,)
    user_pos: np.ndarray  # (n_users, 2)
    ris_to_user_m: np.ndarray  # (n_users,)
    bs_to_user_m: np.ndarray  # (n_users,)
    bs_to_ris_m: float


def generate_topology(config, rng: np.random.Generator) -> Topology:
    """BS fixed at origin. RIS fixed at (ris_offset_m, 0). Users placed
    uniformly at random in the annulus [d_min_m, d_max_m] around the RIS,
    unless `config.user_distances_m` is set (Experiment 7), in which case
    those exact RIS-to-user distances are used with angle=0 (colinear
    placement, to isolate the distance effect cleanly)."""
    bs_pos = np.array([0.0, 0.0])
    ris_pos = np.array([config.ris_offset_m, 0.0])
    n = config.n_users

    if config.user_distances_m is not None:
        distances = np.array(config.user_distances_m, dtype=float)
        angles = np.zeros(n)
    else:
        distances = rng.uniform(config.d_min_m, config.d_max_m, size=n)
        angles = rng.uniform(0, 2 * np.pi, size=n)

    offsets = np.stack([distances * np.cos(angles), distances * np.sin(angles)], axis=1)
    user_pos = ris_pos + offsets
    bs_to_user_m = np.linalg.norm(user_pos - bs_pos, axis=1)

    return Topology(
        bs_pos=bs_pos,
        ris_pos=ris_pos,
        user_pos=user_pos,
        ris_to_user_m=distances,
        bs_to_user_m=bs_to_user_m,
        bs_to_ris_m=config.ris_offset_m,
    )
