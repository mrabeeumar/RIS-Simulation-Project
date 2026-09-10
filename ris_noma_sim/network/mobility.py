"""Random-waypoint mobility for Experiment 8. See PLAN.md Section 8.

The cell is a disk centered at the RIS position with radius `d_max_m` (the
same disk the default topology places users in). Every destination is drawn
inside this disk and users move straight toward their current target, so the
path never exits the disk -- no boundary reflection is needed for this cell
shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ris_noma_sim.network.topology import Topology


@dataclass
class MobilityState:
    topology: Topology
    targets: np.ndarray  # (n_users, 2)


def _random_point_in_disk(center: np.ndarray, radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    distances = radius * np.sqrt(rng.uniform(0, 1, size=n))  # uniform-area sampling
    angles = rng.uniform(0, 2 * np.pi, size=n)
    return center + np.stack([distances * np.cos(angles), distances * np.sin(angles)], axis=1)


def init_mobility(topology: Topology, config, rng: np.random.Generator) -> MobilityState:
    n = topology.user_pos.shape[0]
    targets = _random_point_in_disk(topology.ris_pos, config.d_max_m, n, rng)
    return MobilityState(topology=topology, targets=targets)


def step(state: MobilityState, config, rng: np.random.Generator) -> MobilityState:
    """Advance every user one `config.mobility_dt_s` step toward its target
    at `config.user_speed_mps`; on arrival, snap to the target and draw a new
    one for the next step (PLAN.md Sec 8)."""
    positions = state.topology.user_pos
    targets = state.targets
    step_dist = config.user_speed_mps * config.mobility_dt_s

    to_target = targets - positions
    dist_to_target = np.linalg.norm(to_target, axis=1)

    arrived = dist_to_target <= step_dist
    new_positions = positions.copy()

    # Users still en route: move `step_dist` toward their target.
    moving = ~arrived
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.divide(
            to_target, dist_to_target[:, np.newaxis], out=np.zeros_like(to_target), where=dist_to_target[:, np.newaxis] > 0
        )
    new_positions[moving] = positions[moving] + step_dist * direction[moving]

    # Arrived users: snap to target, draw a fresh target for next step.
    new_positions[arrived] = targets[arrived]
    new_targets = targets.copy()
    n_arrived = int(np.sum(arrived))
    if n_arrived > 0:
        new_targets[arrived] = _random_point_in_disk(state.topology.ris_pos, config.d_max_m, n_arrived, rng)

    new_ris_to_user = np.linalg.norm(new_positions - state.topology.ris_pos, axis=1)
    new_bs_to_user = np.linalg.norm(new_positions - state.topology.bs_pos, axis=1)
    new_topology = replace(
        state.topology, user_pos=new_positions, ris_to_user_m=new_ris_to_user, bs_to_user_m=new_bs_to_user
    )
    return MobilityState(topology=new_topology, targets=new_targets)
