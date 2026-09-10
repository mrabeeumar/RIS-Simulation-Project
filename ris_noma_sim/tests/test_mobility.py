import numpy as np
import pytest

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.network import mobility
from ris_noma_sim.network.topology import generate_topology


def test_init_mobility_targets_within_disk():
    cfg = SimConfig(n_users=5, d_max_m=15.0)
    rng = np.random.default_rng(0)
    topo = generate_topology(cfg, rng)
    state = mobility.init_mobility(topo, cfg, rng)
    dist_from_ris = np.linalg.norm(state.targets - topo.ris_pos, axis=1)
    assert np.all(dist_from_ris <= 15.0 + 1e-9)


def test_step_moves_user_toward_target():
    cfg = SimConfig(n_users=2, d_max_m=15.0, user_speed_mps=1.0, mobility_dt_s=0.1)
    rng = np.random.default_rng(1)
    topo = generate_topology(cfg, rng)
    state = mobility.init_mobility(topo, cfg, rng)

    # Force a distant target so the user is still "en route" after one step.
    state.targets[0] = topo.ris_pos + np.array([10.0, 0.0])
    dist_before = np.linalg.norm(state.topology.user_pos[0] - state.targets[0])

    new_state = mobility.step(state, cfg, rng)
    dist_after = np.linalg.norm(new_state.topology.user_pos[0] - state.targets[0])

    assert dist_after == pytest.approx(dist_before - cfg.user_speed_mps * cfg.mobility_dt_s, abs=1e-6)


def test_step_arrival_retargets():
    cfg = SimConfig(n_users=2, d_max_m=15.0, user_speed_mps=1.0, mobility_dt_s=0.1)
    rng = np.random.default_rng(2)
    topo = generate_topology(cfg, rng)
    state = mobility.init_mobility(topo, cfg, rng)

    # Place the target just barely closer than one step's travel distance.
    state.topology.user_pos[0] = topo.ris_pos.copy()
    state.targets[0] = topo.ris_pos + np.array([0.05, 0.0])  # 0.05 m < 0.1 m/s*0.1s=0.1m step

    new_state = mobility.step(state, cfg, rng)
    np.testing.assert_allclose(new_state.topology.user_pos[0], state.targets[0])
    # a fresh target should have been drawn (arrival triggers retarget)
    assert not np.allclose(new_state.targets[0], state.targets[0])


def test_step_preserves_shapes_for_multiple_users():
    cfg = SimConfig(n_users=4, d_max_m=15.0, user_speed_mps=1.0, mobility_dt_s=0.1)
    rng = np.random.default_rng(3)
    topo = generate_topology(cfg, rng)
    state = mobility.init_mobility(topo, cfg, rng)
    for _ in range(20):
        state = mobility.step(state, cfg, rng)
    assert state.topology.user_pos.shape == (4, 2)
    assert state.targets.shape == (4, 2)
    assert state.topology.ris_to_user_m.shape == (4,)
    assert state.topology.bs_to_user_m.shape == (4,)
