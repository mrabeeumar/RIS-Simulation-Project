import numpy as np
import pytest

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.network.topology import generate_topology


def test_bs_and_ris_positions():
    cfg = SimConfig(ris_offset_m=10.0)
    topo = generate_topology(cfg, np.random.default_rng(0))
    np.testing.assert_allclose(topo.bs_pos, [0.0, 0.0])
    np.testing.assert_allclose(topo.ris_pos, [10.0, 0.0])
    assert topo.bs_to_ris_m == 10.0


def test_users_placed_within_annulus():
    cfg = SimConfig(n_users=6, d_min_m=5.0, d_max_m=15.0)
    rng = np.random.default_rng(1)
    topo = generate_topology(cfg, rng)
    assert np.all(topo.ris_to_user_m >= 5.0 - 1e-9)
    assert np.all(topo.ris_to_user_m <= 15.0 + 1e-9)

    computed = np.linalg.norm(topo.user_pos - topo.ris_pos, axis=1)
    np.testing.assert_allclose(computed, topo.ris_to_user_m, atol=1e-9)


def test_bs_to_user_distance_matches_geometry():
    cfg = SimConfig(n_users=3, ris_offset_m=10.0)
    topo = generate_topology(cfg, np.random.default_rng(2))
    computed = np.linalg.norm(topo.user_pos - topo.bs_pos, axis=1)
    np.testing.assert_allclose(computed, topo.bs_to_user_m, atol=1e-9)


def test_user_distances_override_uses_exact_colinear_placement():
    cfg = SimConfig(n_users=3, user_distances_m=(5.0, 10.0, 15.0), ris_offset_m=10.0)
    topo = generate_topology(cfg, np.random.default_rng(3))
    np.testing.assert_allclose(topo.ris_to_user_m, [5.0, 10.0, 15.0])
    # angle=0 => users lie on the positive-x axis beyond the RIS
    np.testing.assert_allclose(topo.user_pos[:, 1], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(topo.user_pos[:, 0], [15.0, 20.0, 25.0], atol=1e-9)
