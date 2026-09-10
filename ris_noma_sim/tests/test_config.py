import pytest

from ris_noma_sim.core.config import SimConfig


def test_default_config_is_valid():
    SimConfig()


def test_n_ris_zero_is_valid_no_ris_baseline():
    SimConfig(n_ris=0, direct_link_blocked=False)


def test_n_ris_zero_with_blocked_direct_link_is_invalid():
    with pytest.raises(ValueError):
        SimConfig(n_ris=0, direct_link_blocked=True)


@pytest.mark.parametrize("n_ris", [1, 15, 257, -1])
def test_n_ris_out_of_range_rejected(n_ris):
    with pytest.raises(ValueError):
        SimConfig(n_ris=n_ris)


@pytest.mark.parametrize("n_users", [1, 0, 11, -3])
def test_n_users_out_of_range_rejected(n_users):
    with pytest.raises(ValueError):
        SimConfig(n_users=n_users)


@pytest.mark.parametrize("bits", [0, 5, "1bit", 1.5])
def test_invalid_ris_bits_rejected(bits):
    with pytest.raises(ValueError):
        SimConfig(ris_bits=bits)


def test_valid_ris_bits_accepted():
    for bits in (1, 2, 3, 4, "continuous"):
        SimConfig(ris_bits=bits)


@pytest.mark.parametrize("snr_db", [-10.1, 30.1])
def test_snr_out_of_range_rejected(snr_db):
    with pytest.raises(ValueError):
        SimConfig(snr_db=snr_db)


def test_odd_n_users_with_pair_cluster_size_is_valid():
    # Experiment 4 sweeps n_users=3 with the default cluster_size=2; pairing.py
    # handles the leftover user as a singleton cluster, so this must not raise.
    SimConfig(n_users=3, cluster_size=2)


def test_cluster_size_one_rejected():
    with pytest.raises(ValueError):
        SimConfig(n_users=4, cluster_size=1)


def test_cluster_size_equal_n_users_is_valid():
    SimConfig(n_users=5, cluster_size=5)


@pytest.mark.parametrize("epsilon", [-0.01, 1.01])
def test_sic_epsilon_out_of_range_rejected(epsilon):
    with pytest.raises(ValueError):
        SimConfig(sic_epsilon=epsilon)


def test_user_distances_length_mismatch_rejected():
    with pytest.raises(ValueError):
        SimConfig(n_users=3, user_distances_m=(10.0, 20.0))


def test_config_is_frozen_and_hashable():
    cfg = SimConfig()
    with pytest.raises(Exception):
        cfg.n_ris = 128  # type: ignore[misc]
    hash(cfg)  # must not raise
