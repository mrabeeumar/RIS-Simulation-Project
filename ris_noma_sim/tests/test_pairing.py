import numpy as np
import pytest

from ris_noma_sim.core.pairing import pair_users


def test_all_users_partitioned_exactly_once():
    gains = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    clusters = pair_users(gains, cluster_size=2)
    all_members = sorted(idx for cluster in clusters for idx in cluster)
    assert all_members == list(range(5))


def test_even_users_cluster_size_two_strong_weak_pairing():
    gains = np.array([1.0, 4.0, 3.0, 2.0])  # sorted desc indices: 1(4),2(3),3(2),0(1)
    clusters = pair_users(gains, cluster_size=2)
    assert len(clusters) == 2
    for cluster in clusters:
        assert len(cluster) == 2
    # first cluster pairs global strongest (idx1) with global weakest (idx0)
    assert set(clusters[0]) == {1, 0}
    # second cluster pairs next-strongest (idx2) with next-weakest (idx3)
    assert set(clusters[1]) == {2, 3}


def test_odd_n_users_leaves_singleton_cluster():
    gains = np.array([3.0, 1.0, 2.0])
    clusters = pair_users(gains, cluster_size=2)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_size_equals_n_users_returns_single_cluster():
    gains = np.array([3.0, 1.0, 2.0, 5.0])
    clusters = pair_users(gains, cluster_size=4)
    assert len(clusters) == 1
    assert set(clusters[0]) == {0, 1, 2, 3}


@pytest.mark.parametrize("n_users,cluster_size", [(2, 2), (3, 2), (4, 2), (6, 2), (8, 2), (10, 2), (7, 3)])
def test_various_sizes_partition_correctly(n_users, cluster_size):
    rng = np.random.default_rng(0)
    gains = rng.uniform(0.1, 10.0, size=n_users)
    clusters = pair_users(gains, cluster_size=cluster_size)
    all_members = sorted(idx for cluster in clusters for idx in cluster)
    assert all_members == list(range(n_users))
    assert all(1 <= len(c) <= cluster_size for c in clusters)
