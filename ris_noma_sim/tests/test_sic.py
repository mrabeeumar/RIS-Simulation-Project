import numpy as np
import pytest

from ris_noma_sim.core.sic import cluster_rate_bps_hz, cluster_sinr, decoding_order, noise_power_w


def test_noise_power_formula():
    assert noise_power_w(tx_power_w=1.0, snr_db=0.0) == pytest.approx(1.0)
    assert noise_power_w(tx_power_w=1.0, snr_db=10.0) == pytest.approx(0.1)
    assert noise_power_w(tx_power_w=2.0, snr_db=10.0) == pytest.approx(0.2)


def test_decoding_order_descending_gain():
    gains = np.array([1.0, 5.0, 3.0])
    order = decoding_order(gains)
    np.testing.assert_array_equal(order, [1, 2, 0])


def test_two_user_hand_derived_sinr():
    # position 0 = strong (g=2.0, a=0.3), position 1 = weak (g=0.5, a=0.7)
    gains = np.array([2.0, 0.5])
    a = np.array([0.3, 0.7])
    sinr = cluster_sinr(gains, a, tx_power_w=1.0, noise_w=0.1, sic_epsilon=0.1)
    np.testing.assert_allclose(sinr, [2.5, 1.4], rtol=1e-9)


def test_perfect_sic_epsilon_zero_removes_all_interference_at_strong_position():
    gains = np.array([2.0, 0.5])
    a = np.array([0.3, 0.7])
    sinr = cluster_sinr(gains, a, tx_power_w=1.0, noise_w=0.1, sic_epsilon=0.0)
    expected_strong = 0.3 * 1.0 * 2.0 / 0.1  # no interference at all for position 0
    assert sinr[0] == pytest.approx(expected_strong)


def test_no_sic_epsilon_one_matches_full_interference():
    gains = np.array([2.0, 0.5])
    a = np.array([0.3, 0.7])
    sinr = cluster_sinr(gains, a, tx_power_w=1.0, noise_w=0.1, sic_epsilon=1.0)
    expected_strong = (0.3 * 1.0 * 2.0) / (1.0 * 2.0 * 0.7 + 0.1)
    assert sinr[0] == pytest.approx(expected_strong)


def test_strong_user_sinr_improves_as_epsilon_decreases():
    gains = np.array([2.0, 0.5])
    a = np.array([0.3, 0.7])
    sinrs = [
        cluster_sinr(gains, a, tx_power_w=1.0, noise_w=0.1, sic_epsilon=eps)[0]
        for eps in (1.0, 0.5, 0.2, 0.05, 0.0)
    ]
    assert all(earlier < later for earlier, later in zip(sinrs, sinrs[1:]))


def test_weak_user_sinr_independent_of_epsilon():
    gains = np.array([2.0, 0.5])
    a = np.array([0.3, 0.7])
    sinrs = [
        cluster_sinr(gains, a, tx_power_w=1.0, noise_w=0.1, sic_epsilon=eps)[1]
        for eps in (0.0, 0.1, 0.5, 1.0)
    ]
    assert all(s == pytest.approx(sinrs[0]) for s in sinrs)


def test_singleton_cluster_no_interference():
    gains = np.array([3.0])
    a = np.array([1.0])
    sinr = cluster_sinr(gains, a, tx_power_w=1.0, noise_w=0.1, sic_epsilon=0.3)
    assert sinr[0] == pytest.approx(1.0 * 1.0 * 3.0 / 0.1)


def test_rate_formula():
    sinr = np.array([0.0, 1.0, 3.0])
    rates = cluster_rate_bps_hz(sinr)
    np.testing.assert_allclose(rates, [0.0, 1.0, 2.0])
