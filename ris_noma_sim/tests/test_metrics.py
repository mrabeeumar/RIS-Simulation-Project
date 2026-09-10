import numpy as np
import pytest

from ris_noma_sim.core.metrics import (
    ber,
    energy_efficiency_bit_per_j,
    jain_fairness,
    oma_sum_rate_bps_hz,
    outage_indicator,
    system_sum_rate_bps_hz,
)


def test_jain_fairness_equal_rates_is_one():
    rates = np.array([2.0, 2.0, 2.0, 2.0])
    assert jain_fairness(rates) == pytest.approx(1.0)


def test_jain_fairness_unequal_rates_less_than_one():
    rates = np.array([10.0, 0.1, 0.1, 0.1])
    assert jain_fairness(rates) < 1.0


def test_jain_fairness_all_zero_is_trivially_fair():
    rates = np.array([0.0, 0.0, 0.0])
    assert jain_fairness(rates) == pytest.approx(1.0)


def test_outage_indicator():
    rates = np.array([0.2, 0.5, 0.8, 0.49])
    out = outage_indicator(rates, threshold_bps_hz=0.5)
    np.testing.assert_array_equal(out, [True, False, False, True])


def test_energy_efficiency_formula():
    ee = energy_efficiency_bit_per_j(
        r_sum_bps_hz=10.0, tx_power_w=1.0, p_bs_static_w=1.0, n_ris=64, p_element_w=0.005,
    )
    expected = 10.0 / (1.0 + 1.0 + 64 * 0.005)
    assert ee == pytest.approx(expected)


def test_ber_bpsk_zero_sinr_is_half():
    assert ber(np.array([0.0]), "bpsk")[0] == pytest.approx(0.5)


def test_ber_bpsk_decreases_with_sinr():
    b = ber(np.array([0.0, 1.0, 10.0, 100.0]), "bpsk")
    assert all(earlier > later for earlier, later in zip(b, b[1:]))


def test_ber_qpsk_worse_than_bpsk_at_same_sinr_low_regime():
    sinr = np.array([2.0])
    assert ber(sinr, "qpsk")[0] >= ber(sinr, "bpsk")[0]


def test_ber_invalid_modulation_raises():
    with pytest.raises(ValueError):
        ber(np.array([1.0]), "16qam")


def test_system_sum_rate_single_cluster_no_normalization():
    cluster_rates = [np.array([1.0, 2.0])]
    assert system_sum_rate_bps_hz(cluster_rates) == pytest.approx(3.0)


def test_system_sum_rate_two_clusters_normalized_by_n_clusters():
    cluster_rates = [np.array([2.0, 2.0]), np.array([4.0, 4.0])]
    # (4 + 8) / 2 clusters = 6.0
    assert system_sum_rate_bps_hz(cluster_rates) == pytest.approx(6.0)


def test_system_sum_rate_empty_is_zero():
    assert system_sum_rate_bps_hz([]) == 0.0


def test_oma_sum_rate_formula():
    gains = np.array([4.0, 4.0])
    result = oma_sum_rate_bps_hz(gains, tx_power_w=1.0, noise_w=1.0)
    expected = 2 * (0.5 * np.log2(1 + 4.0))
    assert result == pytest.approx(expected)
