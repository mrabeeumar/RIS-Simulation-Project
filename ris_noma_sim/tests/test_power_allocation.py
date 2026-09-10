import numpy as np
import pytest

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.core.sic import cluster_rate_bps_hz, cluster_sinr, noise_power_w
from ris_noma_sim.optimization.power_allocation import allocate


def _assert_valid(a: np.ndarray):
    assert a.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(a) >= -1e-9)  # non-decreasing (weaker gets more power)
    assert np.all(a >= 0)


def test_fixed_two_user():
    cfg = SimConfig(power_algo="fixed", power_alloc_fixed_weak=0.7)
    a = allocate(np.array([5.0, 1.0]), cfg)
    np.testing.assert_allclose(a, [0.3, 0.7])
    _assert_valid(a)


def test_fixed_three_user_split():
    cfg = SimConfig(n_users=3, cluster_size=3, power_algo="fixed", power_alloc_fixed_weak=0.6)
    a = allocate(np.array([5.0, 3.0, 1.0]), cfg)
    np.testing.assert_allclose(a, [0.2, 0.2, 0.6])
    _assert_valid(a)


def test_inverse_gain_ordering_and_sum():
    cfg = SimConfig(power_algo="inverse_gain")
    a = allocate(np.array([5.0, 1.0]), cfg)
    _assert_valid(a)
    assert a[0] < a[1]


def test_singleton_cluster_gets_full_power():
    cfg = SimConfig(power_algo="fixed")
    a = allocate(np.array([2.0]), cfg)
    np.testing.assert_allclose(a, [1.0])


def test_fair_constrained_two_user_meets_qos_floor():
    cfg = SimConfig(power_algo="fair_constrained", snr_db=15.0, tx_power_w=1.0, outage_rate_threshold_bps_hz=0.5, sic_epsilon=0.1)
    gains = np.array([4.0, 0.5])
    a = allocate(gains, cfg)
    _assert_valid(a)
    noise = noise_power_w(cfg.tx_power_w, cfg.snr_db)
    sinr = cluster_sinr(gains, a, cfg.tx_power_w, noise, cfg.sic_epsilon)
    rate_weak = cluster_rate_bps_hz(sinr)[1]
    assert rate_weak >= cfg.outage_rate_threshold_bps_hz - 1e-6


def test_fair_constrained_two_user_low_floor_splits_evenly():
    # a very low QoS threshold should make a1_min < 0.5, clipped to 0.5.
    cfg = SimConfig(power_algo="fair_constrained", snr_db=20.0, tx_power_w=1.0, outage_rate_threshold_bps_hz=0.01, sic_epsilon=0.0)
    gains = np.array([4.0, 3.0])
    a = allocate(gains, cfg)
    np.testing.assert_allclose(a, [0.5, 0.5], atol=1e-9)


@pytest.mark.parametrize("k", [3, 4])
def test_fair_constrained_sca_k_users_meets_qos(k):
    rng = np.random.default_rng(0)
    gains = np.sort(rng.uniform(0.5, 5.0, size=k))[::-1]  # descending
    cfg = SimConfig(n_users=k, cluster_size=k, power_algo="fair_constrained", snr_db=15.0, outage_rate_threshold_bps_hz=0.3, sic_epsilon=0.05)
    a = allocate(gains, cfg)
    _assert_valid(a)
    noise = noise_power_w(cfg.tx_power_w, cfg.snr_db)
    sinr = cluster_sinr(gains, a, cfg.tx_power_w, noise, cfg.sic_epsilon)
    rates = cluster_rate_bps_hz(sinr)
    assert np.all(rates >= cfg.outage_rate_threshold_bps_hz - 1e-3)


def test_max_sumrate_qos_valid_output():
    cfg = SimConfig(n_users=4, cluster_size=4, power_algo="max_sumrate_qos", snr_db=15.0)
    gains = np.array([5.0, 3.0, 2.0, 1.0])
    a = allocate(gains, cfg)
    _assert_valid(a)


def test_unknown_power_algo_raises():
    cfg = SimConfig(power_algo="fixed")
    object.__setattr__(cfg, "power_algo", "not_a_real_algo")  # bypass frozen dataclass for this negative test
    with pytest.raises(ValueError):
        allocate(np.array([2.0, 1.0]), cfg)
