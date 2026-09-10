import numpy as np
import pytest

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.network.controller import NetworkController


def test_simulate_batch_returns_valid_ranges():
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, n_trials=20, power_algo="inverse_gain", ris_algo="max_snr")
    controller = NetworkController(cfg, np.random.default_rng(0))
    result = controller.simulate_batch()

    assert result.sum_rate_bps_hz >= 0
    assert 0.0 <= result.jain_fairness_index <= 1.0 + 1e-9
    assert 0.0 <= result.outage_probability <= 1.0
    assert 0.0 <= result.avg_ber <= 0.5 + 1e-9
    assert result.energy_efficiency_bit_per_j >= 0
    assert result.per_trial_sum_rate_bps_hz.shape == (20,)
    assert result.avg_per_user_rate_bps_hz.shape == (2,)
    assert np.all(result.avg_per_user_rate_bps_hz >= 0)


def test_topology_fixed_for_controller_lifetime():
    cfg = SimConfig(n_ris=16, n_users=3, cluster_size=3, n_trials=5, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    pos_before = controller.topology.user_pos.copy()
    controller.simulate_batch()
    controller.simulate_batch()
    np.testing.assert_allclose(controller.topology.user_pos, pos_before)


def test_n_ris_zero_batch_runs():
    cfg = SimConfig(n_ris=0, n_users=2, cluster_size=2, n_trials=10, power_algo="inverse_gain", direct_link_blocked=False)
    controller = NetworkController(cfg, np.random.default_rng(0))
    result = controller.simulate_batch()
    assert result.sum_rate_bps_hz >= 0


def test_simulate_noma_vs_oma_batch_returns_two_nonneg_floats():
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, n_trials=10, power_algo="inverse_gain", ris_algo="max_snr")
    controller = NetworkController(cfg, np.random.default_rng(0))
    noma_rate, oma_rate = controller.simulate_noma_vs_oma_batch()
    assert noma_rate >= 0
    assert oma_rate >= 0


def test_init_dynamic_always_reconfigures_at_t0():
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    state = controller.init_dynamic()
    assert state.time_s == 0.0
    assert state.reconfigured_this_step is True
    assert state.reconfigure_cause == "init"


def test_sinr_drop_trigger_fires_on_forced_large_drop():
    cfg = SimConfig(reconfig_mode="sinr_drop", reconfig_sinr_drop_db=3.0, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    controller._last_reconfig_sinr = np.array([10.0, 10.0])
    controller._last_reconfig_time_s = 0.0

    current_sinr = np.array([10.0 * 10 ** (-6.0 / 10.0), 10.0])  # 6 dB drop on user 0
    should, cause = controller.maybe_reconfigure(current_sinr, current_time_s=1.0)
    assert should is True
    assert cause == "sinr_drop"


def test_sinr_drop_trigger_does_not_fire_on_small_drop():
    cfg = SimConfig(reconfig_mode="sinr_drop", reconfig_sinr_drop_db=3.0, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    controller._last_reconfig_sinr = np.array([10.0, 10.0])
    controller._last_reconfig_time_s = 0.0

    current_sinr = np.array([10.0 * 10 ** (-1.0 / 10.0), 10.0])  # 1 dB drop, below threshold
    should, cause = controller.maybe_reconfigure(current_sinr, current_time_s=1.0)
    assert should is False
    assert cause is None


def test_sinr_drop_trigger_ignores_improvement():
    cfg = SimConfig(reconfig_mode="sinr_drop", reconfig_sinr_drop_db=3.0, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    controller._last_reconfig_sinr = np.array([10.0])
    controller._last_reconfig_time_s = 0.0

    current_sinr = np.array([50.0])  # SINR improved, not dropped
    should, cause = controller.maybe_reconfigure(current_sinr, current_time_s=1.0)
    assert should is False


@pytest.mark.parametrize(
    # maybe_reconfigure() is a pure function of (current_time_s -
    # _last_reconfig_time_s); it does NOT itself advance
    # _last_reconfig_time_s (only step() does, when a reconfig actually
    # fires), so with a fixed baseline of 0.0, elapsed>=period stays true for
    # every t past the first threshold crossing.
    "t,expected",
    [(0.4, False), (0.5, True), (0.6, True), (1.0, True)],
)
def test_periodic_trigger_fires_once_elapsed_exceeds_period(t, expected):
    cfg = SimConfig(reconfig_mode="periodic", reconfig_period_s=0.5, mobility_dt_s=0.1, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    controller._last_reconfig_time_s = 0.0
    should, cause = controller.maybe_reconfigure(np.array([1.0]), current_time_s=t)
    assert should == expected
    if expected:
        assert cause == "periodic"


def test_periodic_trigger_resets_baseline_after_firing_in_step_loop():
    """End-to-end cadence check via step(): with period=0.3s and dt=0.1s,
    reconfiguration should fire roughly every 3rd step, not on every step
    past the first threshold crossing (i.e. the baseline really does reset)."""
    cfg = SimConfig(
        n_ris=16, n_users=2, cluster_size=2, power_algo="inverse_gain",
        reconfig_mode="periodic", reconfig_period_s=0.3, mobility_dt_s=0.1,
    )
    controller = NetworkController(cfg, np.random.default_rng(0))
    controller.init_dynamic()
    fired = [controller.step().reconfigured_this_step for _ in range(9)]
    # steps are at t=0.1..0.9; period=0.3 => fires at t=0.3,0.6,0.9 (steps 3,6,9)
    assert fired == [False, False, True, False, False, True, False, False, True]


def test_step_sequence_advances_time_and_moves_users():
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, power_algo="inverse_gain", mobility_dt_s=0.1, user_speed_mps=1.0)
    controller = NetworkController(cfg, np.random.default_rng(0))
    state0 = controller.init_dynamic()
    pos0 = state0.topology.user_pos.copy()

    states = [controller.step() for _ in range(10)]
    assert states[-1].time_s == pytest.approx(1.0)
    assert not np.allclose(states[-1].topology.user_pos, pos0)  # users actually moved
    for s in states:
        assert s.user_rates_bps_hz.shape == (2,)
        assert isinstance(s.reconfigured_this_step, (bool, np.bool_))


def test_step_before_init_dynamic_raises():
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, power_algo="inverse_gain")
    controller = NetworkController(cfg, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        controller.step()
