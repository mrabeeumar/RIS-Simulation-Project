import numpy as np
import pytest

from ris_noma_sim.core.channel import effective_channel, generate_channels
from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.core.ris import quantized_phase_levels
from ris_noma_sim.optimization import exhaustive, fairness_aware, fixed_phase, max_snr, max_sumrate, random_phase
from ris_noma_sim.optimization.objective import evaluate, sum_rate_score

_CFG = SimConfig(n_ris=16, n_users=2, cluster_size=2, ris_bits=1, power_algo="inverse_gain", snr_db=15.0)


def _make_channels(rng, n_ris=8, n_users=2):
    return generate_channels(
        n_ris=n_ris, n_users=n_users, m_antennas=1, channel_type="rician",
        rician_k_factor_db=5.0, path_loss_exponent=2.0, bs_to_ris_m=10.0,
        ris_to_user_m=np.full(n_users, 10.0), bs_to_user_m=np.full(n_users, 20.0),
        direct_link_blocked=True, rng=rng,
    )


def test_random_output_is_quantized_and_correct_shape():
    rng = np.random.default_rng(0)
    channels = _make_channels(rng)
    theta = random_phase.optimize(channels, _CFG, rng)
    assert theta.shape == (8,)
    levels = quantized_phase_levels(_CFG.ris_bits)
    for t in theta:
        assert np.any(np.isclose(t, levels))


def test_fixed_is_all_zero():
    rng = np.random.default_rng(0)
    channels = _make_channels(rng)
    theta = fixed_phase.optimize(channels, _CFG, rng)
    np.testing.assert_allclose(theta, 0.0)


def test_max_snr_beats_fixed_for_single_user_gain():
    rng = np.random.default_rng(1)
    channels = _make_channels(rng, n_users=1)
    # cfg.n_users/cluster_size are unused by max_snr.optimize/effective_channel
    # (they only need channels + cfg.ris_bits), so a valid-but-unrelated
    # 2-user config is fine here; only ris_bits matters for this test.
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, ris_bits="continuous", power_algo="inverse_gain")
    theta_fixed = np.zeros(8)
    theta_max_snr = max_snr.optimize(channels, cfg)

    g_fixed = np.abs(effective_channel(channels, theta_fixed)[0, 0]) ** 2
    g_max_snr = np.abs(effective_channel(channels, theta_max_snr)[0, 0]) ** 2
    assert g_max_snr >= g_fixed


def test_n_ris_zero_returns_empty_for_all_algorithms():
    rng = np.random.default_rng(0)
    channels = _make_channels(rng, n_ris=0)
    cfg = SimConfig(n_ris=0, n_users=2, cluster_size=2, direct_link_blocked=False)
    for algo in (random_phase, fixed_phase, max_snr, max_sumrate, fairness_aware):
        theta = algo.optimize(channels, cfg, rng)
        assert theta.shape == (0,)


@pytest.mark.parametrize("bits", [1, 2])
def test_max_sumrate_matches_or_beats_exhaustive_within_tolerance(bits):
    rng = np.random.default_rng(2)
    n_ris = 5
    channels = _make_channels(rng, n_ris=n_ris)
    cfg = SimConfig(n_ris=16, n_users=2, cluster_size=2, ris_bits=bits, power_algo="inverse_gain", snr_db=10.0)
    # n_ris field on cfg isn't used by these algorithms directly (channels carry
    # the true element count); only ris_bits/power_algo/snr_db matter here.
    theta_best = exhaustive.optimize(channels, cfg)
    theta_search = max_sumrate.optimize(channels, cfg, rng)

    best_rate = sum_rate_score(evaluate(channels, theta_best, cfg), cfg)
    search_rate = sum_rate_score(evaluate(channels, theta_search, cfg), cfg)
    # Multi-start coordinate ascent (Max-SNR seed + 3 random restarts, PLAN.md
    # Sec 7) should get within 1% of the true optimum for such a small N.
    assert search_rate >= best_rate * 0.99 or search_rate == pytest.approx(best_rate, abs=1e-9)


def test_statistical_ordering_baselines_le_maxsnr_le_maxsumrate():
    """PLAN.md Sec 13: max(Fixed, Random) <= Max-SNR <= Max-Sum-Rate. Fixed
    and Random are NOT compared against each other -- under the
    per-element-random-LOS-phase channel model (Sec 4) they are both
    "non-intelligent" baselines with statistically indistinguishable expected
    performance, confirmed empirically during implementation."""
    n_trials = 60
    n_ris = 32
    cfg = SimConfig(n_ris=n_ris, n_users=2, cluster_size=2, ris_bits=1, power_algo="inverse_gain", snr_db=15.0)
    rates = {"fixed": [], "random": [], "max_snr": [], "max_sumrate": []}
    for seed in range(n_trials):
        rng = np.random.default_rng(2000 + seed)
        channels = _make_channels(rng, n_ris=n_ris)

        theta_fixed = fixed_phase.optimize(channels, cfg, rng)
        theta_random = random_phase.optimize(channels, cfg, rng)
        theta_max_snr = max_snr.optimize(channels, cfg)
        theta_max_sumrate = max_sumrate.optimize(channels, cfg, rng)

        rates["fixed"].append(sum_rate_score(evaluate(channels, theta_fixed, cfg), cfg))
        rates["random"].append(sum_rate_score(evaluate(channels, theta_random, cfg), cfg))
        rates["max_snr"].append(sum_rate_score(evaluate(channels, theta_max_snr, cfg), cfg))
        rates["max_sumrate"].append(sum_rate_score(evaluate(channels, theta_max_sumrate, cfg), cfg))

    means = {k: float(np.mean(v)) for k, v in rates.items()}
    baseline = max(means["fixed"], means["random"])
    assert baseline <= means["max_snr"] * 1.05
    assert means["max_snr"] <= means["max_sumrate"] * 1.05


def test_fairness_aware_does_not_crash_and_is_valid():
    rng = np.random.default_rng(3)
    channels = _make_channels(rng, n_ris=8, n_users=4)
    cfg = SimConfig(n_ris=16, n_users=4, cluster_size=2, ris_bits=1, power_algo="inverse_gain", fairness_alpha=0.5, fairness_beta=0.5)
    theta = fairness_aware.optimize(channels, cfg, rng)
    assert theta.shape == (8,)
    levels = quantized_phase_levels(cfg.ris_bits)
    for t in theta:
        assert np.any(np.isclose(t, levels))
