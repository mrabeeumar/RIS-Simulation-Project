import numpy as np
import pytest

from ris_noma_sim.core.channel import effective_channel, generate_channels
from ris_noma_sim.core.sic import noise_power_w


def _default_channels(n_ris=64, n_users=3, rng=None, **overrides):
    rng = rng or np.random.default_rng(0)
    kwargs = dict(
        n_ris=n_ris,
        n_users=n_users,
        m_antennas=1,
        channel_type="rician",
        rician_k_factor_db=5.0,
        path_loss_exponent=2.2,
        bs_to_ris_m=30.0,
        ris_to_user_m=np.full(n_users, 15.0),
        bs_to_user_m=np.full(n_users, 40.0),
        direct_link_blocked=False,
        rng=rng,
    )
    kwargs.update(overrides)
    return generate_channels(**kwargs)


def test_shapes():
    c = _default_channels(n_ris=64, n_users=3)
    assert c.H_BR.shape == (64, 1)
    assert c.h_RU.shape == (64, 3)
    assert c.h_BU.shape == (3, 1)


def test_rayleigh_average_power_near_pathloss():
    # E[|h|^2] should equal the path-loss factor L(d) for Rayleigh (unit-power fading).
    rng = np.random.default_rng(1)
    n_ris = 1
    n_trials = 20000
    gains = []
    for _ in range(n_trials):
        c = generate_channels(
            n_ris=n_ris, n_users=1, m_antennas=1, channel_type="rayleigh",
            rician_k_factor_db=0.0, path_loss_exponent=2.0, bs_to_ris_m=10.0,
            ris_to_user_m=np.array([10.0]), bs_to_user_m=np.array([10.0]),
            direct_link_blocked=True, rng=rng,
        )
        gains.append(np.abs(c.H_BR[0, 0]) ** 2)
    expected = (10.0 / 1.0) ** -2.0
    assert np.mean(gains) == pytest.approx(expected, rel=0.1)


def test_rician_stronger_los_reduces_variance_relative_to_rayleigh():
    rng_ray = np.random.default_rng(2)
    rng_ric = np.random.default_rng(2)
    n_trials = 5000
    ray_gains, ric_gains = [], []
    for _ in range(n_trials):
        c_ray = generate_channels(
            n_ris=1, n_users=1, m_antennas=1, channel_type="rayleigh",
            rician_k_factor_db=0.0, path_loss_exponent=0.0, bs_to_ris_m=1.0,
            ris_to_user_m=np.array([1.0]), bs_to_user_m=np.array([1.0]),
            direct_link_blocked=True, rng=rng_ray,
        )
        c_ric = generate_channels(
            n_ris=1, n_users=1, m_antennas=1, channel_type="rician",
            rician_k_factor_db=15.0, path_loss_exponent=0.0, bs_to_ris_m=1.0,
            ris_to_user_m=np.array([1.0]), bs_to_user_m=np.array([1.0]),
            direct_link_blocked=True, rng=rng_ric,
        )
        ray_gains.append(np.abs(c_ray.H_BR[0, 0]) ** 2)
        ric_gains.append(np.abs(c_ric.H_BR[0, 0]) ** 2)
    assert np.var(ric_gains) < np.var(ray_gains)


def test_direct_link_blocked_forces_zero_h_bu():
    c = _default_channels(direct_link_blocked=True)
    assert np.all(c.h_BU == 0)


def test_n_ris_zero_special_case_shapes_and_effective_channel():
    c = _default_channels(n_ris=0, direct_link_blocked=False)
    assert c.H_BR.shape == (0, 1)
    assert c.h_RU.shape == (0, 3)
    theta = np.array([])
    h_eff = effective_channel(c, theta)
    np.testing.assert_allclose(h_eff, c.h_BU)


def test_effective_channel_theta_length_mismatch_raises():
    c = _default_channels(n_ris=64, n_users=2)
    with pytest.raises(ValueError):
        effective_channel(c, np.zeros(32))


def test_effective_channel_matches_manual_formula():
    rng = np.random.default_rng(3)
    c = _default_channels(n_ris=8, n_users=2, rng=rng)
    theta = rng.uniform(0, 2 * np.pi, size=8)
    h_eff = effective_channel(c, theta)

    Theta = np.diag(np.exp(1j * theta))
    for k in range(2):
        manual = c.h_BU[k, :] + c.h_RU[:, k].conj().T @ Theta @ c.H_BR
        np.testing.assert_allclose(h_eff[k, :], manual, atol=1e-10)


def test_geometry_sanity_far_user_snr_not_degenerate():
    """Numeric sanity check from PLAN.md Sec 4: at n_ris=64, snr_db=15, with
    per-element phase perfectly aligned to a single user (best-case coherent
    combining), a user at d_max_m must not be floor-clipped to near-zero SINR.
    Averaged over several trials since a single fading draw is noisy; this
    guards against reintroducing a degenerate default topology/path-loss
    combination (an earlier draft of these defaults measured -13 dB here).
    """
    n_ris = 64
    snr_db = 15.0
    tx_power_w = 1.0
    ris_offset_m = 10.0
    d_max_m = 15.0
    trials = 20

    sinr_db_values = []
    for trial in range(trials):
        rng = np.random.default_rng(100 + trial)
        c = generate_channels(
            n_ris=n_ris, n_users=1, m_antennas=1, channel_type="rician",
            rician_k_factor_db=5.0, path_loss_exponent=2.0, bs_to_ris_m=ris_offset_m,
            ris_to_user_m=np.array([d_max_m]), bs_to_user_m=np.array([ris_offset_m + d_max_m]),
            direct_link_blocked=True, rng=rng,
        )
        # Coherent phase alignment for the single user (best-case Max-SNR bound).
        h_ru = c.h_RU[:, 0]
        h_br = c.H_BR[:, 0]
        theta_star = -np.angle(h_ru.conj() * h_br)
        h_eff = effective_channel(c, theta_star)[0, 0]

        n0 = noise_power_w(tx_power_w, snr_db)
        sinr = tx_power_w * np.abs(h_eff) ** 2 / n0
        sinr_db_values.append(10 * np.log10(sinr))

    assert min(sinr_db_values) > -5.0
