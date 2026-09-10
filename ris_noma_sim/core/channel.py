"""BS<->RIS<->user channel generation and the effective end-to-end channel.

Implements PLAN.md Section 4. Distances are supplied by the caller (normally
`network/topology.py`) rather than generated here, so this module has no
dependency on user/RIS placement logic -- it only turns distances + a fading
family into channel coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PATH_LOSS_REFERENCE_M = 1.0  # d0


@dataclass
class ChannelSet:
    """All channel coefficients for one Monte-Carlo trial.

    H_BR: (n_ris, m_antennas) complex -- BS -> RIS.
    h_RU: (n_ris, n_users) complex -- RIS -> user k (column k).
    h_BU: (n_users, m_antennas) complex -- direct BS -> user k (row k).
    """

    H_BR: np.ndarray
    h_RU: np.ndarray
    h_BU: np.ndarray


def _path_loss(distance_m: np.ndarray | float, path_loss_exponent: float) -> np.ndarray:
    """L(d) = (d/d0)^-path_loss_exponent. Returns a linear (not dB) factor."""
    d = np.asarray(distance_m, dtype=float)
    return (d / _PATH_LOSS_REFERENCE_M) ** (-path_loss_exponent)


def _fading(shape: tuple[int, ...], channel_type: str, rician_k_factor_db: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-average-power small-scale fading, CN(0,1) (Rayleigh) or
    Rician with the given K-factor.

    The LOS component's phase varies per element/link (drawn fresh each
    trial from `rng`), NOT a shared constant phase. This simulator does not
    model a physical antenna-array steering vector (element spacing, carrier
    wavelength, angle of arrival are out of scope), so a random per-element
    LOS phase is the standard simplified stand-in for "the deterministic
    component's phase depends on unmodeled array/link geometry". Giving
    every element an IDENTICAL LOS phase (e.g. always 0) would silently make
    theta=0 ("no RIS intelligence") a pre-aligned, near-optimal
    configuration purely as a modeling artifact, defeating the premise of
    every RIS-optimization experiment -- this was caught by
    `test_ris_opt.py::test_statistical_ordering_...` during implementation
    (Fixed was beating Random by 5x on average) and is exactly the failure
    mode this docstring warns against.
    """
    nlos = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    if channel_type == "rayleigh":
        return nlos
    if channel_type == "rician":
        k_linear = 10.0 ** (rician_k_factor_db / 10.0)
        los_phase = rng.uniform(0, 2 * np.pi, size=shape)
        los = np.exp(1j * los_phase)
        return np.sqrt(k_linear / (k_linear + 1.0)) * los + np.sqrt(1.0 / (k_linear + 1.0)) * nlos
    raise ValueError(f"unknown channel_type: {channel_type!r}")


def generate_channels(
    n_ris: int,
    n_users: int,
    m_antennas: int,
    channel_type: str,
    rician_k_factor_db: float,
    path_loss_exponent: float,
    bs_to_ris_m: float,
    ris_to_user_m: np.ndarray,
    bs_to_user_m: np.ndarray,
    direct_link_blocked: bool,
    rng: np.random.Generator,
) -> ChannelSet:
    """Generate one fresh Monte-Carlo trial's channel coefficients.

    `n_ris == 0` is an explicit special case (no-RIS baseline, PLAN.md Sec 4):
    H_BR and h_RU are returned as empty (0, ...) arrays rather than degenerate
    NumPy broadcasting artifacts, and callers (`effective_channel`) must skip
    the RIS term entirely rather than relying on 0-length sums happening to
    work.
    """
    if n_ris == 0:
        H_BR = np.zeros((0, m_antennas), dtype=complex)
        h_RU = np.zeros((0, n_users), dtype=complex)
    else:
        pl_br = _path_loss(bs_to_ris_m, path_loss_exponent)
        H_BR = np.sqrt(pl_br) * _fading((n_ris, m_antennas), channel_type, rician_k_factor_db, rng)

        pl_ru = _path_loss(np.asarray(ris_to_user_m, dtype=float), path_loss_exponent)  # (n_users,)
        ru_fading = _fading((n_ris, n_users), channel_type, rician_k_factor_db, rng)
        h_RU = np.sqrt(pl_ru)[np.newaxis, :] * ru_fading

    if direct_link_blocked:
        h_BU = np.zeros((n_users, m_antennas), dtype=complex)
    else:
        pl_bu = _path_loss(np.asarray(bs_to_user_m, dtype=float), path_loss_exponent)  # (n_users,)
        bu_fading = _fading((n_users, m_antennas), channel_type, rician_k_factor_db, rng)
        h_BU = np.sqrt(pl_bu)[:, np.newaxis] * bu_fading

    return ChannelSet(H_BR=H_BR, h_RU=h_RU, h_BU=h_BU)


def effective_channel(channels: ChannelSet, theta: np.ndarray) -> np.ndarray:
    """h_k = h_BU,k + h_RU,k^H @ Theta @ H_BR, Theta = diag(exp(j*theta)).

    Returns shape (n_users, m_antennas). `theta` must have length n_ris
    (channels.H_BR.shape[0]); for the n_ris==0 special case theta must be a
    length-0 array and the RIS term is skipped entirely.
    """
    n_ris = channels.H_BR.shape[0]
    if theta.shape[0] != n_ris:
        raise ValueError(f"theta length {theta.shape[0]} does not match n_ris {n_ris}")

    if n_ris == 0:
        return channels.h_BU.copy()

    # h_RU^H @ Theta @ H_BR, vectorized over users: for user k,
    # sum_n conj(h_RU[n,k]) * exp(j*theta[n]) * H_BR[n, :]
    weighted = channels.h_RU.conj() * np.exp(1j * theta)[:, np.newaxis]  # (n_ris, n_users)
    ris_term = weighted.T @ channels.H_BR  # (n_users, m_antennas)
    return channels.h_BU + ris_term
