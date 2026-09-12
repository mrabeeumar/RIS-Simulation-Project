"""Sidebar controls bound directly to SimConfig fields. See PLAN.md Section 12."""

from __future__ import annotations

import streamlit as st

from ris_noma_sim.core.config import SimConfig


def build_sidebar() -> SimConfig | None:
    st.sidebar.header("RIS-NOMA Simulator Controls")

    n_ris = st.sidebar.select_slider(
        "Number of RIS elements",
        options=[16, 32, 64, 128, 256],
        value=64,
        help=(
            "Number of reconfigurable reflecting elements on the RIS surface. "
            "**Increase**: stronger beamforming gain, higher SNR/rate and better "
            "outage performance, but more static power drawn by the RIS and "
            "slower phase-optimization solve time. **Decrease**: weaker "
            "reflected signal, so users -- especially far ones -- get lower "
            "rate and higher outage, but the RIS uses less power and optimizes faster."
        ),
    )
    n_users = st.sidebar.slider(
        "Number of users",
        min_value=2,
        max_value=10,
        value=4,
        help=(
            "Number of users served by the base station via NOMA. "
            "**Increase**: more users share the same time/frequency resource, "
            "so per-user rate and fairness typically drop and clusters get "
            "bigger (or more numerous); total system sum-rate can still rise. "
            "**Decrease**: fewer users to serve, so each gets more rate and "
            "SIC/interference structure is simpler."
        ),
    )
    snr_db = st.sidebar.slider(
        "SNR (dB)",
        min_value=-10.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help=(
            "Transmit SNR: the ratio of transmit power to noise power, in dB "
            "(P is fixed; this controls the noise floor N0). "
            "**Increase**: less noise relative to signal, so rate goes up and "
            "outage/BER go down. **Decrease**: noisier channel, so rate drops "
            "and outage/BER rise, most severely for far or weak users."
        ),
    )

    ris_bits_label = st.sidebar.selectbox(
        "RIS phase resolution",
        ["1-bit", "2-bit", "3-bit", "Continuous"],
        index=1,
        help=(
            "Number of discrete phase-shift states each RIS element can take "
            "(1-bit = 2 states, 2-bit = 4, 3-bit = 8, Continuous = unlimited). "
            "**Increase resolution**: phase shifts can more closely match the "
            "ideal continuous solution, so beamforming gain and rate improve "
            "and approach the continuous-phase upper bound. **Decrease "
            "resolution**: coarser phase quantization, so some beamforming "
            "gain is lost and rate/SINR degrade, with 1-bit being the "
            "roughest approximation."
        ),
    )
    ris_bits = {"1-bit": 1, "2-bit": 2, "3-bit": 3, "Continuous": "continuous"}[ris_bits_label]

    ris_algo = st.sidebar.selectbox(
        "RIS algorithm",
        ["max_sumrate", "fairness_aware", "max_snr", "random", "fixed"],
        format_func=lambda a: {
            "max_sumrate": "Max Sum Rate", "fairness_aware": "Fairness-Aware",
            "max_snr": "Max-SNR", "random": "Random", "fixed": "Fixed",
        }[a],
        index=0,
        help=(
            "Objective the RIS phase-shift optimizer targets. **Max Sum Rate** "
            "maximizes total throughput (can favor already-strong users). "
            "**Fairness-Aware** balances throughput against per-user fairness. "
            "**Max-SNR** maximizes the strongest single link's SNR. **Random** "
            "sets phases with no optimization (worst case, useful as a "
            "baseline). **Fixed** uses a static, non-adaptive phase profile. "
            "Expected ordering: Fixed <= Random <= Max-SNR <= Max-Sum-Rate in "
            "sum-rate performance."
        ),
    )
    power_algo = st.sidebar.selectbox(
        "Power allocation",
        ["fair_constrained", "inverse_gain", "fixed", "max_sumrate_qos"],
        format_func=lambda a: {
            "fair_constrained": "QoS-Fair (SCA)", "inverse_gain": "Inverse-Gain",
            "fixed": "Fixed Split", "max_sumrate_qos": "Max Sum-Rate",
        }[a],
        index=0,
        help=(
            "How transmit power is split between the superposed users in each "
            "NOMA cluster. **QoS-Fair (SCA)** solves for the split that meets "
            "per-user rate targets as evenly as possible. **Inverse-Gain** "
            "gives more power to weaker-channel users (the standard NOMA rule). "
            "**Fixed Split** uses a constant ratio regardless of channel "
            "conditions (simplest, least adaptive). **Max Sum-Rate** "
            "prioritizes total throughput, which can starve weak users of power."
        ),
    )

    channel_type = st.sidebar.selectbox(
        "Channel condition",
        ["rician", "rayleigh"],
        format_func=str.capitalize,
        help=(
            "Small-scale fading model for the wireless links. **Rician** "
            "assumes a dominant line-of-sight path plus scattered multipath "
            "(more stable, higher effective SNR) -- typical when the RIS has "
            "clear sight of the BS and users. **Rayleigh** assumes no "
            "dominant path, pure scattering (more variable, generally worse "
            "and less predictable performance) -- typical in richly "
            "obstructed environments."
        ),
    )
    rician_k_factor = 5.0
    if channel_type == "rician":
        rician_k_factor = st.sidebar.slider(
            "Rician K-factor (dB)",
            0.0,
            15.0,
            5.0,
            step=0.5,
            help=(
                "Ratio of line-of-sight path power to scattered multipath "
                "power, in dB. **Increase**: channel becomes more line-of-sight "
                "dominated -- less fading variance, more predictable and "
                "generally higher rate. **Decrease**: channel behaves more "
                "like Rayleigh fading -- more variance and typically worse "
                "worst-case performance (more outage)."
            ),
        )

    d_max_m = st.sidebar.slider(
        "Max user distance from RIS (m)",
        min_value=5.5,
        max_value=25.0,
        value=15.0,
        step=0.5,
        help=(
            "Farthest a user can be placed from the RIS (users are drawn "
            "randomly between the minimum distance and this value). "
            "**Increase**: users are spread further out, so path loss grows "
            "and average rate/SNR drop, with more users near the coverage edge. "
            "**Decrease**: users stay closer to the RIS, so path loss shrinks "
            "and average rate/SNR improve."
        ),
    )
    sic_epsilon = st.sidebar.slider(
        "SIC imperfection (epsilon)",
        0.0,
        1.0,
        0.0,
        step=0.05,
        help=(
            "Residual interference left over after Successive Interference "
            "Cancellation, as a fraction of the canceled signal's power "
            "(0 = perfect SIC). **Increase**: more leftover interference after "
            "cancellation, so SINR/rate for the stronger-channel users in each "
            "cluster (who do the canceling) degrades and BER/outage rise. "
            "**Decrease toward 0**: cancellation becomes cleaner, so those "
            "users' SINR approaches the perfect-SIC case. The weakest user in "
            "a cluster is unaffected by epsilon, since it does no cancellation."
        ),
    )
    cluster_size = st.sidebar.selectbox(
        "NOMA cluster size",
        [2, n_users],
        format_func=lambda k: f"{k}-user {'pairs' if k == 2 else 'full cluster'}",
        help=(
            "How many users are grouped together into one NOMA superposition "
            "cluster. **2-user pairs**: users are grouped into pairs (simpler "
            "SIC with at most one cancellation step per user, the standard "
            "NOMA setup). **Full cluster**: all users share one cluster "
            "(more spectral reuse but each user may need to cancel multiple "
            "stronger signals, increasing SIC complexity and error "
            "sensitivity)."
        ),
    )

    with st.sidebar.expander("Advanced"):
        modulation = st.selectbox(
            "Modulation (for BER)",
            ["bpsk", "qpsk"],
            format_func=str.upper,
            help=(
                "Modulation scheme used only for the analytic BER calculation. "
                "**BPSK**: 1 bit/symbol, more robust to noise, lower BER at a "
                "given SINR. **QPSK**: 2 bits/symbol, doubles the raw bit rate "
                "per symbol but is less robust, giving higher BER at the same "
                "SINR."
            ),
        )
        outage_threshold = st.slider(
            "Outage rate threshold (bps/Hz)",
            0.1,
            2.0,
            0.5,
            step=0.1,
            help=(
                "A user is counted as in outage if its instantaneous rate "
                "falls below this threshold. **Increase**: a stricter bar, so "
                "more trials/users are counted as outages, inflating the "
                "reported outage probability. **Decrease**: a looser bar, so "
                "fewer trials count as outages, making outage probability look "
                "better without the underlying channel changing."
            ),
        )
        live_n_trials = st.slider(
            "Live Monte-Carlo trials",
            20,
            300,
            100,
            step=10,
            help=(
                "Number of independent channel realizations averaged for the "
                "Live tab's metrics. **Increase**: smoother, more statistically "
                "reliable estimates, at the cost of a slower run when you "
                "press Run. **Decrease**: faster runs, but metrics become "
                "noisier and can vary more between runs of the same config."
            ),
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
            help=(
                "Seed for the random number generator that drives channel "
                "generation and noise. Changing it produces a different, but "
                "still fully reproducible, random realization -- useful for "
                "checking whether a result is a general trend or a lucky/unlucky "
                "draw. The same seed with the same config always reproduces "
                "the exact same run."
            ),
        )

    try:
        return SimConfig(
            n_ris=n_ris, n_users=n_users, snr_db=snr_db, ris_bits=ris_bits,
            ris_algo=ris_algo, power_algo=power_algo, channel_type=channel_type,
            rician_k_factor=rician_k_factor, cluster_size=cluster_size,
            d_max_m=d_max_m, sic_epsilon=sic_epsilon, modulation=modulation,
            outage_rate_threshold_bps_hz=outage_threshold, n_trials=live_n_trials, seed=int(seed),
        )
    except ValueError as exc:
        st.sidebar.error(f"Invalid configuration: {exc}")
        return None
