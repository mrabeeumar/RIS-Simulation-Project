"""Sidebar controls bound directly to SimConfig fields. See PLAN.md Section 12."""

from __future__ import annotations

import streamlit as st

from ris_noma_sim.core.config import SimConfig


def build_sidebar() -> SimConfig | None:
    st.sidebar.header("RIS-NOMA Simulator Controls")

    n_ris = st.sidebar.select_slider("Number of RIS elements", options=[16, 32, 64, 128, 256], value=64)
    n_users = st.sidebar.slider("Number of users", min_value=2, max_value=10, value=4)
    snr_db = st.sidebar.slider("SNR (dB)", min_value=-10.0, max_value=30.0, value=15.0, step=1.0)

    ris_bits_label = st.sidebar.selectbox("RIS phase resolution", ["1-bit", "2-bit", "3-bit", "Continuous"], index=1)
    ris_bits = {"1-bit": 1, "2-bit": 2, "3-bit": 3, "Continuous": "continuous"}[ris_bits_label]

    ris_algo = st.sidebar.selectbox(
        "RIS algorithm",
        ["max_sumrate", "fairness_aware", "max_snr", "random", "fixed"],
        format_func=lambda a: {
            "max_sumrate": "Max Sum Rate", "fairness_aware": "Fairness-Aware",
            "max_snr": "Max-SNR", "random": "Random", "fixed": "Fixed",
        }[a],
        index=0,
    )
    power_algo = st.sidebar.selectbox(
        "Power allocation",
        ["fair_constrained", "inverse_gain", "fixed", "max_sumrate_qos"],
        format_func=lambda a: {
            "fair_constrained": "QoS-Fair (SCA)", "inverse_gain": "Inverse-Gain",
            "fixed": "Fixed Split", "max_sumrate_qos": "Max Sum-Rate",
        }[a],
        index=0,
    )

    channel_type = st.sidebar.selectbox("Channel condition", ["rician", "rayleigh"], format_func=str.capitalize)
    rician_k_factor = 5.0
    if channel_type == "rician":
        rician_k_factor = st.sidebar.slider("Rician K-factor (dB)", 0.0, 15.0, 5.0, step=0.5)

    d_max_m = st.sidebar.slider("Max user distance from RIS (m)", min_value=5.5, max_value=25.0, value=15.0, step=0.5)
    sic_epsilon = st.sidebar.slider("SIC imperfection (epsilon)", 0.0, 1.0, 0.0, step=0.05)
    cluster_size = st.sidebar.selectbox("NOMA cluster size", [2, n_users], format_func=lambda k: f"{k}-user {'pairs' if k == 2 else 'full cluster'}")

    with st.sidebar.expander("Advanced"):
        modulation = st.selectbox("Modulation (for BER)", ["bpsk", "qpsk"], format_func=str.upper)
        outage_threshold = st.slider("Outage rate threshold (bps/Hz)", 0.1, 2.0, 0.5, step=0.1)
        live_n_trials = st.slider("Live Monte-Carlo trials", 20, 300, 100, step=10)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

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
