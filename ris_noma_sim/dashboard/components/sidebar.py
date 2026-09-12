"""Sidebar controls bound directly to SimConfig fields. See PLAN.md Section 12."""

from __future__ import annotations

import json
from dataclasses import asdict

import streamlit as st

from ris_noma_sim.core.config import SimConfig

# Defaults for every sidebar-exposed control, keyed by widget session-state
# key. Used by the "Reset to Defaults" button and as the widgets' initial
# values. Kept separate from SimConfig's own dataclass defaults because a
# couple of controls intentionally diverge (e.g. live_n_trials=100 for
# dashboard responsiveness vs. SimConfig's n_trials=500 for experiments).
_DEFAULTS: dict[str, object] = {
    "sb_n_ris": 64,
    "sb_n_users": 4,
    "sb_snr_db": 15.0,
    "sb_ris_bits_label": "2-bit",
    "sb_ris_algo": "max_sumrate",
    "sb_power_algo": "fair_constrained",
    "sb_channel_type": "rician",
    "sb_rician_k_factor": 5.0,
    "sb_d_max_m": 15.0,
    "sb_sic_epsilon": 0.0,
    "sb_cluster_size": 2,
    "sb_modulation": "bpsk",
    "sb_outage_threshold": 0.5,
    "sb_live_n_trials": 100,
    "sb_seed": 42,
}

_RIS_BITS_LABEL_TO_VALUE = {"1-bit": 1, "2-bit": 2, "3-bit": 3, "Continuous": "continuous"}
_RIS_BITS_VALUE_TO_LABEL = {v: k for k, v in _RIS_BITS_LABEL_TO_VALUE.items()}

# Maps a SimConfig field name (as it appears in an exported/imported JSON
# config) to its sidebar widget's session-state key, plus an optional
# converter from the raw field value to the widget's stored value.
_IMPORT_FIELD_MAP: dict[str, tuple[str, object]] = {
    "n_ris": ("sb_n_ris", None),
    "n_users": ("sb_n_users", None),
    "snr_db": ("sb_snr_db", None),
    "ris_bits": ("sb_ris_bits_label", lambda v: _RIS_BITS_VALUE_TO_LABEL.get(v)),
    "ris_algo": ("sb_ris_algo", None),
    "power_algo": ("sb_power_algo", None),
    "channel_type": ("sb_channel_type", None),
    "rician_k_factor": ("sb_rician_k_factor", None),
    "d_max_m": ("sb_d_max_m", None),
    "sic_epsilon": ("sb_sic_epsilon", None),
    "cluster_size": ("sb_cluster_size", None),
    "modulation": ("sb_modulation", None),
    "outage_rate_threshold_bps_hz": ("sb_outage_threshold", None),
    "n_trials": ("sb_live_n_trials", lambda v: min(300, max(20, int(v)))),
    "seed": ("sb_seed", lambda v: min(10_000, max(0, int(v)))),
}


def _apply_loaded_config(data: dict) -> list[str]:
    """Push values from an imported config dict into the sidebar widgets'
    session state (must run before those widgets are instantiated). Returns
    the list of field names that were skipped (missing or invalid)."""
    skipped = []
    for field_name, (widget_key, convert) in _IMPORT_FIELD_MAP.items():
        if field_name not in data:
            continue
        raw_value = data[field_name]
        value = convert(raw_value) if convert else raw_value
        if value is None:
            skipped.append(field_name)
            continue
        st.session_state[widget_key] = value
    # cluster_size must be one of [2, n_users]; clamp after n_users is applied.
    n_users = st.session_state.get("sb_n_users", _DEFAULTS["sb_n_users"])
    if st.session_state.get("sb_cluster_size") not in (2, n_users):
        st.session_state["sb_cluster_size"] = 2
    return skipped


def build_sidebar() -> SimConfig | None:
    # Seed session state before any widget is instantiated. A widget must
    # not be given both `key=` and `value=`/`index=` when the key is already
    # present in session state (Streamlit warns and ignores the latter), so
    # every default lives here instead of being passed to the widgets below.
    for _key, _value in _DEFAULTS.items():
        st.session_state.setdefault(_key, _value)

    st.sidebar.header("RIS-NOMA Simulator Controls")

    with st.sidebar.expander("Load config"):
        uploaded = st.file_uploader("Load config (JSON)", type=["json"], key="sb_config_upload")
        if uploaded is not None:
            file_id = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("_sb_last_loaded_id") != file_id:
                try:
                    data = json.loads(uploaded.getvalue().decode("utf-8"))
                    skipped = _apply_loaded_config(data)
                    st.session_state["_sb_last_loaded_id"] = file_id
                    if skipped:
                        st.warning(f"Loaded config, but couldn't apply: {', '.join(skipped)}")
                    else:
                        st.success("Config loaded.")
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    st.error(f"Could not read config file: {exc}")
        st.caption("Loading applies only to the sidebar-exposed fields below.")

    if st.sidebar.button("Reset to Defaults", use_container_width=True):
        for key, value in _DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

    n_ris = st.sidebar.select_slider(
        "Number of RIS elements",
        options=[16, 32, 64, 128, 256],
        key="sb_n_ris",
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
        key="sb_n_users",
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
        key="sb_snr_db",
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
        key="sb_ris_bits_label",
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
    ris_bits = _RIS_BITS_LABEL_TO_VALUE[ris_bits_label]

    ris_algo = st.sidebar.selectbox(
        "RIS algorithm",
        ["max_sumrate", "fairness_aware", "max_snr", "random", "fixed"],
        format_func=lambda a: {
            "max_sumrate": "Max Sum Rate", "fairness_aware": "Fairness-Aware",
            "max_snr": "Max-SNR", "random": "Random", "fixed": "Fixed",
        }[a],
        key="sb_ris_algo",
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
        key="sb_power_algo",
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
        key="sb_channel_type",
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
    rician_k_factor = _DEFAULTS["sb_rician_k_factor"]
    if channel_type == "rician":
        rician_k_factor = st.sidebar.slider(
            "Rician K-factor (dB)",
            0.0,
            15.0,
            key="sb_rician_k_factor",
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
        key="sb_d_max_m",
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
        key="sb_sic_epsilon",
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
    cluster_options = sorted({2, n_users})
    if st.session_state.get("sb_cluster_size") not in cluster_options:
        st.session_state["sb_cluster_size"] = cluster_options[0]
    cluster_size = st.sidebar.selectbox(
        "NOMA cluster size",
        cluster_options,
        format_func=lambda k: f"{k}-user {'pairs' if k == 2 else 'full cluster'}",
        key="sb_cluster_size",
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
            key="sb_modulation",
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
            key="sb_outage_threshold",
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
            key="sb_live_n_trials",
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
            key="sb_seed",
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
        config = SimConfig(
            n_ris=n_ris, n_users=n_users, snr_db=snr_db, ris_bits=ris_bits,
            ris_algo=ris_algo, power_algo=power_algo, channel_type=channel_type,
            rician_k_factor=rician_k_factor, cluster_size=cluster_size,
            d_max_m=d_max_m, sic_epsilon=sic_epsilon, modulation=modulation,
            outage_rate_threshold_bps_hz=outage_threshold, n_trials=live_n_trials, seed=int(seed),
        )
    except ValueError as exc:
        st.sidebar.error(f"Invalid configuration: {exc}")
        return None

    with st.sidebar.expander("Save config"):
        st.download_button(
            "Save current config (JSON)",
            data=json.dumps(asdict(config), indent=2),
            file_name="ris_noma_config.json",
            mime="application/json",
            key="sb_config_download",
            use_container_width=True,
        )

    return config
