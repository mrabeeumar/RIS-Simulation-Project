"""Live Simulation tab: recomputes only when the user presses Run, cached by
the full frozen SimConfig. See PLAN.md Section 12."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.dashboard.components.topology_plot import render_topology_figure
from ris_noma_sim.network.controller import BatchResult, NetworkController

_RESULT_KEY = "live_tab_result"
_TOPOLOGY_KEY = "live_tab_topology"
_CONFIG_KEY = "live_tab_config"


@st.cache_data(show_spinner="Running Monte-Carlo simulation...")
def _run_batch(config: SimConfig) -> BatchResult:
    controller = NetworkController(config, np.random.default_rng(config.seed))
    return controller.simulate_batch()


@st.cache_data(show_spinner=False)
def _get_topology(config: SimConfig):
    controller = NetworkController(config, np.random.default_rng(config.seed))
    return controller.topology


def render_live_tab(config: SimConfig | None) -> None:
    if config is None:
        st.warning("Fix the configuration errors in the sidebar to run the simulation.")
        return

    run_col, status_col = st.columns([1, 4])
    with run_col:
        run_clicked = st.button(
            "Run Simulation",
            type="primary",
            use_container_width=True,
            help="Runs the Monte-Carlo simulation for the current sidebar configuration.",
        )

    if run_clicked:
        st.session_state[_RESULT_KEY] = _run_batch(config)
        st.session_state[_TOPOLOGY_KEY] = _get_topology(config)
        st.session_state[_CONFIG_KEY] = config

    result: BatchResult | None = st.session_state.get(_RESULT_KEY)
    topology = st.session_state.get(_TOPOLOGY_KEY)
    last_config: SimConfig | None = st.session_state.get(_CONFIG_KEY)

    with status_col:
        if result is None:
            st.info("Configure parameters in the sidebar, then press **Run Simulation**.")
        elif last_config != config:
            st.warning("Sidebar configuration has changed. Press **Run Simulation** to update the results below.")
        else:
            st.success("Results below reflect the current configuration.")

    if result is None or topology is None:
        return

    col_topo, col_metrics = st.columns([1, 1.4])

    with col_topo:
        st.pyplot(render_topology_figure(topology), clear_figure=True)

    with col_metrics:
        n = last_config.n_trials
        st.metric(f"Sum-rate (n_trials={n}, live)", f"{result.sum_rate_bps_hz:.4f} bps/Hz")
        m1, m2, m3 = st.columns(3)
        m1.metric("Jain Fairness", f"{result.jain_fairness_index:.3f}")
        m2.metric("Energy Efficiency", f"{result.energy_efficiency_bit_per_j:.3f} bit/J")
        m3.metric("Outage Probability", f"{result.outage_probability * 100:.1f}%")
        m4, m5 = st.columns(2)
        m4.metric("Avg. per-user rate", f"{result.avg_user_rate_bps_hz:.4f} bps/Hz")
        m5.metric("Avg. BER", f"{result.avg_ber:.4e}")

        st.caption(
            f"Live tab uses n_trials={n} for responsiveness; the Experiments tab's "
            "precomputed figures use 500-1000 trials, so exact numbers may differ "
            "-- this is expected, not an inconsistency (PLAN.md Sec 12)."
        )

    st.subheader("Per-User Rate")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    users = [f"U{i+1}" for i in range(last_config.n_users)]
    ax.bar(users, result.avg_per_user_rate_bps_hz, color="#1f77b4")
    ax.set_ylabel("Rate (bps/Hz)")
    ax.set_title(f"Average Per-User Rate (n_trials={n})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
