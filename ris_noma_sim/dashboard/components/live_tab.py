"""Live Simulation tab: recomputes on every parameter change, cached by the
full frozen SimConfig. See PLAN.md Section 12."""

from __future__ import annotations

import numpy as np
import streamlit as st

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.dashboard.components.topology_plot import render_topology_figure
from ris_noma_sim.network.controller import BatchResult, NetworkController


@st.cache_data(show_spinner="Running Monte-Carlo simulation...")
def _run_batch(config: SimConfig) -> BatchResult:
    controller = NetworkController(config, np.random.default_rng(config.seed))
    result = controller.simulate_batch()
    # topology isn't picklable-stable across cache reruns trivially; regenerate
    # deterministically for display using the same seed (topology generation
    # consumes the same rng draws as the controller did internally, but a
    # fresh controller with the same seed reproduces the identical topology).
    return result


@st.cache_data(show_spinner=False)
def _get_topology(config: SimConfig):
    controller = NetworkController(config, np.random.default_rng(config.seed))
    return controller.topology


def render_live_tab(config: SimConfig | None) -> None:
    if config is None:
        st.warning("Fix the configuration errors in the sidebar to run the simulation.")
        return

    result = _run_batch(config)
    topology = _get_topology(config)

    col_topo, col_metrics = st.columns([1, 1.4])

    with col_topo:
        st.pyplot(render_topology_figure(topology), clear_figure=True)

    with col_metrics:
        n = config.n_trials
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
    users = [f"U{i+1}" for i in range(config.n_users)]
    ax.bar(users, result.avg_per_user_rate_bps_hz, color="#1f77b4")
    ax.set_ylabel("Rate (bps/Hz)")
    ax.set_title(f"Average Per-User Rate (n_trials={n})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
