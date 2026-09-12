"""Live Simulation tab: recomputes only when the user presses Run, cached by
the full frozen SimConfig. See PLAN.md Section 12."""

from __future__ import annotations

import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.dashboard.components.plot_utils import figure_download_button
from ris_noma_sim.dashboard.components.topology_plot import render_topology_figure
from ris_noma_sim.network.controller import BatchResult, NetworkController

_RESULT_KEY = "live_tab_result"
_TOPOLOGY_KEY = "live_tab_topology"
_CONFIG_KEY = "live_tab_config"
_SOLVE_TIME_KEY = "live_tab_solve_time_s"
_PREV_RESULT_KEY = "live_tab_prev_result"
_PREV_CONFIG_KEY = "live_tab_prev_config"


@st.cache_data(show_spinner="Running Monte-Carlo simulation...")
def _run_batch(config: SimConfig) -> BatchResult:
    controller = NetworkController(config, np.random.default_rng(config.seed))
    return controller.simulate_batch()


@st.cache_data(show_spinner=False)
def _get_topology(config: SimConfig):
    controller = NetworkController(config, np.random.default_rng(config.seed))
    return controller.topology


def _per_user_dataframe(config: SimConfig, result: BatchResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "User": [f"U{i + 1}" for i in range(config.n_users)],
            "Rate (bps/Hz)": result.avg_per_user_rate_bps_hz,
            "SINR (dB)": 10.0 * np.log10(np.maximum(result.avg_per_user_sinr, 1e-15)),
            "BER": result.avg_per_user_ber,
            "Outage prob.": result.avg_per_user_outage_probability,
        }
    )


def _summary_dataframe(config: SimConfig, result: BatchResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [
                "Sum-rate (bps/Hz)", "Avg per-user rate (bps/Hz)", "Jain fairness",
                "Energy efficiency (bit/J)", "Outage probability", "Avg BER", "n_trials",
            ],
            "Value": [
                result.sum_rate_bps_hz, result.avg_user_rate_bps_hz, result.jain_fairness_index,
                result.energy_efficiency_bit_per_j, result.outage_probability, result.avg_ber, config.n_trials,
            ],
        }
    )


def _config_diff(a: SimConfig, b: SimConfig) -> list[tuple[str, object, object]]:
    """Fields that differ between two configs, as (field, old_value, new_value)."""
    da, db = asdict(a), asdict(b)
    return [(k, da[k], db[k]) for k in da if da[k] != db[k]]


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
        prev_result = st.session_state.get(_RESULT_KEY)
        prev_config = st.session_state.get(_CONFIG_KEY)
        if prev_result is not None and prev_config is not None and prev_config != config:
            st.session_state[_PREV_RESULT_KEY] = prev_result
            st.session_state[_PREV_CONFIG_KEY] = prev_config

        start = time.perf_counter()
        st.session_state[_RESULT_KEY] = _run_batch(config)
        st.session_state[_SOLVE_TIME_KEY] = time.perf_counter() - start
        st.session_state[_TOPOLOGY_KEY] = _get_topology(config)
        st.session_state[_CONFIG_KEY] = config

    result: BatchResult | None = st.session_state.get(_RESULT_KEY)
    topology = st.session_state.get(_TOPOLOGY_KEY)
    last_config: SimConfig | None = st.session_state.get(_CONFIG_KEY)
    solve_time_s: float | None = st.session_state.get(_SOLVE_TIME_KEY)
    prev_result: BatchResult | None = st.session_state.get(_PREV_RESULT_KEY)
    prev_config: SimConfig | None = st.session_state.get(_PREV_CONFIG_KEY)

    with status_col:
        if result is None:
            st.info("Configure parameters in the sidebar, then press **Run Simulation**.")
        elif last_config != config:
            st.warning("Sidebar configuration has changed. Press **Run Simulation** to update the results below.")
        else:
            msg = "Results below reflect the current configuration."
            if solve_time_s is not None:
                msg += f" (solved in {solve_time_s:.2f}s)"
            st.success(msg)

    if result is None or topology is None:
        return

    col_topo, col_metrics = st.columns([1, 1.4])

    with col_topo:
        topology_fig = render_topology_figure(topology)
        st.pyplot(topology_fig, clear_figure=False)
        figure_download_button(topology_fig, filename="topology.png", key="download_topology")
        plt.close(topology_fig)

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

    st.subheader("Per-User Breakdown")
    per_user_df = _per_user_dataframe(last_config, result)
    table_col, chart_col = st.columns([1, 1])
    with table_col:
        st.dataframe(per_user_df.style.format({
            "Rate (bps/Hz)": "{:.4f}", "SINR (dB)": "{:.2f}", "BER": "{:.4e}", "Outage prob.": "{:.1%}",
        }), width="stretch", hide_index=True)
    with chart_col:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(per_user_df["User"], per_user_df["Rate (bps/Hz)"], color="#1f77b4")
        ax.set_ylabel("Rate (bps/Hz)")
        ax.set_title(f"Average Per-User Rate (n_trials={n})")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=False)
        figure_download_button(fig, filename="per_user_rate.png", key="download_per_user_rate")
        plt.close(fig)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download per-user results (CSV)",
        data=per_user_df.to_csv(index=False),
        file_name="live_per_user_results.csv",
        mime="text/csv",
        key="download_per_user_csv",
        use_container_width=True,
    )
    dl2.download_button(
        "Download summary metrics (CSV)",
        data=_summary_dataframe(last_config, result).to_csv(index=False),
        file_name="live_summary_metrics.csv",
        mime="text/csv",
        key="download_summary_csv",
        use_container_width=True,
    )

    if prev_result is not None and prev_config is not None:
        st.subheader("Compare with Previous Run")
        show_compare = st.checkbox("Show comparison", value=False, key="live_show_compare")
        if show_compare:
            diffs = _config_diff(prev_config, last_config)
            if diffs:
                diff_text = ", ".join(f"**{field}**: {old} -> {new}" for field, old, new in diffs)
                st.caption(f"Changed since previous run: {diff_text}")
            else:
                st.caption("Configuration is identical to the previous run (re-run with the same seed).")

            def _delta(curr: float, prev: float) -> str:
                d = curr - prev
                return f"{d:+.4f}"

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Sum-rate (bps/Hz)", f"{result.sum_rate_bps_hz:.4f}",
                delta=_delta(result.sum_rate_bps_hz, prev_result.sum_rate_bps_hz),
            )
            c2.metric(
                "Jain Fairness", f"{result.jain_fairness_index:.3f}",
                delta=_delta(result.jain_fairness_index, prev_result.jain_fairness_index),
            )
            c3.metric(
                "Outage Probability", f"{result.outage_probability * 100:.1f}%",
                delta=f"{(result.outage_probability - prev_result.outage_probability) * 100:+.1f}pp",
            )

            n_users_common = min(last_config.n_users, prev_config.n_users)
            fig, ax = plt.subplots(figsize=(8, 3))
            x = np.arange(n_users_common)
            width = 0.35
            ax.bar(x - width / 2, prev_result.avg_per_user_rate_bps_hz[:n_users_common], width, label="Previous", color="#aec7e8")
            ax.bar(x + width / 2, result.avg_per_user_rate_bps_hz[:n_users_common], width, label="Current", color="#1f77b4")
            ax.set_xticks(x)
            ax.set_xticklabels([f"U{i + 1}" for i in range(n_users_common)])
            ax.set_ylabel("Rate (bps/Hz)")
            ax.set_title("Per-User Rate: Previous vs. Current Run")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=False)
            figure_download_button(fig, filename="compare_per_user_rate.png", key="download_compare_per_user_rate")
            plt.close(fig)
