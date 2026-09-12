"""RIS-Assisted NOMA Communication Simulator -- Streamlit dashboard.
See PLAN.md Section 12. Launch with:
    streamlit run ris_noma_sim/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from ris_noma_sim.dashboard.components.experiments_tab import render_experiments_tab
from ris_noma_sim.dashboard.components.live_tab import render_live_tab
from ris_noma_sim.dashboard.components.sidebar import build_sidebar

st.set_page_config(
    page_title="RIS-NOMA Wireless Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("RIS-Assisted NOMA Communication Simulator")
st.caption("BS -> RIS -> Users | NOMA superposition + SIC | RIS phase optimization")

config = build_sidebar()

tab_live, tab_experiments = st.tabs(["Live Simulation", "Experiments"])
with tab_live:
    render_live_tab(config)
with tab_experiments:
    render_experiments_tab()
