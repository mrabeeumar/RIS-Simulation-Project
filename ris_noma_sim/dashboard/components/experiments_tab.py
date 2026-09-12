"""Experiments tab: loads precomputed CSVs/plots from results/, produced
offline by `python -m ris_noma_sim.experiments.run_all`. Never recomputed
live -- see PLAN.md Section 12."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ris_noma_sim.experiments.plotting import RESULTS_DIR

_EXPERIMENTS = [
    ("Experiment 1: Does RIS Help?", "exp1_ris_gain", ["exp1_ris_gain.png"]),
    ("Experiment 2: RIS Phase Resolution", "exp2_resolution", ["exp2_resolution.png"]),
    ("Experiment 3: Number of RIS Elements", "exp3_num_elements", ["exp3_num_elements.png"]),
    ("Experiment 4: Number of Users", "exp4_num_users", ["exp4_num_users.png"]),
    ("Experiment 5: NOMA vs OMA", "exp5_noma_vs_oma", ["exp5_noma_vs_oma.png"]),
    ("Experiment 6: SIC Imperfection", "exp6_sic_imperfection", ["exp6_sic_imperfection.png"]),
    ("Experiment 7: User Distance", "exp7_user_distance", ["exp7_power_allocation.png", "exp7_throughput.png"]),
    ("Experiment 8: Mobility", "exp8_mobility", ["exp8_mobility.png"]),
]


def render_experiments_tab() -> None:
    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.glob("*.csv")):
        st.info(
            "No precomputed experiment results found. Generate them offline with:\n\n"
            "```\npython -m ris_noma_sim.experiments.run_all\n```\n\n"
            "(add `--quick` for a fast smoke run with fewer Monte-Carlo trials)"
        )
        return

    labels = [name for name, _, _ in _EXPERIMENTS]
    choice = st.selectbox("Experiment", labels)
    _, stem, png_files = next(e for e in _EXPERIMENTS if e[0] == choice)

    csv_path = RESULTS_DIR / f"{stem}.csv"
    for png_name in png_files:
        png_path = RESULTS_DIR / png_name
        if png_path.exists():
            st.image(str(png_path), width="stretch")
            with png_path.open("rb") as f:
                st.download_button(
                    f"Download {png_name}",
                    data=f.read(),
                    file_name=png_name,
                    mime="image/png",
                    key=f"download_{png_name}",
                )

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, width="stretch")
    else:
        st.warning(f"{csv_path.name} not found -- run run_all.py to generate it.")
