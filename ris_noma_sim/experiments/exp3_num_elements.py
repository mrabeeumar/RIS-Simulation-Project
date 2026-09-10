"""Experiment 3: number of RIS elements vs sum-rate. See PLAN.md Section 11."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.experiments.plotting import line_plot, save_csv
from ris_noma_sim.network.controller import NetworkController

N_RIS_VALUES = [16, 32, 64, 128, 256]


def run(n_trials: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for n_ris in N_RIS_VALUES:
        cfg = SimConfig(
            n_ris=n_ris, n_users=2, cluster_size=2, snr_db=15.0, ris_bits=2,
            ris_algo="max_sumrate", power_algo="fair_constrained", n_trials=n_trials, seed=seed,
        )
        result = NetworkController(cfg, np.random.default_rng(seed)).simulate_batch()
        rows.append(dict(n_ris=n_ris, sum_rate_bps_hz=result.sum_rate_bps_hz))

    df = pd.DataFrame(rows)
    save_csv(df, "exp3_num_elements.csv")
    line_plot(
        df, x="n_ris", y_cols=["sum_rate_bps_hz"], labels=["Max-Sum-Rate RIS"],
        xlabel="Number of RIS elements (N)", ylabel="Sum-rate (bps/Hz)",
        title="Experiment 3: RIS Size vs Sum-Rate", filename="exp3_num_elements.png",
    )
    return df


if __name__ == "__main__":
    run()
