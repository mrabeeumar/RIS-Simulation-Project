"""Experiment 1: does RIS actually help? No-RIS vs Random RIS vs Optimized
RIS, sum-rate vs SNR. See PLAN.md Section 11."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.experiments.plotting import line_plot, save_csv
from ris_noma_sim.network.controller import NetworkController

SNR_RANGE_DB = np.arange(-10, 31, 5)


def run(n_trials: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for snr_db in SNR_RANGE_DB:
        base = dict(n_users=2, cluster_size=2, snr_db=float(snr_db), power_algo="fair_constrained", n_trials=n_trials, seed=seed)

        no_ris_cfg = SimConfig(n_ris=0, direct_link_blocked=False, **base)
        random_cfg = SimConfig(n_ris=64, direct_link_blocked=True, ris_algo="random", **base)
        optimized_cfg = SimConfig(n_ris=64, direct_link_blocked=True, ris_algo="max_sumrate", **base)

        no_ris = NetworkController(no_ris_cfg, np.random.default_rng(seed)).simulate_batch()
        random_ris = NetworkController(random_cfg, np.random.default_rng(seed)).simulate_batch()
        optimized_ris = NetworkController(optimized_cfg, np.random.default_rng(seed)).simulate_batch()

        rows.append(
            dict(
                snr_db=snr_db,
                no_ris_sum_rate=no_ris.sum_rate_bps_hz,
                random_ris_sum_rate=random_ris.sum_rate_bps_hz,
                optimized_ris_sum_rate=optimized_ris.sum_rate_bps_hz,
            )
        )

    df = pd.DataFrame(rows)
    save_csv(df, "exp1_ris_gain.csv")
    line_plot(
        df, x="snr_db",
        y_cols=["no_ris_sum_rate", "random_ris_sum_rate", "optimized_ris_sum_rate"],
        labels=["No RIS", "Random RIS", "Optimized RIS (Max-Sum-Rate)"],
        xlabel="SNR (dB)", ylabel="Sum-rate (bps/Hz)",
        title="Experiment 1: Does RIS Help?", filename="exp1_ris_gain.png",
    )
    return df


if __name__ == "__main__":
    run()
