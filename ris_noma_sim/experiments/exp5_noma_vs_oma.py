"""Experiment 5: NOMA vs OMA, paired comparison under identical per-trial
channel/RIS realizations. See PLAN.md Section 11."""

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
        cfg = SimConfig(
            n_ris=64, n_users=2, cluster_size=2, snr_db=float(snr_db), ris_bits=2,
            ris_algo="max_sumrate", power_algo="fair_constrained", n_trials=n_trials, seed=seed,
        )
        noma_rate, oma_rate = NetworkController(cfg, np.random.default_rng(seed)).simulate_noma_vs_oma_batch()
        rows.append(dict(snr_db=snr_db, noma_sum_rate=noma_rate, oma_sum_rate=oma_rate))

    df = pd.DataFrame(rows)
    save_csv(df, "exp5_noma_vs_oma.csv")
    line_plot(
        df, x="snr_db", y_cols=["noma_sum_rate", "oma_sum_rate"], labels=["NOMA", "OMA"],
        xlabel="SNR (dB)", ylabel="Sum-rate (bps/Hz)",
        title="Experiment 5: NOMA vs OMA", filename="exp5_noma_vs_oma.png",
    )
    return df


if __name__ == "__main__":
    run()
