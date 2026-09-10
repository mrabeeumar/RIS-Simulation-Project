"""Experiment 4: number of users vs sum-rate/avg rate/fairness/outage/EE.
See PLAN.md Section 11."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.experiments.plotting import grid_plot, save_csv
from ris_noma_sim.network.controller import NetworkController

N_USERS_VALUES = [2, 3, 4, 6, 8, 10]


def run(n_trials: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for n_users in N_USERS_VALUES:
        cfg = SimConfig(
            n_ris=64, n_users=n_users, cluster_size=2, snr_db=15.0, ris_bits=2,
            ris_algo="max_sumrate", power_algo="fair_constrained", n_trials=n_trials, seed=seed,
        )
        result = NetworkController(cfg, np.random.default_rng(seed)).simulate_batch()
        rows.append(
            dict(
                n_users=n_users,
                sum_rate_bps_hz=result.sum_rate_bps_hz,
                avg_user_rate_bps_hz=result.avg_user_rate_bps_hz,
                jain_fairness_index=result.jain_fairness_index,
                outage_probability=result.outage_probability,
                energy_efficiency_bit_per_j=result.energy_efficiency_bit_per_j,
            )
        )

    df = pd.DataFrame(rows)
    save_csv(df, "exp4_num_users.csv")
    grid_plot(
        df, x="n_users",
        panels=[
            ("sum_rate_bps_hz", "Sum-rate (bps/Hz)"),
            ("avg_user_rate_bps_hz", "Avg. per-user rate (bps/Hz)"),
            ("jain_fairness_index", "Jain fairness index"),
            ("outage_probability", "Outage probability"),
            ("energy_efficiency_bit_per_j", "Energy efficiency (bit/J)"),
        ],
        xlabel="Number of users (K)",
        title="Experiment 4: Number of Users vs Performance",
        filename="exp4_num_users.png",
    )
    return df


if __name__ == "__main__":
    run()
