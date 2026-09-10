"""Experiment 6: perfect vs imperfect SIC (sic_epsilon sweep) vs
BER/SINR/throughput/outage. See PLAN.md Section 11."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.experiments.plotting import grid_plot, save_csv
from ris_noma_sim.network.controller import NetworkController

SIC_EPSILON_VALUES = [0.0, 0.05, 0.10, 0.20]


def run(n_trials: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for eps in SIC_EPSILON_VALUES:
        cfg = SimConfig(
            n_ris=64, n_users=2, cluster_size=2, snr_db=15.0, ris_bits=2, sic_epsilon=eps,
            ris_algo="max_sumrate", power_algo="fair_constrained", n_trials=n_trials, seed=seed,
        )
        result = NetworkController(cfg, np.random.default_rng(seed)).simulate_batch()
        rows.append(
            dict(
                sic_epsilon=eps,
                avg_ber=result.avg_ber,
                throughput_bps_hz=result.sum_rate_bps_hz,
                outage_probability=result.outage_probability,
            )
        )

    df = pd.DataFrame(rows)
    save_csv(df, "exp6_sic_imperfection.csv")
    grid_plot(
        df, x="sic_epsilon",
        panels=[
            ("avg_ber", "Average BER"),
            ("throughput_bps_hz", "Throughput / sum-rate (bps/Hz)"),
            ("outage_probability", "Outage probability"),
        ],
        xlabel="SIC residual interference fraction (epsilon)",
        title="Experiment 6: Perfect vs Imperfect SIC",
        filename="exp6_sic_imperfection.png",
        ncols=3,
    )
    return df


if __name__ == "__main__":
    run()
