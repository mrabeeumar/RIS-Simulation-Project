"""Experiment 2: RIS phase resolution vs sum-rate (1/2/3/4-bit/continuous).
See PLAN.md Section 11. Note the resolution sweep here uses the full
1/2/3/4-bit/continuous set directly via config, going beyond the dashboard
slider's 1/2/3-bit/continuous range (PLAN.md Sec 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.experiments.plotting import save_csv
from ris_noma_sim.network.controller import NetworkController

RESOLUTIONS = [1, 2, 3, 4, "continuous"]


def run(n_trials: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for bits in RESOLUTIONS:
        cfg = SimConfig(
            n_ris=64, n_users=2, cluster_size=2, snr_db=15.0, ris_bits=bits,
            ris_algo="max_sumrate", power_algo="fair_constrained", n_trials=n_trials, seed=seed,
        )
        result = NetworkController(cfg, np.random.default_rng(seed)).simulate_batch()
        rows.append(dict(ris_bits=str(bits), sum_rate_bps_hz=result.sum_rate_bps_hz))

    df = pd.DataFrame(rows)
    save_csv(df, "exp2_resolution.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ris_noma_sim.experiments.plotting import RESULTS_DIR, ensure_results_dir

    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["ris_bits"], df["sum_rate_bps_hz"], color="#1f77b4")
    ax.set_xlabel("RIS phase resolution")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.set_title("Experiment 2: RIS Phase Resolution vs Sum-Rate")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_resolution.png", dpi=150)
    plt.close(fig)
    return df


if __name__ == "__main__":
    run()
