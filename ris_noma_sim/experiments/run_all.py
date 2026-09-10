"""Run all 8 experiments sequentially and write results/summary.md.
See PLAN.md Section 11.

Usage: python -m ris_noma_sim.experiments.run_all [--quick]
"""

from __future__ import annotations

import argparse
import time

from ris_noma_sim.experiments import (
    exp1_ris_gain,
    exp2_resolution,
    exp3_num_elements,
    exp4_num_users,
    exp5_noma_vs_oma,
    exp6_sic_imperfection,
    exp7_user_distance,
    exp8_mobility,
)
from ris_noma_sim.experiments.plotting import RESULTS_DIR, ensure_results_dir

_EXPERIMENTS = [
    ("Experiment 1: RIS Gain", exp1_ris_gain),
    ("Experiment 2: RIS Resolution", exp2_resolution),
    ("Experiment 3: Number of RIS Elements", exp3_num_elements),
    ("Experiment 4: Number of Users", exp4_num_users),
    ("Experiment 5: NOMA vs OMA", exp5_noma_vs_oma),
    ("Experiment 6: SIC Imperfection", exp6_sic_imperfection),
    ("Experiment 7: User Distance", exp7_user_distance),
    ("Experiment 8: Mobility", exp8_mobility),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a small n_trials for a fast smoke run")
    args = parser.parse_args()
    n_trials = 20 if args.quick else 500

    ensure_results_dir()
    summary_lines = ["# Experiment Results Summary\n"]

    for name, module in _EXPERIMENTS:
        print(f"Running {name}...")
        start = time.time()
        if module is exp8_mobility:
            df = module.run()  # exp8 doesn't take n_trials (it's a time-stepped mobility run)
        else:
            df = module.run(n_trials=n_trials)
        elapsed = time.time() - start
        print(f"  done in {elapsed:.1f}s, {len(df)} rows")
        summary_lines.append(f"## {name}\n")
        summary_lines.append("```\n" + df.to_string(index=False) + "\n```\n")

    (RESULTS_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nAll experiments complete. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
