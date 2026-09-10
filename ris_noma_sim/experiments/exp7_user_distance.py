"""Experiment 7: user distance vs power allocation vs throughput.
See PLAN.md Section 11.

Near user held fixed at d_min_m; far user's distance is swept from d_min_m to
d_max_m. Reports the resulting NOMA power fraction and rate for each user,
averaged over n_trials fading realizations at each fixed pair of distances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ris_noma_sim.core.channel import effective_channel, generate_channels
from ris_noma_sim.core.config import SimConfig
from ris_noma_sim.core.pairing import pair_users
from ris_noma_sim.core.sic import cluster_rate_bps_hz, cluster_sinr, decoding_order, noise_power_w
from ris_noma_sim.experiments.plotting import line_plot, save_csv
from ris_noma_sim.network.topology import generate_topology
from ris_noma_sim.optimization import max_sumrate

NEAR_USER_DISTANCE_M = 5.0
FAR_USER_DISTANCES_M = [5.0, 8.0, 10.0, 12.0, 15.0]


def run(n_trials: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for far_d in FAR_USER_DISTANCES_M:
        cfg = SimConfig(
            n_ris=64, n_users=2, cluster_size=2, snr_db=15.0, ris_bits=2,
            ris_algo="max_sumrate", power_algo="fair_constrained",
            user_distances_m=(NEAR_USER_DISTANCE_M, far_d), n_trials=n_trials, seed=seed,
        )
        rng = np.random.default_rng(seed)
        topology = generate_topology(cfg, rng)
        noise_w = noise_power_w(cfg.tx_power_w, cfg.snr_db)

        a_near_vals, a_far_vals, rate_near_vals, rate_far_vals = [], [], [], []
        for _ in range(n_trials):
            channels = generate_channels(
                n_ris=cfg.n_ris, n_users=cfg.n_users, m_antennas=1, channel_type=cfg.channel_type,
                rician_k_factor_db=cfg.rician_k_factor, path_loss_exponent=cfg.path_loss_exponent,
                bs_to_ris_m=topology.bs_to_ris_m, ris_to_user_m=topology.ris_to_user_m,
                bs_to_user_m=topology.bs_to_user_m, direct_link_blocked=cfg.direct_link_blocked, rng=rng,
            )
            theta = max_sumrate.optimize(channels, cfg, rng)
            gains = np.abs(effective_channel(channels, theta)[:, 0]) ** 2

            cluster = pair_users(gains, cfg.cluster_size)[0]
            order = decoding_order(gains[cluster])
            ordered_idx = np.array(cluster)[order]
            gains_desc = gains[ordered_idx]

            from ris_noma_sim.optimization.power_allocation import allocate

            a = allocate(gains_desc, cfg)
            sinr = cluster_sinr(gains_desc, a, cfg.tx_power_w, noise_w, cfg.sic_epsilon)
            rates = cluster_rate_bps_hz(sinr)

            # ordered_idx[0] is user 0 (near, index 0 in user_distances_m) if it
            # has the stronger channel, else user 1 (far); map back to near/far.
            near_pos = int(np.where(ordered_idx == 0)[0][0])
            far_pos = int(np.where(ordered_idx == 1)[0][0])
            a_near_vals.append(a[near_pos])
            a_far_vals.append(a[far_pos])
            rate_near_vals.append(rates[near_pos])
            rate_far_vals.append(rates[far_pos])

        rows.append(
            dict(
                far_user_distance_m=far_d,
                a_near_user=float(np.mean(a_near_vals)),
                a_far_user=float(np.mean(a_far_vals)),
                rate_near_user_bps_hz=float(np.mean(rate_near_vals)),
                rate_far_user_bps_hz=float(np.mean(rate_far_vals)),
            )
        )

    df = pd.DataFrame(rows)
    save_csv(df, "exp7_user_distance.csv")
    line_plot(
        df, x="far_user_distance_m", y_cols=["a_near_user", "a_far_user"],
        labels=["Near-user power fraction", "Far-user power fraction"],
        xlabel="Far-user distance from RIS (m)", ylabel="NOMA power fraction",
        title="Experiment 7: User Distance vs Power Allocation", filename="exp7_power_allocation.png",
    )
    line_plot(
        df, x="far_user_distance_m", y_cols=["rate_near_user_bps_hz", "rate_far_user_bps_hz"],
        labels=["Near-user rate", "Far-user rate"],
        xlabel="Far-user distance from RIS (m)", ylabel="Rate (bps/Hz)",
        title="Experiment 7: User Distance vs Throughput", filename="exp7_throughput.png",
    )
    return df


if __name__ == "__main__":
    run()
