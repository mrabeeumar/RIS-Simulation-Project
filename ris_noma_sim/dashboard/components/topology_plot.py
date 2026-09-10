"""Topology schematic (BS/RIS/user positions). See PLAN.md Section 12."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ris_noma_sim.network.topology import Topology


def render_topology_figure(topology: Topology):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(*topology.bs_pos, s=200, marker="^", color="#1f77b4", label="Base Station", zorder=3)
    ax.scatter(*topology.ris_pos, s=250, marker="s", color="#2ca02c", label="RIS", zorder=3)
    ax.scatter(topology.user_pos[:, 0], topology.user_pos[:, 1], s=100, marker="o", color="#d62728", label="Users", zorder=3)

    for i, pos in enumerate(topology.user_pos):
        ax.annotate(f"U{i+1}", pos, textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.plot([topology.bs_pos[0], topology.ris_pos[0]], [topology.bs_pos[1], topology.ris_pos[1]], "--", color="gray", alpha=0.6, zorder=1)
    for pos in topology.user_pos:
        ax.plot([topology.ris_pos[0], pos[0]], [topology.ris_pos[1], pos[1]], "-", color="#2ca02c", alpha=0.3, zorder=1)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Network Topology")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
