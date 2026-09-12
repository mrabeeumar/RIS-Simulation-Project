"""Topology schematic (BS/RIS/user positions). See PLAN.md Section 12."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ris_noma_sim.network.topology import Topology

_LABEL_OFFSET_PX = 18.0  # radial distance of a label from its marker, in pixels
_LABEL_PAD_PX = 3.0  # extra clearance kept between two label bounding boxes
_REPULSION_ITERS = 300


def _radial_start_positions(display_points: np.ndarray) -> np.ndarray:
    """Initial label anchors: each point's marker position offset radially
    away from the point cloud's centroid, so labels start by pointing outward
    into empty space rather than toward the middle of the plot."""
    n = len(display_points)
    centroid = display_points.mean(axis=0)
    directions = display_points - centroid
    norms = np.linalg.norm(directions, axis=1)

    fallback_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    zero_mask = norms < 1e-6
    directions[zero_mask] = np.stack(
        [np.cos(fallback_angles[zero_mask]), np.sin(fallback_angles[zero_mask])], axis=1
    )
    norms[zero_mask] = 1.0
    unit_dirs = directions / norms[:, None]
    return display_points + unit_dirs * _LABEL_OFFSET_PX


def _resolve_label_overlaps(anchors: np.ndarray, half_sizes: np.ndarray) -> np.ndarray:
    """Iteratively push apart label boxes (each `anchors[i]` +/- `half_sizes[i]`)
    until none overlap, using minimum-translation-vector separation along the
    axis of least overlap. Works for any number/arrangement of labels."""
    anchors = anchors.copy()
    n = len(anchors)
    for _ in range(_REPULSION_ITERS):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = anchors[i] - anchors[j]
                overlap = (half_sizes[i] + half_sizes[j] + _LABEL_PAD_PX) - np.abs(delta)
                if overlap[0] > 0 and overlap[1] > 0:
                    moved = True
                    axis = 0 if overlap[0] < overlap[1] else 1
                    sign = 1.0 if delta[axis] >= 0 else -1.0
                    push = np.zeros(2)
                    push[axis] = sign * overlap[axis] / 2.0
                    anchors[i] += push
                    anchors[j] -= push
        if not moved:
            break
    return anchors


def _place_labels(ax, fig, texts) -> None:
    """Position already-created `ax.text` artists so their bounding boxes
    start radially outward from the marker cloud and never overlap each
    other, regardless of how many points there are or how they're arranged."""
    renderer = fig.canvas.get_renderer()
    display_points = np.array([ax.transData.transform(t.get_position()) for t in texts])
    half_sizes = np.array(
        [(bb.width / 2.0, bb.height / 2.0) for bb in (t.get_window_extent(renderer) for t in texts)]
    )

    anchors = _radial_start_positions(display_points)
    anchors = _resolve_label_overlaps(anchors, half_sizes)

    inv = ax.transData.inverted()
    for text, anchor in zip(texts, anchors):
        text.set_position(inv.transform(anchor))


def render_topology_figure(topology: Topology):
    fig, ax = plt.subplots(figsize=(5, 5.6), constrained_layout=True)
    ax.scatter(*topology.bs_pos, s=200, marker="^", color="#1f77b4", label="Base Station", zorder=3)
    ax.scatter(*topology.ris_pos, s=250, marker="s", color="#2ca02c", label="RIS", zorder=3)
    ax.scatter(topology.user_pos[:, 0], topology.user_pos[:, 1], s=100, marker="o", color="#d62728", label="Users", zorder=3)

    ax.plot([topology.bs_pos[0], topology.ris_pos[0]], [topology.bs_pos[1], topology.ris_pos[1]], "--", color="gray", alpha=0.6, zorder=1)
    for pos in topology.user_pos:
        ax.plot([topology.ris_pos[0], pos[0]], [topology.ris_pos[1], pos[1]], "-", color="#2ca02c", alpha=0.3, zorder=1)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Network Topology")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # A draw is needed before text bounding boxes / transData are usable for
    # the label-placement pass below.
    fig.canvas.draw()

    user_texts = [
        ax.text(*pos, f"U{i+1}", fontsize=9, ha="center", va="center", zorder=4)
        for i, pos in enumerate(topology.user_pos)
    ]
    fig.canvas.draw()
    _place_labels(ax, fig, user_texts)

    # Legend lives entirely below the axes so it can never overlap markers
    # or labels, no matter how the point cloud is shaped.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=True)

    return fig
