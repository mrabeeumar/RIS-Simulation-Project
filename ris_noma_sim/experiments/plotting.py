"""Shared plotting helper so every experiment's figures share one style.
See PLAN.md Section 11.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for experiment scripts / CI
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_results_dir()
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def line_plot(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    xscale: str = "linear",
) -> Path:
    ensure_results_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (y_col, label) in enumerate(zip(y_cols, labels)):
        ax.plot(
            df[x], df[y_col], label=label,
            color=_COLORS[i % len(_COLORS)], marker=_MARKERS[i % len(_MARKERS)],
            markersize=6, linewidth=1.8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def grid_plot(
    df: pd.DataFrame,
    x: str,
    panels: list[tuple[str, str]],  # (y_col, ylabel) per panel
    xlabel: str,
    title: str,
    filename: str,
    ncols: int = 2,
) -> Path:
    ensure_results_dir()
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, (y_col, ylabel) in enumerate(panels):
        ax = axes[i]
        ax.plot(df[x], df[y_col], color=_COLORS[i % len(_COLORS)], marker=_MARKERS[i % len(_MARKERS)], linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
