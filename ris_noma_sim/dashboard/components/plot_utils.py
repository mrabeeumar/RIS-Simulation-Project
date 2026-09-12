"""Shared helper for offering downloads of rendered plots. See PLAN.md
Section 12."""

from __future__ import annotations

import io

import streamlit as st
from matplotlib.figure import Figure


def figure_download_button(fig: Figure, *, filename: str, key: str, dpi: int = 200) -> None:
    """Render a small download button for a matplotlib figure as PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    st.download_button(
        "Download PNG",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
        key=key,
    )
