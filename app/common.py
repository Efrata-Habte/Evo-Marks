"""Shared helpers for the web app: cached loaders, colours, small plotly builders. Read-only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evomarks.archive import default_readers  # noqa: E402
from evomarks.utils import load_json  # noqa: E402

ARCH, OUT = ROOT / "archive", ROOT / "results"
GREEN, GREY, RED, BLUE, ORANGE = "#2a9d5c", "#9aa0a6", "#d1495b", "#3a6ea5", "#e08a2c"


@st.cache_data(show_spinner=False)
def _jload(path: str, mtime: float):
    """Cached JSON read; mtime is part of the cache key so re-running the pipeline refreshes the app."""
    return load_json(Path(path))


def J(path: Path):
    """Load a JSON file (cached) or None if it does not exist."""
    return _jload(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_data(show_spinner=False)
def _npz(path: str, mtime: float) -> dict:
    """Cached .npz read as a dict of arrays."""
    z = np.load(path)
    return {k: z[k] for k in z.files}


def NPZ(path: Path):
    """Load an .npz (cached) or None."""
    return _npz(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_resource(show_spinner="Loading models...")
def readers(stamp: float):
    """v000 / best / student readers (cached per results timestamp)."""
    return default_readers(ARCH, OUT)


def get_readers():
    """Readers keyed by name; refreshed when archive/index.json changes."""
    idx = ARCH / "index.json"
    return readers(idx.stat().st_mtime if idx.exists() else 0.0)


def index() -> list[dict]:
    """archive/index.json records."""
    return J(ARCH / "index.json") or []


def pipeline_records() -> list[dict]:
    """Pipeline version records only."""
    return [r for r in index() if r["kind"] == "pipeline"]


def require_results() -> None:
    """Stop the page with a friendly hint if the pipeline has not been run."""
    if not (ARCH / "index.json").exists() or not (OUT / "distill_results.json").exists():
        st.warning("No results yet. Run `python run_all.py --quick` first, then reload this page.")
        st.stop()


def metrics_of(vid: str, name: str = "metrics.json"):
    """Full metrics of a version, or None."""
    return J(ARCH / vid / name)


def heatmap(z, title: str, colorscale="Blues", zmid=None) -> go.Figure:
    """10x10 confusion-style heatmap with digit axes."""
    fig = go.Figure(go.Heatmap(z=z, colorscale=colorscale, zmid=zmid, x=list(range(10)), y=list(range(10)),
                               text=z, texttemplate="%{text}", textfont={"size": 9}))
    fig.update_layout(title=title, xaxis_title="predicted", yaxis_title="true", yaxis_autorange="reversed",
                      height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def digit_image(a: np.ndarray, size: int = 72) -> np.ndarray:
    """Upscale a 28x28 uint8 digit for display (nearest neighbour keeps pixels crisp)."""
    return np.kron(a, np.ones((size // 28 + 1, size // 28 + 1), dtype=a.dtype))
