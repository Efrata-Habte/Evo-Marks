"""Page 7: read a real photo or a random synthetic sheet with v000, best teacher and student."""
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from common import GREEN, OUT, ROOT, BLUE, GREY, ORANGE, digit_image, get_readers, require_results
from config import MAXIMA
from evomarks.data import load_split
from evomarks.photo import read_photo
from evomarks.sheets import render_sheet
from evomarks.verifier import verify

LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5", "tens", "ones"]
COLORS = {"v000": GREY, "best": BLUE, "student": ORANGE}


@st.cache_data(show_spinner=False)
def test_split():
    """Synthetic test sheets, for the demo only (nothing is selected or tuned from them here)."""
    return load_split("test")


def verdict_banner(name: str, digits: np.ndarray) -> None:
    """Big green tick or red warning for one model."""
    marks, total = [int(d) for d in digits[:5]], int(digits[5]) * 10 + int(digits[6])
    v = verify(marks, total, MAXIMA)
    text = f"**{name}**: marks {marks} → sum **{sum(marks)}**, written total **{total}**"
    if v.accepted:
        st.success(f"✓ Total checks out. {text}")
    else:
        st.error(f"⚠ Check this sheet ({v.reason}). {text}")


def render() -> None:
    """Try-it page."""
    require_results()
    st.title("Try it · read a mark sheet")
    template = ROOT / "template" / "evomarks_template.pdf"
    if template.exists():
        st.download_button("Download printable template (A4 PDF)", template.read_bytes(), "evomarks_template.pdf")
    src = st.radio("Input", ["Upload a photo of a filled template", "Random synthetic test sheet"], horizontal=True)
    truth = None
    if src.startswith("Upload"):
        up = st.file_uploader("Photo (JPG/PNG), all four black corner squares visible", type=["jpg", "jpeg", "png"])
        if up is None:
            st.info("Upload a photo, or switch to a synthetic sheet.")
            return
        res = read_photo(Image.open(up))
        for w in res.warnings:
            st.warning(w)
        crops, shown = res.crops, Image.fromarray(res.warped)
    else:
        sp = test_split()
        if "idx" not in st.session_state or st.button("New random sheet"):
            st.session_state["idx"] = int(np.random.default_rng().integers(len(sp)))
        i = st.session_state["idx"]
        crops, shown, truth = sp.images[i], render_sheet(sp, i), sp.sheet(i)
    c1, c2 = st.columns([1, 2])
    c1.image(shown, caption="aligned sheet" if truth is None else f"synthetic sheet {truth.id}", width="stretch")
    c2.image([digit_image(c) for c in crops], width=70, caption=LABELS)
    models = get_readers()
    out = {}
    for name, rd in models.items():
        t0 = time.perf_counter()
        p = rd.probs(crops)
        out[name] = (p, (time.perf_counter() - t0) * 1000)
    st.subheader("Verdict")
    for name, (p, _) in out.items():
        verdict_banner(name, p.argmax(1))
    st.subheader("What each model read")
    st.dataframe([{"model": n, **{LABELS[j]: f"{p[j].argmax()} ({p[j].max():.0%})" for j in range(7)}, "inference (ms / sheet)": round(ms, 2)}
                  for n, (p, ms) in out.items()], hide_index=True, width="stretch")
    fig = go.Figure()
    for name, (p, _) in out.items():
        for j in range(7):
            fig.add_bar(x=[f"{LABELS[j]}:{d}" for d in range(10)], y=p[j], marker_color=COLORS.get(name, GREEN), showlegend=j == 0, name=name,
                        legendgroup=name, visible=True)
    fig.update_layout(barmode="group", title="Probability of each class for every box (models overlaid)", height=360, xaxis=dict(showticklabels=False))
    with st.expander("Probability bars per box"):
        st.plotly_chart(fig, width="stretch")
    if truth is not None:
        st.caption(f"Revealed after reading: true marks {truth.true_marks.tolist()}, written total {truth.written_total}"
                   + (" (this sheet was genuinely miscounted)" if truth.is_miscounted else ""))
