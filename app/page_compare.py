"""Page 4 (key demo page): compare a version with its parent, walk the lineage, compare any two."""
import difflib

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import ARCH, BLUE, GREY, GREEN, RED, heatmap, index, metrics_of, pipeline_records, require_results


def source(vid: str) -> str:
    """pipeline.py text of a version."""
    return (ARCH / vid / "pipeline.py").read_text()


def compare(a: str, b: str, label_a: str, label_b: str) -> None:
    """Show version b against version a: metric deltas, per-digit bars, confusions, code diff."""
    ma, mb = metrics_of(a), metrics_of(b)
    if not ma or not mb:
        st.warning("One of the versions has no metrics (it was rejected before evaluation: unsafe, crashed or timed out).")
    else:
        cols = st.columns(5)
        for col, (k, lab, fmt, inv) in zip(cols, (("dev_acc", "Dev accuracy", "{:.4f}", False), ("meta_acc", "meta_dev accuracy", "{:.4f}", False),
                                                  ("sheet_read_rate", "Sheet exact-read rate", "{:.4f}", False),
                                                  ("n_params", "Parameters", "{:,.0f}", True), ("train_time_s", "Train time (s)", "{:.1f}", True))):
            col.metric(f"{lab} ({label_b})", fmt.format(mb[k]), f"{mb[k] - ma[k]:+.4g}", delta_color="inverse" if inv else "normal")
        fig = go.Figure()
        fig.add_bar(x=list(range(10)), y=ma["per_digit_acc"], name=label_a, marker_color=GREY)
        fig.add_bar(x=list(range(10)), y=mb["per_digit_acc"], name=label_b, marker_color=BLUE)
        fig.update_layout(barmode="group", title="Per-digit dev accuracy", xaxis_title="digit", yaxis=dict(range=[0.5, 1]), height=330)
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(heatmap(ma["confusion"], f"Confusion: {label_a}"), width="stretch")
        c2.plotly_chart(heatmap(mb["confusion"], f"Confusion: {label_b}"), width="stretch")
    diff = "\n".join(difflib.unified_diff(source(a).splitlines(), source(b).splitlines(), label_a, label_b, lineterm="", n=2))
    st.subheader("Code diff of pipeline.py")
    st.code(diff or "(identical source)", language="diff")


def render() -> None:
    """Version-vs-previous-version page."""
    require_results()
    st.title("Version vs previous version")
    recs = pipeline_records()
    ids = [r["id"] for r in recs]
    by = {r["id"]: r for r in recs}
    best = max((r for r in recs if r["accepted"]), key=lambda r: r.get("dev_acc") or 0)["id"]
    any_two = st.toggle("Compare any two versions instead")
    if any_two:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Version A", ids, index=0)
        b = c2.selectbox("Version B", ids, index=ids.index(best))
        compare(a, b, a, b)
        return
    vid = st.selectbox("Version", ids, index=ids.index(best), help="Default = best accepted version")
    r = by[vid]
    st.markdown(f"**{vid}** · {'✅ accepted' if r['accepted'] else '❌ rejected (' + r['reason'] + ')'} · generation {r['gen']}")
    st.info(f"Edit: `{r['edit']}`")
    if not r["parent_id"]:
        st.write("This is the root version v000 (the hand-written starting point); it has no parent.")
        return
    compare(r["parent_id"], vid, f"{r['parent_id']} (parent)", vid)
    st.subheader("Walk the lineage")
    chain = []
    cur = vid
    while cur:
        chain.append(cur)
        cur = by[cur]["parent_id"]
    chain = chain[::-1]
    fig = go.Figure(go.Scatter(x=chain, y=[by[c].get("dev_acc") for c in chain], mode="lines+markers+text", line=dict(color=BLUE),
                               text=[f"{by[c].get('dev_acc', 0):.3f}" for c in chain], textposition="top center",
                               hovertext=[by[c]["edit"] for c in chain]))
    fig.update_layout(title="Dev accuracy along the lineage v000 -> ... -> selected (hover = edit)", height=320)
    st.plotly_chart(fig, width="stretch")
    st.table([{"version": c, "edit": by[c]["edit"], "dev_acc": by[c].get("dev_acc")} for c in chain])
