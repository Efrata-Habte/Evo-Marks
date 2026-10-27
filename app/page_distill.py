"""Page 6: distillation (accuracy vs latency, T/alpha heatmap, soft-target explorer)."""
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, GREY, ORANGE, OUT, J, NPZ, digit_image, require_results


def render() -> None:
    """Distillation page."""
    require_results()
    st.title("Distillation · teacher to tiny student")
    st.latex(r"L = \alpha\,CE(y,\ student) + (1-\alpha)\,T^2\,KL\big(softmax(teacher/T)\,\|\,softmax(student/T)\big)")
    st.caption("alpha = 1 means no distillation (the hard-label baseline). T > 1 softens the teacher so its 'dark knowledge' "
               "(a 4 that looks a bit like a 9) becomes visible; T^2 restores the gradient scale.")
    kd = J(OUT / "distill_results.json")
    by = defaultdict(list)
    for r in kd["rows"]:
        by[r["name"]].append(r)
    fig = go.Figure()
    for name, rs in by.items():
        col = BLUE if rs[0]["kind"] == "teacher" else ORANGE
        acc = [r["digit_acc"] for r in rs]
        fig.add_scatter(x=[np.mean([r["latency_ms"] for r in rs])], y=[np.mean(acc)], mode="markers+text", text=[name], textposition="top center",
                        textfont=dict(size=9), name=rs[0]["kind"], showlegend=False, error_y=dict(type="data", array=[np.std(acc)], visible=len(rs) > 1),
                        marker=dict(size=max(rs[0]["size_kb"] / 4, 12), color=col, opacity=0.75, line=dict(width=1, color="#333")),
                        hovertext=f"{name}<br>{rs[0]['params']:,} params, {rs[0]['size_kb']:.0f} KB")
    fig.update_layout(title="What do we lose by shrinking? (bubble = KB, bars = std over 3 seeds; blue teachers, orange students)",
                      xaxis_title="CPU latency per sheet (ms, 1 thread)", yaxis_title="test digit accuracy", height=420)
    st.plotly_chart(fig, width="stretch")
    cols = st.columns(2)
    for col, (tk, rows) in zip(cols, kd["sweep"].items()):
        Ts, As = sorted({r["T"] for r in rows}), sorted({r["alpha"] for r in rows})
        z = [[next(r["dev_acc"] for r in rows if r["T"] == T and r["alpha"] == a) for a in As] for T in Ts]
        h = go.Figure(go.Heatmap(z=z, x=[str(a) for a in As], y=[str(t) for t in Ts], colorscale="Viridis", text=[[f"{v:.3f}" for v in row] for row in z],
                                 texttemplate="%{text}"))
        h.update_layout(title=f"Student dev accuracy, teacher {tk} (chosen: T={kd['chosen'][tk]['T']:g}, alpha={kd['chosen'][tk]['alpha']:g})",
                        xaxis_title="alpha", yaxis_title="T", height=330)
        col.plotly_chart(h, width="stretch")
    pd_fig = go.Figure()
    for name, col in (("teacher best (evolved)", BLUE), ("student distilled <- best", ORANGE), ("student hard-label", GREY)):
        pd_fig.add_bar(x=list(range(10)), y=np.mean([r["per_digit_acc"] for r in by[name]], axis=0), name=name, marker_color=col)
    pd_fig.update_layout(barmode="group", title="Where do students lag? (averages hide the tail)", xaxis_title="digit", yaxis=dict(range=[0.5, 1]), height=330)
    st.plotly_chart(pd_fig, width="stretch")

    st.subheader("Soft-target explorer")
    z = NPZ(OUT / "kd_softtargets.npz")
    c1, c2 = st.columns([1, 3])
    i = c1.slider("Test digit #", 0, len(z["labels"]) - 1, 0)
    T = c1.slider("Temperature T", 1.0, 8.0, 4.0, 0.5)
    c1.image(digit_image(z["images"][i], 112), caption=f"true label {z['labels'][i]}")
    p = np.exp(z["logits"][i] / T)
    p = p / p.sum()
    b = go.Figure(go.Bar(x=list(range(10)), y=p, marker_color=[ORANGE if k != z["labels"][i] else BLUE for k in range(10)]))
    b.update_layout(title=f"Teacher probabilities at T = {T:g} (blue = true class)", xaxis=dict(dtick=1, title="class"), yaxis=dict(range=[0, 1]), height=330)
    c2.plotly_chart(b, width="stretch")
