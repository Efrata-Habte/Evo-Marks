"""Page 2: Level 0 verified self-training (interactive versions of figures 1-6)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import ARCH, BLUE, GREEN, GREY, ORANGE, RED, NPZ, digit_image, heatmap, metrics_of, require_results


def render() -> None:
    """Level 0 page."""
    require_results()
    st.title("Level 0 · Verified self-training")
    st.markdown("The teacher reads **unlabeled** sheets. A sheet's digits become pseudo-labels only if the read marks "
                "**sum to the read total** and each mark is within its maximum. Accepted sheets train a *copy*; it replaces "
                "the teacher only if it passes the dev gate.")
    run = st.radio("Run", ["best", "v000"], horizontal=True, format_func=lambda r: {"best": "best evolved version", "v000": "v000 (before evolution)"}[r])
    m, conf = metrics_of(run, "full/metrics.json"), metrics_of(run, "full_conf/metrics.json")
    rs = m["rounds"]
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=[0] + [r["round"] for r in rs], y=[m["baseline_dev_acc"]] + [r["dev_acc"] for r in rs],
                               mode="lines", line=dict(color=BLUE), name="teacher"))
    for r in rs:
        ok = r["decision"] == "accept"
        fig.add_trace(go.Scatter(x=[r["round"]], y=[r["candidate_dev_acc"]], mode="markers", name=r["decision"], showlegend=False,
                                 marker=dict(size=13, symbol="circle" if ok else "x", color=GREEN if ok else RED),
                                 hovertext=f"round {r['round']}: {r['decision']} ({r['reason']})"))
    fig.update_layout(title="Did learning from verified sheets help? (1 seed)", xaxis_title="round (0 = seed-only)",
                      yaxis_title="dev digit accuracy", height=380)
    c1.plotly_chart(fig, width="stretch")
    bars = go.Figure()
    xs = [r["round"] for r in rs]
    bars.add_bar(x=xs, y=[r["accepted"] for r in rs], name="accepted", marker_color=GREEN)
    bars.add_bar(x=xs, y=[r["rejected"].get("sum_mismatch", 0) for r in rs], name="rejected: sum_mismatch", marker_color=GREY)
    bars.add_bar(x=xs, y=[r["rejected"].get("mark_exceeds_max", 0) for r in rs], name="rejected: mark_exceeds_max", marker_color=RED)
    bars.update_layout(barmode="stack", title="Accepted vs rejected sheets", xaxis_title="round", height=380)
    c2.plotly_chart(bars, width="stretch")
    c3, c4 = st.columns(2)
    pr = [r for r in rs if r["verifier_precision"] is not None]
    pf = go.Figure()
    pf.add_scatter(x=[r["round"] for r in pr], y=[r["verifier_precision"] for r in pr], name="precision", line=dict(color=BLUE))
    pf.add_bar(x=[r["round"] for r in pr], y=[r["compensating_errors"] for r in pr], name="compensating errors", yaxis="y2", opacity=0.4, marker_color=ORANGE)
    pf.update_layout(title="How often does the sum check accept a wrong label?", yaxis=dict(title="precision"),
                     yaxis2=dict(overlaying="y", side="right", title="compensating-error sheets"), height=380)
    c3.plotly_chart(pf, width="stretch")
    d = np.array(m["confusion"]) - np.array(m["baseline_confusion"])
    c4.plotly_chart(heatmap(d, "Confusion change (after - before): which confusions were fixed?", "RdBu", 0), width="stretch")

    st.subheader("Ablation: arithmetic vs the model's own confidence")
    rows = []
    for lab, mm in (("sum check", m), ("confidence >= 0.9 per digit", conf)):
        n = sum(r["accepted_digits"] for r in mm["rounds"])
        p = sum((r["verifier_precision"] or 0) * r["accepted_digits"] for r in mm["rounds"]) / n if n else None
        rows.append({"method": lab, "digits kept": n, "pseudo-label precision": "n/a" if p is None else round(p, 4),
                     "final dev accuracy": round(mm["dev_acc"], 4)})
    st.dataframe(rows, hide_index=True)

    st.subheader("Round details")
    k = st.select_slider("Round", options=xs, value=xs[-1])
    r = rs[k - 1]
    st.json({x: r[x] for x in r if x != "per_digit_acc"}, expanded=False)
    ex = NPZ(ARCH / run / "full" / "l0_examples.npz")
    if ex is not None and len(ex["r_img"]):
        st.subheader("Hard digits the sum check rescued (teacher confidence < 0.6; first ones found, not cherry-picked)")
        n = min(16, len(ex["r_img"]))
        st.image([digit_image(a) for a in ex["r_img"][:n]], width=64,
                 caption=[f"guess {p} ({c:.2f}) {'✓' if p == t else '✗'}" for p, c, t in zip(ex["r_pred"][:n], ex["r_conf"][:n], ex["r_true"][:n])])
        st.subheader("Sheets rejected for sum_mismatch (✗ = a digit was misread; all ✓ = the total really was wrong)")
        for i in range(min(4, len(ex["j_img"]))):
            st.image([digit_image(a) for a in ex["j_img"][i]], width=64,
                     caption=[f"read {p} {'✓' if p == t else '✗'}" for p, t in zip(ex["j_pred"][i], ex["j_true"][i])])
