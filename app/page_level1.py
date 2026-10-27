"""Page 3: Level 1 code evolution (timeline, lineage tree, edit effectiveness, all versions)."""
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, GREEN, GREY, RED, pipeline_records, require_results


def lineage_figure(recs: list[dict]) -> go.Figure:
    """Parent -> child tree; x = generation, colour = dev accuracy, rejected greyed."""
    by = {r["id"]: r for r in recs}
    kids = defaultdict(list)
    for r in recs:
        if r["parent_id"]:
            kids[r["parent_id"]].append(r["id"])
    ypos, row = {}, [0]

    def walk(v: str) -> None:
        """Depth-first layout: leaves take consecutive rows, parents sit between their children."""
        if not kids[v]:
            ypos[v], row[0] = row[0], row[0] + 1
        else:
            for c in kids[v]:
                walk(c)
            ypos[v] = float(np.mean([ypos[c] for c in kids[v]]))
    walk("v000")
    fig = go.Figure()
    for r in recs:
        if r["parent_id"]:
            p = by[r["parent_id"]]
            fig.add_scatter(x=[p["gen"], r["gen"]], y=[ypos[p["id"]], ypos[r["id"]]], mode="lines", line=dict(color="#cccccc"), showlegend=False, hoverinfo="skip")
    for ok, name, col in ((True, "accepted", None), (False, "rejected", GREY)):
        sel = [r for r in recs if r["accepted"] == ok]
        fig.add_scatter(x=[r["gen"] for r in sel], y=[ypos[r["id"]] for r in sel], mode="markers+text", name=name,
                        text=[r["id"] for r in sel], textposition="middle center", textfont=dict(size=9),
                        marker=dict(size=30, color=col or [r.get("dev_acc") for r in sel], colorscale="Viridis",
                                    showscale=ok, colorbar=dict(title="dev acc") if ok else None, line=dict(width=1, color="#444")),
                        hovertext=[f"{r['id']}<br>{r['edit']}<br>dev {r.get('dev_acc')}<br>{r['reason']}" for r in sel], hoverinfo="text")
    fig.update_layout(title="Lineage: which versions descend from which?", xaxis_title="generation", yaxis=dict(visible=False), height=420)
    return fig


def render() -> None:
    """Level 1 page."""
    require_results()
    st.title("Level 1 · Self-modifying code")
    st.markdown("An **improver** rewrites `pipeline.py` (preprocessing, augmentation, model, training settings) with AST edits. "
                "A **fixed evaluator** scores each version; better versions enter the archive, worse ones are rejected and logged.")
    recs = pipeline_records()
    done = [r for r in recs if r.get("dev_acc") is not None]
    fig = go.Figure()
    best, xs, ys = -1.0, [], []
    for r in done:
        ok = r["accepted"]
        fig.add_scatter(x=[r["gen"]], y=[r["dev_acc"]], mode="markers", showlegend=False,
                        marker=dict(size=12, color=GREEN if ok else "rgba(0,0,0,0)", line=dict(color=GREEN if ok else GREY, width=2)),
                        hovertext=f"{r['id']}: {r['edit']}<br>{'accepted' if ok else 'rejected: ' + r['reason']}", hoverinfo="text")
        best = max(best, r["dev_acc"]) if ok else best
        xs.append(r["gen"]), ys.append(best)
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, shape="hv"), name="best accepted so far")
    fig.update_layout(title="Did self-edited code get better over generations? (filled = accepted, hollow = rejected)",
                      xaxis_title="generation", yaxis_title="dev accuracy (evolution budget)", height=380)
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(lineage_figure(recs), width="stretch")

    by = {r["id"]: r for r in recs}
    ops, blk = defaultdict(list), defaultdict(list)
    for r in done:
        if r["parent_id"] and by[r["parent_id"]].get("dev_acc") is not None:
            d = r["dev_acc"] - by[r["parent_id"]]["dev_acc"]
            ops[r["edit"].split(":")[0]].append(d)
            if r["edit"].startswith(("add_block", "remove_block")):
                blk[r["edit"].split(": ")[1]].append(d)
    c1, c2, c3 = st.columns(3)
    for col, groups, t in ((c1, ops, "edit operator"), (c2, blk, "block edited")):
        keys = sorted(groups)
        f = go.Figure(go.Bar(y=keys, x=[np.mean(groups[k]) for k in keys], orientation="h",
                             marker_color=[GREEN if np.mean(groups[k]) > 0 else RED for k in keys],
                             text=[f"n={len(groups[k])}" for k in keys]))
        f.update_layout(title=f"Which edits helped? by {t}", xaxis_title="mean change in dev accuracy", height=330)
        col.plotly_chart(f, width="stretch")
    cnt = defaultdict(int)
    for r in recs:
        if not r["accepted"]:
            cnt[r["reason"].split(":")[0]] += 1
    f = go.Figure(go.Bar(y=list(cnt), x=list(cnt.values()), orientation="h", marker_color=GREY))
    f.update_layout(title="Why rejected?", height=330)
    c3.plotly_chart(f, width="stretch")
    st.subheader("All versions")
    st.dataframe([{"id": r["id"], "parent": r["parent_id"], "gen": r["gen"], "edit": r["edit"], "dev_acc": r.get("dev_acc"),
                   "meta_acc": r.get("meta_acc"), "decision": "ACCEPTED" if r["accepted"] else "rejected", "reason": r["reason"]} for r in recs],
                 hide_index=True, width="stretch")
