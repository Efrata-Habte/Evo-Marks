"""Page 5: Level 2, the improver improving itself."""
import ast
import difflib
from collections import defaultdict

import plotly.graph_objects as go
import streamlit as st

from common import ARCH, BLUE, GREEN, GREY, OUT, RED, J, index, require_results


def render() -> None:
    """Level 2 page."""
    require_results()
    st.title("Level 2 · The improver improves itself")
    st.markdown("Every K generations the improver rewrites **its own source** (`improver.py`: operator weights, step size, "
                "parent-selection rule). The new improver is kept only if, with the same budget, it improves the pipeline "
                "**more** than the old one, measured on a separate split (`meta_dev`). Inspired by STOP (Zelikman et al., 2023).")
    st.warning("Honest framing: this is bounded recursion over a few numeric settings, not open-ended self-improvement.")
    imps = [r for r in index() if r["kind"] == "improver"]
    steps = [r for r in imps if r["parent_id"]]
    if not steps:
        st.info("No recursive step was run in this setting.")
        return
    fig = go.Figure()
    fig.add_bar(x=[f"gen {r['gen']}" for r in steps], y=[r.get("utility_old", 0) for r in steps], name="current improver", marker_color=GREY)
    fig.add_bar(x=[f"gen {r['gen']}" for r in steps], y=[r.get("utility", 0) for r in steps], name="proposed improver",
                marker_color=[GREEN if r["accepted"] else RED for r in steps], hovertext=[r["edit"] for r in steps])
    fig.update_layout(barmode="group", title="Utility = mean meta_dev gain per edit from a fixed parent (green = kept, red = rolled back)", height=360)
    st.plotly_chart(fig, width="stretch")
    st.dataframe([{"id": r["id"], "parent": r["parent_id"], "gen": r["gen"], "self-edit": r["edit"], "utility (old)": r.get("utility_old"),
                   "utility (new)": r.get("utility"), "decision": "KEPT" if r["accepted"] else "rolled back", "reason": r["reason"]} for r in steps],
                 hide_index=True, width="stretch")
    chain = [r["id"] for r in imps if r["accepted"]]
    series = defaultdict(list)
    for vid in chain:
        for node in ast.parse((ARCH / vid / "improver.py").read_text()).body:
            if isinstance(node, ast.Assign) and node.targets[0].id == "OPERATOR_WEIGHTS":
                for k, v in ast.literal_eval(node.value).items():
                    series[k].append(v)
    f = go.Figure([go.Scatter(x=chain, y=v, mode="lines+markers", name=k) for k, v in series.items()])
    f.update_layout(title="The improver's own operator weights across kept versions", height=320)
    st.plotly_chart(f, width="stretch")
    pick = st.selectbox("Diff of improver.py for", [r["id"] for r in steps])
    r = next(x for x in steps if x["id"] == pick)
    diff = "\n".join(difflib.unified_diff((ARCH / r["parent_id"] / "improver.py").read_text().splitlines(),
                                          (ARCH / pick / "improver.py").read_text().splitlines(), r["parent_id"], pick, lineterm="", n=1))
    st.code(diff or "(identical)", language="diff")
    summ = J(OUT / "evolution_summary.json")
    if summ:
        fa = go.Figure()
        for key, lab, col in (("l1_l2", "Level 1 + Level 2", BLUE), ("l1_only", "Level 1 only", GREY)):
            for h in summ["runs"].get(key, []):
                fa.add_scatter(x=[g["gen"] for g in h], y=[g["best_dev"] for g in h], name=lab, line=dict(color=col), mode="lines+markers")
        fa.update_layout(title="Ablation: does letting the improver edit itself help? (best dev accuracy so far)", xaxis_title="generation", height=360)
        st.plotly_chart(fa, width="stretch")
        st.caption(f"Runs per variant: {len(summ['runs'].get('l1_l2', []))} / {len(summ['runs'].get('l1_only', []))}. "
                   "Few runs: differences smaller than evaluation noise should not be over-interpreted.")
