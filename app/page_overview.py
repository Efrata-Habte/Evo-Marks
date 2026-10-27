"""Page 1: the story, the diagram and headline numbers."""
import numpy as np
import streamlit as st

from common import OUT, J, require_results


def _mean(rows: list, key: str) -> float:
    """Mean of a metric over the rows of one model (students have 3 seeds)."""
    return float(np.mean([r[key] for r in rows]))


def render() -> None:
    """Overview page."""
    require_results()
    st.title("EvoMarks: a mark-sheet reader that improves itself")
    st.markdown(
        "Exam cover sheets hold handwritten marks per question plus a handwritten total. **EvoMarks** reads a photo, "
        "recognises each digit, checks that the marks add up to the written total and flags sheets that do not. "
        "It is small enough to run offline on a cheap phone. It improves itself on three levels, "
        "always under a **fixed, independent evaluator with rollback** (bounded, sandboxed recursive self-improvement; "
        "only two whitelisted files may be edited).")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontname="Helvetica"];
      sheets [label="unlabeled sheets"]; v [label="Sum-check verifier\\n(arithmetic, not confidence)", fillcolor="#dff3e7"];
      l0 [label="Level 0\\nverified self-training\\n(changes weights)"]; l1 [label="Level 1\\npipeline.py rewrites itself\\n(changes code)"];
      l2 [label="Level 2\\nimprover.py rewrites itself\\n(recursive step)"]; ev [label="FIXED evaluator\\ndev / meta_dev", fillcolor="#fbe9d0"];
      kd [label="Distillation\\nbest teacher -> tiny student", fillcolor="#fbe9d0"]; app [label="This app"];
      sheets -> v -> l0; l0 -> ev; l1 -> l0 [label=" edits"]; ev -> l1 [label=" accept / rollback"]; l2 -> l1 [label=" edits improver"];
      ev -> l2 [label=" utility on meta_dev"]; l0 -> kd -> app; }""")
    kd = J(OUT / "distill_results.json")
    by = {}
    for r in kd["rows"]:
        by.setdefault(r["name"], []).append(r)
    base, mid, stu = by["teacher v000 (+L0)"], by["teacher best (evolved)"], by["student distilled <- best"]
    st.subheader("Headline numbers (synthetic test split, unseen writers)")
    st.caption("Mode: " + ("quick (small budgets, expect noisy differences)" if kd["quick"] else "full"))
    cols = st.columns(3)
    for col, title, rows in zip(cols, ("v000 teacher (+ Level 0)", "Best evolved teacher", "Distilled student (tiny)"), (base, mid, stu)):
        with col:
            st.markdown(f"**{title}**")
            d = lambda k: _mean(rows, k) - _mean(base, k)
            st.metric("Digit accuracy", f"{_mean(rows, 'digit_acc'):.3f}", None if rows is base else f"{d('digit_acc'):+.3f}")
            st.metric("Sheets read exactly (7/7 digits)", f"{_mean(rows, 'exact_read_rate'):.3f}", None if rows is base else f"{d('exact_read_rate'):+.3f}")
            st.metric("Parameters", f"{int(rows[0]['params']):,}", None if rows is base else f"{rows[0]['params'] - base[0]['params']:+,}", delta_color="inverse")
            st.metric("CPU latency per sheet", f"{_mean(rows, 'latency_ms'):.2f} ms", None if rows is base else f"{d('latency_ms'):+.2f} ms", delta_color="inverse")
    st.info("Deltas are relative to the v000 teacher. In quick mode differences of 1-2 points are within noise; "
            "see the analysis report for the 2-standard-error verdicts.")
