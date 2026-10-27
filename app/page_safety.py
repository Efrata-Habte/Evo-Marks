"""Page 8: safety guard-rails and limitations."""
from collections import Counter

import streamlit as st

from common import OUT, J, index, require_results
from evomarks import safety


def render() -> None:
    """Safety & limitations page."""
    require_results()
    st.title("Safety & limitations")
    st.markdown("**Bounded, sandboxed recursive self-improvement.** The course slides note that unrestricted recursive "
                "self-improvement has not been demonstrated; this system edits only two whitelisted files, under a fixed evaluator, with rollback.")
    st.subheader("Who may edit what")
    st.table([{"path": "evolvable/pipeline.py", "access": "EDITABLE (Level 1)"}, {"path": "evolvable/improver.py", "access": "EDITABLE (Level 2, edits itself)"},
              {"path": "archive/, results/", "access": "written by the loop (versions, logs)"},
              {"path": "evomarks/ (evaluator, verifier, safety, data...)", "access": "LOCKED, SHA-256 checked before/after every evaluation"},
              {"path": "data/generated/, tests/, config.py", "access": "LOCKED, SHA-256 checked"}])
    st.subheader("Static check rules (live)")
    st.write(f"Allowed imports (pipeline): `{sorted(safety.ALLOWED_IMPORTS)}`; improver: `{sorted(safety.IMPROVER_IMPORTS)}`. "
             f"Forbidden names: `{sorted(safety.FORBIDDEN_NAMES)}`; forbidden identifier words: `{sorted(safety.FORBIDDEN_TOKENS)}`; no `while`, no dunder access.")
    snippet = st.text_area("Try a candidate snippet", "import os\nos.system('echo hi')", height=90)
    res = safety.check(snippet)
    st.write("static check →", f"`{res.reason}`" if "missing" not in res.reason else "passes the import/name rules (but is missing required definitions)")
    summ = J(OUT / "evolution_summary.json") or {}
    t = summ.get("tamper", {})
    st.subheader("Tamper checks and rejections in this run")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tamper checks performed", t.get("checks", "n/a"))
    c2.metric("Violations", t.get("violations", "n/a"))
    recs = index()
    rej = Counter(r["reason"].split(":")[0] for r in recs if not r["accepted"])
    c3.metric("Rejected versions", sum(rej.values()))
    st.write(dict(rej))
    st.subheader("Limitations")
    st.markdown("""
- **Constrained edit space.** Edits are AST operators over constants and whitelisted blocks: constrained program search, not a code-writing LLM.
- **MNIST is cleaner than phone photos.** Real photos shift the domain (stroke width, lighting, crops); see the final real-sheet figure if it was run.
- **One digit per mark.** Marks above 9 and half marks are not supported; the template is fixed.
- **Compensating errors.** Two misreads that still add up pass the sum check; measured, not eliminated.
- **Dev overfitting.** Selecting the best of many versions on dev makes its dev score optimistic; meta_dev and test numbers are the honest ones.
- **Small budgets and few seeds.** Differences of ~1 point are within noise; evolution used few seeds (see the analysis report).
- **Latency measured on one machine,** single thread, including preprocessing.
- **Static checks are not an OS sandbox.** Candidates run in a subprocess with a timeout and a whitelist; do not run untrusted code with this design.
""")
