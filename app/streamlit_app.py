"""EvoMarks web app:  streamlit run app/streamlit_app.py
Read-only over archive/ and results/ (plus model inference on the Try-it page)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import page_compare
import page_distill
import page_level0
import page_level1
import page_level2
import page_overview
import page_safety
import page_tryit

st.set_page_config(page_title="EvoMarks", page_icon="📝", layout="wide")

PAGES = {"Overview": page_overview, "Try it": page_tryit, "Version vs previous version": page_compare,
         "Distillation": page_distill, "Level 1 · Code evolution": page_level1,
         "Level 0 · Self-training": page_level0, "Level 2 · Improver evolution": page_level2,
         "Safety & limitations": page_safety}

st.sidebar.title("📝 EvoMarks")
st.sidebar.caption("Bounded, sandboxed recursive self-improvement for reading handwritten mark sheets.")
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice].render()
