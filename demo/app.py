"""Streamlit demo entry point: navigation + language toggle.

Run:  uv run --group demo streamlit run demo/app.py

Three pages — Guide (landing), Process a note, Results — all bilingual via
the sidebar toggle. The note content itself is and stays German, since that
is the care record's language.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DEMO_DIR = Path(__file__).parent
ROOT = DEMO_DIR.parent
for path in (str(DEMO_DIR), str(ROOT / "src"), str(ROOT / "evals")):
    if path not in sys.path:
        sys.path.insert(0, path)

from i18n import t  # noqa: E402
from views import guide, process, results  # noqa: E402

st.set_page_config(page_title="CareNote AI", page_icon="📝", layout="wide")

with st.sidebar:
    st.toggle("🇬🇧 English UI", key="english_ui")

T = t()
st.navigation(
    [
        st.Page(guide.main, title=T["nav"]["guide"], icon="📖", url_path="guide", default=True),
        st.Page(process.main, title=T["nav"]["process"], icon="📝", url_path="process"),
        st.Page(results.main, title=T["nav"]["results"], icon="📊", url_path="results"),
    ]
).run()
