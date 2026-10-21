"""Shared state of the three Ragbench pages: the selected run and question, cached loading, the run picker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from hgrag import ragbench, ragbench_review

PAGES = {"results": "views/rb_results.py", "detail": "views/rb_detail.py", "summary": "views/rb_summary.py"}


@st.cache_data(show_spinner=False)
def _load(run_dir: str, _mtime: float) -> dict[str, Any]:
    rows, problems = ragbench_review.load_results(run_dir)
    return {"rows": rows, "problems": problems, "summary": ragbench_review.load_summary(run_dir)}


def load(run_dir: Path) -> dict[str, Any]:
    """Rows, problems and summary of a run (cached until results.jsonl changes), the review and the diagnoses."""
    data = _load(str(run_dir), (run_dir / "results.jsonl").stat().st_mtime)
    disputed = ragbench.load_disputed()
    rows = data["rows"]
    return {**data, "dir": run_dir, "k": data["summary"].get("k_sections", 10), "disputed": disputed,
            "review": ragbench_review.load_review(run_dir, rows, disputed),
            "diag": {r["id"]: ragbench_review.diagnose(r, disputed) for r in rows}}


def pick_run(*, picker: bool) -> Path | None:
    """The run in session (``rb_run``); with ``picker``, bench then run select boxes that set it."""
    benches = ragbench_review.list_benches()
    if not benches:
        st.info("No ragbench run yet: `uv run hgrag ragbench --docs 30 --negatives 30 -k 10 --generate`.")
        return None
    current = Path(st.session_state["rb_run"]) if st.session_state.get("rb_run") else None
    if picker:
        keys = list(benches)
        cur_key = next((k for k, runs in benches.items() if current in runs), keys[0])
        c1, c2 = st.columns([1, 2])
        key = c1.selectbox("Bench", keys, index=keys.index(cur_key), help="Sample key: papers, seed, negatives")
        runs = benches[key]
        names = [r.name for r in runs]
        idx = names.index(current.name) if current in runs else 0
        name = c2.selectbox("Run", names, index=idx, help="Latest first")
        chosen = runs[names.index(name)]
        if chosen != current:
            st.session_state["rb_run"] = str(chosen)
            st.session_state["rb_qid"] = None
        return chosen
    if current is None or not current.is_dir():
        current = next(iter(benches.values()))[0]
        st.session_state["rb_run"] = str(current)
    st.caption(f"Run `{current.name}` · change it on the Results page.")
    return current


def short(qid: str) -> str:
    return qid[:8]
