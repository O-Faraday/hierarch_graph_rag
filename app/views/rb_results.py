"""Ragbench page 1: pick a bench and a run, every question in a sortable table, select one to open its detail."""

import os

import pandas as pd
import ragbench_view as rv
import streamlit as st
import ui

from hgrag import config, ragbench, ragbench_review, refine

ui.inject_css()
ui.page_header(
    "Ragbench results",
    "Every question of an Open RAG Bench run: judge score, retrieval ranks and a rule-based diagnosis. Sort by any "
    "column, filter, then select a row to read the question in detail. Metrics are home-made (5-gram overlap or "
    "same heading, judge 1-5), not Vectara's official ones.",
)
st.session_state.setdefault("rb_run", None)
st.session_state.setdefault("rb_qid", None)

run_dir = rv.pick_run(picker=True)
if run_dir is None:
    st.stop()
data = rv.load(run_dir)
rows, review, diag, s = data["rows"], data["review"], data["diag"], data["summary"]
k = data["k"]
if data["problems"]:
    st.warning(f"{len(data['problems'])} unreadable line(s) in results.jsonl, left out: " + "; ".join(data["problems"][:5]))

cfg = s.get("config", {})
facts = [(s.get("n"), "questions"), (s.get("section_hit_at_1"), "section hit@1"), (s.get(f"section_hit_at_{k}"), f"section hit@{k}"),
         (s.get("mrr"), "MRR"), (s.get("context_recall"), "context recall"), (s.get("mean_score"), "mean score /5"),
         (s.get("pct_score_4_plus"), "share ≥ 4")]
if "mean_refined_score" in s:
    facts.append((s["mean_refined_score"], "refined mean /5"))
cols = st.columns(len(facts))
for c, (v, label) in zip(cols, facts):
    c.metric(label, "–" if v is None else v)
st.caption(f"embedder `{cfg.get('embed_model')}` · reranker `{cfg.get('rerank_model') if cfg.get('rerank') else 'off'}` · "
           f"k {cfg.get('k_chunks')} · {cfg.get('papers_indexed')} papers indexed")

# ---- second opinion -------------------------------------------------------------------------------
done = st.session_state.pop("rb_refine_done", None)
if done:
    st.success(done)
with st.expander("Second opinion on the low scores", icon=":material/gavel:"):
    st.caption("For every question scored below the threshold, a **context judge** reads the whole PDF and the context "
               "the generator received and scores it (1-5). If the context is good, a **generation judge** grades the "
               "answer again against the PDF and gives a refined score and a cause (reference, judge, generation). "
               "The first score is kept; results.jsonl, summary.json and questions.md are updated.")
    st.session_state.setdefault("rb_refine_threshold", config.REFINE_THRESHOLD)
    c1, c2, c3 = st.columns([1, 1, 2])
    threshold = c1.number_input("Score threshold", min_value=2, max_value=5, step=1, key="rb_refine_threshold",
                                help="questions scored strictly below are judged again")
    force = c2.checkbox("Redo refined ones", key="rb_refine_force", help="judge again the questions already refined")
    an_key = c3.text_input("Anthropic key (this session only)", type="password", key="rb_refine_key",
                           placeholder="taken from ANTHROPIC_API_KEY if empty")
    todo = refine.candidates(rows, int(threshold), force)
    has_key = bool(an_key or os.environ.get("ANTHROPIC_API_KEY"))
    st.caption(f"{len(todo)} question(s) to judge with `{config.REFINE_MODEL}` (two calls each when the context is good, "
               "the whole PDF sent each time)." + ("" if has_key else " `ANTHROPIC_API_KEY` is not set."))
    if st.button("Run the judges", type="primary", icon=":material/gavel:", disabled=not todo or not has_key):
        bar = st.progress(0.0, text="Starting…")
        try:
            res = refine.refine_run(run_dir, threshold=int(threshold), force=force, api_key=an_key or None,
                                    workdir=ragbench_review.run_workdir(run_dir),
                                    on_progress=lambda n, total, msg: bar.progress(n / total, text=msg))
        except ragbench.RagBenchError as exc:
            st.error(str(exc))
        else:
            rv.clear_cache()
            st.session_state["rb_refine_done"] = (f"{res['n_candidates']} question(s) judged again by {res['model']} in "
                                                  f"{res['seconds']} s, {res['n_errors']} error(s).")
            st.rerun()

# ---- filters -------------------------------------------------------------------------------------
ui.step_header(1, "Questions", "lowest score first; click a column header to sort")
f1, f2, f3, f4 = st.columns(4)
sources = f1.multiselect("Source", sorted({r["source"] for r in rows}))
types = f2.multiselect("Type", sorted({r["type"] for r in rows}))
labels = {c: v[0] for c, v in ragbench_review.CATEGORIES.items()}
cats = f3.multiselect("Diagnosis", list(labels), format_func=labels.get)
kept = f4.selectbox("In the bench", ["all", "kept", "excluded"])
score_range = st.slider("Judge score", 1, 5, (1, 5))


def keep_row(r):
    sc = r.get("score")
    return ((not sources or r["source"] in sources) and (not types or r["type"] in types)
            and (not cats or diag[r["id"]]["category"] in cats)
            and (kept == "all" or review[r["id"]]["keep"] == (kept == "kept"))
            and (sc is None or score_range[0] <= sc <= score_range[1]))


shown = sorted((r for r in rows if keep_row(r)), key=lambda r: (r.get("score") or 0, r.get("context_recall") or 0))
st.session_state["rb_order"] = [r["id"] for r in shown]
df = pd.DataFrame([{
    "id": rv.short(r["id"]), "paper": r["paper"], "source": r["source"], "type": r["type"], "score": r.get("score"),
    "refined": r.get("refine_score"), "cause": r.get("refine_cause"), "context": r.get("context_score"),
    "section rank": r.get("section_rank"), "doc rank": r.get("doc_rank"), "recall": r.get("context_recall"),
    "diagnosis": diag[r["id"]]["label"], "kept": review[r["id"]]["keep"], "question": r["question"],
    "judge": (r.get("rationale") or r.get("error") or "")[:160],
} for r in shown])
st.caption(f"{len(shown)} / {len(rows)} questions")
if df.empty:
    st.info("No question matches the filters.")
    st.stop()
event = st.dataframe(
    df, hide_index=True, width="stretch", height=480, key="rb_table", on_select="rerun", selection_mode="single-row",
    column_config={
        "score": st.column_config.NumberColumn("score", format="%d"),
        "refined": st.column_config.NumberColumn("refined", format="%d", help="score of the generation judge (second opinion)"),
        "context": st.column_config.NumberColumn("context", format="%d", help="context judge: the context against the PDF, 1-5"),
        "recall": st.column_config.ProgressColumn("recall", min_value=0.0, max_value=1.0, format="percent"),
        "kept": st.column_config.CheckboxColumn("kept"),
        "question": st.column_config.TextColumn("question", width="large"),
        "judge": st.column_config.TextColumn("judge comment", width="large"),
    },
)
if event.selection.rows:
    st.session_state["rb_qid"] = shown[event.selection.rows[0]]["id"]
qid = st.session_state.get("rb_qid")
if qid in review:
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"Selected: **{rv.short(qid)}** · {next(r['question'] for r in rows if r['id'] == qid)}")
    if c2.button("Open the detail", type="primary", icon=":material/arrow_forward:", width="stretch"):
        st.switch_page(rv.PAGES["detail"])
else:
    st.caption("Select a row to open its detail.")
