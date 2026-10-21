"""Ragbench page 2: one question in detail (answers, judge, diagnosis, retrieved sections) and the keep/exclude decision."""

import ragbench_view as rv
import streamlit as st
import ui

from hgrag import ragbench_review

ui.inject_css()
ui.page_header(
    "Question detail",
    "Expected and generated answers side by side, the judge's comment, a rule-based diagnosis with a lead to fix it, "
    "and the sections the retrieval returned. Untick « Keep in the bench » to leave a question out of this run's "
    "kept-selection metrics (saved in the run's review.json, nothing else changes).",
)
st.session_state.setdefault("rb_run", None)
st.session_state.setdefault("rb_qid", None)

run_dir = rv.pick_run(picker=False)
if run_dir is None:
    st.stop()
data = rv.load(run_dir)
rows = {r["id"]: r for r in data["rows"]}
order = [q for q in st.session_state.get("rb_order") or [] if q in rows] or list(rows)
qid = st.session_state.get("rb_qid")
if qid not in rows:
    qid = order[0]
    st.session_state["rb_qid"] = qid
r, d, review = rows[qid], data["diag"][qid], data["review"]

# ---- navigation ----------------------------------------------------------------------------------
pos = order.index(qid) if qid in order else 0
n1, n2, n3 = st.columns([1, 3, 1])
if n1.button("Previous", icon=":material/arrow_back:", disabled=pos == 0, width="stretch"):
    st.session_state["rb_qid"] = order[pos - 1]
    st.rerun()
n2.markdown(f"<div style='text-align:center'>question {pos + 1} / {len(order)} (filtered order of the Results page)</div>",
            unsafe_allow_html=True)
if n3.button("Next", icon=":material/arrow_forward:", disabled=pos >= len(order) - 1, width="stretch"):
    st.session_state["rb_qid"] = order[pos + 1]
    st.rerun()

# ---- header --------------------------------------------------------------------------------------
st.subheader(f"{rv.short(qid)} · {r['paper']}")
facts = [("score /5", r.get("score")), ("diagnosis", d["label"]), ("source", r["source"]), ("type", r["type"]),
         ("section rank", r.get("section_rank")), ("doc rank", r.get("doc_rank")), ("context recall", r.get("context_recall"))]
for c, (label, v) in zip(st.columns(len(facts)), facts):
    c.metric(label, "–" if v is None else v)

st.markdown(f"**Question.** {r['question']}")
left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Expected answer**")
    st.write(r.get("expected_answer") or "–")
with right, st.container(border=True):
    st.markdown("**Generated answer**")
    st.write(r.get("answer") if r.get("answer") is not None else "(not generated: run with --generate)")
if r.get("rationale"):
    st.markdown(f"**Judge.** {r['rationale']}")
if r.get("error"):
    st.error(r["error"])

ui.step_header(1, "Diagnosis", d["label"])
(st.success if d["category"] == "ok" else st.info)(d["fix"], icon=":material/lightbulb:")

# ---- decision ------------------------------------------------------------------------------------
ui.step_header(2, "Keep in the bench?", "this run only")
cur = review[qid]
wkeep, wnote = f"rb_keep_{qid}", f"rb_note_{qid}"
st.session_state.setdefault(wkeep, cur["keep"])
st.session_state.setdefault(wnote, cur.get("note", ""))
if qid in data["disputed"]:
    st.caption(f"Listed in data/ragbench_disputed.json: {data['disputed'][qid]}")
st.checkbox("Keep in the bench", key=wkeep)
st.text_area("Note (reason, correction idea, check done…)", key=wnote, height=90)
if st.button("Save", type="primary", icon=":material/save:"):
    review[qid] = {"keep": bool(st.session_state[wkeep]), "note": st.session_state[wnote].strip()}
    path = ragbench_review.save_review(run_dir, review)
    st.success(f"Saved in `{path}`.")

# ---- evidence ------------------------------------------------------------------------------------
ui.step_header(3, "Evidence", "reference section and retrieved sections")
pdf = ragbench_review.pdf_path(r)
st.caption(f"PDF: `{pdf}`" + ("" if pdf.exists() else " (not on disk)"))
if r.get("gold_text"):
    with st.expander("Reference section (benchmark parse, excerpt)"):
        st.markdown(r["gold_text"])
for h in r.get("retrieved") or []:
    path = h["section_path"] if isinstance(h["section_path"], str) else " > ".join(h["section_path"])
    mark = "✅" if h.get("on_gold") else ("📄" if h.get("doc") else "·")
    with st.expander(f"{mark} #{h['rank']} {path} (score {h['score']})", expanded=h.get("on_gold", False) and h["rank"] <= 3):
        st.caption(f"same paper: {h.get('doc')} · on the reference: {h.get('on_gold')}"
                   + (f" (by {h['on_gold_by']})" if h.get("on_gold_by") else "") + f" · recall so far {h.get('recall_so_far')}")
        st.text(h.get("preview") or "")
