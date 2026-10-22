"""Ragbench page 3: metrics of the whole bench against the kept selection, first judge against the second opinion,
rank distribution, improvement leads."""

import pandas as pd
import ragbench_view as rv
import streamlit as st
import ui

from hgrag import ragbench_review

ui.inject_css()
ui.page_header(
    "Ragbench summary",
    "The whole bench next to the kept selection (questions not excluded on the Detail page), by source and by type, "
    "the first judge's scores next to their refinement (second opinion), "
    "then the failure categories ranked by size with a lead for each. Leads come from rules, not from a measure: "
    "an improvement only counts once a new run shows it.",
)
st.session_state.setdefault("rb_run", None)
st.session_state.setdefault("rb_qid", None)

run_dir = rv.pick_run(picker=False)
if run_dir is None:
    st.stop()
data = rv.load(run_dir)
rows, review, k = data["rows"], data["review"], data["k"]
refined = any(r.get("refine_score") is not None for r in rows)
cmp = ragbench_review.compare(rows, review, k, corrected=True)  # refined score where it exists
first = ragbench_review.compare(rows, review, k) if refined else cmp  # first judge only
METRICS = ["n", "doc_hit_at_1", "section_hit_at_1", f"section_hit_at_{k}", "mrr", "context_recall", "mean_score", "pct_score_4_plus"]

# ---- whole vs kept -------------------------------------------------------------------------------
ui.step_header(1, "Whole bench and kept selection", f"{len(cmp['kept']['excluded'])} question(s) excluded")
if refined:
    st.caption("Scores: the **corrected** score (second opinion) where the question was judged again, the first "
               "judge's elsewhere. The first judge's alone are in the last two rows, for reference.")
names = {"mean_score": "mean_score (corrected)", "pct_score_4_plus": "pct_score_4_plus (corrected)"} if refined else {}
lines = [(names.get(m, m), cmp["all"].get(m), cmp["kept"].get(m)) for m in METRICS]
if refined:
    lines += [(f"{m} (first judge)", first["all"].get(m), first["kept"].get(m)) for m in ("mean_score", "pct_score_4_plus")]
table = pd.DataFrame(lines, columns=["metric", "whole bench", "kept selection"])
table["difference"] = [None if a is None or b is None or m == "n" else round(b - a, 3)
                       for m, a, b in zip(table["metric"], table["whole bench"], table["kept selection"])]
st.dataframe(table, hide_index=True, width="stretch")

score_label = "score (corrected)" if refined else "score"
for title, group in (("By source", "by_source"), ("By type", "by_type")):
    st.markdown(f"**{title}**")
    st.dataframe(pd.DataFrame([
        {"group": g, "n (whole)": b["n"], "section hit@1 (whole)": b["section_hit_at_1"], f"{score_label} (whole)": b.get("mean_score"),
         "n (kept)": cmp["kept"][group].get(g, {}).get("n"), "section hit@1 (kept)": cmp["kept"][group].get(g, {}).get("section_hit_at_1"),
         f"{score_label} (kept)": cmp["kept"][group].get(g, {}).get("mean_score"),
         **({"first judge (whole)": first["all"][group].get(g, {}).get("mean_score"),
             "first judge (kept)": first["kept"][group].get(g, {}).get("mean_score")} if refined else {})}
        for g, b in cmp["all"][group].items()]), hide_index=True, width="stretch")

# ---- first judge vs second opinion ---------------------------------------------------------------
ref = ragbench_review.refine_comparison(rows)
ui.step_header(2, "Scores and their refinement", "first judge against the second opinion (context judge, then generation judge)")
if not ref["n"]:
    st.info("No question refined in this run: use *Second opinion on the low scores* on the Results page "
            "(or `uv run hgrag ragbench-refine <run>`)." + (f" {ref['n_errors']} refine error(s)." if ref["n_errors"] else ""))
else:
    info = data["summary"].get("refine") or {}
    c = st.columns(4)
    c[0].metric("questions judged again", ref["n"], help=f"scored below {info.get('threshold', '?')} by the first judge")
    c[1].metric("mean score, first judge", ref["mean_before"])
    c[2].metric("mean score, refined", ref["mean_after"], delta=round(ref["mean_after"] - ref["mean_before"], 2))
    c[3].metric("raised to ≥ 4", ref["n_raised_to_4_plus"])
    st.caption(f"Model `{info.get('model', '?')}` · context judged good from {info.get('context_ok', '?')}/5 · "
               f"{ref['n_errors']} error(s). Whole bench and kept selection with the corrected scores: table 1. "
               "Home-made measure: read the rationales before trusting a gain.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Cause of the gap**")
        causes = pd.DataFrame([{"cause": ragbench_review.REFINE_CAUSES.get(c_, c_), "questions": n}
                               for c_, n in ref["causes"].items()])
        st.bar_chart(causes, x="cause", y="questions", horizontal=True, height=220)
    with right:
        st.markdown("**First score → refined score** (questions)")
        matrix = pd.DataFrame.from_dict(ref["matrix"], orient="index")
        matrix.index = [f"first {i}" for i in matrix.index]
        matrix.columns = [f"→ {j}" for j in matrix.columns]
        st.dataframe(matrix, width="stretch")
    st.dataframe(pd.DataFrame([{**x, "id": rv.short(x["id"]), "cause": ragbench_review.REFINE_CAUSES.get(x["cause"], x["cause"])}
                               for x in ref["rows"]]),
                 hide_index=True, width="stretch",
                 column_order=["id", "source", "score", "refined", "gain", "context", "cause", "rationale", "question"],
                 column_config={"gain": st.column_config.NumberColumn("gain", format="%+d"),
                                "rationale": st.column_config.TextColumn("generation judge", width="large"),
                                "question": st.column_config.TextColumn("question", width="medium")})

# ---- ranks ---------------------------------------------------------------------------------------
ui.step_header(3, "Where the right section lands", "rank of the reference section among the retrieved ones")
dist = ragbench_review.rank_distribution(rows)
st.bar_chart(pd.DataFrame({"rank": list(dist), "questions": list(dist.values())}), x="rank", y="questions", height=240)

if cmp["kept"]["excluded"]:
    with st.expander(f"Excluded questions ({len(cmp['kept']['excluded'])})"):
        by_id = {r["id"]: r for r in rows}
        st.dataframe(pd.DataFrame([{"id": rv.short(q), "score": by_id[q].get("score"), "question": by_id[q]["question"],
                                    "note": review[q].get("note", "")} for q in cmp["kept"]["excluded"]]),
                     hide_index=True, width="stretch")

# ---- leads ---------------------------------------------------------------------------------------
ui.step_header(4, "Improvement leads", "failure categories of the kept selection, largest first")
kept_rows = [r for r in rows if review[r["id"]]["keep"]]
leads = ragbench_review.improvements(kept_rows, data["disputed"])
if not leads:
    st.success("No failure left in the kept selection.")
for x in leads:
    with st.container(border=True):
        st.markdown(f"**{x['label']}** · {x['n']} question(s), {x['share']:.0%} · {x['sources']}")
        st.write(x["fix"])
        st.caption("ids: " + ", ".join(rv.short(q) for q in x["ids"][:12]) + (" …" if len(x["ids"]) > 12 else ""))
