"""Ragbench page 3: metrics of the whole bench against the kept selection, rank distribution, improvement leads."""

import pandas as pd
import ragbench_view as rv
import streamlit as st
import ui

from hgrag import ragbench_review

ui.inject_css()
ui.page_header(
    "Ragbench summary",
    "The whole bench next to the kept selection (questions not excluded on the Detail page), by source and by type, "
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
cmp = ragbench_review.compare(rows, review, k)
METRICS = ["n", "doc_hit_at_1", "section_hit_at_1", f"section_hit_at_{k}", "mrr", "context_recall", "mean_score", "pct_score_4_plus"]

# ---- whole vs kept -------------------------------------------------------------------------------
ui.step_header(1, "Whole bench and kept selection", f"{len(cmp['kept']['excluded'])} question(s) excluded")
table = pd.DataFrame({"metric": METRICS, "whole bench": [cmp["all"].get(m) for m in METRICS],
                      "kept selection": [cmp["kept"].get(m) for m in METRICS]})
table["difference"] = [None if a is None or b is None or m == "n" else round(b - a, 3)
                       for m, a, b in zip(METRICS, table["whole bench"], table["kept selection"])]
st.dataframe(table, hide_index=True, width="stretch")

for title, group in (("By source", "by_source"), ("By type", "by_type")):
    st.markdown(f"**{title}**")
    st.dataframe(pd.DataFrame([
        {"group": g, "n (whole)": b["n"], "section hit@1 (whole)": b["section_hit_at_1"], "score (whole)": b.get("mean_score"),
         "n (kept)": cmp["kept"][group].get(g, {}).get("n"), "section hit@1 (kept)": cmp["kept"][group].get(g, {}).get("section_hit_at_1"),
         "score (kept)": cmp["kept"][group].get(g, {}).get("mean_score")}
        for g, b in cmp["all"][group].items()]), hide_index=True, width="stretch")

# ---- ranks ---------------------------------------------------------------------------------------
ui.step_header(2, "Where the right section lands", "rank of the reference section among the retrieved ones")
dist = ragbench_review.rank_distribution(rows)
st.bar_chart(pd.DataFrame({"rank": list(dist), "questions": list(dist.values())}), x="rank", y="questions", height=240)

if cmp["kept"]["excluded"]:
    with st.expander(f"Excluded questions ({len(cmp['kept']['excluded'])})"):
        by_id = {r["id"]: r for r in rows}
        st.dataframe(pd.DataFrame([{"id": rv.short(q), "score": by_id[q].get("score"), "question": by_id[q]["question"],
                                    "note": review[q].get("note", "")} for q in cmp["kept"]["excluded"]]),
                     hide_index=True, width="stretch")

# ---- leads ---------------------------------------------------------------------------------------
ui.step_header(3, "Improvement leads", "failure categories of the kept selection, largest first")
kept_rows = [r for r in rows if review[r["id"]]["keep"]]
leads = ragbench_review.improvements(kept_rows, data["disputed"])
if not leads:
    st.success("No failure left in the kept selection.")
for x in leads:
    with st.container(border=True):
        st.markdown(f"**{x['label']}** · {x['n']} question(s), {x['share']:.0%} · {x['sources']}")
        st.write(x["fix"])
        st.caption("ids: " + ", ".join(rv.short(q) for q in x["ids"][:12]) + (" …" if len(x["ids"]) > 12 else ""))
