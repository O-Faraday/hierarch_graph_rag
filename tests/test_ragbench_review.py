"""Review of ragbench runs: grouping, tolerant loading, rule-based diagnosis, per-run keep/exclude, leads (offline)."""

from __future__ import annotations

import json

import pytest

from hgrag import ragbench_review as rr


def row(qid, score=5, sec=1, doc=1, source="text", answer="Yes [1].", rationale="matches", **kw):
    return {"id": qid, "paper": "2409.00001v1", "source": source, "type": "extractive", "question": f"q {qid}?",
            "expected_answer": "Yes", "answer": answer, "score": score, "rationale": rationale, "section_rank": sec,
            "doc_rank": doc, "context_recall": 0.5, **kw}


def write_run(runs, name, key, rows, extra_line=None):
    d = runs / name
    d.mkdir(parents=True)
    lines = [json.dumps(r) for r in rows] + ([extra_line] if extra_line else [])
    (d / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"k_sections": 10, "config": {"key": key}}), encoding="utf-8")
    return d


@pytest.mark.parametrize("r, disputed, cat", [
    (row("a", score=None, error="no JSON in the judge reply"), {}, "error"),
    (row("b", score=1), {"b": "wrong table"}, "disputed"),
    (row("c", score=4), {}, "ok"),
    (row("d", score=1, sec=None, doc=None), {}, "doc_missed"),
    (row("e", score=2, sec=None), {}, "section_missed"),
    (row("f", score=1, answer="I cannot answer this from the documents."), {}, "refusal"),
    (row("g", score=1, source="text-image", answer="About -77 dBm, read from the chart [2]."), {}, "figure"),
    (row("h", score=1, rationale="The candidate contradicts the ground truth."), {}, "contradiction"),
    (row("i", score=2, sec=6), {}, "ranked_low"),
    (row("j", score=3, sec=2), {}, "partial"),
    (row("k", score=None, answer=None, sec=2), {}, "ok"),  # retrieval-only run
    (row("l", score=None, answer=None, sec=7), {}, "ranked_low"),
])
def test_diagnose_one_category_per_rule(r, disputed, cat):
    d = rr.diagnose(r, disputed)
    assert d["category"] == cat and d["label"] and d["fix"]


def test_benches_grouped_by_sample_latest_first_and_bad_lines_reported(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "20261016T000000-a", "n30_s0_neg30", [row("x")])
    new = write_run(runs, "20261019T000000-b", "n30_s0_neg30", [row("x")],
                    extra_line=json.dumps({k: v for k, v in row("bad").items() if k != "question"} | {"qu59e85175estion": "?"}))
    write_run(runs, "20261017T000000-c", "n3_s0_neg0", [row("y")])
    (runs / "not-a-run").mkdir()
    benches = rr.list_benches(runs)
    assert list(benches["n30_s0_neg30"]) == [new, runs / "20261016T000000-a"] and len(benches["n3_s0_neg0"]) == 1
    rows, problems = rr.load_results(new)
    assert [r["id"] for r in rows] == ["x"] and len(problems) == 1 and "missing question" in problems[0]
    (new / "results.jsonl").write_text("{not json\n", encoding="utf-8")
    assert "not JSON" in rr.load_results(new)[1][0]


def test_review_defaults_round_trip_and_kept_metrics(tmp_path):
    rows = [row("ok", score=5), row("wrong", score=1), row("miss", score=3, sec=None)]
    d = write_run(tmp_path, "r", "k", rows)
    review = rr.load_review(d, rows, {"wrong": "inverted table"})
    assert review["wrong"] == {"keep": False, "note": "inverted table"} and review["ok"]["keep"] is True
    review["miss"] = {"keep": False, "note": "generic question"}
    rr.save_review(d, review)
    again = rr.load_review(d, rows, {})
    assert again["miss"]["keep"] is False and again["wrong"]["keep"] is False  # saved decisions win over the list
    assert not list(d.glob(".review-*"))  # temporary file replaced
    cmp = rr.compare(rows, again, 10)
    assert cmp["all"]["n"] == 3 and cmp["kept"]["n"] == 1 and sorted(cmp["kept"]["excluded"]) == ["miss", "wrong"]
    assert cmp["kept"]["mean_score"] == 5.0 and "by_source" in cmp["kept"]
    none_out = rr.compare(rows, {q: {"keep": True, "note": ""} for q in again}, 10)
    assert none_out["kept"]["n"] == 3 and none_out["kept"]["excluded"] == []


def test_improvements_ranked_by_size_without_ok():
    rows = [row("a"), row("b", score=2, sec=None), row("c", score=1, sec=None, source="text-image"),
            row("d", score=1, answer="I cannot answer from the documents.")]
    leads = rr.improvements(rows)
    assert [x["category"] for x in leads] == ["section_missed", "refusal"]
    assert leads[0]["n"] == 2 and leads[0]["share"] == 0.5 and "text-image 1" in leads[0]["sources"]
    assert rr.rank_distribution(rows) == {"1": 2, "2-3": 0, "4-10": 0, "absent": 2}


def test_refine_comparison_matrix_causes_and_gains():
    base = {"source": "text-image", "type": "extractive", "question": "q"}
    rows = [{**base, "id": "a", "score": 2, "refine_score": 5, "refine_cause": "reference", "context_score": 5},
            {**base, "id": "b", "score": 3, "refine_score": 3, "refine_cause": "context", "context_score": 2},
            {**base, "id": "c", "score": 1, "refine_score": 2, "refine_cause": "generation", "context_score": 4},
            {**base, "id": "d", "score": 5},
            {**base, "id": "e", "score": 2, "refine_error": "JudgeError: no JSON"}]
    c = rr.refine_comparison(rows)
    assert c["n"] == 3 and c["n_errors"] == 1 and c["mean_before"] == 2.0 and c["mean_after"] == 3.33
    assert c["n_raised_to_4_plus"] == 1 and c["causes"] == {"reference": 1, "context": 1, "generation": 1}
    assert c["matrix"] == {1: {1: 0, 2: 1, 3: 0, 4: 0, 5: 0}, 2: {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}, 3: {1: 0, 2: 0, 3: 1, 4: 0, 5: 0}}
    assert [x["id"] for x in c["rows"]] == ["a", "c", "b"] and c["rows"][0]["gain"] == 3
    assert rr.refine_comparison([{**base, "id": "d", "score": 5}]) == {"n": 0, "n_errors": 0}


def test_compare_uses_the_corrected_score_where_it_exists():
    base = {"paper": "p", "source": "text", "type": "extractive", "question": "q", "doc_rank": 1, "section_rank": 1,
            "context_recall": 1.0}
    rows = [{**base, "id": "a", "score": 2, "refine_score": 5}, {**base, "id": "b", "score": 3}, {**base, "id": "c", "score": 1, "refine_score": 4}]
    review = {"a": {"keep": True}, "b": {"keep": True}, "c": {"keep": False}}
    first, corrected = rr.compare(rows, review, 10), rr.compare(rows, review, 10, corrected=True)
    assert first["all"]["mean_score"] == 2.0 and corrected["all"]["mean_score"] == 4.0
    assert first["kept"]["mean_score"] == 2.5 and corrected["kept"]["mean_score"] == 4.0
    assert corrected["kept"]["by_source"]["text"]["pct_score_4_plus"] == 0.5
    assert rows[0]["score"] == 2  # the rows themselves are not changed


def test_runs_of_other_workdirs_are_listed_with_their_label(tmp_path, monkeypatch):
    from hgrag import ragbench

    monkeypatch.setattr(ragbench, "RAGBENCH_DIR", tmp_path)
    old = write_run(tmp_path / "runs", "20261021T000000-a", "n30", [row("x")])
    check = write_run(tmp_path / "check_full" / "runs", "20261022T000000-b", "n30", [row("x")])
    write_run(tmp_path / "check_full" / "runs", "20261020T000000-c", "n3", [row("y")])
    benches = rr.list_benches()
    assert benches["n30"] == [check, old] and len(benches["n3"]) == 1  # latest first, all workdirs mixed
    assert rr.run_label(old) == "20261021T000000-a" and rr.run_label(check) == "check_full / 20261022T000000-b"
    assert rr.run_workdir(check) == tmp_path / "check_full"
    assert rr.pdf_path({"paper": "2409.1"}, rr.run_workdir(check)) == tmp_path / "check_full" / "pdfs" / "2409.1.pdf"


def test_compare_runs_statuses_sections_and_metrics():
    ref = [row("a", score=2, sec=None), row("b", score=5), row("c", score=3), row("d", score=1, refine_score=4),
           row("e", score=4), row("only-ref")]
    new = [row("a", score=4), row("b", score=2, sec=None, refine_score=2, refine_cause="context"), row("c", score=3),
           row("d", score=1), row("e", score=5), row("only-new")]
    c = rr.compare_runs(ref, new, 10)
    status = {x["id"]: x["status"] for x in c["questions"]}
    # d: refined 4 in the reference, 1 now -> regressed on corrected scores
    assert status == {"a": "fixed", "b": "regressed", "d": "regressed", "e": "better", "c": "same"}
    assert c["counts"] == {"fixed": 1, "regressed": 2, "better": 1, "worse": 0, "same": 1}
    assert c["sections"] == {"gained": 1, "lost": 1} and c["only_ref"] == 1 and c["only_new"] == 1 and c["n_common"] == 5
    assert [x["status"] for x in c["questions"]] == ["fixed", "regressed", "regressed", "better", "same"]
    m = {name: (a, b, d) for name, a, b, d in c["metrics"]}
    assert m["section_hit_at_1"] == (0.8, 0.8, 0.0) and m["mean_score"] == (3.0, 3.0, 0.0)
    assert m["mean_score (corrected)"] == (3.6, 3.0, -0.6) and m["n"][2] is None
    assert rr.compare_runs(ref, new, 10, {"b": "wrong"})["n_common"] == 4
    assert rr.config_diff({"embed_model": "m", "rerank": True}, {"embed_model": "m", "rerank": True, "keyword_weight": 0.3}) == \
        [("keyword_weight", None, 0.3)]
