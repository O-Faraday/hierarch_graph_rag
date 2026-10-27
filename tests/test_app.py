"""Streamlit pages smoke tests (AppTest, no Airflow, no model)."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from test_retrieval import BagOfWordsEmbedder, store  # noqa: E402,F401


def test_pipeline_page_renders_board_and_runs(tmp_path, monkeypatch):
    from hgrag import runner

    runs = tmp_path / "runs"
    monkeypatch.setattr(runner, "RUNS_DIR", runs)
    w = runner.StatusWriter("20260101T000000-abc", runner.INDEXING_GRAPH, {"pdf_dir": "/x/pdfs"}, runs)
    w.finish(w.start("list_pdfs"))
    w.start("parse", "a.pdf")
    w.data["pid"] = __import__("os").getpid()  # alive: the run stays "running"
    w.flush()

    at = AppTest.from_file(str(APP / "views" / "pipeline.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    html = " ".join(str(h.value) for h in at.get("html"))
    assert "hg-pill running" in html and "a.pdf" in html and "hg-cell running" in html
    # trigger is disabled while a run is active, cancel is enabled
    labels = {b.label: b.disabled for b in at.button}
    for refresh in ("Run whole pipeline", "Rebuild hierarchy + index", "Re-index only"):
        assert labels[refresh] is True  # disabled while a run is active
    assert "Cancel run" in labels


def test_pipeline_page_empty_and_refresh_buttons_launch(tmp_path, monkeypatch):
    from hgrag import config, runner

    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "none")
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "a.pdf").write_bytes(b"%PDF a")
    monkeypatch.setattr(config, "PDF_DIR", pdfs)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    launched = []
    monkeypatch.setattr(runner, "start_indexing", lambda d, **kw: launched.append(kw) or "k")

    at = AppTest.from_file(str(APP / "views" / "pipeline.py"), default_timeout=30).run()
    assert not at.exception and any("No run yet" in i.value for i in at.info)
    next(b for b in at.button if b.label == "Rebuild hierarchy + index").click().run()
    assert launched[-1]["steps"] == ["hierarchy", "index"]
    next(b for b in at.button if b.label == "Run whole pipeline").click().run()
    assert launched[-1]["steps"] == ["parse", "hierarchy", "index"]
    next(b for b in at.button if b.label == "Re-index only").click().run()
    assert launched[-1]["steps"] == ["index"]
    assert launched[-1]["assess_judges"] is False  # config default in the tests
    next(c for c in at.checkbox if c.label.startswith("LLM judges")).check().run()
    assert any("ANTHROPIC_API_KEY" in w.value for w in at.warning)
    next(b for b in at.button if b.label == "Re-index only").click().run()
    assert launched[-1]["assess_judges"] is True


def test_run_board_states_and_merged_cells(tmp_path):
    import ui

    from hgrag import runner

    w = runner.StatusWriter("k", runner.INDEXING_GRAPH, {"steps": ["hierarchy", "index"]}, tmp_path)
    w.finish(w.start("list_pdfs"))
    w.skip("parse", "output already on disk")
    w.finish(w.start("hierarchy", "a.pdf"))
    w.finish(w.start("hierarchy", "b<1>.pdf"), error="boom")
    board = ui.run_board(w.data)
    assert "hg-cell success" in board and "hg-cell failed" in board and "hg-cell skipped merged" in board
    assert "b&lt;1&gt;.pdf" in board and "boom" in board  # escaped
    assert "--hg-run" in ui.CSS and "--hg-ok" in ui.CSS

def test_query_page_streams_steps(store, monkeypatch, tmp_path):  # noqa: F811
    client, out = store
    from hgrag import config, indexing

    monkeypatch.setattr(indexing, "load_embedder", lambda _id: BagOfWordsEmbedder())
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    import hgrag.retrieval as retrieval

    # reuse the fixture store instead of reopening Qdrant by path
    monkeypatch.setattr(config, "COLLECTION_NAME", "c")
    real = retrieval.retrieve_steps

    def patched(query, k, **kw):
        return real(query, k, client=client, collection="c", output_dir=out, **kw)

    monkeypatch.setattr(retrieval, "retrieve_steps", patched)
    at = AppTest.from_file(str(APP / "views" / "query.py"), default_timeout=60).run()
    next(t for t in at.text_input if t.label == "Query").set_value("settlement delay")
    next(c for c in at.checkbox if c.label.startswith("Track in MLflow")).uncheck()
    next(c for c in at.checkbox if c.label.startswith("Generate the answer")).uncheck()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("sections rebuilt" in m.value for m in at.success)
    assert any("Average settlement delay" in md.value for md in at.markdown)
    assert not any("Answer" in str(h.value) for h in at.get("html") if "hg-step" in str(h.value))


def test_query_page_generates_the_answer_from_the_sections(store, monkeypatch):  # noqa: F811
    client, out = store
    from hgrag import config, generation, indexing
    import hgrag.retrieval as retrieval

    monkeypatch.setattr(indexing, "load_embedder", lambda _id: BagOfWordsEmbedder())
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(config, "COLLECTION_NAME", "c")
    real = retrieval.retrieve_steps
    monkeypatch.setattr(retrieval, "retrieve_steps",
                        lambda q, k, **kw: real(q, k, client=client, collection="c", output_dir=out, **kw))
    prompts = []

    def fake_stream(cfg, system, user):
        user = user if isinstance(user, str) else user[-1]["text"]
        prompts.append(user)
        yield {"text": "The delay dropped sharply [1]."}
        yield {"usage": {"input_tokens": 99, "output_tokens": 6, "stop_reason": "end_turn"}}

    monkeypatch.setattr(generation, "stream_completion", fake_stream)
    judged = []
    real_events = generation._answer_events

    def spy_events(*a, judges, **kw):  # records the switch, never calls a real judge
        judged.append(judges)
        return (yield from real_events(*a, judges=False, **kw))

    monkeypatch.setattr(generation, "_answer_events", spy_events)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    at = AppTest.from_file(str(APP / "views" / "query.py"), default_timeout=60).run()
    next(t for t in at.text_input if t.label == "Query").set_value("settlement delay")
    next(c for c in at.checkbox if c.label.startswith("Track in MLflow")).uncheck()
    next(c for c in at.checkbox if c.label.startswith("LLM judges")).check()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert judged == [True]  # the box reaches the answer checks
    assert prompts and "Average settlement delay dropped sharply" in prompts[0] and "Question: settlement delay" in prompts[0]
    assert any("The delay dropped sharply [1]." in m.value for m in at.markdown)
    assert any("Sources:" in c.value and "[1]" in c.value for c in at.caption)
    assert any("99 tokens in" in c.value for c in at.caption)


def test_query_page_shows_a_readable_error_without_llm(store, monkeypatch):  # noqa: F811
    client, out = store
    from hgrag import config, indexing
    import hgrag.retrieval as retrieval

    monkeypatch.setattr(indexing, "load_embedder", lambda _id: BagOfWordsEmbedder())
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(config, "COLLECTION_NAME", "c")
    real = retrieval.retrieve_steps
    monkeypatch.setattr(retrieval, "retrieve_steps",
                        lambda q, k, **kw: real(q, k, client=client, collection="c", output_dir=out, **kw))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HGRAG_LLM_PROVIDER", "anthropic")
    at = AppTest.from_file(str(APP / "views" / "query.py"), default_timeout=60).run()
    next(t for t in at.text_input if t.label == "Query").set_value("settlement delay")
    next(c for c in at.checkbox if c.label.startswith("Track in MLflow")).uncheck()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("ANTHROPIC_API_KEY is not set" in e.value for e in at.error)
    assert any("Average settlement delay" in md.value for md in at.markdown)  # the retrieval output stays


def test_pipeline_page_inspect_and_control_check(tmp_path, monkeypatch):
    from test_control import CONTROL, EXPECTED, _process

    from hgrag import config, runner, verify

    _, out, res = _process(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "none")
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(config, "PDF_DIR", tmp_path / "elsewhere")
    monkeypatch.setattr(config, "CONTROL_PDF_DIR", CONTROL)
    import models

    monkeypatch.setattr(models, "embedder", lambda: object())
    import hgrag.indexing as indexing

    monkeypatch.setattr(indexing, "open_store", lambda *a, **k: type("C", (), {"close": lambda self: None})())
    canned = {"ok": False, "n_checks": 2, "n_ok": 1, "checks": [
        {"pdf": "a.pdf", "check": "x", "ok": True, "expected": 1, "actual": 1},
        {"pdf": "a.pdf", "check": "y", "ok": False, "expected": ["a"], "actual": "b"}]}
    monkeypatch.setattr(verify, "verify", lambda *a, **k: canned)

    import logging

    records: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    grab = Grab(level=logging.WARNING)
    logging.getLogger("streamlit").addHandler(grab)
    at = AppTest.from_file(str(APP / "views" / "pipeline.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Use the 3 control PDFs").click().run()
    logging.getLogger("streamlit").removeHandler(grab)
    # regressions seen on the user's machine: widget with both `value=` and session state, mixed-type Arrow table
    assert not [r for r in records if "default value" in r or "Arrow" in r], records
    for df in at.dataframe:
        for col in df.value.columns:
            kinds = {type(v).__name__ for v in df.value[col].dropna()}
            assert len(kinds) <= 1, (col, kinds)
    assert at.text_input[0].value == str(CONTROL)
    assert [t.label for t in at.tabs] == ["Parse", "Hierarchy", "Chunks", "Figures"]
    assert any("Aurora" in str(c.value) or "Executive summary" in str(c.value) for c in at.code), "tree not shown"
    next(b for b in at.button if b.label == "Run the check").click().run()
    assert not at.exception, at.exception
    assert any("1 of 2 checks passed" in e.value for e in at.error)


def test_export_button_builds_a_zip(tmp_path, monkeypatch):
    from hgrag import bundle, config, runner

    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "none")
    monkeypatch.setattr(config, "PDF_DIR", tmp_path / "p")
    z = tmp_path / "x.zip"
    z.write_bytes(b"PK")
    monkeypatch.setattr(bundle, "build_bundle", lambda *a, **k: z)
    at = AppTest.from_file(str(APP / "views" / "pipeline.py"), default_timeout=30).run()
    next(b for b in at.button if b.label == "Prepare the analysis zip").click().run()
    assert not at.exception, at.exception
    assert any("x.zip" in str(c.value) for c in at.caption)


def _ragbench_runs(tmp_path, monkeypatch):
    import json

    from hgrag import config, ragbench

    monkeypatch.setattr(ragbench, "RAGBENCH_DIR", tmp_path)
    disputed = tmp_path / "disputed.json"
    disputed.write_text(json.dumps({"q-wrong-ref": "inverted table"}), encoding="utf-8")
    monkeypatch.setattr(config, "RAGBENCH_DISPUTED", disputed)
    run = tmp_path / "runs" / "20261019T000000-abc"
    run.mkdir(parents=True)
    base = {"paper": "2409.00001v1", "type": "extractive", "expected_answer": "Yes", "doc_rank": 1, "context_recall": 0.5,
            "gold_text": "## 5 Results\n\nTable 1", "retrieved": [{"rank": 1, "paper": "2409.00001v1", "section_path": "p > 5 Results",
                                                               "score": 0.7, "preview": "Table 1", "doc": True, "on_gold": True}]}
    rows = [{**base, "id": "q-good", "source": "text", "question": "Good?", "answer": "Yes [1].", "score": 5,
             "rationale": "matches", "section_rank": 1},
            {**base, "id": "q-wrong-ref", "source": "text-table", "question": "Inverted?", "answer": "No [1].", "score": 1,
             "rationale": "contradicts the ground truth", "section_rank": 1},
            {**base, "id": "q-missed", "source": "text-image", "question": "Missed?", "answer": "Maybe.", "score": 2,
             "rationale": "partial", "section_rank": None}]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({**ragbench.summarize(rows, 10), "config": {"key": "n3_s0_neg0"}}), encoding="utf-8")
    return run


def test_ragbench_pages_list_detail_decide_and_summarize(tmp_path, monkeypatch):
    import json

    run = _ragbench_runs(tmp_path, monkeypatch)
    at = AppTest.from_file(str(APP / "views" / "rb_results.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert at.selectbox[0].value == "n3_s0_neg0" and at.selectbox[1].value == run.name
    table = at.dataframe[0].value
    assert list(table["id"]) == ["q-wrong-", "q-missed", "q-good"]  # lowest score first
    assert not dict(zip(table["id"], table["kept"]))["q-wrong-"]  # disputed: excluded by default
    assert "Section not retrieved" in list(table["diagnosis"])

    detail = AppTest.from_file(str(APP / "views" / "rb_detail.py"), default_timeout=30)
    detail.session_state["rb_run"] = str(run)
    detail.session_state["rb_qid"] = "q-missed"
    detail.run()
    assert not detail.exception, detail.exception
    assert any("Section not retrieved" in str(m.value) for m in detail.get("metric"))
    detail.checkbox(key="rb_keep_q-missed").uncheck()
    detail.text_area(key="rb_note_q-missed").input("generic question")
    next(b for b in detail.button if b.label == "Save").click().run()
    assert not detail.exception, detail.exception
    saved = json.loads((run / "review.json").read_text(encoding="utf-8"))
    assert saved["q-missed"] == {"keep": False, "note": "generic question"} and saved["q-wrong-ref"]["keep"] is False

    summary = AppTest.from_file(str(APP / "views" / "rb_summary.py"), default_timeout=30)
    summary.session_state["rb_run"] = str(run)
    summary.run()
    assert not summary.exception, summary.exception
    metrics = summary.dataframe[0].value.set_index("metric")
    assert metrics.loc["n", "whole bench"] == 3 and metrics.loc["n", "kept selection"] == 1
    assert metrics.loc["mean_score", "kept selection"] == 5.0
    assert any("No failure left" in s.value for s in summary.success)


def test_ragbench_results_page_runs_the_refine_judges(tmp_path, monkeypatch):
    import json

    from hgrag import refine

    run = _ragbench_runs(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen = {}

    def fake_refine(run_dir, **kw):
        seen.update(run_dir=run_dir, **kw)
        rows = [json.loads(x) for x in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        for r in rows:
            if r["score"] < kw["threshold"]:
                r.update(context_score=5, refine_score=4, refine_cause="reference")
        (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        kw["on_progress"](1, 1, "done")
        return {"n_candidates": 2, "model": "m", "seconds": 1.0, "n_errors": 0}

    monkeypatch.setattr(refine, "refine_run", fake_refine)
    at = AppTest.from_file(str(APP / "views" / "rb_results.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    assert at.number_input(key="rb_refine_threshold").value == 4
    at.number_input(key="rb_refine_threshold").set_value(3).run()
    next(b for b in at.button if b.label == "Run the judges").click().run()
    assert not at.exception, at.exception
    assert seen["threshold"] == 3 and seen["run_dir"] == run and seen["force"] is False
    assert any("judged again by m" in s.value for s in at.success)
    table = at.dataframe[0].value
    assert dict(zip(table["id"], table["cause"]))["q-wrong-"] == "reference"


def test_ragbench_summary_compares_scores_with_their_refinement(tmp_path, monkeypatch):
    import json

    from hgrag import ragbench

    run = _ragbench_runs(tmp_path, monkeypatch)
    summary = AppTest.from_file(str(APP / "views" / "rb_summary.py"), default_timeout=30)
    summary.session_state["rb_run"] = str(run)
    summary.run()
    assert not summary.exception, summary.exception
    assert any("No question refined" in i.value for i in summary.info)

    rows = [json.loads(x) for x in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    extra = {"q-wrong-ref": {"context_score": 5, "refine_score": 4, "refine_cause": "reference", "refine_rationale": "PDF says No"},
             "q-missed": {"context_score": 2, "refine_score": 2, "refine_cause": "context"}}
    for r in rows:
        r.update(extra.get(r["id"], {}))
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({**ragbench.summarize(rows, 10), "config": {"key": "n3_s0_neg0"},
                                                  "refine": {"model": "m", "threshold": 4, "context_ok": 4}}), encoding="utf-8")
    summary.run()
    assert not summary.exception, summary.exception
    metrics = summary.dataframe[0].value.set_index("metric")
    # corrected scores where they exist: q-good 5, q-wrong-ref 1 -> 4, q-missed 2 (context: kept); q-wrong-ref is excluded
    assert metrics.loc["mean_score (corrected)", "whole bench"] == 3.67 and metrics.loc["mean_score (first judge)", "whole bench"] == 2.67
    assert metrics.loc["mean_score (corrected)", "kept selection"] == 3.5 and metrics.loc["pct_score_4_plus (corrected)", "whole bench"] == 0.667
    by_source = summary.dataframe[1].value.set_index("group")
    assert by_source.loc["text-table", "score (corrected) (whole)"] == 4.0 and by_source.loc["text-table", "first judge (whole)"] == 1.0
    m = {x.label: x.value for x in summary.metric}
    assert m["questions judged again"] == "2" and m["mean score, refined"] == "3.0" and m["raised to ≥ 4"] == "1"
    detail = next(d.value for d in summary.dataframe if "gain" in d.value.columns)
    assert list(detail["id"]) == ["q-wrong-", "q-missed"] and list(detail["gain"]) == [3, 0]
    assert "Reference wrong or imprecise" in list(detail["cause"])


def test_compare_page_lists_regressions_and_opens_the_detail(tmp_path, monkeypatch):
    import json

    from hgrag import ragbench

    run = _ragbench_runs(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    new = tmp_path / "check_full" / "runs" / "20261022T000000-new"  # another workdir: listed too
    new.mkdir(parents=True)
    moved = {"q-good": {"score": 2, "section_rank": None}, "q-missed": {"score": 5, "section_rank": 1},
             "q-wrong-ref": {"score": 5}}
    for r in rows:
        r.update(moved[r["id"]])
    rows[0].update(context_score=2, refine_score=2, refine_cause="context", context_rationale="no section 5",
                   refine_rationale="kept", context_same=False)
    (new / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (new / "summary.json").write_text(json.dumps({**ragbench.summarize(rows, 10), "config": {
        "key": "n3_s0_neg0", "keyword_weight": 0.3, "reference_run": str(run)}}), encoding="utf-8")

    at = AppTest.from_file(str(APP / "views" / "rb_compare.py"), default_timeout=30)
    at.session_state["rb_run"] = str(new)
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox[1].value == run.name and at.selectbox[2].value == "check_full / 20261022T000000-new"
    settings = at.dataframe[0].value
    assert list(settings["setting"]) == ["keyword_weight"] and list(settings["new"]) == ["0.3"]
    metrics = at.dataframe[1].value.set_index("metric")
    assert metrics.loc["n", "new"] == 2  # q-wrong-ref is disputed: left out by default
    m = {x.label: x.value for x in at.metric}
    assert m["≥ 4 → < 4"] == "1" and m["< 4 → ≥ 4"] == "1" and m["net"] == "0"
    assert m["section lost"] == "1" and m["section gained"] == "1" and "better" not in m and "worse" not in m
    assert list(at.dataframe[2].value.columns) == ["retrieval change", "questions", "< 4 → ≥ 4", "≥ 4 → < 4", "net"]
    table = at.dataframe[3].value
    assert list(table["id"]) == ["q-missed", "q-good"]  # fixed first, then regressed
    assert list(table["status"]) == ["Fixed (now ≥ 4)", "Regressed (now < 4)"]
    at.session_state["rb_cmp_status"] = ["regressed"]
    at.run()
    assert list(at.dataframe[3].value["id"]) == ["q-good"]

    detail = AppTest.from_file(str(APP / "views" / "rb_detail.py"), default_timeout=30)
    detail.session_state["rb_run"] = str(new)
    detail.session_state["rb_qid"] = "q-good"
    detail.run()
    assert not detail.exception, detail.exception
    md = " ".join(str(x.value) for x in detail.markdown)
    assert "Second opinion" in md and "no section 5" in md
    assert any("differ from the reference run" in str(c.value) for c in detail.caption)
    labels = {x.label for x in detail.metric}
    assert {"refined /5", "context /5", "cause"} <= labels
