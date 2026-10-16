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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    at = AppTest.from_file(str(APP / "views" / "query.py"), default_timeout=60).run()
    next(t for t in at.text_input if t.label == "Query").set_value("settlement delay")
    next(c for c in at.checkbox if c.label.startswith("Track in MLflow")).uncheck()
    at.button[0].click().run()
    assert not at.exception, at.exception
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
