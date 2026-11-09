"""Traces and checks of the Databricks steps, in a real local MLflow store with fake steps: nothing runs on Databricks."""

import json
from types import SimpleNamespace

import mlflow
import pytest

from hgrag import assessments, config, generation, tracing, tracking
from hgrag.assessments import Assessment
from hgrag.dbx import indexing, parsing, retrieval
from test_dbx_retrieval import Client, _row, _tree
from test_tracking import mlflow_dir  # noqa: F401  (fixture)


@pytest.fixture
def uc(monkeypatch, tmp_path, mlflow_dir):  # noqa: F811
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_VOLUME", "pdf_store")
    monkeypatch.setattr(config, "DBX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DBX_SEARCH_ENDPOINT", "hgrag")
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_ROOT", "")
    return tmp_path


def _traces(pipeline, run_key="rk"):
    mlflow.flush_trace_async_logging()
    exp = mlflow.get_experiment_by_name(pipeline)
    return mlflow.search_traces(locations=[exp.experiment_id], return_type="list",
                                filter_string=f"tag.`hgrag.run_key` = '{run_key}'")


def _names(trace):
    return sorted(a.name for a in trace.info.assessments)


class Session:
    """``tables``: the rows of each query, by a piece of its text."""

    def __init__(self, tables=None):
        self.tables = tables or {}

    def createDataFrame(self, rows, _schema):
        return SimpleNamespace(createOrReplaceTempView=lambda _name: None)

    def sql(self, statement, args=None):
        rows = next((r for key, r in self.tables.items() if key in statement), [])
        return SimpleNamespace(collect=lambda: rows)


PENDING = [SimpleNamespace(doc_id="d1", path="/Volumes/x/a.pdf")]
CHUNKS = [{"chunk_index": 0, "text": "some words", "embed_text": "x", "kinds": [], "headings": []}]


def test_each_parsed_document_has_a_trace_given_to_parse_pdf(uc, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    seen = []

    def parse(_pdf, *, trace, **_kwargs):
        seen.append(trace)
        trace.assess([Assessment("parse.text_coverage", 0.9, passed=True)])
        return {"n_pages": 1, "doc_id": "d1"}

    parsing.parse_paths(Session(), [str(pdf)], parse=parse, converter=object(), run_key="rk")
    assert seen[0].active
    trace = _traces(tracking.INDEXING)[0]
    assert trace.info.tags["hgrag.pdf"] == "a.pdf" and _names(trace) == ["parse.text_coverage"]


def test_a_failed_parse_is_an_error_trace_and_does_not_stop_the_batch(uc, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")

    def parse(_pdf, **_kwargs):
        raise ValueError("bad pdf")

    summary = parsing.parse_paths(Session(), [str(pdf)], parse=parse, converter=object(), run_key="rk")
    assert summary["failed"] == 1 and "ERROR" in str(_traces(tracking.INDEXING)[0].info.state)


def test_chunking_traces_the_hierarchy_and_chunk_checks(uc, monkeypatch):
    calls = []

    def assess_document(_doc_dir, *, steps, judge):
        calls.append((steps, judge))
        return {"hierarchy": [Assessment("hierarchy.numbering", 1, passed=True),
                              Assessment("hierarchy.outline", 4, source="llm_judge", passed=True)]}

    monkeypatch.setattr(assessments, "assess_document", assess_document)
    monkeypatch.setattr(config, "ASSESS_JUDGES", True)
    (uc / "d1").mkdir()
    summary = indexing.chunk_pending(Session({"LEFT ANTI JOIN": PENDING}), chunker=object(), build=lambda *_: CHUNKS,
                                     run_key="rk")
    assert summary["documents"] == 1 and calls == [(("hierarchy",), True)]
    trace = next(t for t in _traces(tracking.INDEXING) if t.info.tags.get("hgrag.doc_id") == "d1")
    assert _names(trace) == ["chunks.duplicates", "chunks.empty_texts", "chunks.size_distribution",
                             "hierarchy.numbering", "hierarchy.outline"]
    saved = json.loads((uc / "d1" / "assessments.json").read_text(encoding="utf-8"))
    assert saved["chunks"][0]["name"] == "chunks.size_distribution"


def test_a_failing_check_does_not_lose_the_stored_chunks(uc, monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("no document.json")

    monkeypatch.setattr(assessments, "assess_document", boom)
    summary = indexing.chunk_pending(Session({"LEFT ANTI JOIN": PENDING}), chunker=object(), build=lambda *_: CHUNKS,
                                     run_key="rk")
    assert summary["documents"] == 1 and summary["failed"] == {}


@pytest.mark.parametrize("rows, passed", [(7, "true"), (9, "false")])
def test_the_index_rows_are_checked_against_the_table(uc, rows, passed):
    index = SimpleNamespace(wait_until_ready=lambda **_k: None, sync=lambda: None, describe=lambda: {
        "status": {"detailed_state": "ONLINE", "indexed_row_count": 7}})
    client = SimpleNamespace(endpoint_exists=lambda _n: True, index_exists=lambda *_a: True, get_index=lambda *_a: index)
    indexing.sync_index(client, session=Session({"count(*)": [SimpleNamespace(n=rows)]}), run_key="rk")
    check = _traces(tracking.INDEXING)[0].info.assessments[0]
    assert check.name == "index.rows_indexed" and check.metadata["passed"] == passed


def test_reassess_covers_existing_documents_without_parsing_nor_chunking(uc, monkeypatch):
    calls = []

    def assess_document(doc_dir, *, steps, judge):
        calls.append((doc_dir.name, steps, judge))
        return {"parse": [Assessment("parse.text_coverage", 0.9, passed=True)],
                "describe": [Assessment("describe.figure_fidelity", 5, source="llm_judge", passed=True)]}

    monkeypatch.setattr(assessments, "assess_document", assess_document)
    monkeypatch.setattr(config, "ASSESS_JUDGES", True)
    (uc / "d1").mkdir()
    session = Session({"status IN": PENDING, "chunk_index, text": [SimpleNamespace(chunk_index=0, text="some words")]})
    summary = indexing.assess_documents(session, run_key="rk")
    assert summary["documents"] == 1 and summary["failed"] == {}
    assert calls == [("d1", ("parse", "describe", "hierarchy"), True)]
    names = _names(next(t for t in _traces(tracking.INDEXING) if t.info.tags.get("hgrag.doc_id") == "d1"))
    assert "describe.figure_fidelity" in names and "chunks.size_distribution" in names


def test_reassess_goes_on_after_a_failing_document(uc, monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("no document.json")

    monkeypatch.setattr(assessments, "assess_document", boom)
    summary = indexing.assess_documents(Session({"status IN": PENDING}), run_key="rk")
    assert summary["documents"] == 0 and "no document.json" in summary["failed"]["/Volumes/x/a.pdf"]


@pytest.fixture
def llm(monkeypatch, uc):
    (uc / "d1").mkdir()
    (uc / "d1" / "hierarchy.json").write_text(json.dumps(_tree("d1")), encoding="utf-8")

    def stream(_cfg, _system, _user):
        yield {"text": "Answer [1]"}
        yield {"usage": {"input_tokens": 10, "output_tokens": 2, "stop_reason": "end_turn"}}

    monkeypatch.setattr(generation, "stream_completion", stream)


CFG = generation.LLMConfig(provider="anthropic", api_key="test", send_images=False)


def test_a_question_has_the_checks_of_the_search(uc, llm):
    retrieval.ask("q", client=Client([_row("d1", "#/s1", 0, 0.5)]), cfg=CFG, judges=False, rerank=False, run_key="rk")
    trace = _traces(tracking.RETRIEVAL)[0]
    assert "retrieval.top_score" in _names(trace) and any(n.startswith("prompt.") for n in _names(trace))
    assert all(a.source.source_type != "LLM_JUDGE" for a in trace.info.assessments)


def test_a_question_with_judges_has_their_verdicts(uc, llm, monkeypatch):
    from hgrag.assessments import answer as answer_checks
    from hgrag.assessments import retrieval as retrieval_checks

    monkeypatch.setattr(retrieval_checks, "judge_context", lambda *_a, **_k: [
        Assessment("retrieval.context_sufficiency", 5, source="llm_judge", passed=True)])
    monkeypatch.setattr(answer_checks, "judge_answer", lambda *_a, **_k: [
        Assessment("answer.judge", 5, source="llm_judge", passed=True)])
    retrieval.ask("q", client=Client([_row("d1", "#/s1", 0, 0.5)]), cfg=CFG, judges=True, rerank=False, run_key="rk")
    names = _names(_traces(tracking.RETRIEVAL)[0])
    assert "retrieval.context_sufficiency" in names and "answer.judge" in names


def test_the_threshold_of_the_top_score_only_applies_to_the_cross_encoder():
    from hgrag.assessments import retrieval as retrieval_checks

    low = SimpleNamespace(score=0.001, vector_score=0.001)
    assert retrieval_checks.assess(retrieval._diagnostics([low], "huggingface"))[0].passed is False
    assert retrieval_checks.assess(retrieval._diagnostics([low], "ai_search"))[0].passed is None
    assert retrieval_checks.assess(retrieval._diagnostics([low], "none"))[0].passed is None


def test_report_reads_back_what_is_stored(uc, capsys):
    span = tracing.root("question", tracking.RETRIEVAL, "rk")
    with span:
        span.assess([Assessment("answer.ok", 1, passed=True), Assessment("answer.judge", 2, source="llm_judge", passed=False)])
    out = tracing.report(tracking.RETRIEVAL, "rk")
    assert (out["traces"], out["checks"], out["judged"], out["failed"], out["errors"]) == (1, 2, 1, 1, 0)
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["hgrag_mlflow"]["traces"] == 1


def test_report_only_counts_its_execution(uc):
    with tracing.root("question", tracking.RETRIEVAL, "other"):
        pass
    assert tracing.report(tracking.RETRIEVAL, "rk")["traces"] == 0


def test_report_shows_a_store_failure_without_raising(uc, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(mlflow, "search_traces", boom)
    out = tracing.report(tracking.RETRIEVAL, "rk")
    assert "store down" in out["error"] and "store down" in capsys.readouterr().out
