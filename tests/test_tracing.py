"""MLflow traces of the steps (offline, SQLite backend in a temp dir, no model, no API)."""

from __future__ import annotations

import json

import mlflow
import pytest

from hgrag import config, generation, tracing, tracking
from hgrag.retrieval import retrieve_sections, retrieve_steps

from test_generation import SECTIONS, FakeResponse, _anthropic_stream
from test_retrieval import BagOfWordsEmbedder, store  # noqa: F401  (fixture)
from test_tracking import mlflow_dir  # noqa: F401  (fixture)


def _spans(trace_id):
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    return trace, {s.name: s for s in trace.data.spans}


def test_no_trace_without_run_key_or_when_off(mlflow_dir, monkeypatch):  # noqa: F811
    span = tracing.root("x", tracking.RETRIEVAL, None)
    assert not span.active and span.trace_id is None and not span.child("y").active
    span.end({"a": 1})  # no-op
    monkeypatch.setattr(config, "TRACING", False)
    assert not tracing.root("x", tracking.RETRIEVAL, "rk").active


def test_question_trace_from_search_to_answer(store, mlflow_dir, monkeypatch):  # noqa: F811
    client, out = store
    key = tracking.new_run_key()
    root = tracing.root("rag_query", tracking.RETRIEVAL, key, inputs={"question": "new branches"})
    sections = retrieve_sections("new branches portfolio", 3, embedder=BagOfWordsEmbedder(), client=client,
                                 collection="c", output_dir=out, run_key=key, trace=root)
    sent = {}
    monkeypatch.setattr(generation.requests, "post",
                        lambda url, **kw: (sent.update(kw) if url.endswith("/v1/messages") else None)
                        or FakeResponse(_anthropic_stream(["3 new ", "branches [1]."])))
    cfg = generation.LLMConfig(provider="anthropic", api_key="sk-secret-key")
    answer = "".join(generation.answer_stream("new branches", sections, cfg, trace=root))
    root.end({"answer": answer})

    trace, spans = _spans(root.trace_id)
    assert trace.info.trace_metadata["mlflow.trace.session"] == key and trace.info.tags["hgrag.run_key"] == key
    assert mlflow.get_experiment(trace.info.experiment_id).name == tracking.RETRIEVAL
    names = [s.name for s in trace.data.spans]
    for name in ("rag_query", "retrieve", "search", "embed", "dense_search", "keyword_search", "fusion",
                 "reconstruct", "generate", "build_prompt", "llm"):
        assert name in names, name
    parent = {s.span_id: s.name for s in trace.data.spans}
    assert parent[spans["search"].parent_id] == "retrieve" and parent[spans["retrieve"].parent_id] == "rag_query"
    assert parent[spans["embed"].parent_id] == "search" and parent[spans["llm"].parent_id] == "generate"
    assert names.count("reconstruct") == len(sections)
    assert spans["dense_search"].outputs["chunks"][0]["text"]  # full content, not a count
    assert spans["reconstruct"].outputs["text"].startswith("#")
    assert spans["llm"].outputs["answer"] == "3 new branches [1]." and spans["rag_query"].outputs["answer"] == answer
    assert spans["llm"].inputs["model"] == cfg.model and "Sources:" in spans["llm"].inputs["user"]
    dump = json.dumps([s.to_dict() for s in trace.data.spans])
    assert "sk-secret-key" not in dump and sent["headers"]["x-api-key"] == "sk-secret-key"  # the key is never traced


def test_search_alone_is_a_trace_and_stopping_early_is_not_an_error(store, mlflow_dir):  # noqa: F811
    client, out = store
    key = tracking.new_run_key()
    stream = retrieve_steps("settlement delay", 2, embedder=BagOfWordsEmbedder(), client=client, collection="c",
                            output_dir=out, run_key=key)
    next(stream)  # the app read the first event, then moved on
    stream.close()
    mlflow.flush_trace_async_logging()
    exp = mlflow.get_experiment_by_name(tracking.RETRIEVAL)
    traces = mlflow.search_traces(locations=[exp.experiment_id], return_type="list")
    assert len(traces) == 1
    spans = {s.name: s for s in traces[0].data.spans}
    assert spans["retrieve"].attributes.get("closed_early") is True
    assert all(s.status.status_code.name != "ERROR" for s in spans.values())


def test_a_failing_step_is_marked(mlflow_dir):  # noqa: F811
    root = tracing.root("rag_query", tracking.RETRIEVAL, "rk")
    with pytest.raises(ValueError):
        with root.child("search", "RETRIEVER"):
            raise ValueError("Qdrant is locked")
    root.end()
    _trace, spans = _spans(root.trace_id)
    assert spans["search"].status.status_code.name == "ERROR" and spans["rag_query"].status.status_code.name == "OK"


def test_mlflow_failure_never_breaks_the_pipeline(mlflow_dir, monkeypatch):  # noqa: F811
    def broken(*a, **kw):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(mlflow, "start_span_no_context", broken)
    span = tracing.root("rag_query", tracking.RETRIEVAL, "rk")
    assert not span.active
    sections = [dict(SECTIONS[0])]
    msgs = generation.build_messages("q", sections)
    assert msgs["sources"]  # the step itself still runs


def test_hierarchy_step_is_traced(tmp_path, mlflow_dir):  # noqa: F811
    from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size

    from hgrag.pipeline import build_and_save_hierarchy

    doc = DoclingDocument(name="r")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))
    h = doc.add_heading(text="1. Intro", level=1, prov=prov)
    doc.add_text(label="text", text="Some words here.", parent=h, prov=prov)
    d = tmp_path / "doc"
    d.mkdir()
    doc.save_as_json(d / "document.json")
    (d / "figures.json").write_text("[]")
    (d / "manifest.json").write_text(json.dumps({"doc_id": "doc", "source_pdf": "r.pdf"}))
    key = tracking.new_run_key()
    build_and_save_hierarchy(d, run_key=key)
    mlflow.flush_trace_async_logging()
    exp = mlflow.get_experiment_by_name(tracking.INDEXING)
    trace = mlflow.search_traces(locations=[exp.experiment_id], return_type="list")[0]
    span = trace.data.spans[0]
    assert span.name == "hierarchy" and "1. Intro" in span.outputs["tree"] and span.outputs["stats"]["n_sections"] >= 1


# ------------------------------------------------------------------------------- production checks as feedback
def _feedback(trace):
    names = {s.span_id: s.name for s in trace.data.spans}
    return {a.name: (names.get(a.span_id), a) for a in trace.info.assessments}


def test_checks_are_feedback_on_the_span_they_assess(store, mlflow_dir, monkeypatch):  # noqa: F811
    from hgrag.assessments import answer as AN
    from hgrag.assessments import retrieval as RT

    client, out = store
    root = tracing.root("rag_query", tracking.RETRIEVAL, tracking.new_run_key(), inputs={"question": "q"})
    sections = retrieve_sections("new branches portfolio", 3, embedder=BagOfWordsEmbedder(), client=client,
                                 collection="c", output_dir=out, run_key="rk", trace=root)
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: FakeResponse(_anthropic_stream(["3 branches [1]."])))

    def judge(cfg, system, user, **kw):
        if system == RT.CONTEXT_JUDGE_SYSTEM:
            return '{"relevance": {"1": 5}, "sufficiency": 4, "rationale": "enough"}', {}
        assert system == AN.ANSWER_JUDGE_SYSTEM
        return '{"parts": ["count"], "unanswered": [], "relevance": 5, "completeness": 5, "rationale": "ok"}', {}

    monkeypatch.setattr(generation, "complete", judge)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    info: dict = {}
    "".join(generation.answer_stream("new branches", sections, generation.LLMConfig(provider="anthropic", api_key="k"),
                                     info=info, trace=root, judges=True))
    root.end()
    fb = _feedback(mlflow.get_trace(root.trace_id))
    assert fb["retrieval.top_score"][0] == "search" and fb["retrieval.search_agreement"][0] == "search"
    assert fb["prompt.dropped_sections"][0] == "build_prompt" and fb["retrieval.context_sufficiency"][0] == "build_prompt"
    assert fb["answer.citations"][0] == "llm" and fb["answer.completeness"][0] == "generate"
    span, a = fb["retrieval.context_sufficiency"]
    assert a.value == 4 and a.source.source_type == "LLM_JUDGE" and a.rationale == "enough"
    assert fb["answer.citations"][1].source.source_type == "CODE" and fb["answer.citations"][1].metadata["passed"] == "true"
    assert [c["name"] for c in info["context_checks"]] == ["retrieval.context_relevance", "retrieval.context_sufficiency"]
    assert "answer.relevance" in [c["name"] for c in info["answer_checks"]]


def test_judges_are_off_by_default(monkeypatch):
    from hgrag.assessments import answer as AN

    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: FakeResponse(_anthropic_stream(["Yes [1]."])))
    monkeypatch.setattr(AN, "judge_answer", lambda *a, **k: pytest.fail("judge called"))
    info: dict = {}
    "".join(generation.answer_stream("q", SECTIONS, generation.LLMConfig(provider="anthropic", api_key="k"), info=info))
    assert info["context_checks"] == [] and {c["source"] for c in info["answer_checks"]} == {"code"}


def test_hierarchy_checks_are_feedback_and_saved(tmp_path, mlflow_dir):  # noqa: F811
    from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size

    from hgrag.pipeline import build_and_save_hierarchy

    doc = DoclingDocument(name="r")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))
    for title in ("1 Intro", "3 Method"):  # 2 is missing: the numbering check fails
        doc.add_text(label="text", text="Some words here.", parent=doc.add_heading(text=title, level=1, prov=prov), prov=prov)
    d = tmp_path / "doc"
    d.mkdir()
    doc.save_as_json(d / "document.json")
    (d / "figures.json").write_text("[]")
    (d / "manifest.json").write_text(json.dumps({"doc_id": "doc", "source_pdf": "r.pdf"}))
    build_and_save_hierarchy(d, run_key=tracking.new_run_key())
    exp = mlflow.get_experiment_by_name(tracking.INDEXING)
    trace = mlflow.search_traces(locations=[exp.experiment_id], return_type="list")[0]
    fb = _feedback(trace)
    assert fb["hierarchy.numbering"][0] == "hierarchy" and fb["hierarchy.numbering"][1].value == 1
    saved = json.loads((d / "assessments.json").read_text(encoding="utf-8"))
    assert [a["name"] for a in saved["hierarchy"]] == ["hierarchy.numbering", "hierarchy.references_excluded"]


def test_mlflow_ui_runs_one_worker(monkeypatch):
    import subprocess

    from hgrag import cli

    seen = {}
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.update(cmd=cmd) or 0)
    monkeypatch.setattr("sys.argv", ["hgrag", "mlflow-ui"])
    with pytest.raises(SystemExit):
        cli.main()
    assert seen["cmd"][-2:] == ["--workers", "1"] and "--backend-store-uri" in seen["cmd"]


def test_a_check_with_nothing_to_check_is_not_sent(mlflow_dir):  # noqa: F811
    from hgrag.assessments import Assessment

    root = tracing.root("parse", tracking.INDEXING, "rk")
    root.assess([Assessment("describe.figure_descriptions", None, rationale="no figure descriptions"),  # nothing to check
                 Assessment("describe.table_descriptions", 1.0, passed=True),
                 Assessment("parse.text_coverage", None, error="source PDF not found")])  # a real error is kept
    root.end()
    names = sorted(a.name for a in mlflow.get_trace(root.trace_id).info.assessments)
    assert names == ["describe.table_descriptions", "parse.text_coverage"]


def test_embedding_model_missing_offline_says_how_to_fix(monkeypatch):
    import sentence_transformers

    from hgrag.indexing import load_embedder

    def missing(*a, **k):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", missing)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(OSError, match="HGRAG_EMBED_MODEL"):
        load_embedder("BAAI/bge-base-en-v1.5")


def test_index_step_goes_under_the_trace_of_its_pdf(tmp_path, mlflow_dir, monkeypatch):  # noqa: F811
    from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size

    from hgrag.pipeline import build_and_save_hierarchy, index_documents

    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

    monkeypatch.setattr("hgrag.indexing.load_embedder", lambda *a, **k: BagOfWordsEmbedder())
    monkeypatch.setattr("hgrag.indexing.make_chunker", lambda *a, **k: HierarchicalChunker())  # no tokenizer download
    doc = DoclingDocument(name="r")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))
    h = doc.add_heading(text="1 Intro", level=1, prov=prov)
    doc.add_text(label="text", text="The bank opened three new branches in the north last year.", parent=h, prov=prov)
    d = tmp_path / "doc"
    d.mkdir()
    doc.save_as_json(d / "document.json")
    (d / "figures.json").write_text("[]")
    (d / "manifest.json").write_text(json.dumps({"doc_id": "doc", "source_pdf": "r.pdf"}))
    root = tracing.root("indexing", tracking.INDEXING, "rk", tags={"hgrag.pdf": "r.pdf"})
    build_and_save_hierarchy(d, trace=root)
    index_documents([str(d)], qdrant_path=tmp_path / "q", collection="c", run_key="rk", doc_traces={str(d): root})
    root.end()
    trace, spans = _spans(root.trace_id)
    by_id = {s.span_id: s.name for s in trace.data.spans}
    assert [by_id[s.parent_id] for s in trace.data.spans if s.name in ("hierarchy", "index")] == ["indexing", "indexing"]
    assert by_id[spans["chunk"].parent_id] == "index" and by_id[spans["embed_and_store"].parent_id] == "index"
    fb = _feedback(trace)
    assert fb["chunks.points_indexed"][0] == "index" and fb["hierarchy.numbering"][0] == "hierarchy"
    exp = mlflow.get_experiment_by_name(tracking.INDEXING)
    assert len(mlflow.search_traces(locations=[exp.experiment_id], return_type="list")) == 1  # no separate index trace
