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
