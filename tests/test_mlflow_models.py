"""Versions, pyfunc models, registry and genai evaluation (offline: no model, no API, SQLite MLflow in a temp dir)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mlflow
import pytest

from hgrag import config, evaluation, generation, genai_eval, mlflow_models, registry, tracing, tracking, versions
from hgrag.assessments import Assessment

from test_generation import FakeResponse, _anthropic_stream
from test_retrieval import BagOfWordsEmbedder, store  # noqa: F401  (fixture)
from test_tracking import mlflow_dir  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _fresh_ids():
    versions._IDS.clear()
    yield
    versions._IDS.clear()


def test_a_version_is_the_configuration_that_changes_the_output(monkeypatch):
    rag = versions.config_snapshot("rag")
    assert {"embed_model", "rerank_model", "llm_model", "system_prompt_sha", "keyword_weight"} <= set(rag)
    assert all(isinstance(v, str) for v in rag.values())
    idx = versions.config_snapshot("indexing")
    assert {"table_mode", "figure_vlm", "chunk_max_tokens", "embed_model", "excluded_sections"} <= set(idx)
    before = versions.config_hash("rag")
    monkeypatch.setattr(config, "KEYWORD_WEIGHT", 0.5)
    assert versions.config_hash("rag") != before
    with pytest.raises(ValueError):
        versions.config_snapshot("other")


def test_traces_are_linked_to_the_model_of_their_configuration(mlflow_dir):  # noqa: F811
    assert versions.model_id("rag") is None
    exp = tracking._setup(tracking.RETRIEVAL)
    m = mlflow.initialize_logged_model(name="hgrag-rag", experiment_id=exp, tags={versions.HASH_TAG: versions.config_hash("rag")})
    versions._IDS.clear()
    assert versions.model_id("rag") == m.model_id
    root = tracing.root("rag_query", tracking.RETRIEVAL, "rk")
    root.end()
    mlflow.flush_trace_async_logging()
    assert mlflow.get_trace(root.trace_id).info.trace_metadata["mlflow.modelId"] == m.model_id


def test_register_logs_a_pyfunc_model_once_per_configuration(mlflow_dir):  # noqa: F811
    first = registry.register("rag")
    assert first["name"] == "hgrag-rag" and first["version"] == 1 and not first["reused"]
    model = mlflow.get_logged_model(first["model_id"])
    assert model.tags[versions.HASH_TAG] == versions.config_hash("rag") and model.params["embed_model"] == config.EMBED_MODEL_ID
    again = registry.register("rag")
    assert again["reused"] and again["model_id"] == first["model_id"] and again["version"] == 1
    assert registry.register("rag", force=True)["version"] == 2


def test_rag_model_answers_and_traces(store, mlflow_dir, monkeypatch):  # noqa: F811
    client, out = store
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: FakeResponse(_anthropic_stream(["3 branches [1]."])))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    res = mlflow_models.answer_question("new branches portfolio", embedder=BagOfWordsEmbedder(), client=client,
                                        collection="c", output_dir=out, k=3)
    assert res["answer"] == "3 branches [1]." and res["sources"][0]["n"] == 1
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(res["trace_id"])
    assert trace.data.spans[0].name == "rag_query" and {a.name for a in trace.info.assessments} >= {
        "retrieval.top_score", "prompt.dropped_sections", "answer.citations"}
    assert mlflow_models._questions("q") == ["q"] and mlflow_models._questions({"question": ["a", "b"]}) == ["a", "b"]


def test_indexing_model_runs_the_pipeline(monkeypatch):
    from types import SimpleNamespace

    from hgrag import runner

    seen = {}

    def fake_run(run_key, pdf_dir, **kw):
        seen.update(pdf_dir=pdf_dir, **kw)
        return {"run_key": run_key, "state": runner.SUCCESS, "tasks": {}}

    monkeypatch.setattr(runner, "run_indexing", fake_run)
    model = mlflow_models.IndexingModel()
    model.load_context(SimpleNamespace(model_config={"describe_pictures": "false", "steps": "hierarchy,index"}))
    out = model.predict(None, ["/data/pdfs/a.pdf", "/data/pdfs/b.pdf"])
    assert out["state"] == runner.SUCCESS and seen["pdfs"] == ["a.pdf", "b.pdf"] and seen["steps"] == ["hierarchy", "index"]
    assert seen["describe_pictures"] is False


def test_inside_an_mlflow_trace_our_trace_nests_and_keeps_its_checks(mlflow_dir):  # noqa: F811
    tracking._setup(tracking.EVAL)

    @mlflow.trace(name="outer")
    def outer():
        root = tracing.root("rag_query", tracking.RETRIEVAL, "rk")
        child = root.child("search")
        child.assess([Assessment("retrieval.top_score", 0.9, passed=True)])
        child.end()
        root.end({"answer": "a"})
        return root.trace_id

    trace_id = outer()
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    spans = {s.name: s for s in trace.data.spans}
    assert spans["rag_query"].parent_id == spans["outer"].span_id
    assert trace.info.assessments == []  # the enclosing trace was not stored yet: no feedback logged
    rec = spans["rag_query"].attributes[tracing.ASSESSMENTS_ATTRIBUTE]
    assert rec[0]["name"] == "retrieval.top_score" and rec[0]["span_id"] == spans["search"].span_id


def test_spans_from_threads_keep_the_trace_under_evaluate(mlflow_dir):  # noqa: F811
    tracking._setup(tracking.EVAL)
    mlflow.set_experiment(tracking.EVAL)

    def work(root):
        with root.child("reconstruct"):
            pass

    @mlflow.trace(name="predict")
    def predict(question):
        root = tracing.root("rag_query", tracking.RETRIEVAL, "rk")
        with ThreadPoolExecutor(2) as pool:
            [f.result() for f in [tracing.submit(pool, work, root) for _ in range(2)]]
        root.end({"answer": question})
        return {"answer": question}

    result = mlflow.genai.evaluate(data=[{"inputs": {"question": f"q{i}"}} for i in range(3)], predict_fn=predict,
                                   scorers=[])
    assert len(mlflow.search_traces(run_id=result.run_id, return_type="list")) == 3


def test_scorers(monkeypatch, mlflow_dir):  # noqa: F811
    hgrag_checks, correctness, evidence_recall = genai_eval._scorers()
    tracking._setup(tracking.EVAL)

    @mlflow.trace(name="outer")
    def run():
        root = tracing.root("rag_query", tracking.RETRIEVAL, "rk")
        with root.child("reconstruct") as sp:
            sp.set({"text": "Net profit of 412 million euros, up nine percent."})
        root.assess([Assessment("answer.citations", 2, passed=True)])
        root.end()
        return root.trace_id

    trace_id = run()
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    fb = hgrag_checks(trace=trace)
    assert [f.name for f in fb] == ["answer.citations"] and fb[0].value == 2
    ev = evidence_recall(trace=trace, expectations={"evidence": ["net profit of 412 million euros", "dividend of 1.80"]})
    assert ev.value == 0.5 and "dividend of 1.80" in ev.rationale
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(evaluation, "judge_answer", lambda cfg, q, gt, a: {"score": 4, "rationale": "close"})
    c = correctness(inputs={"question": "q"}, outputs={"answer": "a"}, expectations={"expected_answer": "e"})
    assert c.value == 4 and c.source.source_type == "LLM_JUDGE" and c.metadata["passed"] == "true"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert "ANTHROPIC_API_KEY" in correctness(inputs={"question": "q"}, outputs={}, expectations={"expected_answer": "e"}).error.error_message


def test_evaluate_rag_on_the_control_dataset(store, mlflow_dir, monkeypatch):  # noqa: F811
    client, out = store
    monkeypatch.setattr("hgrag.indexing.load_embedder", lambda *a, **k: BagOfWordsEmbedder())
    monkeypatch.setattr("hgrag.indexing.open_store", lambda *a, **k: client)
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: FakeResponse(_anthropic_stream(["412 million [1]."])))
    monkeypatch.setattr(evaluation, "judge_answer", lambda cfg, q, gt, a: {"score": 5, "rationale": "right"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    res = genai_eval.evaluate_rag(ids=["q01", "q02"], store={"output_dir": out, "qdrant_path": "unused", "collection": "c"})
    assert res["metrics"]["correctness/mean"] == 5 and "evidence_recall/mean" in res["metrics"]
    assert "answer.citations/mean" in res["metrics"]  # the production checks, read back from the traces
    traces = mlflow.search_traces(run_id=res["run_id"], return_type="list")
    assert sorted(t.info.tags["hgrag.question_id"] for t in traces) == ["q01", "q02"]
