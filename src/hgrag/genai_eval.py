"""Test the question-answering application with ``mlflow.genai.evaluate`` on the control PDFs (``hgrag evaluate-rag``).

The store is rebuilt from the frozen parsing of the control PDFs (``evaluation.build_store``), the application is
the registered ``hgrag-rag`` model (or the code of the project when none is given) and every question of
``eval_dataset.jsonl`` becomes one trace in the ``hgrag_eval`` experiment, linked to the model version. Scorers:

* ``hgrag_checks``: the production checks of each step (search, prompt, answer; the LLM judges with ``judges``),
  read back from the trace where the application left them;
* ``correctness``: an LLM judge (Claude Haiku) of the answer against the expected answer of the dataset;
* ``evidence_recall``: share of the evidence passages of the dataset found in the rebuilt sections (deterministic).
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import Any

from . import config, tracking

SCORER_NAMES = ("hgrag_checks", "correctness", "evidence_recall")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _scorers() -> list[Any]:
    from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
    from mlflow.genai.scorers import scorer

    from . import evaluation, tracing
    from .assessments import Assessment

    @scorer(name="hgrag_checks")
    def hgrag_checks(trace) -> list[Any]:
        """The checks each step attached to its span (``tracing.ASSESSMENTS_ATTRIBUTE``), as feedback."""
        out = []
        for span in trace.data.spans:
            for rec in span.attributes.get(tracing.ASSESSMENTS_ATTRIBUTE) or []:
                rec = dict(rec)
                span_id = rec.pop("span_id", None)
                out.append(Assessment(**rec).to_feedback(trace.info.trace_id, span_id))
        return out or [Feedback(name="hgrag_checks", value=None, error="no check found in the trace")]

    @scorer(name="correctness")
    def correctness(inputs, outputs, expectations) -> Any:
        cfg = evaluation.judge_config()
        source = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=cfg.model)
        if not cfg.api_key:
            return Feedback(name="correctness", error="ANTHROPIC_API_KEY is not set", source=source)
        try:
            j = evaluation.judge_answer(cfg, inputs["question"], expectations["expected_answer"],
                                        (outputs or {}).get("answer") or "")
        except Exception as exc:  # noqa: BLE001
            return Feedback(name="correctness", error=str(exc), source=source)
        return Feedback(name="correctness", value=j["score"], rationale=j["rationale"], source=source,
                        metadata={"passed": str(j["score"] >= 4).lower()})

    @scorer(name="evidence_recall")
    def evidence_recall(trace, expectations) -> Any:
        context = _norm(" ".join((s.outputs or {}).get("text") or "" for s in trace.data.spans if s.name == "reconstruct"))
        evidence = expectations.get("evidence") or []
        if not evidence:
            return Feedback(name="evidence_recall", error="no evidence in the dataset row")
        found = [e for e in evidence if _norm(e) in context]
        missing = [e for e in evidence if e not in found]
        return Feedback(name="evidence_recall", value=round(len(found) / len(evidence), 3),
                        rationale=f"{len(found)}/{len(evidence)} evidence passages in the rebuilt sections"
                                  + (f"; missing: {missing}" if missing else ""))

    return [hgrag_checks, correctness, evidence_recall]


def dataset_rows(path: str | os.PathLike | None = None, ids: list[str] | None = None) -> list[dict[str, Any]]:
    """``eval_dataset.jsonl`` as ``mlflow.genai.evaluate`` rows (inputs, expectations, tags)."""
    from . import evaluation

    rows = evaluation.load_dataset(path or config.CONTROL_PDF_DIR / "eval_dataset.jsonl", ids=ids)
    return [{"inputs": {"question": r["question"]},
             "expectations": {"expected_answer": r["expected_answer"], "evidence": r.get("evidence") or []},
             "tags": {"hgrag.question_id": r["id"], "hgrag.source_kind": r["source_kind"], "hgrag.pdf": r["pdf"]}}
            for r in rows]


def evaluate_rag(model_uri: str | None = None, *, judges: bool = False, k: int = 5, ids: list[str] | None = None,
                 workers: int = 1, store: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the evaluation; returns the run id, the model id and the aggregated metrics. ``workers``: questions
    answered in parallel (the local Qdrant store and the models are shared: 1 by default)."""
    import mlflow

    from . import evaluation, versions
    from .mlflow_models import RagModel

    store = store or evaluation.build_store()
    model_config = {"output_dir": str(store["output_dir"]), "qdrant_path": str(store["qdrant_path"]),
                    "collection": store["collection"], "k": k, "judges": judges, "embed_model": config.EMBED_MODEL_ID}
    if model_uri:
        model = mlflow.pyfunc.load_model(model_uri, model_config=model_config)
        model_id = model.metadata.model_id
        predict = model.predict
    else:  # the code of the project, linked to the model registered for this configuration if there is one
        app = RagModel()
        app.load_context(SimpleNamespace(model_config=model_config))
        model_id = versions.model_id("rag")
        predict = lambda x: app.predict(None, x)  # noqa: E731
    try:
        tracking._setup(tracking.EVAL)
        mlflow.set_experiment(tracking.EVAL)  # no live span here: the tracer reset is harmless
        os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(workers)

        @mlflow.trace(name="rag_eval", span_type="CHAIN")
        def predict_fn(question: str) -> dict[str, Any]:
            return predict({"question": question})[0]

        result = mlflow.genai.evaluate(data=dataset_rows(ids=ids), predict_fn=predict_fn, scorers=_scorers(),
                                       model_id=model_id)
    finally:
        client = getattr(locals().get("app"), "client", None)
        if client is not None:
            client.close()
    return {"run_id": result.run_id, "model_id": model_id, "metrics": dict(result.metrics)}


def format_metrics(metrics: dict[str, Any]) -> str:
    return "\n".join(f"{k:<45} {v:.3f}" if isinstance(v, (int, float)) else f"{k:<45} {v}"
                     for k, v in sorted(metrics.items()))
