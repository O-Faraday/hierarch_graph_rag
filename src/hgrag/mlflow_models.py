"""The two applications as MLflow pyfunc models (``hgrag register`` logs them "from code", ``model_code/``).

* ``RagModel``: a question -> the answer, its numbered sources and the id of its trace. One ``rag_query`` trace per
  question (search, reranking, rebuilt sections, prompt, answer) carrying the production checks.
* ``IndexingModel``: PDFs -> parse, hierarchy and index (``runner.run_indexing``), one ``indexing`` trace.

``model_config`` (set at logging, overridable at loading with ``mlflow.pyfunc.load_model(..., model_config=...)``)
points to the stores; the models themselves (embedding, reranker, LLM) follow the project configuration, recorded
as the model params (``versions.config_snapshot``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from . import config, tracking


def _questions(model_input: Any) -> list[str]:
    """A question, a list of them, ``{"question": ...}`` or a DataFrame with a ``question`` column."""
    if isinstance(model_input, str):
        return [model_input]
    if isinstance(model_input, dict):
        q = model_input.get("question")
        return list(q) if isinstance(q, (list, tuple)) else [q]
    if hasattr(model_input, "columns"):  # pandas DataFrame
        return [str(q) for q in model_input["question"].tolist()]
    return [str(q) for q in model_input]


def answer_question(question: str, *, embedder: Any, client: Any, collection: str, output_dir: str | Path, k: int,
                    reranker: Any = None, rerank: bool | None = None, judges: bool = False,
                    llm: Any = None) -> dict[str, Any]:
    """Search, rebuild the sections, answer: one ``rag_query`` trace with its checks."""
    from . import generation, tracing
    from .retrieval import retrieve_sections

    run_key = tracking.new_run_key()
    trace = tracing.root("rag_query", tracking.RETRIEVAL, run_key, inputs={"question": question})
    with trace:
        sections = retrieve_sections(question, k, embedder=embedder, client=client, collection=collection,
                                     output_dir=output_dir, reranker=reranker, rerank=rerank, run_key=None, trace=trace)
        info: dict[str, Any] = {}
        answer = "".join(generation.answer_stream(question, sections, llm or generation.LLMConfig(provider="anthropic"),
                                                  info=info, trace=trace, judges=judges)) if sections else ""
        out = {"answer": answer, "sources": info.get("sources", []), "trace_id": trace.trace_id}
        trace.set(out)
    return out


class RagModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from .indexing import load_embedder, open_store
        from .rerank import load_reranker

        cfg = dict(context.model_config or {})
        self.k = int(cfg.get("k", 10))
        self.judges = str(cfg.get("judges", config.ASSESS_JUDGES)).lower() in ("1", "true", "yes")
        self.output_dir = cfg.get("output_dir") or str(config.OUTPUT_DIR)
        self.collection = cfg.get("collection") or config.COLLECTION_NAME
        self.embedder = load_embedder(cfg.get("embed_model") or config.EMBED_MODEL_ID)
        self.reranker = load_reranker(config.RERANK_MODEL) if config.RERANK else None
        self.client = open_store(cfg.get("qdrant_path") or config.QDRANT_PATH)  # Qdrant local: one process per store

    # model_input without type hint: MLflow would infer a signature from it and requires list[...] hints
    def predict(self, context: Any, model_input, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return [answer_question(q, embedder=self.embedder, client=self.client, collection=self.collection,
                                output_dir=self.output_dir, k=self.k, reranker=self.reranker, judges=self.judges)
                for q in _questions(model_input)]


class IndexingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        cfg = dict(context.model_config or {})
        self.describe_pictures = str(cfg.get("describe_pictures", config.DESCRIBE_PICTURES)).lower() in ("1", "true", "yes")
        self.steps = [s for s in str(cfg.get("steps", "parse,hierarchy,index")).split(",") if s]

    def predict(self, context: Any, model_input, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """``{"pdf_dir": ..., "pdfs": [names]}`` or a list of PDF paths (in one folder)."""
        from . import runner

        if isinstance(model_input, dict):
            pdf_dir, pdfs = model_input.get("pdf_dir") or str(config.PDF_DIR), model_input.get("pdfs")
        else:
            paths = [Path(p) for p in (model_input if isinstance(model_input, (list, tuple)) else [model_input])]
            pdf_dir, pdfs = str(paths[0].parent), [p.name for p in paths]
        status = runner.run_indexing(tracking.new_run_key(), pdf_dir, describe_pictures=self.describe_pictures,
                                     steps=self.steps, pdfs=pdfs)
        return {"run_key": status["run_key"], "state": status["state"], "error": status.get("error"),
                "tasks": {t: runner.task_summary(v) for t, v in status["tasks"].items()}}
