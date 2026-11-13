"""Databricks question flow: AI Search finds the chunks, the rest is the local flow.

Chunks ranked by the AI Search reranker, or by a cross-encoder through the Hugging Face API on a workspace without it
-> grouped by section -> sections rebuilt from ``hierarchy.json`` on the volume -> sourced answer by the LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from .. import config, tracing, tracking
from ..assessments import retrieval as retrieval_checks
from ..generation import LLMConfig, answer_stream
from ..rerank import HFRouterReranker, rerank_points
from ..retrieval import _group_points, first_sections, reconstruct_hits
from . import indexing

# The AI Search reranker reads the first 2000 characters of these columns, in this order: the short and generated ones first.
RERANK_COLUMNS = ["section_title", "table_description", "formula_reading", "text"]


def _point(row: dict[str, Any], score: float | None = None) -> SimpleNamespace:
    """A chunk of AI Search; ``score`` replaces the score of the service, which stays in ``vector_score``."""
    payload = {**row, "source_pdf": row.get("path"), "kinds": json.loads(row.get("kinds") or "[]"),
               "headings": json.loads(row.get("headings") or "[]")}
    raw = float(row.get("score") or 0.0)
    return SimpleNamespace(id=row.get("chunk_id"), payload=payload, score=raw if score is None else score, vector_score=raw)


def _reranker_not_enabled(error: Exception) -> bool:
    """Only the refusal of a workspace without the reranker: a wrong column or a denied access must stay visible, and
    must not send documents to an external service."""
    from databricks.ai_search.exceptions import BadRequest

    message = str(error).lower()
    return isinstance(error, BadRequest) and "rerank" in message and "enabled" in message


def _native_search(query: str, k: int, client: Any) -> list[dict[str, Any]] | None:
    """The ``k`` best chunks by the AI Search reranker, ``None`` if the workspace does not have it."""
    from databricks.ai_search.reranker import DatabricksReranker

    try:
        return indexing.search(query, k, client=client, reranker=DatabricksReranker(columns_to_rerank=RERANK_COLUMNS))
    except Exception as error:
        if _reranker_not_enabled(error):
            return None
        raise


def _diagnostics(points: list[Any], backend: str) -> dict[str, Any]:
    """What the checks of the search read: only the score of the Hugging Face cross-encoder has a known scale, so the
    threshold of ``retrieval.top_score`` only applies to it."""
    if not points:
        return {"top_score": None}
    if backend == "huggingface":
        return {"top_score": points[0].score, "top_score_kind": "rerank"}
    return {"top_score": points[0].vector_score, "top_score_kind": backend}


def search_sections(query: str, k: int = 10, *, rerank: bool | None = None, reranker: Any = None, client: Any = None,
                    run_key: str | None = None, trace: tracing.Span | None = None,
                    k_sections: int | None = None) -> list[dict[str, Any]]:
    """The sections of the ``k`` best chunks, best first. The AI Search reranker is tried first; on a workspace without
    it, AI Search fetches ``RERANK_CANDIDATES`` chunks and the cross-encoder of Hugging Face (or ``reranker``) keeps
    ``k``. The order of the service is kept whatever its score means, so the score follows the rank.
    With ``k_sections`` (``HGRAG_K_SECTIONS`` by default), ``RERANK_CANDIDATES`` ranked chunks are read and the first
    ``k_sections`` sections met are kept, as on the local backend: ``k`` is then unused."""
    if not query.strip():
        raise ValueError("empty query")
    use_rerank = config.DBX_RERANK if rerank is None else rerank
    n_sections = config.K_SECTIONS if k_sections is None else k_sections
    if n_sections:
        k = max(k, config.RERANK_CANDIDATES)
    span = (trace or tracing.Span()).child("search", "RETRIEVER", inputs={"query": query, "k": k, "rerank": use_rerank,
                                                                          "k_sections": n_sections})
    with span, tracking.track_step(tracking.RETRIEVAL, "search", run_key, query=query, k=k, rerank=use_rerank,
                                   k_sections=n_sections) as tracker:
        rows = _native_search(query, k, client) if use_rerank else indexing.search(query, k, client=client)
        backend = "ai_search" if use_rerank else "none"
        if rows is None:
            backend = "huggingface"
            rows = indexing.search(query, max(k, config.RERANK_CANDIDATES), client=client)
            with span.child("rerank", "RERANKER", inputs={"model": config.RERANK_MODEL, "candidates": len(rows), "top_k": k}) as sp:
                points = rerank_points(query, [_point(row) for row in rows], reranker or HFRouterReranker(), k)
                sp.set({"chunk_ids": [p.id for p in points], "scores": [p.score for p in points]})
        else:
            points = [_point(row, 1.0 / rank) for rank, row in enumerate(rows, start=1)]
        sections = _group_points(first_sections(points, n_sections) if n_sections else points)
        span.set({"chunks": rows, "reranker": backend})
        span.assess(retrieval_checks.assess(_diagnostics(points, backend)))
        tracker.metrics(n_chunks=len(rows), n_kept=len(points), n_sections=len(sections))
        tracker.dict({"backend": backend, "model": config.RERANK_MODEL if backend == "huggingface" else None}, "reranker.json")
        tracker.dict(sections, "sections.json")
    print({"hgrag_reranker": backend})
    return sections


def ask(question: str, k: int = 10, *, rerank: bool | None = None, reranker: Any = None, include_children: bool = False,
        cfg: LLMConfig | None = None, judges: bool | None = None, client: Any = None,
        run_key: str | None = None) -> dict[str, Any]:
    """Answer a question from the index: the answer, its numbered sources, usage and production checks. Without any
    chunk found the LLM is not called."""
    span = tracing.root("question", tracking.RETRIEVAL, run_key, inputs={"question": question, "k": k})
    with span:
        hits = search_sections(question, k, rerank=rerank, reranker=reranker, client=client, run_key=run_key, trace=span)
        sections = reconstruct_hits(hits, include_children=include_children, output_dir=config.dbx_output_dir(),
                                    run_key=run_key, trace=span)
        info: dict[str, Any] = {"sources": []}
        answer = ""
        if sections:
            answer = "".join(answer_stream(question, sections, cfg, run_key=run_key, info=info, trace=span, judges=judges))
        span.set({"answer": answer, "sources": info["sources"]})
    if run_key is not None:
        tracking.finish_execution(tracking.RETRIEVAL, run_key, n_sections=len(sections))
    return {**info, "question": question, "answer": answer}
