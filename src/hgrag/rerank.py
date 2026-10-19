"""Second-stage ranking: a cross-encoder rescores the chunks found by the vector search.

The bi-encoder (embeddings) compares two vectors computed apart; the cross-encoder reads the question and the chunk
together, so it separates a chunk that *talks about* the topic from one that *answers* it. It is slower, so it only
sees the ``RERANK_CANDIDATES`` best chunks of the vector search.
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from . import config


@lru_cache(maxsize=2)
def load_reranker(model_id: str = config.RERANK_MODEL) -> Any:
    """CrossEncoder returning scores in [0, 1] (sigmoid of the logit), cached for the process."""
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, max_length=config.RERANK_MAX_LENGTH, activation_fn=torch.nn.Sigmoid())


def pair_text(payload: dict[str, Any]) -> str:
    """What the cross-encoder reads for a chunk: its two closest headings, then the text."""
    headings = [h for h in (payload.get("headings") or []) if h][-2:]
    return "\n".join([*headings, payload.get("text") or ""])


def rerank_points(query: str, points: list[Any], reranker: Any, top_k: int, *, batch_size: int = 16) -> list[Any]:
    """The ``top_k`` best points by cross-encoder score (highest first). ``score`` becomes the reranker score and the
    vector similarity is kept as ``vector_score``; Qdrant payloads are untouched."""
    if not points:
        return []
    scores = reranker.predict([(query, pair_text(p.payload or {})) for p in points], batch_size=batch_size)
    ranked = sorted(zip(points, (float(s) for s in scores), strict=True), key=lambda t: t[1], reverse=True)
    return [SimpleNamespace(id=p.id, payload=p.payload, score=s, vector_score=float(p.score)) for p, s in ranked[:top_k]]
