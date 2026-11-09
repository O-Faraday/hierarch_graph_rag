"""Second-stage ranking: a cross-encoder rescores the chunks found by the vector search.

The bi-encoder (embeddings) compares two vectors computed apart; the cross-encoder reads the question and the chunk
together, so it separates a chunk that *talks about* the topic from one that *answers* it. It is slower, so it only
sees the ``RERANK_CANDIDATES`` best chunks of the vector search.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import requests

from . import config

HF_RANKING_URL = "https://router.huggingface.co/hf-inference/models/{model}"


class RerankError(RuntimeError):
    pass


class HFRouterReranker:
    """The cross-encoder through the Hugging Face API, for a compute without sentence-transformers: same ``predict``
    as ``load_reranker``, same scores in [0, 1] (the API returns the logits). The server truncates long texts."""

    def __init__(self, model_id: str | None = None, *, token: str | None = None, timeout: float = 60.0) -> None:
        self.model_id = model_id or config.RERANK_MODEL
        self.token = token
        self.timeout = timeout

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 0) -> list[float]:
        """Scores of the (query, text) pairs, in their order. ``batch_size`` is ignored: one request per query."""
        token = self.token or os.environ.get("HF_TOKEN")
        if not token:
            raise RerankError("HF_TOKEN is not set (token with the 'Make calls to Inference Providers' permission).")
        queries = {query for query, _ in pairs}
        if len(queries) != 1:
            raise RerankError(f"expected the pairs of one query, got {len(queries)} queries")
        texts = [text for _, text in pairs]
        response = requests.post(HF_RANKING_URL.format(model=self.model_id), timeout=self.timeout,
                                 headers={"Authorization": f"Bearer {token}"},
                                 json={"inputs": {"query": pairs[0][0], "texts": texts}})
        if not response.ok:
            raise RerankError(f"Hugging Face reranking with {self.model_id}: HTTP {response.status_code} "
                              f"{response.text[:200]}")
        ranking = response.json()
        scores = {item["index"]: item["score"] for item in ranking}
        if set(scores) != set(range(len(texts))):
            raise RerankError(f"Hugging Face reranking with {self.model_id}: {len(scores)} scores for {len(texts)} texts")
        return [1.0 / (1.0 + math.exp(-scores[i])) for i in range(len(texts))]


@lru_cache(maxsize=2)
def load_reranker(model_id: str = config.RERANK_MODEL) -> Any:
    """CrossEncoder returning scores in [0, 1] (sigmoid of the logit), cached for the process."""
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, max_length=config.RERANK_MAX_LENGTH, activation_fn=torch.nn.Sigmoid())


def pair_text(payload: dict[str, Any]) -> str:
    """What the cross-encoder reads for a chunk: its two closest headings, the generated table description or formula
    reading if any (a LaTeX formula or a table in Markdown alone scores near 0), then the text."""
    headings = [h for h in (payload.get("headings") or []) if h][-2:]
    generated = [payload[k] for k in ("table_description", "formula_reading") if payload.get(k)]
    return "\n".join([*headings, *generated, payload.get("text") or ""])


def rerank_points(query: str, points: list[Any], reranker: Any, top_k: int, *, batch_size: int = 16) -> list[Any]:
    """The ``top_k`` best points by cross-encoder score (highest first). ``score`` becomes the reranker score and the
    vector similarity is kept as ``vector_score``; Qdrant payloads are untouched."""
    if not points:
        return []
    scores = reranker.predict([(query, pair_text(p.payload or {})) for p in points], batch_size=batch_size)
    ranked = sorted(zip(points, (float(s) for s in scores), strict=True), key=lambda t: t[1], reverse=True)
    return [SimpleNamespace(id=p.id, payload=p.payload, score=s, vector_score=float(p.score)) for p, s in ranked[:top_k]]
