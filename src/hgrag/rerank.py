"""Second-stage ranking: a cross-encoder rescores the chunks found by the vector search.

The bi-encoder (embeddings) compares two vectors computed apart; the cross-encoder reads the question and the chunk
together, so it separates a chunk that *talks about* the topic from one that *answers* it. It is slower, so it only
sees the ``RERANK_CANDIDATES`` best chunks of the vector search.
"""

from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import requests

from . import config

HF_ROUTER = "https://router.huggingface.co/hf-inference/models"


class RemoteReranker:
    """A reranker served by the Hugging Face router (``hf:<model>``, task text-ranking), for models too slow on a CPU
    (Qwen3-Reranker-0.6B: about 5 minutes per question locally). Same interface as a CrossEncoder: ``predict(pairs)``
    returns one score in [0, 1] per (query, text) pair, the sigmoid of the raw score the API returns."""

    def __init__(self, model_id: str, token: str | None = None, *, timeout: float = 120.0, retries: int = 3,
                 max_texts: int = 32):
        self.model_id, self.timeout, self.retries, self.max_texts = model_id, timeout, retries, max_texts
        self.token = token or os.environ.get("HF_TOKEN")  # never logged nor stored
        if not self.token:
            raise RuntimeError("HF_TOKEN is not set: needed for a reranker served by the Hugging Face router.")

    def _rank(self, query: str, texts: list[str]) -> list[float]:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(f"{HF_ROUTER}/{self.model_id}", timeout=self.timeout,
                                     headers={"Authorization": f"Bearer {self.token}"},
                                     json={"inputs": {"query": query, "texts": texts}})
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                scores = [0.0] * len(texts)
                for item in resp.json():
                    scores[item["index"]] = 1 / (1 + math.exp(-float(item["score"])))
                return scores
            except (requests.RequestException, RuntimeError) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"reranker {self.model_id}: {last}")

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16, **_kw: Any) -> list[float]:
        out: list[float] = []
        i = 0
        while i < len(pairs):  # consecutive pairs of the same query go in one request
            query = pairs[i][0]
            j = i
            while j < len(pairs) and pairs[j][0] == query and j - i < self.max_texts:
                j += 1
            out += self._rank(query, [t for _q, t in pairs[i:j]])
            i = j
        return out


@lru_cache(maxsize=2)
def load_reranker(model_id: str = config.RERANK_MODEL) -> Any:
    """CrossEncoder returning scores in [0, 1] (sigmoid of the logit), cached for the process; ``hf:<model>`` calls the
    Hugging Face router instead (RemoteReranker)."""
    if model_id.startswith("hf:"):
        return RemoteReranker(model_id[3:])
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
