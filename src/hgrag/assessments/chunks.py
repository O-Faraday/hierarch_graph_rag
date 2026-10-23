"""Step 4 - chunks and indexing: size distribution, duplicates, empty texts, and every chunk stored in Qdrant."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from .. import config
from . import Assessment

_ALNUM = re.compile(r"[^\W_]")


def _quantile(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def size_distribution(chunks: list[dict[str, Any]], *, small_words: int | None = None,
                      max_small_share: float | None = None) -> Assessment:
    """Words per chunk; fails when too many chunks are smaller than ``small_words``: a short chunk carries little
    meaning of its own and wins queries on its headings alone."""
    small_words = config.ASSESS_SMALL_CHUNK_WORDS if small_words is None else small_words
    max_small_share = config.ASSESS_MAX_SMALL_CHUNKS if max_small_share is None else max_small_share
    if not chunks:
        return Assessment("chunks.size_distribution", None, passed=False, rationale="no chunk")
    words = [len((c.get("text") or "").split()) for c in chunks]
    share = sum(w < small_words for w in words) / len(words)
    dist = {"n": len(words), "p10": _quantile(words, 0.1), "median": statistics.median(words),
            "p90": _quantile(words, 0.9), "max": max(words)}
    return Assessment("chunks.size_distribution", round(share, 3), passed=share <= max_small_share, metadata=dist,
                      rationale=f"{share:.0%} of {len(words)} chunks under {small_words} words (threshold "
                                f"{max_small_share:.0%}); words p10 {dist['p10']}, median {dist['median']:g}, "
                                f"p90 {dist['p90']}, max {dist['max']}")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def duplicates(chunks: list[dict[str, Any]], *, max_duplicates: int | None = None) -> Assessment:
    """Chunks of the same document with the same text (case and spaces aside): a running header, an affiliation, a
    block parsed twice. They crowd out the other candidates of the search. A few are normal (a short sentence of a
    proof repeated): fails above ``max_duplicates`` extra copies."""
    max_duplicates = config.ASSESS_MAX_DUPLICATES if max_duplicates is None else max_duplicates
    counts = Counter(_norm(c.get("text") or "") for c in chunks if _norm(c.get("text") or ""))
    dups = {t: n for t, n in counts.items() if n > 1}
    extra = sum(n - 1 for n in dups.values())
    top = sorted(dups.items(), key=lambda kv: -kv[1])[:5]
    return Assessment("chunks.duplicates", extra, passed=extra <= max_duplicates,
                      metadata={"texts": [{"text": t[:120], "count": n} for t, n in top]},
                      rationale=f"{extra} chunk(s) repeating another one (threshold {max_duplicates})"
                                + (": " + "; ".join(f"'{t[:50]}' x{n}" for t, n in top) if top else ""))


def empty_texts(chunks: list[dict[str, Any]]) -> Assessment:
    """Chunks without a letter or a digit (empty, spaces, punctuation, control characters such as the \\x00 that
    Docling extracts from some vector charts)."""
    empty = [c.get("chunk_index") for c in chunks if not _ALNUM.search(c.get("text") or "")]
    return Assessment("chunks.empty_texts", len(empty), passed=not empty, metadata={"chunks": empty[:50]},
                      rationale=f"{len(empty)} chunk(s) without any letter or digit out of {len(chunks)}")


def points_count(client: Any, collection: str, doc_id: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    return client.count(collection_name=collection, count_filter=flt, exact=True).count


def points_indexed(chunks: list[dict[str, Any]], n_points: int | None, *, error: str | None = None) -> Assessment:
    """As many points in the Qdrant collection for the document as chunks written (``chunks.jsonl``): fewer = an
    indexing that stopped half-way, more = points left by an earlier indexing with other chunks."""
    if n_points is None:
        return Assessment("chunks.points_indexed", None, error=error or "Qdrant not available")
    return Assessment("chunks.points_indexed", n_points, passed=n_points == len(chunks),
                      metadata={"chunks": len(chunks)},
                      rationale=f"{n_points} point(s) in Qdrant for {len(chunks)} chunk(s)")


def assess(chunks: list[dict[str, Any]], doc_id: str, *, client: Any = None,
           collection: str | None = None) -> list[Assessment]:
    out = [size_distribution(chunks), duplicates(chunks), empty_texts(chunks)]
    if client is None:
        out.append(points_indexed(chunks, None, error="no Qdrant client given"))
    else:
        collection = collection or config.COLLECTION_NAME
        try:
            n = points_count(client, collection, doc_id) if client.collection_exists(collection) else 0
            out.append(points_indexed(chunks, n))
        except Exception as exc:  # noqa: BLE001
            out.append(points_indexed(chunks, None, error=f"Qdrant: {exc}"))
    return out
