"""Keyword side of the hybrid search: BM25 sparse vectors stored next to the dense vector in Qdrant.

A chunk gets one weight per term: its frequency, saturated and normalised by the chunk length (BM25, k1 and b below).
The collection's sparse vector has the IDF modifier, so Qdrant multiplies each weight by the term's inverse document
frequency over the collection: the score of a query (each of its terms with weight 1) is the BM25 score of the chunk.
Rare terms (Laplace, Cauchy, b/τ) therefore weigh much more than the words shared by a whole paper.

Terms are lowercase word tokens (Greek letters and digits included) without English stop words, hashed to 31-bit ids:
no vocabulary to store, and collisions are negligible at this scale.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from typing import Any

from qdrant_client.models import SparseVector

SPARSE_NAME = "keywords"  # name of the sparse vector in the collection
K1, B = 1.2, 0.75
AVG_WORDS = 60.0  # typical chunk length in words (chunks are cut at 256 tokens); fixed, so indexing stays per document

_TOKEN = re.compile(r"\w+")
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves
""".split())


def terms(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOP_WORDS]


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def _vector(weights: dict[int, float]) -> SparseVector | None:
    if not weights:
        return None
    ids = sorted(weights)
    return SparseVector(indices=ids, values=[weights[i] for i in ids])


def document_vector(text: str) -> SparseVector | None:
    """BM25 term weights of a chunk (without the IDF, applied by Qdrant); None if it has no term."""
    toks = terms(text)
    norm = K1 * (1 - B + B * len(toks) / AVG_WORDS)
    weights: dict[int, float] = {}
    for term, tf in Counter(toks).items():
        i = term_id(term)
        weights[i] = weights.get(i, 0.0) + tf * (K1 + 1) / (tf + norm)
    return _vector(weights)


def query_vector(text: str) -> SparseVector | None:
    """Each distinct term of the query with weight 1; None if it has no term."""
    return _vector({term_id(t): 1.0 for t in set(terms(text))})


def has_keywords(client: Any, collection: str) -> bool:
    """The collection holds keyword vectors (made after the hybrid search was added): hybrid search possible."""
    if not client.collection_exists(collection):
        return False
    return SPARSE_NAME in (client.get_collection(collection).config.params.sparse_vectors or {})
