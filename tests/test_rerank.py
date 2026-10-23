"""Reranking, model prefixes and the vector-size guard (offline, fake models)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hgrag import config
from hgrag.indexing import ensure_collection, open_store
from hgrag.rerank import pair_text, rerank_points
from hgrag.retrieval import search_hits, search_steps
from test_retrieval import BagOfWordsEmbedder, store  # noqa: F401  (fixture)


class WordOverlapReranker:
    """Cross-encoder stand-in: score = share of the question words found in the chunk (query, chunk read together)."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, batch_size=16):
        self.pairs += list(pairs)
        return np.array([len(set(q.lower().split()) & set(t.lower().replace(".", " ").split())) / len(q.split())
                         for q, t in pairs], dtype="float32")


def test_prefixes_follow_the_model(monkeypatch):
    monkeypatch.delenv("HGRAG_EMBED_QUERY_PREFIX")
    monkeypatch.delenv("HGRAG_EMBED_DOC_PREFIX")
    assert config.embed_prefixes("BAAI/bge-base-en-v1.5") == ("Represent this sentence for searching relevant passages: ", "")
    assert config.embed_prefixes("intfloat/e5-base-v2") == ("query: ", "passage: ")
    assert config.embed_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
    monkeypatch.setenv("HGRAG_EMBED_QUERY_PREFIX", "Q: ")
    assert config.embed_prefixes("sentence-transformers/all-MiniLM-L6-v2")[0] == "Q: "


def test_query_and_passages_get_their_prefix(store, monkeypatch):  # noqa: F811
    client, _ = store
    monkeypatch.setenv("HGRAG_EMBED_QUERY_PREFIX", "QUERY ")
    seen = []

    class Spy(BagOfWordsEmbedder):
        def encode(self, sentences, **kw):
            seen.extend(sentences)
            return super().encode(sentences, **kw)

    search_hits("settlement delay", 2, embedder=Spy(), client=client, collection="c")
    assert seen == ["QUERY settlement delay"]


def test_rerank_points_keeps_the_best_and_the_vector_score():
    pts = [SimpleNamespace(id=i, score=0.9 - i / 10, payload={"text": t, "headings": ["A", "B", "C"]})
           for i, t in enumerate(["unrelated words here", "average settlement delay dropped", "settlement"])]
    rr = WordOverlapReranker()
    out = rerank_points("average settlement delay", pts, rr, 2)
    assert [p.id for p in out] == [1, 2]
    assert out[0].score == pytest.approx(1.0) and out[0].vector_score == pytest.approx(0.8)
    assert rr.pairs[0][1].startswith("B\nC\n")  # the two closest headings, then the text
    assert pair_text({"text": "x"}) == "x"
    assert rerank_points("q", [], rr, 3) == []


def test_search_fetches_candidates_then_keeps_k(store):  # noqa: F811
    client, _ = store
    rr = WordOverlapReranker()
    events = list(search_steps("settlement delay dropped", 1, embedder=BagOfWordsEmbedder(), client=client,
                               collection="c", rerank=True, reranker=rr, candidates=10))
    steps = [e["step"] for e in events]
    assert steps == ["embed", "vector_search", "rerank", "group"]
    assert len(events[1]["data"]) > 1 and len(events[2]["data"]) == 1  # candidates, then k
    best = events[3]["data"][0]
    assert "settlement" in best["hits"][0]["text"] and "vector_score" in best["hits"][0]
    assert 0 <= best["score"] <= 1


def test_without_rerank_nothing_changes(store):  # noqa: F811
    client, _ = store
    steps = [e["step"] for e in search_steps("settlement", 2, embedder=BagOfWordsEmbedder(), client=client,
                                             collection="c", rerank=False)]
    assert steps == ["embed", "vector_search", "group"]


def test_collection_with_another_vector_size_is_refused(tmp_path):
    client = open_store(tmp_path / "q")
    try:
        ensure_collection(client, "c", 384)
        ensure_collection(client, "c", 384)  # same size: fine
        with pytest.raises(ValueError, match="384-dimension"):
            ensure_collection(client, "c", 768)
    finally:
        client.close()


def test_ragbench_store_folder_depends_on_the_model():
    from hgrag.ragbench import qdrant_dirname

    assert qdrant_dirname("sentence-transformers/all-MiniLM-L6-v2") == "qdrant"
    assert qdrant_dirname("BAAI/bge-base-en-v1.5") == "qdrant-baai-bge-base-en-v1-5"


def test_reranker_reads_the_generated_formula_reading_and_table_description():
    formula = {"headings": ["4 Series", "4.7 Envelope"], "text": r"$$E(t)=\pm C$$",
               "formula_reading": "If the ratio is negative, the envelope takes both signs."}
    assert pair_text(formula).split("\n") == ["4 Series", "4.7 Envelope", "If the ratio is negative, the envelope takes both signs.",
                                              r"$$E(t)=\pm C$$"]
    table = {"text": "| a | b |", "table_description": "Ablation of the enhancement components."}
    assert pair_text(table) == "Ablation of the enhancement components.\n| a | b |"
