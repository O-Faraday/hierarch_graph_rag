"""Hybrid search: BM25 keyword vectors next to the dense vector, fused with HGRAG_KEYWORD_WEIGHT (offline, no model)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from hgrag import config, keywords
from hgrag.indexing import index_chunks, load_embedder
from hgrag.retrieval import fuse, search_hits, search_steps
from test_retrieval import BagOfWordsEmbedder, store  # noqa: F401


class ConstantEmbedder:
    """Every text gets the same direction: the dense side cannot tell chunks apart, only the keywords can."""

    def encode(self, sentences, **kwargs):
        out = np.ones((len(sentences), 8), dtype="float32")
        return out / np.linalg.norm(out[0])


def _chunks(texts):
    return [{"chunk_index": i, "doc_id": "d", "text": t, "embed_text": t, "section_path": f"p > s{i}", "section_ref": f"#/s{i}",
             "page_no": 1} for i, t in enumerate(texts)]


def test_bm25_weights_ignore_stop_words_and_saturate():
    v = keywords.document_vector("The step size of the Laplace target and the Laplace tails")
    weights = dict(zip(v.indices, v.values))
    assert keywords.term_id("the") not in weights and keywords.term_id("of") not in weights
    assert weights[keywords.term_id("laplace")] > weights[keywords.term_id("step")]  # tf 2 > tf 1, below 2x (saturation)
    assert weights[keywords.term_id("laplace")] < 2 * weights[keywords.term_id("step")]
    longer = keywords.document_vector("step " + "filler " * 200)
    assert dict(zip(longer.indices, longer.values))[keywords.term_id("step")] < weights[keywords.term_id("step")]
    assert keywords.terms("b/τ>0 and E(t)") == ["b", "τ", "0", "e", "t"]
    assert keywords.query_vector("the of and") is None and keywords.document_vector("") is None
    q = keywords.query_vector("Laplace laplace Cauchy")
    assert sorted(q.indices) == q.indices and q.values == [1.0, 1.0]


def test_fusion_weights_normalised_scores():
    dense = [SimpleNamespace(id=i, score=s, payload={"n": i}) for i, s in (("a", 0.9), ("b", 0.8), ("c", 0.5))]
    kw = [SimpleNamespace(id=i, score=s, payload={"n": i}) for i, s in (("c", 12.0), ("d", 6.0))]
    out = fuse(dense, kw, 0.3, 3)
    # a: 0.7*1, b: 0.7*0.75, c: 0.7*0 + 0.3*1, d: 0.3*0
    assert [p.id for p in out] == ["a", "b", "c"] and out[0].score == 0.7 and out[2].keyword_score == 12.0
    assert out[0].keyword_score is None and out[2].dense_score == 0.5 and out[0].payload == {"n": "a"}
    assert [p.id for p in fuse(dense, kw, 0.0, 4)][:3] == ["a", "b", "c"]
    assert [p.id for p in fuse(dense, kw, 1.0, 2)] == ["c", "d"]


def test_rare_term_found_by_keywords_when_the_dense_side_is_blind(tmp_path):
    client = QdrantClient(path=str(tmp_path / "q"))
    texts = ["step size selector tuning", "step size selector theory", "symmetric criterion on Laplace and Cauchy targets"]
    index_chunks(client, _chunks(texts), ConstantEmbedder(), doc_id="d", collection="c")
    assert keywords.has_keywords(client, "c")
    hits = search_hits("Laplace Cauchy step size", 1, embedder=ConstantEmbedder(), client=client, collection="c",
                       keyword_weight=0.3)
    assert hits[0]["section_path"] == "p > s2" and hits[0]["hits"][0]["text"].startswith("symmetric")
    raw = next(e["data"] for e in search_steps("Laplace", 3, embedder=ConstantEmbedder(), client=client, collection="c",
                                               keyword_weight=0.3) if e["step"] == "vector_search")
    assert raw[0]["keyword_score"] > 0 and "dense_score" in raw[0]
    dense_only = next(e["data"] for e in search_steps("Laplace", 3, embedder=ConstantEmbedder(), client=client,
                                                      collection="c", keyword_weight=0.0) if e["step"] == "vector_search")
    assert all("keyword_score" not in r for r in dense_only)
    client.close()


def test_collection_made_before_the_hybrid_search_stays_dense(tmp_path):
    client = QdrantClient(path=str(tmp_path / "q"))
    client.create_collection("old", vectors_config=VectorParams(size=8, distance=Distance.COSINE))
    index_chunks(client, _chunks(["alpha beta gamma", "delta"]), ConstantEmbedder(), doc_id="d", collection="old")
    assert not keywords.has_keywords(client, "old")
    hits = search_hits("delta", 2, embedder=ConstantEmbedder(), client=client, collection="old", keyword_weight=0.3)
    assert len(hits) == 2 and all("keyword_score" not in h["hits"][0] for h in hits)
    client.close()


def test_hybrid_is_on_for_the_existing_retrieval_tests(store):  # noqa: F811
    client, _ = store
    assert keywords.has_keywords(client, "c") and config.KEYWORD_WEIGHT == 0.3


def test_qwen3_embedding_prefix_and_left_padding(monkeypatch):
    monkeypatch.delenv("HGRAG_EMBED_QUERY_PREFIX")  # emptied for every test by conftest
    monkeypatch.delenv("HGRAG_EMBED_DOC_PREFIX")
    query, doc = config.embed_prefixes("Qwen/Qwen3-Embedding-0.6B")
    assert query.startswith("Instruct: Given a question") and query.endswith("\nQuery:") and doc == ""
    seen = {}

    class FakeST:
        def __init__(self, model_id, **kw):
            seen[model_id] = kw

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    load_embedder("Qwen/Qwen3-Embedding-0.6B")
    load_embedder("BAAI/bge-base-en-v1.5")
    assert seen["Qwen/Qwen3-Embedding-0.6B"] == {"tokenizer_kwargs": {"padding_side": "left"}}
    assert seen["BAAI/bge-base-en-v1.5"] == {}
