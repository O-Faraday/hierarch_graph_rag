"""Retrieval flow: query -> similar chunks in Qdrant -> sections -> reconstructed documents.

The vector search returns paragraph chunks. Each hit carries ``doc_id`` and ``section_ref``;
the section is then rebuilt from ``hierarchy.json`` with *all* its elements (paragraphs,
lists, tables, figures with their VLM description, footnotes), not only the matching chunk.
Sections are reconstructed in parallel.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import config, keywords, tracing, tracking
from .tracking import track_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
def _min_max(points: list[Any]) -> dict[Any, float]:
    scores = [float(p.score) for p in points]
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    return {p.id: (float(p.score) - lo) / (hi - lo) if hi > lo else 1.0 for p in points}


def fuse(dense: list[Any], kw: list[Any], weight: float, limit: int) -> list[Any]:
    """Hybrid ranking: ``(1 - weight) * dense + weight * keyword``, each score min-max normalised over its own list
    (a chunk absent from a list gets 0 there). Points keep their payload and carry ``dense_score``/``keyword_score``."""
    d, k = _min_max(dense), _min_max(kw)
    raw_d = {p.id: float(p.score) for p in dense}
    raw_k = {p.id: float(p.score) for p in kw}
    by_id = {p.id: p for p in kw} | {p.id: p for p in dense}
    fused = {i: (1 - weight) * d.get(i, 0.0) + weight * k.get(i, 0.0) for i in by_id}
    best = sorted(fused, key=lambda i: fused[i], reverse=True)[:limit]
    return [SimpleNamespace(id=i, payload=by_id[i].payload, score=fused[i], dense_score=raw_d.get(i),
                            keyword_score=raw_k.get(i)) for i in best]


def _part_scores(p: Any) -> dict[str, float]:
    return {k: getattr(p, k) for k in ("dense_score", "keyword_score") if getattr(p, k, None) is not None}
def _group_points(points: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for p in points:  # already sorted by decreasing similarity
        pl = p.payload or {}
        key = (pl.get("doc_id"), pl.get("section_ref"))
        if key[0] is None or key[1] is None:
            logger.warning("hit %s has no doc_id/section_ref, skipped", p.id)
            continue
        entry = grouped.setdefault(
            key,
            {
                "doc_id": key[0],
                "section_ref": key[1],
                "section_path": pl.get("section_path"),
                "source_pdf": pl.get("source_pdf"),
                "score": float(p.score),
                "hits": [],
            },
        )
        entry["hits"].append(
            {
                "chunk_index": pl.get("chunk_index"),
                "score": float(p.score),
                **({"vector_score": float(p.vector_score)} if getattr(p, "vector_score", None) is not None else {}),
                "page_no": pl.get("page_no"),
                "kinds": pl.get("kinds"),
                "text": pl.get("text"),
            }
        )
    return sorted(grouped.values(), key=lambda e: e["score"], reverse=True)


def search_steps(
    query: str,
    k: int = 5,
    *,
    embedder: Any | None = None,
    client: Any | None = None,
    collection: str = config.COLLECTION_NAME,
    qdrant_path: str | Path = config.QDRANT_PATH,
    model_id: str = config.EMBED_MODEL_ID,
    min_score: float | None = None,
    run_key: str | None = None,
    rerank: bool | None = None,
    reranker: Any | None = None,
    candidates: int | None = None,
    keyword_weight: float | None = None,
    diagnostics: dict[str, Any] | None = None,
    trace: tracing.Span | None = None,
) -> Iterator[dict[str, Any]]:
    """Search as a generator of step events ``{"step", "seconds", "data"}``.

    Steps: ``embed`` (query vector), ``vector_search`` (the raw chunks: ``k``, or ``candidates`` when reranking; on a
    collection with keyword vectors, the dense and BM25 lists fused with ``keyword_weight``, default
    ``HGRAG_KEYWORD_WEIGHT``, each chunk carrying ``dense_score`` and ``keyword_score``),
    ``rerank`` (only when reranking: the best ``k`` after the cross-encoder), ``group`` (chunks grouped by section).
    ``rerank=None`` follows ``HGRAG_RERANK``; a ``reranker`` object (anything with ``predict(pairs)``) is loaded from
    ``HGRAG_RERANK_MODEL`` when omitted. The UI consumes it incrementally; ``search_hits`` drains it.
    The whole search is one MLflow step (``search``) when ``run_key`` is given. ``diagnostics`` (optional dict) is
    filled with what the production checks need (``assessments.retrieval``): ``top_score`` (best reranker score, or
    best dense similarity without reranking) and, on a hybrid collection, how many candidates both searches found.
    ``trace``: the span of a larger operation (a question); without it the search is a trace of its own.
    """
    diag = diagnostics if diagnostics is not None else {}
    if not query.strip():
        raise ValueError("empty query")
    use_rerank = config.RERANK if rerank is None else rerank
    n_fetch = max(k, candidates or config.RERANK_CANDIDATES) if use_rerank else k
    weight = config.KEYWORD_WEIGHT if keyword_weight is None else keyword_weight
    span = tracing.child_or_root(trace, "search", tracking.RETRIEVAL, run_key, span_type="RETRIEVER", inputs={
        "query": query, "k": k, "candidates": n_fetch, "rerank": use_rerank, "keyword_weight": weight,
        "collection": collection, "embedding_model": model_id})
    sections: list[dict[str, Any]] = []
    try:
        for event in _search_events(query, k, embedder=embedder, client=client, collection=collection,
                                    qdrant_path=qdrant_path, model_id=model_id, min_score=min_score, run_key=run_key,
                                    use_rerank=use_rerank, reranker=reranker, n_fetch=n_fetch, weight=weight,
                                    diag=diag, span=span):
            if event["step"] == "group":
                sections = event["data"]
            yield event
    except GeneratorExit:  # the consumer stopped early (the app moved on): not an error
        span.end(closed_early=True)
        raise
    except BaseException as exc:
        span.end(error=exc)
        raise
    span.end({"sections": [{k_: s.get(k_) for k_ in ("section_path", "doc_id", "section_ref", "score")}
                           | {"n_chunks": len(s["hits"])} for s in sections], "diagnostics": diag})


def _chunk(p: Any, **scores: Any) -> dict[str, Any]:
    """A candidate chunk as a span shows it: scores, where it comes from and its full text."""
    pl = p.payload or {}
    return {"id": str(p.id), "score": float(p.score), **_part_scores(p), **scores,
            **{k: pl.get(k) for k in ("section_path", "page_no", "kinds", "text")}}


def _search_events(query: str, k: int, *, embedder: Any, client: Any, collection: str, qdrant_path: str | Path,
                   model_id: str, min_score: float | None, run_key: str | None, use_rerank: bool, reranker: Any,
                   n_fetch: int, weight: float, diag: dict[str, Any], span: tracing.Span) -> Iterator[dict[str, Any]]:
    with track_step(tracking.RETRIEVAL, "search", run_key, query=query, k=k, collection=collection,
                    model_id=model_id, rerank=use_rerank, **({"candidates": n_fetch} if use_rerank else {})) as tr:
        owns_client = client is None
        if client is None:
            from qdrant_client import QdrantClient

            if not Path(qdrant_path).exists():
                raise FileNotFoundError(f"No Qdrant store at {qdrant_path}: run the indexing first.")
            client = QdrantClient(path=str(qdrant_path))
        try:
            t0 = time.perf_counter()
            if embedder is None:
                from .indexing import load_embedder

                embedder = load_embedder(model_id)
            query_prefix, _ = config.embed_prefixes(model_id)
            with span.child("embed", "EMBEDDING", inputs={"text": query_prefix + query, "model": model_id}) as sp:
                vector = embedder.encode([query_prefix + query], normalize_embeddings=True)[0].tolist()
                sp.set({"dim": len(vector)})
            dt = time.perf_counter() - t0
            tr.metrics(embed_s=dt, vector_dim=len(vector))
            yield {"step": "embed", "seconds": dt,
                   "data": {"query": query, "model": model_id, "dim": len(vector), "vector": vector}}

            t0 = time.perf_counter()
            with span.child("dense_search", "RETRIEVER", inputs={"limit": n_fetch, "min_score": min_score}) as sp:
                points = client.query_points(
                    collection_name=collection, query=vector, limit=n_fetch,
                    with_payload=True, score_threshold=min_score,
                ).points
                sp.set({"chunks": [_chunk(p) for p in points]})
            sparse = keywords.query_vector(query) if weight > 0 and keywords.has_keywords(client, collection) else None
            if sparse is not None:
                with span.child("keyword_search", "RETRIEVER", inputs={"limit": n_fetch, "n_terms": len(sparse.indices)}) as sp:
                    kw_points = client.query_points(collection_name=collection, query=sparse, using=keywords.SPARSE_NAME,
                                                    limit=n_fetch, with_payload=True).points
                    sp.set({"chunks": [_chunk(p) for p in kw_points]})
                with span.child("fusion", "RETRIEVER", inputs={"keyword_weight": weight, "limit": n_fetch}) as sp:
                    points = fuse(points, kw_points, weight, n_fetch)
                    sp.set({"chunks": [_chunk(p) for p in points]})
                both = sum(1 for p in points if p.dense_score is not None and p.keyword_score is not None)
                diag.update(n_candidates=len(points), n_both=both, agreement=both / len(points) if points else None,
                            n_dense_only=sum(1 for p in points if p.keyword_score is None),
                            n_keyword_only=sum(1 for p in points if p.dense_score is None))
            dense = ([p.dense_score for p in points if p.dense_score is not None] if sparse is not None
                     else [float(p.score) for p in points])
            diag.update(top_score=max(dense, default=None), top_score_kind="dense")
            dt = time.perf_counter() - t0
            raw = [{"id": str(p.id), "score": float(p.score), **_part_scores(p), **(p.payload or {})} for p in points]
            tr.metrics(qdrant_s=dt, n_chunks=len(raw), keyword_weight=weight if sparse is not None else 0.0,
                       **({"top_score": raw[0]["score"]} if raw else {}))
            yield {"step": "vector_search", "seconds": dt, "data": raw}

            if use_rerank:
                from .rerank import load_reranker, rerank_points

                t0 = time.perf_counter()
                if reranker is None:
                    reranker = load_reranker(config.RERANK_MODEL)
                with span.child("rerank", "RERANKER", inputs={"model": getattr(reranker, "model_id", None) or config.RERANK_MODEL,
                                                              "candidates": len(points), "top_k": k}) as sp:
                    points = rerank_points(query, points, reranker, k)
                    sp.set({"chunks": [_chunk(p, vector_score=p.vector_score) for p in points]})
                dt = time.perf_counter() - t0
                reranked = [{"id": str(p.id), "score": p.score, "vector_score": p.vector_score, **(p.payload or {})}
                            for p in points]
                tr.metrics(rerank_s=dt, n_reranked=len(reranked), **({"top_rerank_score": reranked[0]["score"]} if reranked else {}))
                diag.update(top_score=reranked[0]["score"] if reranked else None, top_score_kind="rerank")
                yield {"step": "rerank", "seconds": dt, "data": reranked}
        finally:
            if owns_client:
                client.close()

        t0 = time.perf_counter()
        sections = _group_points(points)
        dt = time.perf_counter() - t0
        tr.metrics(n_sections=len(sections))
        tr.dict(sections, "sections.json")
        yield {"step": "group", "seconds": dt, "data": sections}


def search_hits(query: str, k: int = 5, *, run_key: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Return the sections matched by the ``k`` closest chunks, best score first.

    Chunks of the same section are grouped: one entry per ``(doc_id, section_ref)`` with the
    best score and the list of matching chunks. Output is JSON-friendly (Airflow XCom).
    """
    sections: list[dict[str, Any]] = []
    for event in search_steps(query, k, run_key=run_key, **kwargs):
        if event["step"] == "group":
            sections = event["data"]
    return sections


# ---------------------------------------------------------------------------
# Section reconstruction
# ---------------------------------------------------------------------------
def load_hierarchy(doc_id: str, output_dir: str | Path = config.OUTPUT_DIR) -> dict[str, Any]:
    path = Path(output_dir) / doc_id / "hierarchy.json"
    if not path.exists():
        raise FileNotFoundError(f"hierarchy.json not found for document {doc_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_section(tree: dict[str, Any], ref: str) -> dict[str, Any] | None:
    stack = [tree]
    while stack:
        node = stack.pop()
        if node["ref"] == ref:
            return node
        stack.extend(node.get("children", []))
    return None


def _png_abs(doc_dir: Path, png_path: str | None) -> str | None:
    if not png_path:
        return None
    # Outputs produced on Windows store "figures\\figure-001.png".
    return str(doc_dir / Path(*png_path.replace("\\", "/").split("/")))


def _render_element(el: dict[str, Any]) -> str:
    kind, text = el["kind"], (el.get("text") or "").strip()
    extra = el.get("extra") or {}
    if kind == "list_item":
        return f"- {text}"
    if kind == "figure":
        label = extra.get("caption") or "Figure"
        desc = extra.get("description")
        return f"[Figure] {label}" + (f"\nDescription (machine-generated, unverified): {desc}" if desc else "")
    if kind == "table":  # caption first, then the rows (authoritative), then the generated description if any
        parts = ([f"[Table] {extra['caption']}"] if extra.get("caption") else []) + [text]
        if extra.get("description"):
            parts.append(f"Table description (machine-generated, unverified): {extra['description']}")
        return "\n\n".join(parts)
    if kind == "footnote":
        return f"[Footnote] {text}"
    if kind == "code":
        return f"```\n{text}\n```"
    return text  # paragraph, table (markdown), formula, other


def _render_section(
    node: dict[str, Any], include_children: bool, depth: int = 0
) -> tuple[list[str], list[dict[str, Any]], int]:
    blocks = [f"{'#' * min(max(node['level'], 1) + depth, 6)} {node['title']}"]
    elements: list[dict[str, Any]] = list(node["elements"])
    previous_kind = None
    for el in node["elements"]:
        rendered = _render_element(el)
        if not rendered:
            continue
        # Consecutive list items stay in one block, everything else is separated by a blank line.
        if el["kind"] == "list_item" and previous_kind == "list_item":
            blocks[-1] += "\n" + rendered
        else:
            blocks.append(rendered)
        previous_kind = el["kind"]
    if include_children:
        for child in node.get("children", []):
            sub_blocks, sub_elements, _ = _render_section(child, True, depth)
            blocks.extend(sub_blocks)
            elements.extend(sub_elements)
    return blocks, elements, len(elements)


def reconstruct_section(
    doc_id: str,
    section_ref: str,
    *,
    include_children: bool = False,
    output_dir: str | Path = config.OUTPUT_DIR,
    tree: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a document (Markdown) from the elements of one section, in reading order."""
    tree = tree or load_hierarchy(doc_id, output_dir)
    node = find_section(tree, section_ref)
    if node is None:
        raise KeyError(f"section {section_ref} not found in document {doc_id}")
    doc_dir = Path(output_dir) / doc_id
    blocks, elements, n = _render_section(node, include_children)
    figures = [
        {
            "ref": e["ref"],
            "caption": (e.get("extra") or {}).get("caption"),
            "png_path": _png_abs(doc_dir, (e.get("extra") or {}).get("png_path")),
        }
        for e in elements
        if e["kind"] == "figure"
    ]
    return {
        "doc_id": doc_id,
        "section_ref": section_ref,
        "section_path": node["path"],
        "title": node["title"],
        "level": node["level"],
        "page_no": node.get("page_no"),
        "n_elements": n,
        "include_children": include_children,
        "text": "\n\n".join(blocks),
        "figures": figures,
    }


def reconstruct_hit(
    hit: dict[str, Any],
    *,
    include_children: bool = False,
    output_dir: str | Path = config.OUTPUT_DIR,
    tree: dict[str, Any] | None = None,
    run_key: str | None = None,
    trace: tracing.Span | None = None,
) -> dict[str, Any]:
    """Reconstruct the section of one ``search_hits`` entry and keep its score and chunks. ``trace``: the span of
    the question, under which the rebuilt section is recorded."""
    span = trace.child("reconstruct", "RETRIEVER", inputs={
        "doc_id": hit["doc_id"], "section_ref": hit["section_ref"], "section_path": hit.get("section_path"),
        "score": hit["score"], "include_children": include_children}) if trace is not None else tracing.Span()
    with span, track_step(tracking.RETRIEVAL, "reconstruct", run_key,
                          item=f"{hit['doc_id']}{hit['section_ref']}", doc_id=hit["doc_id"],
                          section_ref=hit["section_ref"], include_children=include_children) as tr:
        doc = reconstruct_section(
            hit["doc_id"], hit["section_ref"], include_children=include_children,
            output_dir=output_dir, tree=tree,
        )
        tr.metrics(n_elements=doc["n_elements"], n_figures=len(doc["figures"]),
                   n_chars=len(doc["text"]), score=hit["score"])
        tr.text(doc["text"], "section.md")
        span.set({k: doc[k] for k in ("section_path", "title", "level", "page_no", "n_elements", "text", "figures")})
    return {**doc, "score": hit["score"], "hits": hit["hits"], "source_pdf": hit.get("source_pdf")}


def _sections_view(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"section_path": d["section_path"], "score": d["score"], "n_elements": d["n_elements"],
             "n_chars": len(d["text"])} for d in docs]


def retrieve_sections(
    query: str,
    k: int = 5,
    *,
    include_children: bool = False,
    output_dir: str | Path = config.OUTPUT_DIR,
    max_workers: int = 8,
    run_key: str | None = None,
    trace: tracing.Span | None = None,
    **search_kwargs: Any,
) -> list[dict[str, Any]]:
    """Full flow: vector search, then parallel reconstruction of every matched section. Traced as ``retrieve``
    (search and reconstructions), under ``trace`` when given."""
    span = tracing.child_or_root(trace, "retrieve", tracking.RETRIEVAL, run_key, span_type="RETRIEVER",
                                 inputs={"query": query, "k": k, "include_children": include_children})
    with span:
        hits = search_hits(query, k, run_key=run_key, trace=span, **search_kwargs)
        if not hits:
            span.set({"sections": []})
            return []
        # Each hierarchy.json is read once, then shared by the workers.
        trees = {d: load_hierarchy(d, output_dir) for d in {h["doc_id"] for h in hits}}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(hits))) as pool:
            futures = [
                pool.submit(
                    reconstruct_hit, h, include_children=include_children,
                    output_dir=output_dir, tree=trees[h["doc_id"]], run_key=run_key, trace=span,
                )
                for h in hits
            ]
            docs = [f.result() for f in futures]  # same order as hits: best score first
        span.set({"sections": _sections_view(docs)})
        return docs


def retrieve_steps(
    query: str,
    k: int = 5,
    *,
    include_children: bool = False,
    output_dir: str | Path = config.OUTPUT_DIR,
    max_workers: int = 8,
    run_key: str | None = None,
    trace: tracing.Span | None = None,
    **search_kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Full flow as step events, for the UI: ``embed``, ``vector_search``, ``group``, then one
    ``reconstruct`` event per section (in parallel, emitted as each finishes), then ``done``.
    Traced as ``retrieve``, under ``trace`` when given.
    """
    span = tracing.child_or_root(trace, "retrieve", tracking.RETRIEVAL, run_key, span_type="RETRIEVER",
                                 inputs={"query": query, "k": k, "include_children": include_children})
    try:
        results = yield from _retrieve_events(query, k, include_children=include_children, output_dir=output_dir,
                                              max_workers=max_workers, run_key=run_key, span=span, **search_kwargs)
    except GeneratorExit:  # the app stopped reading: not an error
        span.end(closed_early=True)
        raise
    except BaseException as exc:
        span.end(error=exc)
        raise
    span.end({"sections": _sections_view(results)})


def _retrieve_events(query: str, k: int, *, include_children: bool, output_dir: str | Path, max_workers: int,
                     run_key: str | None, span: tracing.Span, **search_kwargs: Any) -> Iterator[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for event in search_steps(query, k, run_key=run_key, trace=span, **search_kwargs):
        if event["step"] == "group":
            sections = event["data"]
        yield event
    if not sections:
        if run_key is not None:
            tracking.finish_execution(tracking.RETRIEVAL, run_key, n_sections=0)
        yield {"step": "done", "seconds": 0.0, "data": []}
        return []

    from concurrent.futures import as_completed

    trees = {d: load_hierarchy(d, output_dir) for d in {s["doc_id"] for s in sections}}
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(sections))) as pool:
        futures = {}
        for s in sections:
            t0 = time.perf_counter()
            fut = pool.submit(
                reconstruct_hit, s, include_children=include_children,
                output_dir=output_dir, tree=trees[s["doc_id"]], run_key=run_key, trace=span,
            )
            futures[fut] = t0
        for fut in as_completed(futures):
            doc = fut.result()
            results.append(doc)
            yield {"step": "reconstruct", "seconds": time.perf_counter() - futures[fut], "data": doc}
    results.sort(key=lambda d: d["score"], reverse=True)
    if run_key is not None:
        tracking.finish_execution(tracking.RETRIEVAL, run_key, n_sections=len(results))
    yield {"step": "done", "seconds": 0.0, "data": results}
    return results
