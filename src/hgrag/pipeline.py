"""Pipeline steps, callable from the CLI and from the Airflow DAG (JSON-friendly I/O)."""

from __future__ import annotations

import json
import logging
from collections import Counter
import re
import statistics
import time
from pathlib import Path
from typing import Any

from docling_core.types.doc import DoclingDocument

from . import config, tracing, tracking
from .ids import pdf_name, pdf_stem
from .tracking import track_step
from .hierarchy import build_hierarchy, render_tree, tree_stats

logger = logging.getLogger(__name__)


def _metric_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-./ ]", "_", name)[:200]


def list_pdfs(pdf_dir: str | Path = config.PDF_DIR) -> list[str]:
    return sorted(str(p) for p in Path(pdf_dir).glob("*.pdf"))


def _load(doc_dir: Path) -> tuple[DoclingDocument, list[dict[str, Any]], dict[str, Any]]:
    doc = DoclingDocument.load_from_json(doc_dir / "document.json")
    figures = json.loads((doc_dir / "figures.json").read_text(encoding="utf-8"))
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    return doc, figures, manifest


def parse_step(
    pdf_path: str | Path, *, converter: Any | None = None, run_key: str | None = None,
    describe_pictures: bool = config.DESCRIBE_PICTURES,
) -> dict[str, Any]:
    """Step 1 (tracked): Docling parsing, figures as PNG + metadata."""
    from .parsing import parse_options, parse_pdf

    span = tracing.root("parse", tracking.INDEXING, run_key, span_type="PARSER", tags={"hgrag.pdf": Path(pdf_path).name},
                        inputs={"pdf": Path(pdf_path).name, "describe_pictures": describe_pictures,
                                "options": parse_options()})
    with span, track_step(tracking.INDEXING, "parse", run_key, item=Path(pdf_path).name,
                          pdf=Path(pdf_path).name, describe_pictures=describe_pictures,
                          pdf_bytes=Path(pdf_path).stat().st_size) as tr:
        manifest = parse_pdf(pdf_path, converter=converter, describe_pictures=describe_pictures, trace=span)
        span.set(manifest, doc_id=manifest["doc_id"])
        doc_dir = Path(manifest["doc_dir"])
        tr.metrics(**{k: manifest[k] for k in (
            "n_figures", "n_pages", "n_tables", "n_pictures", "n_headings", "n_paragraphs",
            "n_list_items", "n_figures_described", "n_figures_png", "convert_seconds") if k in manifest})
        for part, tag in (("figure_vlm", "figure_vlm"), ("table_llm", "table_llm")):
            if manifest.get(part) and "seconds" in manifest[part]:
                tr.metrics(**{f"{part}_seconds": manifest[part]["seconds"], f"{part}_errors": manifest[part]["n_errors"]})
                tr.tags(**{tag: manifest[part]["model"]})
        if "n_tables_described" in manifest:
            tr.metrics(n_tables_described=manifest["n_tables_described"])
        tr.metrics(**{f"label.{k}": v for k, v in manifest.get("labels", {}).items()})
        tr.tags(doc_id=manifest["doc_id"])
        tr.dict(manifest, "manifest.json")
        for name in ("figures.json", "document.md"):
            if (doc_dir / name).exists():
                tr.artifact(doc_dir / name)
        tr.artifacts(doc_dir / "figures", "figures")  # the extracted PNGs
    logger.info("parse %s: %s pages, %s figures, %s tables, %s headings in %.1fs", Path(pdf_path).name,
                manifest.get("n_pages"), manifest["n_figures"], manifest.get("n_tables"),
                manifest.get("n_headings"), manifest.get("convert_seconds", 0))
    return manifest


def build_and_save_hierarchy(doc_dir: str | Path, *, run_key: str | None = None) -> dict[str, Any]:
    """Step 2: heading tree + elements, written to ``hierarchy.json`` (and ``hierarchy_tree.txt``)."""
    doc_dir = Path(doc_dir)
    span = tracing.root("hierarchy", tracking.INDEXING, run_key, tags={"hgrag.doc_id": doc_dir.name},
                        inputs={"doc_dir": doc_dir.name})
    with span, track_step(tracking.INDEXING, "hierarchy", run_key, item=doc_dir.name, doc_dir=doc_dir.name) as tr:
        out = _build_and_save_hierarchy(doc_dir)
        span.set({"tree": (doc_dir / "hierarchy_tree.txt").read_text(encoding="utf-8"), "stats": out["tree"]})
        stats = out["tree"]
        tr.metrics(n_sections=out["n_sections"], n_leaf_sections=out["n_leaf_sections"],
                   n_elements=out["n_elements"], max_depth=stats["max_depth"],
                   n_empty_sections=stats["n_empty_sections"], root_elements=stats["root_elements"],
                   **{f"elements.{k}": v for k, v in stats["elements_by_kind"].items()},
                   **{f"sections_level_{k}": v for k, v in stats["sections_by_level"].items()})
        tr.dict(stats, "tree_stats.json")
        tr.artifact(doc_dir / "hierarchy.json")
        tr.artifact(doc_dir / "hierarchy_tree.txt")
    logger.info("hierarchy %s: %d sections (max depth %d), %d elements %s", doc_dir.name, out["n_sections"],
                stats["max_depth"], out["n_elements"], stats["elements_by_kind"])
    if stats["n_empty_sections"] > 1:  # one is normal: the document title
        logger.warning("hierarchy %s: %d leaf sections without any element", doc_dir.name, stats["n_empty_sections"])
    return out


def _build_and_save_hierarchy(doc_dir: Path) -> dict[str, Any]:
    doc, figures, manifest = _load(doc_dir)
    from .describe import load_table_descriptions

    root = build_hierarchy(doc, figures, doc_title=pdf_stem(manifest["source_pdf"]),
                           table_descriptions=load_table_descriptions(doc_dir))
    (doc_dir / "hierarchy.json").write_text(
        json.dumps(root.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (doc_dir / "hierarchy_tree.txt").write_text(render_tree(root), encoding="utf-8")
    stats = tree_stats(root)
    return {
        **manifest,
        "n_sections": stats["n_sections"],
        "n_leaf_sections": stats["n_leaf_sections"],
        "n_elements": stats["n_elements"],
        "tree": stats,
    }


def index_documents(
    doc_dirs: list[str],
    *,
    qdrant_path: str | Path = config.QDRANT_PATH,
    collection: str = config.COLLECTION_NAME,
    model_id: str = config.EMBED_MODEL_ID,
    run_key: str | None = None,
) -> dict[str, Any]:
    """Steps 3-4: paragraph chunks + embeddings into the single Qdrant collection.

    Qdrant local mode allows one process on the storage path, hence one task for all docs.
    """
    span = tracing.root("index", tracking.INDEXING, run_key, inputs={
        "documents": [Path(d).name for d in doc_dirs], "embedding_model": model_id, "collection": collection,
        "chunk_labels": list(config.CHUNK_LABELS), "max_tokens": config.CHUNK_MAX_TOKENS})
    with span, track_step(tracking.INDEXING, "index", run_key, model_id=model_id, collection=collection,
                          min_chunk_words=config.MIN_CHUNK_WORDS, n_documents=len(doc_dirs)) as tr:
        out = _index_documents(doc_dirs, qdrant_path, collection, model_id, trace=span)
        span.set({k: out[k] for k in ("collection", "chunks_per_doc", "total_chunks", "embedding_dim",
                                      "points_in_collection", "seconds")})
        tr.metrics(total_chunks=out["total_chunks"], embedding_dim=out["embedding_dim"],
                   points_in_collection=out["points_in_collection"], **out["seconds"])
        for name, d in out["documents"].items():
            key = _metric_key(name)
            tr.metrics(**{f"chunks.{key}": d["n_chunks"], f"dropped.{key}": d["n_dropped"],
                          f"figure_chunks.{key}": d["n_figure_chunks"],
                          f"table_chunks_described.{key}": d["n_table_chunks_described"], f"chunk_words_mean.{key}": d["words_mean"], f"chunk_words_max.{key}": d["words_max"],
                          f"sections_covered.{key}": d["n_sections_covered"]})
            tr.artifact(Path(d["doc_dir"]) / "chunks.jsonl", "chunks")
        tr.dict(out, "index_summary.json")
    for name, d in out["documents"].items():
        logger.info("index %s: %d chunks (%d words mean, max %d), %d sections covered, dropped %s", name,
                    d["n_chunks"], d["words_mean"], d["words_max"], d["n_sections_covered"], d["dropped"] or "none")
    if run_key is not None:
        tracking.finish_execution(tracking.INDEXING, run_key, total_chunks=out["total_chunks"])
    return out


def _index_documents(doc_dirs, qdrant_path, collection, model_id, trace=None) -> dict[str, Any]:
    from .describe import load_formula_readings, load_table_descriptions
    from .indexing import chunk_paragraphs, figure_chunks, index_chunks, load_embedder, make_chunker, open_store

    t0 = time.perf_counter()
    embedder = load_embedder(model_id)
    # Chunks must fit the encoder window, else they are truncated; they are also capped (HGRAG_CHUNK_MAX_TOKENS, 256)
    # so that changing the model does not change the chunks.
    chunker = make_chunker(model_id, max_tokens=min(config.CHUNK_MAX_TOKENS, getattr(embedder, "max_seq_length", None) or 256))
    seconds = {"seconds_load_model": time.perf_counter() - t0, "seconds_chunk": 0.0, "seconds_embed_store": 0.0}
    client = open_store(qdrant_path)
    per_doc: dict[str, int] = {}
    dspan = espan = tracing.Span()  # spans of the document being indexed (ended with the error if it fails)
    documents: dict[str, Any] = {}
    dim = 0
    try:
        for d in doc_dirs:
            doc_dir = Path(d)
            dspan = (trace or tracing.Span()).child("index_document", inputs={"doc_dir": doc_dir.name})
            doc, figures, manifest = _load(doc_dir)
            tdesc = load_table_descriptions(doc_dir)
            root = build_hierarchy(doc, figures, doc_title=pdf_stem(manifest["source_pdf"]), table_descriptions=tdesc)
            t = time.perf_counter()
            cstats: dict[str, Any] = {}
            with dspan.child("chunk", inputs={"doc_id": manifest["doc_id"], "labels": list(config.CHUNK_LABELS)}) as sp:
                chunks = chunk_paragraphs(doc, chunker, root, manifest["doc_id"], stats=cstats, table_descriptions=tdesc,
                                          formula_readings=load_formula_readings(doc_dir))
                fig_chunks = figure_chunks(root, manifest["doc_id"], start_index=len(chunks))  # described figures
                chunks += fig_chunks
                sp.set({"chunks": [{k: c.get(k) for k in ("chunk_index", "section_path", "page_no", "kinds", "text",
                                                          "embed_text")} for c in chunks], "dropped": cstats.get("dropped")})
            seconds["seconds_chunk"] += time.perf_counter() - t
            with (doc_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:  # one chunk per line, for analysis
                for c in chunks:
                    fh.write(json.dumps({k: c[k] for k in ("chunk_index", "page_no", "section_path", "text")},
                                        ensure_ascii=False) + "\n")
            t = time.perf_counter()
            espan = dspan.child("embed_and_store", "EMBEDDING", inputs={"model": model_id, "collection": collection,
                                                                        "n_chunks": len(chunks)})
            per_doc[manifest["doc_id"]] = index_chunks(
                client,
                chunks,
                embedder,
                doc_id=manifest["doc_id"],
                source_pdf=manifest["source_pdf"],
                collection=collection,
                model_id=model_id,
            )
            seconds["seconds_embed_store"] += time.perf_counter() - t
            espan.end({"points": per_doc[manifest["doc_id"]]})
            words = [len(c["text"].split()) for c in chunks] or [0]
            documents[pdf_name(manifest["source_pdf"])] = {
                "doc_id": manifest["doc_id"], "doc_dir": str(doc_dir), "n_chunks": len(chunks),
                "n_figure_chunks": len(fig_chunks),
                "n_table_chunks_described": sum(1 for c in chunks if c.get("table_description")),
                "n_dropped": cstats.get("n_dropped", 0), "dropped": cstats.get("dropped", {}),
                "words_mean": round(statistics.fmean(words), 1), "words_min": min(words), "words_max": max(words),
                "n_sections_covered": len({c["section_ref"] for c in chunks}),
                "chunks_per_page": {str(p): n for p, n in sorted(
                    Counter(c["page_no"] for c in chunks if c["page_no"]).items())},
            }
            dspan.set({"n_chunks": len(chunks), "n_dropped": cstats.get("n_dropped", 0)}, doc_id=manifest["doc_id"])
            dspan.end()
        if client.collection_exists(collection):
            info = client.get_collection(collection)
            dim = info.config.params.vectors.size
            points = client.count(collection, exact=True).count
        else:
            points = 0
    except BaseException as exc:
        espan.end(error=exc)
        dspan.end(error=exc)
        raise
    finally:
        client.close()
    return {"collection": collection, "chunks_per_doc": per_doc, "total_chunks": sum(per_doc.values()),
            "embedding_dim": dim, "points_in_collection": points,
            "seconds": {k: round(v, 2) for k, v in seconds.items()}, "documents": documents}
