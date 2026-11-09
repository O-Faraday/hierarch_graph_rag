"""Databricks indexing step: parsed documents -> ``chunks`` Delta table -> AI Search Delta Sync index.

The chunks are the same as on the local backend; only the store changes. AI Search computes the embeddings from
``embed_text`` with a model serving endpoint, on the documents and on the queries. Nothing is embedded here.

No duplicate, whatever is replayed: ``chunk_id`` is ``doc_id:chunk_index``, each document is merged as a whole (its
chunks that no longer exist are deleted), and the index follows the table through its change data feed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .. import config, tracing, tracking
from . import parsing, progress

logger = logging.getLogger(__name__)

TABLE = "chunks"
INDEX = "chunks_index"
_VIEW = "hgrag_document_chunks"
_SCHEMA = ("chunk_id string, doc_id string, chunk_index int, path string, page_no int, section_ref string, "
           "section_title string, section_path string, kinds string, headings string, element_refs string, "
           "text string, embed_text string, table_description string, formula_reading string")
_COLUMNS = [c.split()[0] for c in _SCHEMA.split(", ")]  # indexed_at stays out of the index


def ensure_table(session: Any) -> None:
    """The change data feed is what a Delta Sync index reads."""
    session.sql(f"""CREATE TABLE IF NOT EXISTS {config.dbx_table(TABLE)} (
        chunk_id STRING NOT NULL, doc_id STRING, chunk_index INT, path STRING, page_no INT, section_ref STRING,
        section_title STRING, section_path STRING, kinds STRING, headings STRING, element_refs STRING,
        text STRING, embed_text STRING, table_description STRING, formula_reading STRING, indexed_at TIMESTAMP)
        USING DELTA TBLPROPERTIES (delta.enableChangeDataFeed = true)""")


def build_chunks(doc_dir: str | Path, chunker: Any) -> list[dict[str, Any]]:
    """Chunks of a parsed document, like the local backend; ``hierarchy.json`` is written next to them, for retrieval."""
    from ..describe import load_formula_readings, load_table_descriptions
    from ..hierarchy import build_hierarchy
    from ..ids import pdf_stem
    from ..indexing import chunk_paragraphs, figure_chunks
    from ..pipeline import _load

    doc_dir = Path(doc_dir)
    doc, figures, manifest = _load(doc_dir)
    descriptions = load_table_descriptions(doc_dir)
    root = build_hierarchy(doc, figures, doc_title=pdf_stem(manifest["source_pdf"]), table_descriptions=descriptions,
                           outline_headings=manifest.get("outline_headings"))
    (doc_dir / "hierarchy.json").write_text(json.dumps(root.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    chunks = chunk_paragraphs(doc, chunker, root, manifest["doc_id"], table_descriptions=descriptions,
                              formula_readings=load_formula_readings(doc_dir))
    return chunks + figure_chunks(root, manifest["doc_id"], start_index=len(chunks))


def _chunk_checks(chunks: list[dict[str, Any]]) -> list[Any]:
    """The checks of the chunks that do not depend on the store; the rows of the index are checked by ``sync_index``."""
    from ..assessments import chunks as chunk_checks

    return [chunk_checks.size_distribution(chunks), chunk_checks.duplicates(chunks), chunk_checks.empty_texts(chunks)]


def _record_checks(span: tracing.Span, by_step: dict[str, list[Any]]) -> None:
    """One span per step, holding its checks: they become MLflow feedback when the trace ends."""
    for step, records in by_step.items():
        with span.child(step, "CHAIN", inputs={"checks": len(records)}) as sp:
            sp.assess(records)
            sp.set({"checks": [a.to_dict() for a in records]})


def _assess_chunked(doc_dir: Path, chunks: list[dict[str, Any]], span: tracing.Span) -> None:
    """Checks and judges of the hierarchy and of the chunks of a document, saved to ``assessments.json``. Never fails
    the step: its chunks are already stored."""
    try:
        from .. import assessments

        by_step = assessments.assess_document(doc_dir, steps=("hierarchy",), judge=config.ASSESS_JUDGES)
        by_step["chunks"] = _chunk_checks(chunks)
        assessments.save(doc_dir, "chunks", by_step["chunks"])
        _record_checks(span, by_step)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: checks skipped: %s", doc_dir.name, exc)


def _row(chunk: dict[str, Any], doc_id: str, path: str) -> tuple:
    return (f"{doc_id}:{chunk['chunk_index']}", doc_id, chunk["chunk_index"], path, chunk.get("page_no"),
            chunk.get("section_ref"), chunk.get("section_title"), chunk.get("section_path"),
            json.dumps(chunk.get("kinds", [])), json.dumps(chunk.get("headings", []), ensure_ascii=False),
            json.dumps(chunk.get("element_refs", [])), chunk["text"], chunk["embed_text"],
            chunk.get("table_description"), chunk.get("formula_reading"))


def _save(session: Any, rows: list[tuple], doc_id: str) -> None:
    session.createDataFrame(rows, _SCHEMA).createOrReplaceTempView(_VIEW)
    session.sql(f"""MERGE INTO {config.dbx_table(TABLE)} t
        USING (SELECT *, current_timestamp() AS indexed_at FROM {_VIEW}) s ON t.chunk_id = s.chunk_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        WHEN NOT MATCHED BY SOURCE AND t.doc_id = :doc_id THEN DELETE""", args={"doc_id": doc_id})


def chunk_pending(session: Any, *, chunker: Any = None, build: Any = build_chunks,
                  run_key: str | None = None) -> dict[str, Any]:
    """Chunk the usable documents never chunked, or parsed again since, and drop the chunks of superseded versions."""
    with tracking.track_step(tracking.INDEXING, "chunk", run_key, tokenizer=config.DBX_CHUNK_TOKENIZER,
                             max_tokens=config.CHUNK_MAX_TOKENS) as tracker:
        summary = _chunk_pending(session, chunker, build, run_key)
        tracker.metrics(**summary)
        tracker.dict(summary, "chunks_summary.json")
    return summary


def _chunk_pending(session: Any, chunker: Any, build: Any, run_key: str | None = None) -> dict[str, Any]:
    parsing.ensure_table(session)
    ensure_table(session)
    documents, chunks = config.dbx_table(parsing.TABLE), config.dbx_table(TABLE)
    session.sql(f"DELETE FROM {chunks} WHERE doc_id IN (SELECT doc_id FROM {documents} WHERE status = 'superseded')")
    pending = session.sql(f"""SELECT d.doc_id, d.path FROM {documents} d LEFT ANTI JOIN
        (SELECT doc_id, min(indexed_at) AS indexed_at FROM {chunks} GROUP BY doc_id) c
        ON d.doc_id = c.doc_id AND c.indexed_at >= d.updated_at
        WHERE d.status IN ('parsed', 'partial')""").collect()
    summary: dict[str, Any] = {"documents": 0, "chunks": 0, "failed": {}, "chunk_seconds": 0.0}
    t0 = time.perf_counter()
    progress.report(f"chunks: {len(pending)} document(s) to process")
    for position, row in enumerate(pending, 1):
        doc_dir = Path(config.dbx_output_dir()) / row.doc_id
        label = f"chunks {position}/{len(pending)} {Path(row.path).name}"
        progress.report(f"{label}: started")
        span = tracing.root("index", tracking.INDEXING, run_key,
                            tags={"hgrag.pdf": Path(row.path).name, "hgrag.doc_id": row.doc_id},
                            inputs={"doc_id": row.doc_id, "tokenizer": config.DBX_CHUNK_TOKENIZER,
                                    "max_tokens": config.CHUNK_MAX_TOKENS})
        with span:
            try:
                if chunker is None:  # the tokenizer is only loaded when there is something to chunk
                    from ..indexing import make_chunker

                    chunker = make_chunker(config.DBX_CHUNK_TOKENIZER, max_tokens=config.CHUNK_MAX_TOKENS)
                chunks = build(doc_dir, chunker)
                rows = [_row(c, row.doc_id, row.path) for c in chunks]
                _save(session, rows, row.doc_id)
            except Exception as exc:  # noqa: BLE001 - no chunk row: the document is pending again at the next run
                logger.warning("%s: chunking failed: %s", row.path, exc)
                summary["failed"][row.path] = str(exc)[:500]
                span.end(error=exc)
                progress.report(f"{label}: failed ({str(exc)[:200]})")
                continue
            summary["documents"] += 1
            summary["chunks"] += len(rows)
            span.set({"n_chunks": len(rows), "chunks": [
                {k: c.get(k) for k in ("chunk_index", "section_path", "page_no", "kinds", "text")} for c in chunks]})
            _assess_chunked(doc_dir, chunks, span)
        progress.report(f"{label}: {len(rows)} chunk(s), checks done")
    summary["chunk_seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps({"hgrag_chunks": summary}))
    return summary


def _client() -> Any:
    from databricks.ai_search.client import AISearchClient

    return AISearchClient(disable_notice=True)  # credentials of the notebook or job


def sync_index(client: Any = None, *, session: Any = None, run_key: str | None = None) -> dict[str, Any]:
    """Create what is missing (a running endpoint is billed until it is deleted), or sync the existing index, and wait.
    With ``session``, the rows of the index are checked against the rows of the table."""
    span = tracing.root("sync_index", tracking.INDEXING, run_key,
                        inputs={"endpoint": config.DBX_SEARCH_ENDPOINT, "embedding_endpoint": config.DBX_EMBED_ENDPOINT})
    with span, tracking.track_step(tracking.INDEXING, "index", run_key, endpoint=config.DBX_SEARCH_ENDPOINT,
                                   embedding_endpoint=config.DBX_EMBED_ENDPOINT) as tracker:
        summary = _sync_index(client or _client())
        tracker.metrics(**summary)
        tracker.tags(index=summary["index"], index_state=summary["state"])
        span.set(summary)
        if session is not None:
            span.assess([_rows_check(session, summary)])
    return summary


def _rows_check(session: Any, summary: dict[str, Any]) -> Any:
    """As many rows in the index as in the table: fewer means a sync still running or stopped half-way."""
    from ..assessments import Assessment

    indexed = summary.get("indexed_rows")
    try:
        rows = session.sql(f"SELECT count(*) AS n FROM {config.dbx_table(TABLE)}").collect()[0].n
    except Exception as exc:  # noqa: BLE001
        return Assessment("index.rows_indexed", None, error=str(exc)[:500])
    if indexed is None:
        return Assessment("index.rows_indexed", None, error=f"no row count (state {summary.get('state')})")
    return Assessment("index.rows_indexed", indexed, passed=indexed == rows, metadata={"table_rows": rows},
                      rationale=f"{indexed} row(s) in the index for {rows} in the table (state {summary.get('state')})")


def assess_documents(session: Any, *, run_key: str | None = None) -> dict[str, Any]:
    """Run again the checks and judges of the usable documents, without parsing nor chunking them: for the documents
    processed before a check existed, or to compare judges. One ``assess`` trace per document."""
    from .. import assessments

    documents, chunks = config.dbx_table(parsing.TABLE), config.dbx_table(TABLE)
    summary: dict[str, Any] = {"documents": 0, "checks": 0, "failed": {}}
    with tracking.track_step(tracking.INDEXING, "assess", run_key, judges=config.ASSESS_JUDGES) as tracker:
        rows = session.sql(f"SELECT doc_id, path FROM {documents} WHERE status IN ('parsed', 'partial')").collect()
        for position, row in enumerate(rows, 1):
            progress.report(f"assess {position}/{len(rows)} {Path(row.path).name}: started")
            doc_dir = Path(config.dbx_output_dir()) / row.doc_id
            span = tracing.root("assess", tracking.INDEXING, run_key,
                                inputs={"doc_id": row.doc_id, "judges": config.ASSESS_JUDGES},
                                tags={"hgrag.pdf": Path(row.path).name, "hgrag.doc_id": row.doc_id})
            with span:
                try:
                    by_step = assessments.assess_document(doc_dir, steps=("parse", "describe", "hierarchy"),
                                                          judge=config.ASSESS_JUDGES)
                    stored = session.sql(f"SELECT chunk_index, text FROM {chunks} WHERE doc_id = :doc_id",
                                         args={"doc_id": row.doc_id}).collect()
                    by_step["chunks"] = _chunk_checks([{"chunk_index": c.chunk_index, "text": c.text} for c in stored])
                    assessments.save(doc_dir, "chunks", by_step["chunks"])
                    _record_checks(span, by_step)
                except Exception as exc:  # noqa: BLE001 - one bad document must not stop the others
                    logger.warning("%s: checks failed: %s", row.path, exc)
                    summary["failed"][row.path] = str(exc)[:500]
                    span.end(error=exc)
                    continue
            summary["documents"] += 1
            summary["checks"] += sum(len(records) for records in by_step.values())
        tracker.metrics(**summary)
    print(json.dumps({"hgrag_assess": summary}))
    return summary


def _sync_index(client: Any) -> dict[str, Any]:
    endpoint, name = config.dbx_search_endpoint(), config.dbx_uc_name(INDEX)
    t0 = time.perf_counter()
    if not client.endpoint_exists(endpoint):
        client.create_endpoint_and_wait(endpoint, endpoint_type="STANDARD", verbose=True)
    if client.index_exists(endpoint, name):
        index = client.get_index(endpoint, name)
        index.sync()
    else:
        index = client.create_delta_sync_index(
            endpoint_name=endpoint, index_name=name, primary_key="chunk_id",
            source_table_name=config.dbx_uc_name(TABLE), pipeline_type="TRIGGERED",
            embedding_source_column="embed_text", embedding_model_endpoint_name=config.DBX_EMBED_ENDPOINT,
            columns_to_sync=_COLUMNS)
    progress.report(f"index: waiting for {name}, the embeddings are computed now")
    index.wait_until_ready(verbose=True, wait_for_updates=True)
    status = index.describe().get("status", {})
    summary = {"index": name, "state": status.get("detailed_state"), "indexed_rows": status.get("indexed_row_count"),
               "embedding_endpoint": config.DBX_EMBED_ENDPOINT, "sync_seconds": round(time.perf_counter() - t0, 2)}
    print(json.dumps({"hgrag_index": summary}))
    return summary


def raise_on_failures(session: Any, chunks: dict[str, Any]) -> None:
    """Fail the run if a document could not be parsed or chunked. Called last, so that the others are already indexed.
    A ``partial`` document is usable and does not count."""
    rows = session.sql(f"SELECT path FROM {config.dbx_table(parsing.TABLE)} WHERE status = 'failed'").collect()
    failed = sorted({r.path for r in rows} | set(chunks["failed"]))
    if failed:
        raise RuntimeError(f"{len(failed)} document(s) failed, see the documents table: {failed[:10]}")


def stop_search(client: Any = None) -> dict[str, Any]:
    """Delete the index of this kit, never the endpoint nor another index: AI Search has no pause, and an endpoint stops
    being billed 24 h after the deletion of its last index. ``chunks`` stays; ``sync_index`` creates the index again."""
    client = client or _client()
    endpoint, name = config.dbx_search_endpoint(), config.dbx_uc_name(INDEX)
    deleted, others = False, []
    listing: dict[str, Any] = {}
    if client.endpoint_exists(endpoint):
        deleted = client.index_exists(endpoint, name)
        if deleted:
            client.delete_index(endpoint, name)
        listing = client.list_indexes(endpoint) or {}
        others = sorted(i["name"] for i in listing.get("vector_indexes", []) if i["name"] != name)
    # One page only (the client takes no page token): a next page may hold other indexes.
    summary = {"endpoint": endpoint, "index": name, "deleted": deleted, "other_indexes": others,
               "billing_stops": not others and not listing.get("next_page_token")}
    print(json.dumps({"hgrag_stop_search": summary}))
    return summary


def search(query: str, num_results: int = 5, *, client: Any = None, reranker: Any = None) -> list[dict[str, Any]]:
    """Hybrid search (vectors and keywords, as on the local backend), best chunks first."""
    client = client or _client()
    index = client.get_index(config.dbx_search_endpoint(), config.dbx_uc_name(INDEX))
    response = index.similarity_search(query_text=query, columns=_COLUMNS, num_results=num_results, query_type="HYBRID",
                                       reranker=reranker)
    names = [c["name"] for c in response["manifest"]["columns"]]
    return [dict(zip(names, row)) for row in response["result"].get("data_array", [])]
