"""Databricks parsing step: new PDFs of a volume (Auto Loader, queue table) -> ``parse_pdf`` -> files on a volume +
``documents`` table.

The same parsing as the local backend, descriptions by the vision model included. Everything runs in the process of the
job, one PDF after the other, so a single-node cluster is enough.

No duplicate, whatever is replayed (a run can stop midway, a checkpoint can be lost):

* ``doc_id`` is the hash of the content: one row and one folder per content, whatever the file names;
* a document already ``parsed`` with the same options is skipped, so neither Docling nor the vision model run again;
* each document is merged on ``doc_id`` as soon as it is done: an interrupted batch keeps what it paid for;
* a new content under the same path marks the previous version ``superseded``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from .. import config, tracing, tracking
from ..ids import compute_doc_id
from . import progress

logger = logging.getLogger(__name__)

TABLE = "documents"
QUEUE = "parse_queue"
_VIEW = "hgrag_parsed_document"
_SCHEMA = ("doc_id string, path string, status string, error string, parse_options string, n_pages int, "
           "convert_seconds double, parse_seconds double")
_DESCRIPTION_STEPS = ("figure_vlm", "table_llm", "formula_vlm", "formula_reading")

_converter: Any = None
_converter_seconds = 0.0


def get_converter() -> Any:
    """One converter per process: its models are loaded once, not at each micro-batch."""
    global _converter, _converter_seconds
    if _converter is None:
        from docling.datamodel.base_models import InputFormat

        from ..parsing import build_converter

        t0 = time.perf_counter()
        _converter = build_converter()
        _converter.initialize_pipeline(InputFormat.PDF)
        _converter_seconds = time.perf_counter() - t0
    return _converter


def prefetch_models(target: str | Path) -> Path:
    """Download once the Docling models of the configured options to ``target`` (``HGRAG_DOCLING_ARTIFACTS``). Written
    to the local disk first: volumes refuse random writes."""
    from docling.utils.model_downloader import download_models

    target = Path(target)
    done = target / ".complete"
    if done.exists():
        return target
    progress.report(f"models: downloading to {target}")
    with tempfile.TemporaryDirectory() as tmp:
        download_models(output_dir=Path(tmp), with_layout=True, with_tableformer=True, with_rapidocr=True,
                        with_code_formula=config.CODE_ENRICHMENT or config.FORMULA_ENRICHMENT,
                        with_picture_classifier=config.PICTURE_CLASSIFICATION)
        shutil.copytree(tmp, target, dirs_exist_ok=True)
    done.write_text("", encoding="utf-8")
    return target


def ensure_table(session: Any) -> None:
    session.sql(f"""CREATE TABLE IF NOT EXISTS {config.dbx_table(TABLE)} (
        doc_id STRING NOT NULL, path STRING, status STRING, error STRING, parse_options STRING, n_pages INT,
        convert_seconds DOUBLE, parse_seconds DOUBLE, updated_at TIMESTAMP) USING DELTA""")


def _local_path(path: str) -> str:
    """Auto Loader gives ``dbfs:/Volumes/...``, Python opens ``/Volumes/...``."""
    return path.removeprefix("dbfs:").removeprefix("file:")


def _options() -> str:
    from ..parsing import parse_options

    return json.dumps(parse_options(), sort_keys=True)


def _outcome(manifest: dict[str, Any]) -> tuple[str, str | None]:
    """``parsed``, or ``partial`` when the document is usable but incomplete: conversion stopped by the time limit, or
    items without description (``parse_pdf`` goes on without them, this must not pass for a complete document)."""
    problems = []
    if manifest.get("conversion_status") == "partial_success":
        problems.append("conversion stopped before the end")
    for step in _DESCRIPTION_STEPS:
        stats = manifest.get(step) or {}
        if stats.get("error"):
            problems.append(f"{step}: {stats['error']}")
        elif stats.get("errors"):
            problems.append(f"{step}: {len(stats['errors'])} item(s) failed")
    return ("partial", "; ".join(problems)[:2000]) if problems else ("parsed", None)


def _already_parsed(session: Any, options: str) -> set[str]:
    rows = session.sql(f"SELECT doc_id FROM {config.dbx_table(TABLE)} WHERE status = 'parsed' AND parse_options = :options",
                       args={"options": options}).collect()
    return {r.doc_id for r in rows}


def _save(session: Any, row: tuple) -> None:
    table = config.dbx_table(TABLE)
    session.createDataFrame([row], _SCHEMA).createOrReplaceTempView(_VIEW)
    session.sql(f"""MERGE INTO {table} t
        USING (SELECT *, current_timestamp() AS updated_at FROM {_VIEW}) s ON t.doc_id = s.doc_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *""")
    session.sql(f"""UPDATE {table} SET status = 'superseded', updated_at = current_timestamp()
        WHERE path = :path AND doc_id != :doc_id AND status != 'superseded'""", args={"path": row[1], "doc_id": row[0]})


def _parse_one(pdf: str, doc_id: str, options: str, parse: Any, converter: Any, run_key: str | None = None) -> tuple:
    """One trace per parsed document, with the checks and the judges of the parsing and of the descriptions."""
    t0 = time.perf_counter()
    span = tracing.root("parse", tracking.INDEXING, run_key, span_type="PARSER",
                        tags={"hgrag.pdf": Path(pdf).name, "hgrag.doc_id": doc_id},
                        inputs={"pdf": Path(pdf).name, "options": options})
    with span:
        try:
            manifest = parse(pdf, output_root=config.dbx_output_dir(), converter=converter, describe_pictures=True,
                             trace=span)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
            logger.warning("%s: parsing failed: %s", pdf, exc)
            span.end(error=exc)
            return doc_id, pdf, "failed", str(exc)[:2000], options, None, None, round(time.perf_counter() - t0, 2)
        span.set(manifest, doc_id=doc_id)
    status, error = _outcome(manifest)
    return (doc_id, pdf, status, error, options, manifest.get("n_pages"), manifest.get("convert_seconds"),
            round(time.perf_counter() - t0, 2))


def parse_paths(session: Any, paths: Iterable[str], *, parse: Any = None, converter: Any = None,
                run_key: str | None = None) -> dict[str, Any]:
    """Parse the PDFs not parsed yet with the current options; returns the counts and durations of the run, also
    logged to MLflow when ``run_key`` is given."""
    with tracking.track_step(tracking.INDEXING, "parse", run_key, options=_options()) as tracker:
        summary = _parse_paths(session, paths, parse, converter, run_key)
        tracker.metrics(**summary)
    return summary


def _parse_paths(session: Any, paths: Iterable[str], parse: Any, converter: Any,
                 run_key: str | None = None) -> dict[str, Any]:
    if parse is None:
        from ..parsing import parse_pdf as parse

    options = _options()
    seen = _already_parsed(session, options)
    counts = {"parsed": 0, "partial": 0, "failed": 0, "skipped": 0}
    pages, convert_seconds, parse_seconds = 0, 0.0, 0.0
    paths = list(paths)
    progress.report(f"parse: {len(paths)} PDF(s) to process")
    for position, path in enumerate(paths, 1):
        pdf = _local_path(path)
        label = f"parse {position}/{len(paths)} {Path(pdf).name}"
        try:
            doc_id = compute_doc_id(pdf)
        except OSError as exc:  # unreadable: no content to hash, the row is identified by its path
            doc_id = "unreadable-" + hashlib.sha256(pdf.encode("utf-8")).hexdigest()[:16]
            row = (doc_id, pdf, "failed", str(exc)[:2000], options, None, None, 0.0)
        else:
            if doc_id in seen:
                counts["skipped"] += 1
                progress.report(f"{label}: already parsed")
                continue
            progress.report(f"{label}: started")
            row = _parse_one(pdf, doc_id, options, parse, converter if converter is not None else get_converter(), run_key)
        seen.add(doc_id)
        _save(session, row)
        counts[row[2]] += 1
        pages += row[5] or 0
        convert_seconds += row[6] or 0.0
        parse_seconds += row[7] or 0.0
        problem = f" ({row[3][:200]})" if row[3] else ""
        progress.report(f"{label}: {row[2]}, {row[5] or 0} page(s), {row[7] or 0:.0f} s{problem}")
    summary = {**counts, "pages": pages, "converter_load_seconds": round(_converter_seconds, 2),
               "convert_seconds": round(convert_seconds, 2),
               "describe_seconds": round(max(parse_seconds - convert_seconds, 0.0), 2),
               "convert_seconds_per_page": round(convert_seconds / pages, 2) if pages else None}
    logger.info("parsing step: %s", summary)
    print(json.dumps({"hgrag_parsing": summary}))  # in the driver log and the output of the job
    return summary


def parse_pending(session: Any, *, parse: Any = None, converter: Any = None, run_key: str | None = None) -> dict[str, Any]:
    """Parse the discovered PDFs without a row in ``documents``, or changed since that row was written."""
    rows = session.sql(f"""SELECT DISTINCT regexp_replace(q.path, '^(dbfs|file):', '') AS path
        FROM {config.dbx_table(QUEUE)} q LEFT ANTI JOIN {config.dbx_table(TABLE)} d
        ON regexp_replace(q.path, '^(dbfs|file):', '') = d.path AND d.updated_at >= q.modificationTime""").collect()
    return parse_paths(session, [r.path for r in rows], parse=parse, converter=converter, run_key=run_key)


def retry_incomplete(session: Any, *, parse: Any = None, converter: Any = None, run_key: str | None = None) -> dict[str, Any]:
    """Parse again the ``failed`` and ``partial`` documents: Auto Loader will not offer them twice."""
    rows = session.sql(f"SELECT path FROM {config.dbx_table(TABLE)} WHERE status IN ('failed', 'partial')").collect()
    return parse_paths(session, [r.path for r in rows], parse=parse, converter=converter, run_key=run_key)


def discover_pdfs(spark: Any) -> Any:
    """Record the new PDFs of the volume in the queue table, then stop (``availableNow``). The parsing is not done in
    ``foreachBatch``: with Spark Connect it runs in another Python process, without the configuration and the secrets
    of the job, and the secrets must not travel with the function."""
    ensure_table(spark)
    progress.report("discovery: looking for new PDFs")
    query = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.allowOverwrites", str(config.DBX_ALLOW_OVERWRITES).lower())
        .option("pathGlobFilter", "*.pdf")
        .load(config.dbx_pdf_dir())
        .select("path", "modificationTime")
        .writeStream
        .option("checkpointLocation", config.dbx_checkpoint())
        .trigger(availableNow=True)
        .toTable(config.dbx_table(QUEUE))
    )
    query.awaitTermination()
    progress.report("discovery: done")
    return query
