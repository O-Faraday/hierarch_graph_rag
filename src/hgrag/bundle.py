"""One zip with everything needed to analyse a pipeline run away from the machine that ran it."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

from . import config, runner, tracking

LIBS = ("docling", "docling-core", "sentence-transformers", "qdrant-client", "mlflow", "streamlit", "torch",
        "transformers")


def _versions() -> dict[str, str]:
    out = {}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not installed"
    return out


def _add_tree(zf: zipfile.ZipFile, src: Path, arc: str, counter: dict[str, int]) -> None:
    for f in sorted(src.rglob("*")):
        if f.is_file():
            zf.write(f, f"{arc}/{f.relative_to(src).as_posix()}")
            counter[arc.split("/")[0]] = counter.get(arc.split("/")[0], 0) + 1


def build_bundle(
    dest: str | Path | None = None, *, output_dir: Path = config.OUTPUT_DIR, runs_dir: Path | None = None,
    mlflow_dir: Path | None = None, control_dir: Path = config.CONTROL_PDF_DIR, max_runs: int = 10,
) -> Path:
    """Write the zip and return its path. Missing pieces are skipped and listed in ``bundle_info.json``."""
    runs_dir = Path(runs_dir or runner.RUNS_DIR)
    mlflow_dir = Path(mlflow_dir or tracking.TRACKING_DIR)
    dest = Path(dest or config.PROJECT_ROOT / "data" / "bundles" / f"hgrag-analysis-{time.strftime('%Y%m%dT%H%M%S')}.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    skipped: list[str] = []

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        if output_dir.exists() and any(output_dir.iterdir()):
            _add_tree(zf, output_dir, "output", counts)  # per document: Docling json/md, figures, tree, chunks
        else:
            skipped.append("output (nothing parsed yet)")

        runs = sorted((p for p in runs_dir.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)[:max_runs]
        for r in runs:
            for name in ("status.json", "log.txt"):
                if (r / name).exists():
                    zf.write(r / name, f"runs/{r.name}/{name}")
                    counts["runs"] = counts.get("runs", 0) + 1
        if not runs:
            skipped.append("runs (no run recorded)")

        db = mlflow_dir / "mlflow.db"
        if db.exists():  # consistent copy even while MLflow is writing
            with tempfile.TemporaryDirectory() as tmp:
                snap = Path(tmp) / "mlflow.db"
                src, dst = sqlite3.connect(db), sqlite3.connect(snap)
                try:
                    src.backup(dst)
                finally:
                    dst.close()
                    src.close()
                zf.write(snap, "mlflow/mlflow.db")
            counts["mlflow"] = 1
        else:
            skipped.append("mlflow.db (no tracking yet)")

        if (control_dir / "expected.json").exists():
            for f in ("expected.json", "eval_dataset.jsonl"):
                if (control_dir / f).exists():
                    zf.write(control_dir / f, f"control/{f}")
            try:  # structural checks only: no model and no Qdrant needed
                from .verify import format_report, verify

                zf.writestr("control/verify_structure.txt",
                            format_report(verify(control_dir, output_dir=output_dir)) + "\n")
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"verify report ({exc})")

        evals = sorted((p for p in config.EVAL_DIR.glob("*") if (p / "summary.json").exists()),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:3]  # the benchmark runs, not the store
        for e in evals:
            for name in ("summary.json", "results.jsonl", "report.md"):
                if (e / name).exists():
                    zf.write(e / name, f"eval/{e.name}/{name}")
                    counts["eval"] = counts.get("eval", 0) + 1

        lock = config.PROJECT_ROOT / "uv.lock"
        if lock.exists():
            zf.write(lock, "env/uv.lock")
        info: dict[str, Any] = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": sys.version.split()[0],
            "platform": platform.platform(), "libraries": _versions(),
            "config": {"embed_model": config.EMBED_MODEL_ID, "collection": config.COLLECTION_NAME,
                       "min_chunk_words": config.MIN_CHUNK_WORDS, "images_scale": config.IMAGES_SCALE,
                       "figure_vlm": config.FIGURE_VLM, "describe_pictures": config.DESCRIBE_PICTURES,
                       "picture_prompt": config.PICTURE_PROMPT,
                       "docling_options": {"table_mode": config.TABLE_MODE, "formula_enrichment": config.FORMULA_ENRICHMENT,
                                           "code_enrichment": config.CODE_ENRICHMENT,
                                           "picture_classification": config.PICTURE_CLASSIFICATION}},
            "env": {k: v for k, v in os.environ.items() if k.startswith("HGRAG_")},
            "files_per_section": counts, "skipped": skipped,
        }
        zf.writestr("bundle_info.json", json.dumps(info, indent=2, ensure_ascii=False))
    return dest
