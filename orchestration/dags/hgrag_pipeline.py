"""Airflow DAG: PDF -> Docling parsing -> heading tree -> paragraph chunks in Qdrant.

Airflow (3.3.2) lives in ``orchestration/.venv`` with its constraints file. The heavy
pipeline (Docling, torch, sentence-transformers, Qdrant) lives in the project venv and is
called through ``@task.external_python`` so the two dependency sets never have to resolve
together. Executor: LocalExecutor.
"""

from __future__ import annotations

import os
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path(os.environ.get("HGRAG_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
HGRAG_PYTHON = os.environ.get("HGRAG_PYTHON", str(PROJECT_ROOT / ".venv" / "bin" / "python"))
PDF_DIR = os.environ.get("HGRAG_PDF_DIR", str(PROJECT_ROOT / "data" / "pdf_sample"))


@dag(
    dag_id="hgrag_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["hgrag", "docling", "qdrant"],
    params={"pdf_dir": PDF_DIR},
)
def hgrag_pipeline():
    @task
    def list_pdfs(params=None) -> list[str]:
        pdf_dir = Path((params or {}).get("pdf_dir", PDF_DIR))
        return sorted(str(p) for p in pdf_dir.glob("*.pdf"))

    # One mapped task per PDF; max_active_tis_per_dag keeps VLM/layout models from
    # competing for memory on a single machine.
    @task.external_python(python=HGRAG_PYTHON, max_active_tis_per_dag=1)
    def parse(pdf_path: str, run_key: str) -> dict:
        from hgrag.pipeline import parse_step

        return parse_step(pdf_path, run_key=run_key)

    @task.external_python(python=HGRAG_PYTHON)
    def build_hierarchy(manifest: dict, run_key: str) -> dict:
        from hgrag.pipeline import build_and_save_hierarchy

        return build_and_save_hierarchy(manifest["doc_dir"], run_key=run_key)

    @task
    def collect(summaries) -> list[str]:
        # Materialise the lazy mapped XCom into plain strings for the next task.
        return [s["doc_dir"] for s in summaries]

    @task.external_python(python=HGRAG_PYTHON)
    def index(doc_dirs: list[str], run_key: str) -> dict:
        from hgrag.pipeline import index_documents

        return index_documents(doc_dirs, run_key=run_key)

    # `{{ run_id }}` (templated) is the MLflow run_key: it groups all steps of this DAG run
    # under one parent run in the experiment `hgrag_pipeline`.
    run_key = "{{ run_id }}"
    manifests = parse.partial(run_key=run_key).expand(pdf_path=list_pdfs())
    summaries = build_hierarchy.partial(run_key=run_key).expand(manifest=manifests)
    index(collect(summaries), run_key=run_key)


hgrag_pipeline()
