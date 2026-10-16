"""Airflow DAG (retrieval flow, parallel to ``hgrag_pipeline``).

query -> similar chunks in Qdrant (grouped by section) -> one mapped task per matched
section rebuilds the document from the section elements, in parallel (LocalExecutor) -> merge.

Trigger it with the params ``query``, ``k`` and ``include_children``.
Note: Qdrant local mode accepts a single process on the store, so do not run this DAG while
``hgrag_pipeline`` is indexing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

PROJECT_ROOT = Path(os.environ.get("HGRAG_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
HGRAG_PYTHON = os.environ.get("HGRAG_PYTHON", str(PROJECT_ROOT / ".venv" / "bin" / "python"))


@dag(
    dag_id="hgrag_retrieval",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["hgrag", "retrieval", "qdrant"],
    params={
        "query": Param("", type="string", description="Natural-language query"),
        "k": Param(5, type="integer", minimum=1, maximum=50, description="Chunks to retrieve"),
        "include_children": Param(False, type="boolean", description="Also rebuild sub-sections"),
    },
)
def hgrag_retrieval():
    @task
    def prepare(params=None) -> dict:
        query = (params or {}).get("query", "").strip()
        if not query:
            raise ValueError("Set the `query` param when triggering the DAG.")
        return {
            "query": query,
            "k": int(params.get("k", 5)),
            "include_children": bool(params.get("include_children", False)),
        }

    @task.external_python(python=HGRAG_PYTHON)
    def search(request: dict, run_key: str) -> list[dict]:
        from hgrag.retrieval import search_hits

        return search_hits(request["query"], request["k"], run_key=run_key)

    # One mapped task instance per matched section: they run in parallel.
    @task.external_python(python=HGRAG_PYTHON)
    def reconstruct(hit: dict, request: dict, run_key: str) -> dict:
        from hgrag.retrieval import reconstruct_hit

        return reconstruct_hit(hit, include_children=request["include_children"], run_key=run_key)

    @task(trigger_rule="none_failed")
    def merge(sections) -> list[dict]:
        ordered = sorted(sections, key=lambda s: s["score"], reverse=True)
        for s in ordered:
            print(f"score={s['score']:.3f}  {s['section_path']}  ({s['n_elements']} elements)")
            print(s["text"], end="\n\n")
        return ordered

    # Closes the MLflow parent run of this DAG run (experiment `hgrag_retrieval`).
    @task.external_python(python=HGRAG_PYTHON)
    def close_run(ordered: list[dict], run_key: str) -> None:
        from hgrag.tracking import RETRIEVAL, finish_execution

        finish_execution(RETRIEVAL, run_key, n_sections=len(ordered))

    run_key = "{{ run_id }}"
    request = prepare()
    hits = search(request, run_key=run_key)
    sections = reconstruct.partial(request=request, run_key=run_key).expand(hit=hits)
    close_run(merge(sections), run_key=run_key)


hgrag_retrieval()
