"""Read-only helpers to show the MLflow runs of an execution inside the app."""

from __future__ import annotations

import os
from typing import Any

from hgrag import tracking

MLFLOW_UI_URL = os.environ.get("HGRAG_MLFLOW_UI_URL", "http://localhost:5000")


def execution_steps(experiment: str, run_key: str) -> list[dict[str, Any]]:
    """Child runs (steps) of one execution, with their metrics. Empty if MLflow has nothing."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking.TRACKING_URI)
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.`hgrag.run_key` = '{run_key}' and tags.`hgrag.kind` = 'step'",
            output_format="list",
            order_by=["attributes.start_time ASC"],
        )
    except Exception:  # noqa: BLE001 - monitoring must not break the page
        return []
    rows = []
    for r in runs:
        row = {"step": r.data.tags.get("mlflow.runName"), "status": r.info.status}
        row.update({k: round(v, 3) for k, v in r.data.metrics.items()})
        row["run_id"] = r.info.run_id
        rows.append(row)
    return rows
