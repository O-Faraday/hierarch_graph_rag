"""MLflow tracking of the pipeline steps (indexing and retrieval).

Layout in MLflow: one *experiment* per pipeline (``hgrag_pipeline``, ``hgrag_retrieval``),
one *parent run* per execution (identified by ``run_key``: the Airflow ``run_id``, or a
generated id for local runs), one *child run* per step (and per mapped item, e.g. per PDF or
per reconstructed section). Steps run in different processes under Airflow, so the parent is
found-or-created through a tag, guarded by a file lock.

Tracking is opt-in per call: ``run_key=None`` disables it. It never breaks a pipeline:
any MLflow error is logged and swallowed.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

INDEXING = "hgrag_pipeline"
RETRIEVAL = "hgrag_retrieval"
EVAL = "hgrag_eval"
RAGBENCH = "hgrag_ragbench"

TRACKING_DIR = Path(os.environ.get("HGRAG_MLFLOW_DIR", config.PROJECT_ROOT / "data" / "mlflow"))
# SQLite backend: the file store is deprecated in MLflow 3 and SQLite works on Windows too.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}"
_TAG_KEY = "hgrag.run_key"


def new_run_key() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


class StepTracker:
    """Thin wrapper around the active child run; every method is failure-safe."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _safe(self, fn, *args, **kwargs) -> None:
        if not self.enabled:
            return
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tracking must never break the pipeline
            logger.warning("MLflow call %s failed: %s", getattr(fn, "__name__", fn), exc)

    def params(self, **params: Any) -> None:
        import mlflow

        self._safe(mlflow.log_params, {k: str(v)[:500] for k, v in params.items()})

    def metrics(self, **metrics: float) -> None:
        import mlflow

        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._safe(mlflow.log_metrics, clean)

    def tags(self, **tags: Any) -> None:
        import mlflow

        self._safe(mlflow.set_tags, {k: str(v) for k, v in tags.items()})

    def dict(self, data: Any, name: str) -> None:
        """Log a JSON-able object as an artifact (e.g. step output)."""
        import mlflow

        self._safe(mlflow.log_dict, data, name)

    def text(self, text: str, name: str) -> None:
        import mlflow

        self._safe(mlflow.log_text, text, name)

    def artifacts(self, directory: str | Path, artifact_path: str | None = None) -> None:
        import mlflow

        if Path(directory).is_dir():
            self._safe(mlflow.log_artifacts, str(directory), artifact_path)

    def artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        import mlflow

        self._safe(mlflow.log_artifact, str(path), artifact_path)


_DISABLED = StepTracker(False)


def _setup(experiment: str) -> None:
    import mlflow

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if TRACKING_URI.startswith("sqlite") and not mlflow.get_experiment_by_name(experiment):
        try:
            mlflow.create_experiment(experiment, artifact_location=(TRACKING_DIR / "artifacts" / experiment).as_uri())
        except Exception:  # noqa: BLE001 - concurrent creation by another task
            pass
    mlflow.set_experiment(experiment)


def _parent_run_id(experiment: str, run_key: str) -> str:
    """Find or create the parent run of an execution (safe across processes)."""
    import mlflow
    from filelock import FileLock

    exp = mlflow.get_experiment_by_name(experiment)
    with FileLock(str(TRACKING_DIR / f"{experiment}.lock"), timeout=60):
        found = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.`{_TAG_KEY}` = '{run_key}' and tags.`hgrag.kind` = 'execution'",
            max_results=1,
            output_format="list",
        )
        if found:
            return found[0].info.run_id
        run = mlflow.MlflowClient().create_run(
            exp.experiment_id,
            tags={_TAG_KEY: run_key, "hgrag.kind": "execution", "mlflow.runName": run_key},
        )
        return run.info.run_id


@contextmanager
def track_step(
    pipeline: str, step: str, run_key: str | None, *, item: str | None = None, **params: Any
) -> Iterator[StepTracker]:
    """Open a child run for one step. ``item`` names the mapped item (a PDF, a section...).

    ``run_key=None`` (or an MLflow setup failure) yields a no-op tracker.
    """
    if run_key is None:
        yield _DISABLED
        return
    try:
        import mlflow

        _setup(pipeline)
        parent_id = _parent_run_id(pipeline, run_key)
        run_name = f"{step}:{item}" if item else step
        mlflow.start_run(run_name=run_name, parent_run_id=parent_id,
                               tags={_TAG_KEY: run_key, "hgrag.kind": "step", "hgrag.step": step})
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow disabled for step %s: %s", step, exc)
        yield _DISABLED
        return

    tracker = StepTracker(True)
    t0 = time.perf_counter()
    status = "FINISHED"
    try:
        tracker.params(**params)
        yield tracker
    except BaseException as exc:
        status = "FAILED"
        tracker.tags(error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        tracker.metrics(duration_s=time.perf_counter() - t0)
        try:
            import mlflow

            mlflow.end_run(status=status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow end_run failed: %s", exc)


def finish_execution(pipeline: str, run_key: str, *, failed: bool = False, **metrics: float) -> None:
    """Close the parent run of an execution (local runs and the final Airflow task)."""
    try:
        import mlflow

        _setup(pipeline)
        rid = _parent_run_id(pipeline, run_key)
        client = mlflow.MlflowClient()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                client.log_metric(rid, k, float(v))
        client.set_terminated(rid, status="FAILED" if failed else "FINISHED")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow finish_execution failed: %s", exc)
