"""MLflow traces of the pipeline: what each step received and produced.

One trace per parsed document (``parse``, with the description calls), per hierarchy, per indexing (one span per
document) and per question (search, reranking, reconstruction, prompt, answer). Traces sit next to the runs of
``tracking`` in the same experiments, and follow the same switch: ``run_key=None`` traces nothing, and the
``run_key`` is the trace session, so the traces of one execution are found together. ``HGRAG_TRACING=0`` keeps the
runs and turns the traces off.

Spans are created with an explicit parent (``mlflow.start_span_no_context``), not through the context: the search
and the answer are generators consumed step by step by the app, and sections are rebuilt in threads. A span holds
the full content of the step (chunks, sections, prompt, answer), never an API key nor an image (its path only).
Every MLflow call is failure-safe: tracing never breaks the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import config, tracking

logger = logging.getLogger(__name__)

_EXPERIMENT_IDS: dict[str, str] = {}


def _experiment_id(experiment: str) -> str:
    import mlflow

    key = f"{tracking.TRACKING_URI}|{experiment}"
    if key not in _EXPERIMENT_IDS:
        _EXPERIMENT_IDS[key] = tracking._setup(experiment)
    mlflow.set_tracking_uri(tracking.TRACKING_URI)  # no-op when unchanged
    return _EXPERIMENT_IDS[key]


def _jsonable(value: Any) -> Any:
    """Span inputs and outputs are stored as JSON: anything else becomes its string."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


class Span:
    """A live MLflow span, or a no-op one (tracing off, or MLflow failed). Use as a context manager, or ``end()``."""

    def __init__(self, live: Any = None):
        self._live = live
        self._ended = live is None

    @property
    def active(self) -> bool:
        return self._live is not None

    @property
    def trace_id(self) -> str | None:
        return getattr(self._live, "trace_id", None)

    @property
    def span_id(self) -> str | None:
        return getattr(self._live, "span_id", None)

    def child(self, name: str, span_type: str = "UNKNOWN", inputs: Any = None,
              attributes: dict[str, Any] | None = None) -> Span:
        if self._live is None:
            return Span()
        try:
            import mlflow

            return Span(mlflow.start_span_no_context(name, span_type=span_type, parent_span=self._live,
                                                     inputs=_jsonable(inputs) if inputs is not None else None,
                                                     attributes=_jsonable(attributes) if attributes else None))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow span %s not created: %s", name, exc)
            return Span()

    def set(self, outputs: Any = None, **attributes: Any) -> None:
        """Outputs (replaced) and attributes (added) before the end."""
        if self._live is None:
            return
        try:
            if outputs is not None:
                self._live.set_outputs(_jsonable(outputs))
            if attributes:
                self._live.set_attributes(_jsonable(attributes))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow span update failed: %s", exc)

    def end(self, outputs: Any = None, error: BaseException | None = None, **attributes: Any) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self.set(outputs, **attributes)
            if error is not None:
                self._live.record_exception(error)
                self._live.set_status("ERROR")
            self._live.end()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow span end failed: %s", exc)

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if isinstance(exc, GeneratorExit):  # a generator whose consumer stopped reading: not an error
            self.end(closed_early=True)
        else:
            self.end(error=exc)


def root(name: str, pipeline: str, run_key: str | None, *, span_type: str = "CHAIN", inputs: Any = None,
         tags: dict[str, Any] | None = None) -> Span:
    """Start a trace in the experiment of ``pipeline`` (``tracking.INDEXING``, ``tracking.RETRIEVAL``...)."""
    if run_key is None or not config.TRACING:
        return Span()
    try:
        import mlflow

        live = mlflow.start_span_no_context(
            name, span_type=span_type, inputs=_jsonable(inputs) if inputs is not None else None,
            experiment_id=_experiment_id(pipeline),
            tags={"hgrag.run_key": run_key, **{k: str(v) for k, v in (tags or {}).items()}},
            metadata={"mlflow.trace.session": run_key})
        return Span(live)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow trace %s not started: %s", name, exc)
        return Span()


def child_or_root(parent: Span | None, name: str, pipeline: str, run_key: str | None, *, span_type: str = "CHAIN",
                  inputs: Any = None, tags: dict[str, Any] | None = None) -> Span:
    """A child of ``parent`` when the caller traces a larger operation, else a trace of its own."""
    if parent is not None and parent.active:
        return parent.child(name, span_type, inputs)
    return root(name, pipeline, run_key, span_type=span_type, inputs=inputs, tags=tags)
