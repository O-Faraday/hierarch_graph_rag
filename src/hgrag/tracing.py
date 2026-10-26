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

import contextvars
import json
import logging
from typing import Any

from . import config, tracking

logger = logging.getLogger(__name__)

_EXPERIMENT_IDS: dict[str, str] = {}
ASSESSMENTS_ATTRIBUTE = "hgrag.assessments"


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

    def __init__(self, live: Any = None, root: Span | None = None, nested: bool = False):
        self._live = live
        self._ended = live is None
        self._root = root or self
        self._nested = nested  # our root sits inside a trace opened by MLflow (genai.evaluate, @mlflow.trace)
        self._pending: list[tuple[str | None, Any]] = []  # (span id, assessment), on the root span only

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
                                                     attributes=_jsonable(attributes) if attributes else None),
                        root=self._root)
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

    def assess(self, assessments: list[Any]) -> None:
        """Attach production checks (``assessments.Assessment`` or their dicts) to this span. They are logged as MLflow
        feedback when the trace ends: a feedback needs its trace in the store."""
        if self._live is None:
            return
        from .assessments import Assessment

        records = [a if isinstance(a, Assessment) else Assessment(**a) for a in assessments]
        # a check with nothing to check (no figure to describe, no keyword search) has no value and no error: MLflow
        # shows a feedback without value as an error, so it stays in assessments.json only
        self._root._pending += [(self.span_id, a) for a in records if a.value is not None or a.error is not None]

    def end(self, outputs: Any = None, error: BaseException | None = None, **attributes: Any) -> None:
        if self._ended:
            return
        self._ended = True
        if self._root is self and self._nested and self._pending:
            # the enclosing trace is not stored yet, so no feedback can be logged on it: the checks travel in an
            # attribute, read back by the evaluation scorers (``genai_eval.hgrag_checks``)
            self.set(**{ASSESSMENTS_ATTRIBUTE: [{"span_id": sid, **a.to_dict()} for sid, a in self._pending]})
            self._pending = []
        try:
            self.set(outputs, **attributes)
            if error is not None:
                self._live.record_exception(error)
                self._live.set_status("ERROR")
            self._live.end()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow span end failed: %s", exc)
        if self._root is self and self._pending:
            self._log_assessments()

    def _log_assessments(self) -> None:
        try:
            import mlflow

            mlflow.flush_trace_async_logging()  # the trace must be stored before its feedback
            for span_id, a in self._pending:
                fb = a.to_feedback(self.trace_id, span_id)
                mlflow.log_feedback(trace_id=self.trace_id, span_id=span_id, name=fb.name, value=fb.value,
                                    source=fb.source, error=fb.error, rationale=fb.rationale, metadata=fb.metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow feedback not logged: %s", exc)
        self._pending = []

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if isinstance(exc, GeneratorExit):  # a generator whose consumer stopped reading: not an error
            self.end(closed_early=True)
        else:
            self.end(error=exc)


def root(name: str, pipeline: str, run_key: str | None, *, span_type: str = "CHAIN", inputs: Any = None,
         tags: dict[str, Any] | None = None) -> Span:
    """Start a trace in the experiment of ``pipeline`` (``tracking.INDEXING``, ``tracking.RETRIEVAL``...), linked to
    the model registered for the current configuration (``versions``). Inside a trace that MLflow already opened
    (``mlflow.genai.evaluate``, a function under ``@mlflow.trace``), the span is nested in it instead."""
    if run_key is None or not config.TRACING:
        return Span()
    try:
        import mlflow

        from . import versions

        from mlflow.entities.span import NoOpSpan

        outer = mlflow.get_current_active_span()
        if isinstance(outer, NoOpSpan):  # MLflow turned tracing off around this call (genai.evaluate's check run)
            return Span()
        if outer is not None:
            return Span(mlflow.start_span_no_context(
                name, span_type=span_type, parent_span=outer, inputs=_jsonable(inputs) if inputs is not None else None,
                attributes={"hgrag.run_key": run_key}), nested=True)
        model_id = versions.model_id("indexing" if pipeline == tracking.INDEXING else "rag")
        live = mlflow.start_span_no_context(
            name, span_type=span_type, inputs=_jsonable(inputs) if inputs is not None else None,
            experiment_id=_experiment_id(pipeline),
            tags={"hgrag.run_key": run_key, **{k: str(v) for k, v in (tags or {}).items()}},
            metadata={"mlflow.trace.session": run_key, **({"mlflow.modelId": model_id} if model_id else {})})
        return Span(live)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow trace %s not started: %s", name, exc)
        return Span()


def submit(pool: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """``pool.submit`` running ``fn`` in a copy of the caller's context. A span opened in a worker thread without
    it lost its whole trace under ``mlflow.genai.evaluate`` (the prediction context lives in a context variable);
    one copy per task, a context cannot be entered by two threads at once."""
    return pool.submit(contextvars.copy_context().run, fn, *args, **kwargs)


def child_or_root(parent: Span | None, name: str, pipeline: str, run_key: str | None, *, span_type: str = "CHAIN",
                  inputs: Any = None, tags: dict[str, Any] | None = None) -> Span:
    """A child of ``parent`` when the caller traces a larger operation, else a trace of its own."""
    if parent is not None and parent.active:
        return parent.child(name, span_type, inputs)
    return root(name, pipeline, run_key, span_type=span_type, inputs=inputs, tags=tags)
