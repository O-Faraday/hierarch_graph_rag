"""MLflow tracking of both pipelines (offline, SQLite backend in a temp dir)."""

from __future__ import annotations

import mlflow
import pytest

from hgrag import tracking
from hgrag.retrieval import retrieve_sections, retrieve_steps

from test_retrieval import BagOfWordsEmbedder, store  # noqa: F401  (fixture)


@pytest.fixture()
def mlflow_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlflow")
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}")
    return tmp_path / "mlflow"


def _runs(experiment, run_key):
    exp = mlflow.get_experiment_by_name(experiment)
    return mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list",
                              filter_string=f"tags.`hgrag.run_key` = '{run_key}'")


def test_retrieval_steps_are_tracked(store, mlflow_dir):  # noqa: F811
    client, out = store
    key = tracking.new_run_key()
    results = retrieve_sections("new branches portfolio", 3, embedder=BagOfWordsEmbedder(), client=client,
                                collection="c", output_dir=out, run_key=key)
    runs = _runs(tracking.RETRIEVAL, key)
    kinds = sorted(r.data.tags["hgrag.kind"] for r in runs)
    assert kinds.count("execution") == 1  # parent found-or-created once, even from threads
    steps = [r for r in runs if r.data.tags["hgrag.kind"] == "step"]
    assert sorted(r.data.tags["hgrag.step"] for r in steps) == ["reconstruct"] * len(results) + ["search"]
    search = next(r for r in steps if r.data.tags["hgrag.step"] == "search")
    assert search.data.metrics["n_sections"] == len(results)
    assert search.data.params["k"] == "3"
    assert all(r.info.status == "FINISHED" and "duration_s" in r.data.metrics for r in steps)


def test_stream_events_and_parent_closed(store, mlflow_dir):  # noqa: F811
    client, out = store
    key = tracking.new_run_key()
    events = list(retrieve_steps("settlement delay", 2, embedder=BagOfWordsEmbedder(), client=client,
                                 collection="c", output_dir=out, run_key=key))
    names = [e["step"] for e in events]
    assert names[:3] == ["embed", "vector_search", "group"] and names[-1] == "done"
    assert "reconstruct" in names
    parent = next(r for r in _runs(tracking.RETRIEVAL, key) if r.data.tags["hgrag.kind"] == "execution")
    assert parent.info.status == "FINISHED"


def test_tracking_disabled_and_failure_marked(mlflow_dir):
    with tracking.track_step(tracking.INDEXING, "x", None) as tr:
        assert tr.enabled is False
    key = tracking.new_run_key()
    with pytest.raises(ValueError):
        with tracking.track_step(tracking.INDEXING, "boom", key):
            raise ValueError("nope")
    step = next(r for r in _runs(tracking.INDEXING, key) if r.data.tags["hgrag.kind"] == "step")
    assert step.info.status == "FAILED" and "nope" in step.data.tags["error"]
