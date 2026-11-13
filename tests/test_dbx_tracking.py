"""MLflow tracking of the Databricks steps, without MLflow calls or Databricks."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from hgrag import config, tracking
from hgrag.dbx import indexing, parsing


class Tracker:
    def __init__(self):
        self.metric_values, self.tag_values, self.saved = {}, {}, {}

    def metrics(self, **values):
        self.metric_values.update(values)

    def tags(self, **values):
        self.tag_values.update(values)

    def dict(self, data, name):
        self.saved[name] = data


@pytest.fixture
def steps(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_VOLUME", "pdf_store")
    monkeypatch.setattr(config, "DBX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DBX_SEARCH_ENDPOINT", "hgrag")
    monkeypatch.setattr(config, "TRACING", False)  # the runs only: the traces are tested in test_dbx_traces.py
    opened = []

    @contextmanager
    def track_step(pipeline, step, run_key, **params):
        tracker = Tracker()
        opened.append(SimpleNamespace(pipeline=pipeline, step=step, run_key=run_key, params=params, tracker=tracker))
        yield tracker

    monkeypatch.setattr(tracking, "track_step", track_step)
    return opened


class Session:
    def createDataFrame(self, rows, _schema):
        return SimpleNamespace(createOrReplaceTempView=lambda _name: None)

    def sql(self, _statement, args=None):
        return SimpleNamespace(collect=lambda: [])


def test_experiments_go_under_the_configured_folder(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_ROOT", "")
    assert tracking.experiment_name(tracking.INDEXING) == "hgrag_indexing"
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_ROOT", "/Shared")
    assert tracking.experiment_name(tracking.INDEXING) == "/Shared/hgrag_indexing"


def test_setup_creates_the_experiment_under_that_folder(monkeypatch, tmp_path):
    import mlflow

    created = []
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_ROOT", "/Shared")
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path)
    monkeypatch.setattr(tracking, "TRACKING_URI", "databricks")
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda _name: None)
    monkeypatch.setattr(mlflow, "create_experiment", lambda name, **kwargs: created.append((name, kwargs)) or "1")
    assert tracking._setup(tracking.INDEXING) == "1"
    assert created == [("/Shared/hgrag_indexing", {})]


def test_parsing_logs_its_summary(steps):
    summary = parsing.parse_paths(Session(), [], parse=lambda *_a, **_k: {}, converter=object(), run_key="k")
    assert (steps[0].step, steps[0].run_key) == ("parse", "k") and "table_mode" in steps[0].params["options"]
    assert steps[0].tracker.metric_values == summary


def test_chunking_logs_its_summary_and_failures(steps):
    summary = indexing.chunk_pending(Session(), chunker=object(), build=lambda *_: [], run_key="k")
    assert steps[0].step == "chunk" and steps[0].params["tokenizer"] == config.DBX_CHUNK_TOKENIZER
    assert steps[0].tracker.saved["chunks_summary.json"] == summary


def test_index_sync_logs_the_rows_and_the_state(steps):
    index = SimpleNamespace(wait_until_ready=lambda **_k: None, sync=lambda: None, describe=lambda: {
        "status": {"detailed_state": "ONLINE", "indexed_row_count": 7}})
    client = SimpleNamespace(endpoint_exists=lambda _n: True, index_exists=lambda *_a: True, get_index=lambda *_a: index)
    indexing.sync_index(client, run_key="k")
    assert steps[0].step == "index" and steps[0].tracker.metric_values["indexed_rows"] == 7
    assert steps[0].tracker.tag_values["index_state"] == "ONLINE"
