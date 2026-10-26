"""Tests never load a cross-encoder and never add model-specific prefixes, whatever the production defaults are."""

import os

import pytest

# Offline: MLflow would otherwise fetch its telemetry configuration (config.mlflow-telemetry.io) and send usage
# events whenever a test tracks a run or a trace. Set before mlflow is imported.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"

from hgrag import config


@pytest.fixture(autouse=True)
def _plain_models(monkeypatch):
    monkeypatch.setattr(config, "RERANK", False)
    monkeypatch.setenv("HGRAG_EMBED_QUERY_PREFIX", "")
    monkeypatch.setenv("HGRAG_EMBED_DOC_PREFIX", "")
