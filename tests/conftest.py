"""Tests never load a cross-encoder and never add model-specific prefixes, whatever the production defaults are."""

import os

import pytest

# Offline: MLflow would otherwise fetch its telemetry configuration (config.mlflow-telemetry.io) and send usage
# events whenever a test tracks a run or a trace. Set before mlflow is imported.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
# The tests run on the code defaults, not on the project settings of hgrag.toml.
os.environ["HGRAG_CONFIG"] = ""

from hgrag import config


@pytest.fixture(autouse=True)
def _plain_models(monkeypatch):
    # .env is loaded by hgrag.config: its keys and switches must not reach the tests. Without this, HGRAG_ASSESS_JUDGES=1
    # made tests call the real Anthropic API with the user's key. A test that needs a key sets a fake one.
    for key in ("ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "ASSESS_JUDGES", False)
    monkeypatch.setattr(config, "RERANK", False)
    monkeypatch.setenv("HGRAG_EMBED_QUERY_PREFIX", "")
    monkeypatch.setenv("HGRAG_EMBED_DOC_PREFIX", "")
