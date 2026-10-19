"""Tests never load a cross-encoder and never add model-specific prefixes, whatever the production defaults are."""

import pytest

from hgrag import config


@pytest.fixture(autouse=True)
def _plain_models(monkeypatch):
    monkeypatch.setattr(config, "RERANK", False)
    monkeypatch.setenv("HGRAG_EMBED_QUERY_PREFIX", "")
    monkeypatch.setenv("HGRAG_EMBED_DOC_PREFIX", "")
