"""Backend selection and Unity Catalog names of the Databricks backend."""

import os
import subprocess
import sys

import pytest

from hgrag import config


def _backend_in_a_new_process(tmp_path, **env) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if k != "HGRAG_BACKEND"}
    # An empty project root: no .env of this machine.
    clean.update(HGRAG_PROJECT_ROOT=str(tmp_path), HGRAG_CONFIG="", **env)
    return subprocess.run([sys.executable, "-c", "from hgrag import config; print(config.BACKEND)"],
                          capture_output=True, text=True, timeout=120, env=clean)


def test_backend_is_local_by_default(tmp_path):
    assert _backend_in_a_new_process(tmp_path).stdout.strip() == "local"


def test_backend_comes_from_the_environment(tmp_path):
    assert _backend_in_a_new_process(tmp_path, HGRAG_BACKEND="Databricks").stdout.strip() == "databricks"


def test_unknown_backend_stops_at_startup(tmp_path):
    done = _backend_in_a_new_process(tmp_path, HGRAG_BACKEND="qdrant")
    assert done.returncode != 0 and "HGRAG_BACKEND" in done.stderr


def test_table_and_volume_names(monkeypatch):
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_VOLUME", "pdf_store")
    assert config.dbx_table("chunks") == "`demo`.`pdf_bench`.`chunks`"
    assert config.dbx_volume_path() == "/Volumes/demo/pdf_bench/pdf_store"


@pytest.mark.parametrize("catalog", ["", "demo`; DROP TABLE x; --", "a.b", "a/../b"])
def test_missing_or_unsafe_names_are_refused(monkeypatch, catalog):
    monkeypatch.setattr(config, "DBX_CATALOG", catalog)
    monkeypatch.setattr(config, "DBX_SCHEMA", "s")
    monkeypatch.setattr(config, "DBX_VOLUME", "v")
    with pytest.raises(ValueError, match="HGRAG_DBX_CATALOG"):
        config.dbx_table("chunks")
    with pytest.raises(ValueError, match="HGRAG_DBX_CATALOG"):
        config.dbx_volume_path()
