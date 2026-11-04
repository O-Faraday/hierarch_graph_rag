"""The Databricks backend runs without the packages of the local extra; the query side runs without Docling too."""

import os
import subprocess
import sys
import textwrap

import pytest

LOCAL_ONLY = ("qdrant_client", "streamlit", "sentence_transformers", "psutil")


def _import_without(blocked: tuple[str, ...], modules: tuple[str, ...]) -> subprocess.CompletedProcess:
    code = textwrap.dedent(f"""
        import importlib, sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {blocked!r}:
                    raise ImportError(f"blocked: {{name}}")

        sys.meta_path.insert(0, Block())
        for module in {modules!r}:
            importlib.import_module(module)
    """)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "HGRAG_CONFIG": ""})


@pytest.mark.parametrize("blocked, modules", [
    (LOCAL_ONLY, ("hgrag.hierarchy", "hgrag.indexing", "hgrag.pipeline", "hgrag.retrieval")),
    ((*LOCAL_ONLY, "docling", "docling_core"), ("hgrag.config", "hgrag.ids", "hgrag.keywords", "hgrag.retrieval",
                                                "hgrag.generation")),
])
def test_modules_import_without_the_local_extra(blocked, modules):
    done = _import_without(blocked, modules)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_block_really_blocks():
    assert _import_without(LOCAL_ONLY, ("qdrant_client",)).returncode != 0
