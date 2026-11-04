"""The Databricks notebooks are valid source notebooks, stay thin and hold no secret value."""

import ast
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "databricks" / "notebooks").glob("*.py"))


def test_notebooks_exist():
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_a_source_notebook_without_secret(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Databricks notebook source\n")
    ast.parse(text)
    assert not re.search(r"dapi[0-9a-f]{20,}|hf_[A-Za-z0-9]{25,}|sk-ant-", text)
    assert not re.search(r"/Volumes/[a-z]", text), "paths come from the widgets"
    assert not re.search(r"^\s*def ", text, re.M), "the logic belongs to the package"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_magic_commands_are_alone_in_their_cell(path):
    for cell in path.read_text(encoding="utf-8").split("# COMMAND ----------"):
        lines = [line for line in cell.splitlines() if line.strip() and line != "# Databricks notebook source"]
        if any(line.startswith("# MAGIC") for line in lines):
            assert all(line.startswith("# MAGIC") for line in lines), cell


def test_parsing_notebook_keeps_opencv_below_the_fips_abort():
    text = (NOTEBOOKS[0].parent / "01_parse_documents.py").read_text(encoding="utf-8")
    assert '%pip install "opencv-python-headless==4.11.0.86" $wheel_spec' in text


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_configuration_is_set_before_hgrag_is_imported(path):
    text = path.read_text(encoding="utf-8")
    if "from hgrag" in text:
        assert text.index("os.environ.update") < text.index("from hgrag")
