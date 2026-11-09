"""The CLI must not crash on characters the console encoding (cp1252 on Windows) cannot represent."""

from __future__ import annotations

import io
import sys

from hgrag import cli


def test_tolerant_console_replaces_unencodable_characters(monkeypatch):
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    cli._tolerant_console()
    print("sum Σ of costs")
    out.flush()
    assert raw.getvalue().decode("cp1252") == "sum \\u03a3 of costs\n"
