"""The evaluation dataset is well formed and every answer is grounded in the control PDFs."""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest
from pypdf import PdfReader

from hgrag import config

CONTROL = config.CONTROL_PDF_DIR
ROWS = [json.loads(x) for x in (CONTROL / "eval_dataset.jsonl").read_text(encoding="utf-8").splitlines() if x]
EXPECTED = json.loads((CONTROL / "expected.json").read_text(encoding="utf-8"))
KINDS = {"paragraph", "table", "list", "code", "figure", "derived", "unanswerable"}


def _text(name: str) -> str:
    raw = " ".join(p.extract_text() for p in PdfReader(CONTROL / name).pages)
    return re.sub(r"\s+", " ", raw).lower()


def test_schema_and_coverage():
    assert len({r["id"] for r in ROWS}) == len(ROWS)
    assert {"id", "pdf", "section", "source_kind", "question", "expected_answer", "evidence", "requires_vlm"} <= set(ROWS[0])
    assert all(r["source_kind"] in KINDS and r["question"].endswith("?") and r["expected_answer"] for r in ROWS)
    per_pdf = Counter(r["pdf"] for r in ROWS if r["pdf"])
    assert set(per_pdf) == set(EXPECTED) and min(per_pdf.values()) >= 7
    assert KINDS <= {r["source_kind"] for r in ROWS}  # every kind of leaf is tested
    assert sum(r["requires_vlm"] for r in ROWS) >= 3


@pytest.mark.parametrize("row", [r for r in ROWS if r["pdf"]], ids=lambda r: r["id"])
def test_answer_is_grounded_in_the_pdf(row):
    headings = [h for h, _ in EXPECTED[row["pdf"]]["headings"]]
    assert row["section"] in headings, "the expected section must be a heading of the PDF"
    if row["source_kind"] == "figure":
        assert row["requires_vlm"] and not row["evidence"]  # the value lives in the image only
    else:
        assert row["evidence"], "text questions need verbatim evidence"
        text = _text(row["pdf"])
        for snippet in row["evidence"]:
            assert snippet.lower() in text, snippet
