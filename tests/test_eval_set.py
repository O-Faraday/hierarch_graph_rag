"""The evaluation set of data/pdf_eval: every evidence string is in its PDF, every section is a real heading."""

from __future__ import annotations

import json
import re

import pypdfium2 as pdfium
import pytest

from hgrag import config
from hgrag.verify import _norm

EVAL_DIR = config.PROJECT_ROOT / "data" / "pdf_eval"
ROWS = [json.loads(line) for line in (EVAL_DIR / "eval_dataset.jsonl").read_text(encoding="utf-8").splitlines()]
EXPECTED = json.loads((EVAL_DIR / "expected.json").read_text(encoding="utf-8"))


def _text(pdf: str) -> str:
    doc = pdfium.PdfDocument(str(EVAL_DIR / pdf))
    try:
        return re.sub(r"\s+", " ", " ".join(page.get_textpage().get_text_range() for page in doc)).lower()
    finally:
        doc.close()


def test_ids_are_unique_and_rows_complete():
    assert len({r["id"] for r in ROWS}) == len(ROWS)
    for r in ROWS:
        assert r["question"] and r["expected_answer"] and r["source_kind"]
        assert (r["pdf"] is None) == (r["section"] is None) == (r["source_kind"] == "unanswerable"), r["id"]


@pytest.mark.parametrize("pdf", sorted(EXPECTED))
def test_evidence_is_in_the_pdf_and_sections_are_headings(pdf):
    text = _text(pdf)
    headings = {_norm(h) for h, _level in EXPECTED[pdf]["headings"]}
    rows = [r for r in ROWS if r["pdf"] == pdf]
    assert rows, f"no question on {pdf}"
    for r in rows:
        assert _norm(r["section"]) in headings, (r["id"], r["section"])
        for ev in r["evidence"]:
            assert re.sub(r"\s+", " ", ev).lower() in text, (r["id"], ev)
        assert r["evidence"] or r["requires_vlm"], f"{r['id']}: no evidence and no image to read it from"


def test_every_pdf_of_the_set_is_described():
    pdfs = {r["pdf"] for r in ROWS if r["pdf"]}
    assert pdfs - set(EXPECTED) == {"2404.02228v3.pdf"}  # downloaded by scripts/fetch_eval_arxiv.py, not generated
