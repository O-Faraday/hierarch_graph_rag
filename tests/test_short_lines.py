"""Consecutive short lines of a section (a code block cut by Docling) are joined into one chunk instead of dropped."""

from __future__ import annotations

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size

from hgrag import config
from hgrag.hierarchy import build_hierarchy
from hgrag.indexing import chunk_paragraphs
from test_hybrid import WhitespaceTokenizer


def _prov() -> ProvenanceItem:
    return ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _manual() -> DoclingDocument:
    """The WT-90 manual of the evaluation set: the command block of 3.2 is read as lines of 2 to 4 words."""
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    h = d.add_heading(text="3.1 Procedure", level=1, prov=_prov())
    d.add_text(label="text", text="Lock the rotor before you climb into the nacelle of the turbine.", parent=h,
               prov=_prov())
    d.add_text(label="text", text="Date: 2026", parent=h, prov=_prov())  # alone in its run: still dropped
    h2 = d.add_heading(text="3.2 Controller commands", level=1, prov=_prov())
    d.add_text(label="text", text="The controller is operated from the console in the nacelle:", parent=h2, prov=_prov())
    for line in ("wtctl mode maintenance --confirm", "wtctl status", "mode: maintenance", "rotor_lock: engaged"):
        d.add_text(label="text", text=line, parent=h2, prov=_prov())
    return d


def _chunks():
    doc = _manual()
    root = build_hierarchy(doc, [], doc_title="x")
    stats: dict = {}
    chunks = chunk_paragraphs(doc, HybridChunker(tokenizer=WhitespaceTokenizer(), merge_peers=False), root, "d1",
                              stats=stats)
    return chunks, stats


def test_the_lines_of_a_code_block_become_one_chunk_of_their_section():
    chunks, stats = _chunks()
    block = [c for c in chunks if "wtctl mode maintenance --confirm" in c["text"]]
    assert len(block) == 1
    assert block[0]["text"].splitlines() == ["wtctl mode maintenance --confirm", "wtctl status", "mode: maintenance",
                                             "rotor_lock: engaged"]
    assert block[0]["section_title"] == "3.2 Controller commands"
    assert len(block[0]["element_refs"]) == 4
    assert stats["dropped"] == {"too_short": 1}  # "Date: 2026": no short neighbour in its section


def test_without_merging_the_lines_are_dropped(monkeypatch):
    monkeypatch.setattr(config, "MERGE_SHORT_LINES", False)
    chunks, stats = _chunks()
    assert not any("wtctl" in c["text"] for c in chunks)
    assert stats["dropped"] == {"too_short": 5}
