"""Offline tests on a synthetic DoclingDocument (no model download needed)."""

from __future__ import annotations

import numpy as np
import pytest
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import TableData

from hgrag.hierarchy import build_hierarchy, section_index
from hgrag.indexing import chunk_paragraphs, index_chunks, open_store
from hgrag.parsing import picture_metadata


def _prov(page: int = 1) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 5))


@pytest.fixture()
def doc() -> DoclingDocument:
    d = DoclingDocument(name="synthetic")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    d.add_text(label="text", text="Preamble before any heading.", prov=_prov())
    h1 = d.add_heading(text="1. Introduction", level=1, prov=_prov())
    d.add_text(label="text", text="Intro paragraph one.", parent=h1, prov=_prov())
    h11 = d.add_heading(text="1.1 Background", level=2, parent=h1, prov=_prov())
    d.add_text(label="text", text="Background paragraph.", parent=h11, prov=_prov())
    d.add_picture(parent=h11, prov=_prov(), caption=d.add_text(label="caption", text="Fig. 1 A figure"))
    d.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[]), parent=h11, prov=_prov())
    h2 = d.add_heading(text="2. Methods", level=1, prov=_prov(2))
    d.add_text(label="text", text="Methods paragraph.", parent=h2, prov=_prov(2))
    return d


def test_heading_tree_and_leaves(doc):
    root = build_hierarchy(doc, doc_title="synthetic")
    titles = [n.title for n in root.walk()]
    assert titles == ["synthetic", "1. Introduction", "1.1 Background", "2. Methods"]

    intro = root.children[0]
    assert [c.title for c in intro.children] == ["1.1 Background"]
    assert not intro.is_leaf and intro.children[0].is_leaf
    assert root.children[1].is_leaf  # "2. Methods"
    assert intro.children[0].path == "synthetic > 1. Introduction > 1.1 Background"

    assert [e.kind for e in root.elements] == ["paragraph"]  # before any heading
    kinds = [e.kind for e in intro.children[0].elements]
    assert kinds == ["paragraph", "figure", "table"]  # standalone caption is not a leaf


def test_picture_metadata_fields(doc):
    pic = doc.pictures[0]
    rec = picture_metadata(pic, doc, index=1, doc_id="abc", png_path=None, rel_png_path=None)
    for key in ("self_ref", "page_no", "prov", "caption", "image", "description", "classification"):
        assert key in rec
    assert rec["caption"] == "Fig. 1 A figure"
    assert rec["prov"][0]["bbox"]["r"] == 10


class FakeEmbedder:
    def encode(self, sentences, **kwargs):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(sentences), 8)).astype("float32")


def test_paragraph_chunks_into_single_collection(doc, tmp_path):
    root = build_hierarchy(doc, doc_title="synthetic")
    # min_words=0: this test is about indexing, its paragraphs are deliberately short.
    chunks = chunk_paragraphs(doc, HierarchicalChunker(), root, "abc", min_words=0)
    texts = [c["text"] for c in chunks]
    assert texts == [
        "Preamble before any heading.",
        "Intro paragraph one.",
        "Background paragraph.",
        "Fig. 1 A figure",  # captions are indexed (HGRAG_CHUNK_LABELS includes "caption")
        "Methods paragraph.",
    ]
    assert chunks[2]["section_path"].endswith("1.1 Background") and chunks[3]["section_path"].endswith("1.1 Background")
    assert chunks[3]["kinds"] == ["caption"]

    client = open_store(tmp_path / "qdrant")
    n = index_chunks(client, chunks, FakeEmbedder(), doc_id="abc", source_pdf="x.pdf", collection="c")
    assert n == 5 and client.count("c").count == 5
    # idempotent re-indexing, and a second document goes in the same collection
    index_chunks(client, chunks, FakeEmbedder(), doc_id="abc", collection="c")
    index_chunks(client, chunks[:2], FakeEmbedder(), doc_id="def", collection="c")
    assert client.count("c").count == 7
    hit = client.query_points("c", query=[0.1] * 8, limit=1, with_payload=True).points[0]
    assert {"doc_id", "section_path", "page_no", "text"} <= set(hit.payload)
    client.close()


# ------------------------------------------------------------------------------- heading levels (roman numerals)
from hgrag.hierarchy import heading_levels, roman_to_int  # noqa: E402


def _hs(*items):
    return [(str(i), t, l) for i, (t, l) in enumerate(items)]


def test_roman_numbers():
    assert [roman_to_int(x) for x in ("I", "IV", "V", "IX", "XIV", "ABC")] == [1, 4, 5, 9, 14, None]


def test_roman_sections_letters_and_back_matter_get_the_right_levels():
    # what Docling gave on a real arXiv paper: V. CONCLUSION and the back matter nested under IV.
    hs = _hs(("Abstract", 1), ("I. INTRODUCTION", 1), ("II. METHODS", 1), ("III. RESULTS", 1), ("A. Structure", 2),
             ("B. Energetics", 2), ("C. Mobility", 2), ("IV. DISCUSSION", 1), ("A. Second-order", 2), ("B. Small", 2),
             ("C. Large", 3), ("D. Larger", 2), ("V. CONCLUSION", 2), ("REFERENCES", 2), ("SUPPLEMENTARY MATERIALS", 2))
    lv = [heading_levels(hs)[r] for r, _t, _l in hs]
    assert lv == [1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1]


def test_single_letters_are_not_taken_for_roman_numbers():
    # "V." right after "U." (a letter series) and a lone "I." must not become top-level headings
    hs = _hs(("1 Intro", 1), ("U. odd", 2), ("V. odd", 2))
    assert [heading_levels(hs)[r] for r, _t, _l in hs] == [1, 2, 2]
    hs = _hs(("I. Introduction", 1), ("A. Setup", 2))
    assert [heading_levels(hs)[r] for r, _t, _l in hs] == [1, 2]


def test_arabic_numbering_is_untouched_but_references_return_to_the_top():
    hs = _hs(("1 Introduction", 1), ("1.1 Related literature", 2), ("8 Conclusion", 1), ("References", 2),
             ("A Online Appendix", 1), ("A.1 Model", 2))
    assert [heading_levels(hs)[r] for r, _t, _l in hs] == [1, 2, 1, 1, 1, 2]
    assert heading_levels([("a", "Only one heading", 1), ("b", "References", 2)])["b"] == 2  # a single top level: keep


def test_build_hierarchy_uses_the_corrected_levels():
    d = DoclingDocument(name="paper")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    h = {}
    for text, lvl in [("I. INTRODUCTION", 1), ("II. METHODS", 1), ("III. DISCUSSION", 1), ("A. First", 2), ("IV. CONCLUSION", 2),
                      ("REFERENCES", 2)]:
        h[text] = d.add_heading(text=text, level=lvl, prov=_prov())
        d.add_text(label="text", text=f"Body of {text}.", parent=h[text], prov=_prov())
    root = build_hierarchy(d, doc_title="paper")
    paths = {n.title: n.path for n in root.walk()}
    assert paths["IV. CONCLUSION"] == "paper > IV. CONCLUSION" and paths["REFERENCES"] == "paper > REFERENCES"
    assert paths["A. First"] == "paper > III. DISCUSSION > A. First"
