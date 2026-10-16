"""Regression test: HybridChunker exposes generic DocItem objects, not TextItem.

An offline whitespace tokenizer stands in for the Hugging Face one (no download needed).
"""

from __future__ import annotations

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size

from hgrag import config
from hgrag.hierarchy import build_hierarchy
import pytest
from docling_core.types.doc import DocItemLabel

from hgrag.indexing import chunk_label_set, chunk_paragraphs


class WhitespaceTokenizer(BaseTokenizer):
    max_tokens: int = 256

    def count_tokens(self, text):
        return len(text.split())

    def get_max_tokens(self):
        return self.max_tokens

    def get_tokenizer(self):
        # Oversized paragraphs are split with semchunk, which needs a callable token counter.
        return self.count_tokens


def _prov() -> ProvenanceItem:
    return ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _doc():
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    h = d.add_heading(text="1. Intro", level=1, prov=_prov())
    d.add_text(label="text", text="First test paragraph.", parent=h, prov=_prov())
    lst = d.add_list_group(parent=h)
    d.add_list_item(text="a bullet", parent=lst, prov=_prov())
    d.add_text(label="text", text="Claims", parent=h, prov=_prov())  # singleton, filtered out
    d.add_text(label="text", text="Second test paragraph.", parent=h, prov=_prov())
    d.add_text(label="text", text="word " * 600, parent=h, prov=_prov())  # split by max_tokens
    return d, build_hierarchy(d, doc_title="x")


def test_hybrid_chunker_keeps_paragraphs_and_list_items(monkeypatch):
    monkeypatch.setattr(config, "MIN_PROSE_WORDS", 0)
    d, root = _doc()
    chunker = HybridChunker(tokenizer=WhitespaceTokenizer(max_tokens=256), merge_peers=False)
    stats: dict = {}
    chunks = chunk_paragraphs(d, chunker, root, "doc1", stats=stats)  # default labels: all textual elements

    texts = [c["text"] for c in chunks]
    assert "Claims" not in texts
    assert texts[0] == "First test paragraph." and texts[1].endswith("a bullet") and texts[2] == "Second test paragraph."
    assert chunks[1]["kinds"] == ["list_item"] and chunks[0]["kinds"] == ["text"]
    assert len(chunks) == 3 + 3  # the 600-word paragraph is split in 3 chunks of <= 256
    assert all(len(c["text"].split()) <= 256 for c in chunks)
    assert all(c["section_path"] == "x > 1. Intro" for c in chunks)  # list items map to their section too
    assert stats["dropped"] == {"too_short": 1}

    # the filter can be disabled
    all_chunks = chunk_paragraphs(d, chunker, root, "doc1", min_words=0)
    assert "Claims" in [c["text"] for c in all_chunks] and len(all_chunks) == len(chunks) + 1


def test_chunk_labels_are_configurable(monkeypatch):
    monkeypatch.setattr(config, "MIN_PROSE_WORDS", 0)
    d, root = _doc()
    chunker = HybridChunker(tokenizer=WhitespaceTokenizer(max_tokens=256), merge_peers=False)
    only_paragraphs = chunk_paragraphs(d, chunker, root, "doc1", labels=["text", "paragraph"])
    assert len(only_paragraphs) == 2 + 3 and all(c["kinds"] == ["text"] for c in only_paragraphs)
    with pytest.raises(ValueError, match="unknown Docling label 'bogus'"):
        chunk_label_set(["text", "bogus"])
    assert DocItemLabel.LIST_ITEM in chunk_label_set()  # default set includes list items, tables, code
    assert {DocItemLabel.TABLE, DocItemLabel.CODE} <= chunk_label_set()


def test_short_prose_is_dropped_and_only_the_closest_heading_is_embedded(monkeypatch):
    d, root = _doc()
    chunker = HybridChunker(tokenizer=WhitespaceTokenizer(max_tokens=256), merge_peers=False)
    monkeypatch.setattr(config, "MIN_PROSE_WORDS", 5)
    chunks = chunk_paragraphs(d, chunker, root, "doc1")
    assert "First test paragraph." not in [c["text"] for c in chunks]  # 3 words: a date line, an author name...
    monkeypatch.setattr(config, "MIN_PROSE_WORDS", 0)
    monkeypatch.setattr(config, "HEADING_CONTEXT", 1)
    one = chunk_paragraphs(d, chunker, root, "doc1")
    monkeypatch.setattr(config, "HEADING_CONTEXT", 0)
    none = chunk_paragraphs(d, chunker, root, "doc1")
    assert all(c["embed_text"] == c["text"] for c in none)
    assert all(c["embed_text"].count("\n") <= 1 for c in one) and any(c["embed_text"].startswith("1. Intro") for c in one)


def test_title_is_not_prepended_and_references_are_not_indexed(monkeypatch):
    monkeypatch.setattr(config, "MIN_PROSE_WORDS", 0)
    monkeypatch.setattr(config, "HEADING_CONTEXT", 99)
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    t = d.add_text(label="title", text="A Study of Fe Clusters", prov=_prov())
    d.add_text(label="text", text="Princeton University, New Jersey, USA", parent=t, prov=_prov())
    h = d.add_heading(text="2. Methods", level=1, prov=_prov())
    h2 = d.add_heading(text="2.1 Setup", level=2, parent=h, prov=_prov())
    d.add_text(label="text", text="The simulation uses a classical potential for iron atoms.", parent=h2, prov=_prov())
    r = d.add_heading(text="References", level=1, prov=_prov())
    d.add_text(label="text", text="Guryan, J., Kroft, K. (2009). Peer effects in sequential settings. Journal.", parent=r, prov=_prov())
    root = build_hierarchy(d, doc_title="x")
    chunker = HybridChunker(tokenizer=WhitespaceTokenizer(max_tokens=256), merge_peers=False)
    stats: dict = {}
    chunks = chunk_paragraphs(d, chunker, root, "doc1", stats=stats)
    by_text = {c["text"]: c["embed_text"] for c in chunks}
    assert by_text["Princeton University, New Jersey, USA"] == "Princeton University, New Jersey, USA"  # no title prefix
    assert by_text["The simulation uses a classical potential for iron atoms."].startswith("2. Methods\n2.1 Setup\n")
    assert not any("Guryan" in c["text"] for c in chunks) and stats["dropped"].get("excluded_section") == 1
