"""Retrieval flow: similarity search, then section reconstruction (offline, no model)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size

from hgrag.hierarchy import build_hierarchy
from hgrag.indexing import chunk_paragraphs, index_chunks, open_store
from hgrag.retrieval import reconstruct_section, retrieve_sections, search_hits

DOC_ID = "doc1"


class BagOfWordsEmbedder:
    """Deterministic embedder: texts sharing words get close vectors."""

    def encode(self, sentences, **kwargs):
        out = np.zeros((len(sentences), 64), dtype="float32")
        for i, s in enumerate(sentences):
            for w in s.lower().replace(".", " ").split():
                seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                out[i] += np.random.default_rng(seed).normal(size=64)
            n = np.linalg.norm(out[i])
            out[i] /= n or 1.0
        return out


def _prov(page: int = 1) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


@pytest.fixture()
def store(tmp_path):
    d = DoclingDocument(name="rapport")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    h1 = d.add_heading(text="1. Highlights", level=1, prov=_prov())
    d.add_text(label="text", text="Portfolio grew thanks to three new branches.", parent=h1, prov=_prov())
    lst = d.add_list_group(parent=h1)
    d.add_list_item(text="Lille", parent=lst, prov=_prov())
    d.add_list_item(text="Nantes", parent=lst, prov=_prov())
    pic = d.add_picture(parent=h1, prov=_prov(), caption=d.add_text(label="caption", text="Figure 1: Claims"))
    h2 = d.add_heading(text="2. Results", level=1, prov=_prov())
    d.add_text(label="text", text="Average settlement delay dropped sharply.", parent=h2, prov=_prov())

    figures = [{
        "self_ref": pic.self_ref, "figure_index": 1,
        "png_path": "figures\\figure-001.png",  # as written on Windows
        "description": {"text": "A bar chart of claims per quarter."},
    }]
    root = build_hierarchy(d, figures, doc_title="rapport")

    out = tmp_path / "output"
    (out / DOC_ID).mkdir(parents=True)
    (out / DOC_ID / "hierarchy.json").write_text(json.dumps(root.to_dict()))

    client = open_store(tmp_path / "qdrant")
    chunks = chunk_paragraphs(d, HierarchicalChunker(), root, DOC_ID, min_words=3)
    index_chunks(client, chunks, BagOfWordsEmbedder(), doc_id=DOC_ID, source_pdf="rapport.pdf", collection="c")
    return client, out


def test_search_groups_chunks_by_section(store):
    client, _ = store
    hits = search_hits("settlement delay", 3, embedder=BagOfWordsEmbedder(), client=client, collection="c")
    assert hits[0]["section_path"] == "rapport > 2. Results"
    assert hits[0]["hits"][0]["text"].startswith("Average settlement delay")
    assert len(hits) == len({(h["doc_id"], h["section_ref"]) for h in hits})
    assert hits[0]["score"] >= hits[-1]["score"]


def test_reconstruct_uses_all_section_elements(store):
    client, out = store
    results = retrieve_sections(
        "new branches portfolio", 2,
        embedder=BagOfWordsEmbedder(), client=client, collection="c", output_dir=out,
    )
    best = results[0]
    assert best["section_path"] == "rapport > 1. Highlights"
    text = best["text"]
    # the matching paragraph AND the rest of the section, in reading order
    assert text.startswith("# 1. Highlights")
    assert text.index("Portfolio grew") < text.index("- Lille") < text.index("[Figure] Figure 1: Claims")
    assert "- Lille\n- Nantes" in text  # consecutive list items stay together
    assert "Description (machine-generated, unverified): A bar chart of claims per quarter." in text
    assert "Average settlement delay" not in text  # other section is not mixed in
    # Windows-style path is normalised and resolved under the document folder
    assert Path(best["figures"][0]["png_path"]).as_posix().endswith("doc1/figures/figure-001.png")
    assert best["hits"][0]["score"] == pytest.approx(best["score"])


def test_include_children_and_unknown_section(store):
    _, out = store
    doc = reconstruct_section(DOC_ID, "#/root", include_children=True, output_dir=out)
    assert "1. Highlights" in doc["text"] and "2. Results" in doc["text"]
    with pytest.raises(KeyError):
        reconstruct_section(DOC_ID, "#/texts/999", output_dir=out)


def test_table_caption_is_shown_before_the_rows():
    from hgrag.retrieval import _render_element

    el = {"kind": "table", "text": "| a | b |\n|---|---|\n| 1 | 2 |", "extra": {"caption": "Table 2. Results"}}
    out = _render_element(el)
    assert out.startswith("[Table] Table 2. Results\n\n| a | b |") and "description" not in out
    assert _render_element({**el, "extra": {}}) == el["text"]
