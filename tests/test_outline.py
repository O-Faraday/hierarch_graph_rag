"""PDF bookmarks as headings: headings run into their paragraph ("2.2. Title. Text...") are split from it."""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size

from hgrag import config, outline
from hgrag.hierarchy import build_hierarchy, section_index
from hgrag.indexing import chunk_paragraphs
from test_hybrid import WhitespaceTokenizer

ROOT = Path(__file__).resolve().parents[1]
FILLER = " ".join(["word"] * 30)


def _prov(page: int = 1) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _run_in_doc() -> DoclingDocument:
    """The style of the suBART paper: the headings open their paragraph; one long Docling heading; a false heading."""
    d = DoclingDocument(name="x")
    for p in (1, 2):
        d.add_page(page_no=p, size=Size(width=600, height=800))
    d.add_heading(text="A paper title", level=1, prov=_prov())
    d.add_text(label="text", text=f"1. Introduction. Many research questions {FILLER}", prov=_prov())
    d.add_text(label="text", text="2. CEA and the model. Our main motivation is the cost.", prov=_prov())
    d.add_text(label="text", text=f"2.1. Transmural Care Model (TTCM). Our data of interest {FILLER}", prov=_prov(2))
    d.add_heading(text=f"Posterior inference. This section describes the sampler {FILLER}", level=1, prov=_prov(2))
    d.add_heading(text="Friedman #1:", level=1, prov=_prov(2))
    d.add_text(label="text", text="1.5 kg of sand. Not a heading, a measure out of the outline.", prov=_prov(2))
    return d


OUTLINE = [
    {"title": "Introduction", "level": 1, "page_no": 1},
    {"title": "CEA and the model", "level": 1, "page_no": 1},
    {"title": "Transmural Care Model (TTCM)", "level": 2, "page_no": 2},
    {"title": "Posterior inference", "level": 1, "page_no": 2},
]


def _titles(root):
    return [(n.title, n.level) for n in root.walk() if n.level]


def test_split_heading_keeps_the_number_and_the_rest():
    assert outline.split_heading("2.2. Transmural Care Model (TTCM). Our data", "Transmural Care Model (TTCM)",
                                 numbered_only=True) == ("2.2. Transmural Care Model (TTCM)", "Our data")
    # the bookmark keeps the number; Docling glues words ("inCEA")
    assert outline.split_heading("1 Executive summary", "1 Executive summary", numbered_only=True) == (
        "1 Executive summary", "")
    assert outline.split_heading("6.1. Experiment to assess performance inCEA. In this", "assess performance in CEA",
                                 numbered_only=True) is None  # not the start of the title
    assert outline.split_heading("Appendix A: The phase problem", "The phase problem", numbered_only=False) == (
        "Appendix A: The phase problem", "")


def test_split_heading_refuses_a_sentence_that_only_starts_like_a_title():
    assert outline.split_heading("Introduction of the method is slow.", "Introduction", numbered_only=False) is None
    # a paragraph needs a number before the title, unless it is the title alone
    assert outline.split_heading("Introduction. We study", "Introduction", numbered_only=True) is None
    assert outline.split_heading("Introduction", "Introduction", numbered_only=True) == ("Introduction", "")


def test_section_outline_drops_the_bookmark_of_the_document_title():
    marks = [{"title": "SPRING", "level": 1, "page_no": 1}, {"title": "Introduction", "level": 2, "page_no": 1},
             {"title": "Monitoring", "level": 3, "page_no": 6}]
    assert [(b["title"], b["level"]) for b in outline.section_outline(marks)] == [("Introduction", 1), ("Monitoring", 2)]
    assert outline.section_outline(OUTLINE) == OUTLINE


def test_run_in_headings_become_sections_at_the_level_of_their_bookmark():
    doc = _run_in_doc()
    refs_before = [t.self_ref for t in doc.texts]
    split, confirmed = outline.with_outline_headings(doc, OUTLINE)
    assert [t.self_ref for t in split.texts][:len(refs_before)] == refs_before  # no ref moves: new items appended
    assert doc.texts[1].text.startswith("1. Introduction.")  # the original is untouched
    root = build_hierarchy(split, [], doc_title="x", outline_headings=confirmed)
    assert _titles(root) == [("A paper title", 1), ("1. Introduction", 1), ("2. CEA and the model", 1),
                             ("2.1. Transmural Care Model (TTCM)", 2), ("Posterior inference", 1)]
    index = section_index(root)
    assert index[doc.texts[3].self_ref].title == "2.1. Transmural Care Model (TTCM)"
    assert split.texts[3].text.startswith("Our data of interest")  # the heading is no longer in the paragraph
    posterior = next(n for n in root.walk() if n.title == "Posterior inference")
    assert [e.text.split()[0] for e in posterior.elements] == ["This", "Friedman", "1.5"]  # false heading demoted


def test_chunks_carry_the_headings_found_in_the_bookmarks():
    split, confirmed = outline.with_outline_headings(_run_in_doc(), OUTLINE)
    root = build_hierarchy(split, [], doc_title="x", outline_headings=confirmed)
    chunks = chunk_paragraphs(split, HybridChunker(tokenizer=WhitespaceTokenizer()), root, "d1")
    ttcm = next(c for c in chunks if "Our data of interest" in c["text"])
    assert ttcm["section_title"] == "2.1. Transmural Care Model (TTCM)"
    assert "2.1. Transmural Care Model (TTCM)" in ttcm["headings"]
    assert not ttcm["text"].startswith("2.1.")


def test_run_in_headings_read_as_a_numbered_list_leave_their_group():
    """Docling read "1. Introduction. Text..." as a numbered list: the whole list was one chunk, dropped."""
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    d.add_heading(text="An article", level=1, prov=_prov())
    group = d.add_list_group()
    d.add_heading(text=f"Introduction. The basin supplies water {FILLER}", level=1, parent=group, prov=_prov())
    d.add_heading(text=f"Results. The mean recharge was 66 millimetres {FILLER}", level=1, parent=group, prov=_prov())
    marks = [{"title": "Introduction", "level": 1, "page_no": 1}, {"title": "Results", "level": 1, "page_no": 1}]
    split, confirmed = outline.with_outline_headings(d, marks)
    assert all(split.texts[i].parent.cref == "#/body" for i in range(1, 3))
    root = build_hierarchy(split, [], doc_title="x", outline_headings=confirmed)
    chunks = chunk_paragraphs(split, HybridChunker(tokenizer=WhitespaceTokenizer(), merge_peers=False), root, "d1")
    by_section = {c["section_title"]: c["text"] for c in chunks}
    assert by_section["Introduction"].startswith("The basin supplies water")
    assert by_section["Results"].startswith("The mean recharge was 66 millimetres")


def test_without_bookmarks_or_when_off_the_document_is_unchanged(monkeypatch):
    doc = _run_in_doc()
    assert outline.with_outline_headings(doc, []) == (doc, {})
    monkeypatch.setattr(config, "OUTLINE_HEADINGS", False)
    assert outline.with_outline_headings(doc, OUTLINE) == (doc, {})


def test_bookmarks_are_read_from_the_pdf_and_cached(tmp_path):
    pdf = ROOT / "data" / "pdf_control" / "01_aurora_bank_annual_report.pdf"
    marks = outline.read_outline(pdf)
    assert marks[0] == {"title": "1 Executive summary", "level": 1, "page_no": 1}
    assert {"title": "2.1 Revenue", "level": 2, "page_no": 1} in marks
    # a document parsed before outline.json existed: read from its source PDF, then saved
    assert outline.load_outline(tmp_path, str(pdf).replace("/", "\\")) == marks
    assert json.loads((tmp_path / outline.OUTLINE_FILE).read_text(encoding="utf-8")) == marks
    assert outline.load_outline(tmp_path / "missing", None) == []


def test_headings_inside_one_text_region_are_split_when_numbered():
    """The layout model read the whole body as one paragraph (Docling on Databricks, 01_talvera)."""
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=600, height=800))
    d.add_heading(text="A paper title", level=1, prov=_prov())
    d.add_text(label="text", prov=_prov(), text=(
        "1. Introduction. The basin supplies water (see Results. below) to many. 2. Study area and data. It covers "
        "412 km2. 2.1. Monitoring wells. Levels were recorded. Remark 1. A value. 3. Results. The recharge was 66 mm."))
    bookmarks = [{"title": t, "level": lvl, "page_no": 1} for t, lvl in (
        ("A paper title", 1), ("Introduction", 2), ("Study area and data", 2), ("Monitoring wells", 3), ("Results", 2),
        ("Discussion", 2))]
    doc, confirmed = outline.with_outline_headings(d, bookmarks)
    items = [(type(i).__name__, i.text) for i, _ in doc.iterate_items()]
    assert items[1:] == [
        ("SectionHeaderItem", "1. Introduction"), ("TextItem", "The basin supplies water (see Results. below) to many."),
        ("SectionHeaderItem", "2. Study area and data"), ("TextItem", "It covers 412 km2."),
        ("SectionHeaderItem", "2.1. Monitoring wells"), ("TextItem", "Levels were recorded. Remark 1. A value."),
        ("SectionHeaderItem", "3. Results"), ("TextItem", "The recharge was 66 mm.")]
    assert sorted(confirmed.values()) == [1, 1, 1, 2]  # the title bookmark is dropped, the levels follow
    assert outline.inner_heading("We discuss it. Results. were good.", "Results") is None  # no number: a sentence
