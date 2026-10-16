"""Figure descriptions: API call, rewrite of figures.json, searchable figure chunk, graceful failures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hgrag import config, describe, evaluation, generation, vlm_bench
from hgrag.generation import GenerationError
from hgrag.hierarchy import Element, SectionNode
from hgrag.indexing import figure_chunks
from test_eval import offline  # noqa: F401
from test_generation import FakeResponse
from test_vlm_bench import _openai_stream

CONTROL = config.CONTROL_PDF_DIR
DESC = "Bar chart of branches per year, 2021 about 212, decreasing to 201 in 2024."


@pytest.fixture()
def parsed(tmp_path):
    """A writable copy of the frozen parsing of the aurora report."""
    doc_id = vlm_bench.load_truth()[0]["doc_id"]
    dst = tmp_path / "out" / doc_id
    shutil.copytree(CONTROL / "parsed" / doc_id, dst)
    return dst


def _fake_hf(monkeypatch, text=DESC, status=200):
    calls = []

    def post(url, **kw):
        calls.append((url, kw["json"]["model"], kw["headers"].get("authorization")))
        if status != 200:
            return FakeResponse([], status=status, body={"error": {"message": "boom"}})
        return FakeResponse(_openai_stream(text))

    monkeypatch.setattr(generation.requests, "post", post)
    return calls


def test_describe_document_writes_figures_and_manifest(parsed, monkeypatch):
    calls = _fake_hf(monkeypatch, "<think>x</think>" + DESC)
    manifest = describe.describe_document(parsed, api_key="hf_secret")
    fig = json.loads((parsed / "figures.json").read_text(encoding="utf-8"))[0]
    assert fig["description"] == {"text": DESC, "created_by": config.FIGURE_VLM, "confidence": None}
    assert manifest["n_figures_described"] == 1 and manifest["figure_vlm"]["n_errors"] == 0
    model = config.FIGURE_VLM.split(":", 1)[1]
    assert sorted(c[1] for c in calls) == [model, config.TABLE_MODEL.split(":", 1)[1]]  # one figure, one table
    assert all(c[0] == f"{vlm_bench.HF_URL}/chat/completions" and c[2] == "Bearer hf_secret" for c in calls)
    # already described: nothing is sent again, unless forced
    describe.describe_document(parsed, api_key="hf_secret")
    assert len(calls) == 2
    describe.describe_document(parsed, api_key="hf_secret", force=True)
    assert len(calls) == 4


def test_failures_are_recorded_not_raised(parsed, monkeypatch):
    _fake_hf(monkeypatch, status=500)
    m = describe.describe_document(parsed, api_key="hf_secret")
    fig = json.loads((parsed / "figures.json").read_text(encoding="utf-8"))[0]
    assert "description" not in fig or not fig["description"]
    assert "boom" in fig["description_error"] and m["figure_vlm"]["n_errors"] == 1 and m["n_figures_described"] == 0
    assert m["table_llm"]["n_errors"] == 1 and m["n_tables_described"] == 0


def test_missing_token_is_one_clear_error(parsed, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="HF_TOKEN"):
        describe.describe_document(parsed)


def test_figure_chunk_is_built_from_caption_and_description():
    root = SectionNode(ref="#root", title="Report", level=0, path="Report")
    sec = SectionNode(ref="#/texts/1", title="3.1 New branches", level=2, path="Report > 3 Branch network > 3.1 New branches")
    sec.elements = [Element("#/pictures/0", "figure", "picture", 4, DESC, {"caption": "Figure 1. Branches", "description": DESC}),
                    Element("#/pictures/1", "figure", "picture", 4, None, {"caption": "No description", "description": None})]
    root.children.append(sec)
    chunks = figure_chunks(root, "docid", start_index=10)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_index"] == 10 and c["kinds"] == ["figure"] and c["element_refs"] == ["#/pictures/0"]
    assert c["text"] == f"Figure 1. Branches\n{DESC}" and c["page_no"] == 4 and c["section_path"] == sec.path
    assert c["embed_text"].startswith("3 Branch network\n3.1 New branches\n")  # same context as the paragraph chunks


def test_described_figure_becomes_searchable(parsed, tmp_path, monkeypatch, offline):  # noqa: F811
    """End to end on the frozen fixtures: describe, build the hierarchy, index, then find the figure by its content."""
    from hgrag import pipeline
    from hgrag.retrieval import search_hits
    from test_retrieval import BagOfWordsEmbedder

    _fake_hf(monkeypatch)
    describe.describe_document(parsed, api_key="hf_secret")
    ctrl = tmp_path / "ctrl"
    shutil.copytree(CONTROL, ctrl, ignore=shutil.ignore_patterns("parsed"))
    shutil.copytree(parsed.parent, ctrl / "parsed")  # only the aurora document is present
    for name in ("02_alpine_botany_field_guide.pdf", "03_network_operations_handbook.pdf"):
        d = ctrl / "parsed" / evaluation.compute_doc_id(ctrl / name)
        shutil.copytree(CONTROL / "parsed" / d.name, d)
    store = evaluation.build_store(ctrl, workdir=tmp_path / "store")
    aurora = next(v for k, v in store["index"]["documents"].items() if k.startswith("01_"))
    assert aurora["n_figure_chunks"] == 1
    from hgrag.indexing import open_store

    client = open_store(store["qdrant_path"])
    try:
        hits = search_hits("bar chart of branches per year decreasing", 3, embedder=BagOfWordsEmbedder(), client=client,
                           collection=store["collection"])
    finally:
        client.close()
    top = hits[0]["hits"][0]
    assert DESC in top["text"] and top["kinds"] == ["figure"], top


def test_table_description_is_added_to_the_embedding_text_only(parsed, monkeypatch, offline):  # noqa: F811
    from hgrag import pipeline
    from hgrag.indexing import chunk_paragraphs
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_core.types.doc import DoclingDocument
    from hgrag.hierarchy import build_hierarchy

    _fake_hf(monkeypatch, "Table of revenue lines in M EUR for 2024 and 2025: net interest income 720 then 781.")
    describe.describe_document(parsed, api_key="hf_secret", figures=False)
    recs = json.loads((parsed / "tables.json").read_text(encoding="utf-8"))
    assert recs[0]["description"]["text"].startswith("Table of revenue") and "Net interest income" in recs[0]["markdown"]
    tdesc = describe.load_table_descriptions(parsed)
    doc = DoclingDocument.load_from_json(parsed / "document.json")
    root = build_hierarchy(doc, doc_title="x", table_descriptions=tdesc)
    chunks = chunk_paragraphs(doc, HierarchicalChunker(), root, "d", table_descriptions=tdesc)
    tab = [c for c in chunks if "table" in c["kinds"]]
    assert tab
    assert all("Table of revenue" in c["embed_text"] and "Table of revenue" not in c["text"] for c in tab)  # text stays raw
    assert all(c["table_description"].startswith("Table of revenue") for c in tab)
    assert any(e.extra.get("description") for n in root.walk() for e in n.elements if e.kind == "table")
    # unchanged table = no new call; a changed table (other hash) would be described again
    calls = _fake_hf(monkeypatch, "other")
    describe.describe_document(parsed, api_key="hf_secret", figures=False)
    assert calls == []


def test_described_tables_are_indexed_with_their_description(parsed, tmp_path, monkeypatch, offline):  # noqa: F811
    from hgrag.indexing import open_store
    from hgrag.retrieval import search_hits
    from test_retrieval import BagOfWordsEmbedder

    _fake_hf(monkeypatch, "Quarterly headcount summary with seven hundred twenty staff.")
    describe.describe_document(parsed, api_key="hf_secret", figures=False)
    ctrl = tmp_path / "ctrl"
    shutil.copytree(CONTROL, ctrl, ignore=shutil.ignore_patterns("parsed"))
    shutil.copytree(CONTROL / "parsed", ctrl / "parsed", dirs_exist_ok=False)
    shutil.rmtree(ctrl / "parsed" / parsed.name)
    shutil.copytree(parsed, ctrl / "parsed" / parsed.name)
    store = evaluation.build_store(ctrl, workdir=tmp_path / "store")
    aurora = next(v for k, v in store["index"]["documents"].items() if k.startswith("01_"))
    assert aurora["n_table_chunks_described"] >= 1
    client = open_store(store["qdrant_path"])
    try:
        hits = search_hits("headcount staff summary", 3, embedder=BagOfWordsEmbedder(), client=client,
                           collection=store["collection"])
    finally:
        client.close()
    top = hits[0]["hits"][0]
    assert "table" in top["kinds"] and "headcount" not in top["text"], top  # found through its description


def test_reconstructed_section_shows_figure_and_table_descriptions(parsed, monkeypatch):
    from docling_core.types.doc import DoclingDocument

    from hgrag.hierarchy import build_hierarchy
    from hgrag.retrieval import _render_section

    _fake_hf(monkeypatch, "Machine description text.")
    describe.describe_document(parsed, api_key="hf_secret")
    doc = DoclingDocument.load_from_json(parsed / "document.json")
    figs = json.loads((parsed / "figures.json").read_text(encoding="utf-8"))
    root = build_hierarchy(doc, figs, doc_title="x", table_descriptions=describe.load_table_descriptions(parsed))
    blocks, _, _ = _render_section(root.to_dict(), include_children=True)
    text = "\n\n".join(blocks)
    assert "Description (machine-generated, unverified): Machine description text." in text  # figure
    assert "| Net interest income" in text and "Table description (machine-generated, unverified): Machine" in text


def test_formulas_are_transcribed_by_the_vision_model(tmp_path, monkeypatch):
    """The crop of each formula goes to the VLM; the LaTeX replaces the item text and is saved (document.json, formulas.json)."""
    import shutil

    from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ProvenanceItem, Size

    from hgrag import describe

    pdf = Path(__file__).resolve().parents[1] / "data" / "pdf_control" / "01_aurora_bank_annual_report.pdf"
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=595, height=842))
    box = BoundingBox(l=50, t=700, r=300, b=650, coord_origin=CoordOrigin.BOTTOMLEFT)
    doc.add_text(label=DocItemLabel.FORMULA, text="garbled x", orig="garbled x",
                 prov=ProvenanceItem(page_no=1, bbox=box, charspan=(0, 9)))
    d = tmp_path / "doc"
    d.mkdir()
    shutil.copyfile(pdf, tmp_path / "a.pdf")
    doc.save_as_json(d / "document.json")
    (d / "manifest.json").write_text(json.dumps({"source_pdf": str(tmp_path / "a.pdf")}))
    (d / "figures.json").write_text("[]")
    calls = []
    monkeypatch.setattr(describe, "_ask_image", lambda cfg, png, prompt, system: calls.append(png) or "```latex\n$$E = mc^2$$\n```")
    monkeypatch.setenv("HF_TOKEN", "x")
    manifest = describe.describe_document(d, figures=False, tables=False, formulas=True, api_key="x")
    assert manifest["n_formulas_transcribed"] == 1 and len(calls) == 1 and calls[0].exists()
    saved = DoclingDocument.load_from_json(d / "document.json")
    assert [t.text for t in saved.texts] == ["E = mc^2"]
    rec = json.loads((d / "formulas.json").read_text())[0]
    assert rec["latex"] == "E = mc^2" and rec["original_text"] == "garbled x"
    describe.describe_document(d, figures=False, tables=False, formulas=True, api_key="x")
    assert len(calls) == 1  # already transcribed: no second call
