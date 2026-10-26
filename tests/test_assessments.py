"""Production assessments of the parsing and the descriptions (no model, no API: the judge is a stub)."""

import json
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size, TableCell, TableData

from hgrag import assessments, describe
from hgrag.assessments import description as D
from hgrag.assessments import parsing as P

CONTROL_PDF = Path(__file__).resolve().parents[1] / "data" / "pdf_control" / "01_aurora_bank_annual_report.pdf"
FILLER = " ".join(f"word{i}" for i in range(30))


def _prov(page=1):
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _doc(pages=3):
    doc = DoclingDocument(name="t")
    for p in range(1, pages + 1):
        doc.add_page(page_no=p, size=Size(width=595, height=842))
    return doc


def _by_name(recs):
    return {r.name: r for r in recs}


def test_words_unfold_ligatures_and_join_line_end_hyphens():
    assert P.words("ﬁnancial opti-\r\nmization 3 x") == {"financial": 1, "optimization": 1}


def test_text_coverage_flags_the_pages_docling_lost():
    doc = _doc()
    h = doc.add_heading(text="Results", level=1, prov=_prov(1))
    doc.add_text(label="text", text=FILLER, parent=h, prov=_prov(1))
    cells = [TableCell(text=f"word{i}", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=i,
                       end_col_offset_idx=i + 1) for i in range(30, 50)]
    doc.add_table(data=TableData(num_rows=1, num_cols=20, table_cells=cells), parent=h, prov=_prov(2))
    pdf = {1: "Results " + FILLER, 2: " ".join(f"word{i}" for i in range(30, 60)), 3: "page 3"}
    rec = _by_name(P.text_coverage(doc, pdf, min_coverage=0.9, min_page_coverage=0.7))
    cov, low = rec["parse.text_coverage"], rec["parse.low_coverage_pages"]
    assert cov.value == round(51 / 61, 3) and cov.passed is False and cov.metadata["unscored_pages"] == [3]
    assert low.value == 1 and low.metadata["pages"] == [(2, 0.67)] and "2 (0.67)" in low.rationale
    assert not any(r.passed is False for r in P.text_coverage(doc, pdf, min_coverage=0.8, min_page_coverage=0.6))


def test_text_coverage_without_text_layer_is_not_scored():
    rec = _by_name(P.text_coverage(_doc(), {1: "", 2: "a few words"}))
    assert rec["parse.text_coverage"].value is None and rec["parse.text_coverage"].passed is None


def test_formulas_count_by_their_pdf_text_not_their_latex():
    doc = _doc(1)
    doc.add_text(label=DocItemLabel.FORMULA, text="E = mc^2", orig="energy equals mass", prov=_prov(1))
    assert P.docling_page_texts(doc) == {1: ["energy equals mass"]}


def test_pdf_text_layer_is_read_page_by_page():
    pages = P.pdf_page_texts(CONTROL_PDF)
    assert len(pages) >= 1 and sum(P.words(pages[1]).values()) > 20


def test_label_profile(tmp_path):
    doc = _doc(2)
    doc.add_text(label=DocItemLabel.FORMULA, text="", orig="", prov=_prov(1))
    doc.add_table(data=TableData(num_rows=0, num_cols=0, table_cells=[]), prov=_prov(1))
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "figure-001.png").write_bytes(b"png")
    figures = [{"self_ref": "#/pictures/0", "png_path": "figures\\figure-001.png"},  # Windows path of old outputs
               {"self_ref": "#/pictures/1", "png_path": "figures/figure-002.png"}, {"self_ref": "#/pictures/2"}]
    rec = _by_name(P.label_profile(doc, figures, tmp_path, max_headings_per_page=1))
    assert rec["parse.headings"].value == 0 and rec["parse.headings"].passed is False
    assert rec["parse.empty_formulas"].value == 1 and rec["parse.empty_tables"].value == 1
    assert rec["parse.figures_without_png"].metadata["items"] == ["#/pictures/1", "#/pictures/2"]
    for i in range(3):
        doc.add_heading(text=f"H{i}", level=1, prov=_prov(1))
    rec = _by_name(P.label_profile(doc, [], max_headings_per_page=1))
    assert rec["parse.headings"].passed and rec["parse.headings_per_page"].value == 1.5
    assert rec["parse.headings_per_page"].passed is False


def test_description_checks():
    long = "A bar chart of the yearly revenue by region with values between ten and forty million euros."
    figures = [{"self_ref": "#/pictures/0", "png_path": "a.png", "description": {"text": long}},
               {"self_ref": "#/pictures/1", "png_path": "b.png", "description": {"text": "I cannot see the image " + long}},
               {"self_ref": "#/pictures/2", "png_path": "c.png", "description_error": "HTTP 503"},
               {"self_ref": "#/pictures/3"}]  # no PNG: not described by design
    tables = [{"self_ref": "#/tables/0", "description": {"text": "Revenue table."}}]
    rec = _by_name(D.description_checks(figures, tables, [], min_words=10))
    fig = rec["describe.figure_descriptions"]
    assert fig.value == round(1 / 3, 3) and fig.passed is False
    assert [f["problem"] for f in fig.metadata["failed"]] == ["refusal", "missing"]
    assert rec["describe.table_descriptions"].metadata["failed"][0]["problem"] == "too short (2 words)"
    assert rec["describe.formula_readings"].value is None


@pytest.mark.parametrize("latex,original,problem", [
    (r"\frac{a}{b}", None, None),
    (r"\left( x \right) + \{ y \}", None, None),
    (r"\begin{cases} a & b \\ c & d \end{cases}", None, None),
    (r"\max_{0\leq c < m} x", "max 0≤c<m x", None),
    (r"\frac{a}{b", None, "unbalanced braces"),
    (r"\left( x", None, r"\left x1 / \right x0"),
    (r"\begin{array} a", None, r"\begin{array} not closed"),
    (r"a = b " + r"\quad" * 12, None, "command repeated in a loop (output cut)"),
    (r"\max_{0\leq c", "max 0≤c<m x", "'<' of the PDF text missing (output cut?)"),
    ("```latex x```", None, "Markdown fence or unpaired $ in the LaTeX"),
    ("", None, "no LaTeX"),
])
def test_latex_problems(latex, original, problem):
    found = D.latex_problems(latex, original)
    assert (problem in found) if problem else not found


def test_inline_math_is_prose_not_a_defect():
    formulas = [{"self_ref": "#/texts/1", "latex": r"Proof. See Appendix D. \hfill $\square$"},
                {"self_ref": "#/texts/2", "latex": r"\frac{a}{b"}]
    rec = D.latex_checks(["#/texts/1", "#/texts/2", "#/texts/3"], formulas)
    assert rec.value == round(1 / 3, 3) and rec.metadata["prose"] == ["#/texts/1"]
    assert {f["item"] for f in rec.metadata["failed"]} == {"#/texts/2", "#/texts/3"}


def test_sampling_is_deterministic():
    keys = [f"doc#/texts/{i}" for i in range(2000)]
    picked = [k for k in keys if assessments.sampled(k, 0.2)]
    assert picked == [k for k in keys if assessments.sampled(k, 0.2)] and 300 < len(picked) < 500
    assert all(assessments.sampled(k, 1.0) for k in keys) and not any(assessments.sampled(k, 0.0) for k in keys)


def _judged_dir(tmp_path):
    for name in ("figures/figure-001.png", "formulas/formula_0001.png"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"png")
    figures = [{"self_ref": "#/pictures/0", "png_path": "figures/figure-001.png", "description": {"text": "A chart."}},
               {"self_ref": "#/pictures/1", "png_path": "figures/missing.png", "description": {"text": "Gone."}}]
    formulas = [{"self_ref": "#/texts/5", "latex": "E=mc^2", "png_path": "formulas/formula_0001.png",
                 "reading": {"text": "Energy equals mass times c squared."}}]
    return figures, formulas


def test_llm_judges_on_figures_and_formulas(tmp_path, monkeypatch):
    figures, formulas = _judged_dir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    calls = []

    def fake(cfg, png, prompt, system):
        calls.append((png.name, system))
        assert cfg.thinking == "disabled"  # Haiku judges do not think
        if system == D.FORMULA_JUDGE_SYSTEM:  # the reading is judged with the text it was written from
            assert "TEXT BEFORE THE FORMULA:\nLet E be the energy." in prompt and "TEXT AFTER THE FORMULA:\n(none)" in prompt
        if system == D.FIGURE_JUDGE_SYSTEM:
            return '{"unsupported": ["a 40% value"], "missing": [], "score": 3, "rationale": "one invented value"}'
        return 'Here: {"latex_errors": [], "latex_score": 5, "reading_errors": ["wrong limit"], "reading_score": 2, "rationale": "r"}'

    monkeypatch.setattr(describe, "_ask_image", fake)
    recs = _by_name(D.judge_items(figures, formulas, tmp_path, doc_key="d", sample=1.0, max_judged=5, model="claude-haiku-5-5",
                                  contexts={"#/texts/5": ("Let E be the energy.", "")}))
    assert sorted(c[0] for c in calls) == ["figure-001.png", "formula_0001.png"]  # no PNG on disk: not judged
    fig = recs["describe.figure_fidelity"]
    assert fig.value == 3 and fig.passed is False and fig.source == "llm_judge" and fig.target == "#/pictures/0"
    assert fig.metadata["unsupported"] == ["a 40% value"] and fig.source_id == "claude-haiku-5-5"
    assert recs["describe.formula_latex"].passed and recs["describe.formula_reading"].value == 2
    assert recs["describe.formula_reading"].metadata["errors"] == ["wrong limit"]
    assert D.judge_items(figures, formulas, tmp_path, doc_key="d", sample=0.0, max_judged=5) == []


def test_llm_judge_bad_reply_and_missing_key(tmp_path, monkeypatch):
    figures, formulas = _judged_dir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    calls = []
    monkeypatch.setattr(describe, "_ask_image", lambda cfg, png, prompt, system: calls.append(prompt) or "no json")
    recs = D.judge_items(figures[:1], [], tmp_path, doc_key="d", sample=1.0, max_judged=5)
    assert len(calls) == 2 and "JSON object only" in calls[1]  # one retry
    assert recs[0].value is None and "no JSON" in recs[0].error
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    recs = D.judge_items(figures[:1], [], tmp_path, doc_key="d", sample=1.0, max_judged=5)
    assert recs[0].name == "describe.judge" and "ANTHROPIC_API_KEY" in recs[0].error


def test_assessment_as_mlflow_feedback():
    from mlflow.entities import AssessmentSourceType

    fb = assessments.Assessment("parse.text_coverage", 0.93, passed=True, rationale="ok", target="page 3",
                                metadata={"pages": [(3, 0.5)]}).to_feedback()
    assert fb.name == "parse.text_coverage" and fb.value == 0.93 and fb.rationale == "ok"
    assert fb.source.source_type == AssessmentSourceType.CODE
    assert fb.metadata == {"pages": "[[3, 0.5]]", "passed": "true", "target": "page 3"}
    judged = assessments.Assessment("describe.figure_fidelity", 4, source="llm_judge", source_id="claude-haiku-5-5")
    assert judged.to_feedback().source.source_type == AssessmentSourceType.LLM_JUDGE


def test_assess_document_writes_and_keeps_other_steps(tmp_path):
    doc = _doc(1)
    h = doc.add_heading(text="Intro", level=1, prov=_prov(1))
    doc.add_text(label="text", text=FILLER, parent=h, prov=_prov(1))
    doc.save_as_json(tmp_path / "document.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"source_pdf": str(tmp_path / "gone.pdf")}))
    (tmp_path / "assessments.json").write_text(json.dumps({"hierarchy": [{"name": "x", "value": 1}]}))
    out = assessments.assess_document(tmp_path, steps=("parse", "describe"), judge=False)
    cov = _by_name(out["parse"])["parse.text_coverage"]
    assert cov.value is None and "source PDF not found" in cov.error
    saved = json.loads((tmp_path / "assessments.json").read_text(encoding="utf-8"))
    assert set(saved) == {"hierarchy", "parse", "describe"} and saved["hierarchy"] == [{"name": "x", "value": 1}]
    assert assessments.load_assessments(tmp_path)["parse"][0].name == "parse.text_coverage"
    rows = {r["name"]: r for r in assessments.summarize(out["parse"] + out["describe"])}
    assert rows["parse.headings"]["passed"] == 1 and rows["parse.text_coverage"]["errors"] == 1


# ------------------------------------------------------------------------------------------------ step 3: hierarchy
from hgrag.assessments import chunks as C  # noqa: E402
from hgrag.assessments import hierarchy as H  # noqa: E402


def _node(title, level, children=(), elements=()):
    return {"title": title, "level": level, "page_no": 1, "children": list(children),
            "elements": [{"kind": k, "text": t} for k, t in elements]}


def test_heading_numbers():
    assert H.heading_number("2.1 Data") == ("arabic", (2, 1))
    assert H.heading_number("A.3. Proofs") == ("A", (3,))
    assert H.heading_number("IV. RESULTS") == ("roman", (4,))
    assert H.heading_number("A Simple Method") is None and H.heading_number("Results") is None
    assert H.heading_number("2024 was a year") is None


def test_numbering_flags_skips_and_steps_back():
    tree = _node("doc", 0, [
        _node("1 Introduction", 1), _node("2 Method", 1, [_node("2.1 Data", 2), _node("2.3 Model", 2)]),
        _node("3 Results", 1, [_node("3.1 Main", 2), _node("1. Crossover", 2)]), _node("Appendix", 1, [_node("A.1 Proof", 2)]),
    ])
    rec = H.numbering(tree)
    assert rec.value == 2 and rec.passed is False
    assert rec.metadata["problems"] == ["'2.3 Model' skips 2.2", "'1. Crossover' after 3"]
    ok = _node("doc", 0, [_node("1 A", 1, [_node("1.1 B", 2)]), _node("2 C", 1, [_node("2.1 D", 2)]), _node("3.1 E", 1)])
    assert H.numbering(ok).passed  # 3.1 after 2.1: heading 3 lost but the sub-numbering restarts correctly
    assert H.numbering(_node("doc", 0, [_node("Intro", 1)])).value is None


def test_references_and_acknowledgements_out_of_the_index():
    refs = [("list_item", f"[{i}] A. Author, Some title, Journal of Things, 2020.") for i in range(1, 5)]
    tree = _node("doc", 0, [
        _node("1 Introduction", 1, elements=[("paragraph", "As shown by Smith et al. (2020) in Proceedings of X, the method works."),
                                             ("paragraph", "This work was supported by the ANR under grant 123.")]),
        _node("5 Conclusion", 1, elements=[("paragraph", "We conclude."), *refs]),
        _node("References", 1, elements=refs), _node("Acknowledgments", 1, elements=[("paragraph", "We thank X.")]),
    ])
    rec = H.references_excluded(tree)
    assert rec.metadata["excluded_sections"] == ["References", "Acknowledgments"]
    assert [(x["kind"], x["section"], x["elements"]) for x in rec.metadata["leaks"]] == [
        ("acknowledgements", "1 Introduction", 1), ("bibliography", "5 Conclusion", 4)]
    assert rec.value == 2 and rec.passed is False
    assert H.reference_like("Smith, J., Doe, A.: A title. In Proceedings of ICML, 2021.")
    assert not H.reference_like("As shown by Smith et al. (2020), the method works.")


def test_outline_judge(monkeypatch):
    from hgrag import generation

    seen = {}

    def fake(cfg, system, user, **kw):
        seen.update(cfg=cfg, user=user)
        return ('{"problems": [{"heading": "Figure 2: Loss", "problem": "a caption, not a heading"}], "score": 3, '
                '"rationale": "one caption read as a heading"}\nNote: the rest is fine.'), {}

    monkeypatch.setattr(generation, "complete", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    tree = _node("doc", 0, [_node("1 Intro", 1, [_node("Figure 2: Loss", 2)], elements=[("paragraph", "x")])])
    rec = H.judge_outline(tree, model="claude-haiku-5-5")
    assert rec.value == 3 and rec.passed is False and rec.source == "llm_judge"
    assert rec.metadata["problems"][0]["heading"] == "Figure 2: Loss"
    assert "1 Intro  (p.1, 1 el.)\n  Figure 2: Loss" in seen["user"] and seen["cfg"].thinking == "disabled"


def test_judge_reply_with_malformed_json_still_gives_the_score():
    data = D.parse_reply('{"problems": [{"heading": "The "best" model"}], "score": 2, "rationale": "two \\"quoted\\" words"}',
                         ("score",))
    assert data["score"] == 2 and data["rationale"] == 'two "quoted" words'
    with pytest.raises(D.JudgeReplyError):
        D.parse_reply('{"problems": []}', ("score",))


# ------------------------------------------------------------------------------------------------- step 4: chunks
def _chunks(*texts):
    return [{"chunk_index": i, "text": t} for i, t in enumerate(texts)]


def test_chunk_size_duplicates_and_empty_texts():
    rows = _chunks("short one", "x " * 40, "y " * 30, "MCGILL  University", "McGill University", "mcgill university", "\x00\x00", " . ")
    rec = {r.name: r for r in C.assess(rows, "d")}
    size = rec["chunks.size_distribution"]
    assert size.value == round(6 / 8, 3) and size.passed is False and size.metadata["max"] == 40
    assert C.size_distribution(rows, max_small_share=0.8).passed
    dup = C.duplicates(rows, max_duplicates=1)
    assert dup.value == 2 and dup.passed is False and dup.metadata["texts"][0] == {"text": "mcgill university", "count": 3}
    assert C.duplicates(rows, max_duplicates=2).passed
    assert rec["chunks.empty_texts"].value == 2 and rec["chunks.empty_texts"].metadata["chunks"] == [6, 7]
    assert rec["chunks.points_indexed"].error == "no Qdrant client given"


def test_every_chunk_is_in_qdrant(tmp_path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(tmp_path / "q"))
    try:
        client.create_collection("c", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        client.upsert("c", [PointStruct(id=i, vector=[1.0, 0.5], payload={"doc_id": "d" if i < 3 else "other"}) for i in range(5)])
        rec = {r.name: r for r in C.assess(_chunks("a", "b", "c"), "d", client=client, collection="c")}
        assert rec["chunks.points_indexed"].value == 3 and rec["chunks.points_indexed"].passed
        rec = {r.name: r for r in C.assess(_chunks("a", "b", "c", "d"), "d", client=client, collection="c")}
        assert rec["chunks.points_indexed"].passed is False and "3 point(s) in Qdrant for 4 chunk(s)" in rec["chunks.points_indexed"].rationale
        assert {r.name: r for r in C.assess(_chunks("a"), "d", client=client, collection="none")}["chunks.points_indexed"].value == 0
    finally:
        client.close()


def test_assess_document_runs_the_four_steps(tmp_path):
    doc = _doc(1)
    h = doc.add_heading(text="1 Intro", level=1, prov=_prov(1))
    doc.add_text(label="text", text=FILLER, parent=h, prov=_prov(1))
    doc.save_as_json(tmp_path / "document.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"doc_id": "d", "source_pdf": "gone.pdf"}))
    (tmp_path / "hierarchy.json").write_text(json.dumps(_node("doc", 0, [_node("1 Intro", 1, elements=[("paragraph", FILLER)])])))
    (tmp_path / "chunks.jsonl").write_text(json.dumps({"chunk_index": 0, "text": FILLER}) + "\n")
    out = assessments.assess_document(tmp_path)
    assert set(out) == {"parse", "describe", "hierarchy", "chunks"}
    assert {r.name for r in out["hierarchy"]} == {"hierarchy.numbering", "hierarchy.references_excluded"}  # no judge
    assert _by_name(out["chunks"])["chunks.size_distribution"].passed


# ---------------------------------------------------------------------------------------- step 5: search, reranking
from hgrag.assessments import prompt as PR  # noqa: E402
from hgrag.assessments import retrieval as RT  # noqa: E402
from test_generation import with_png  # noqa: E402, F401


def test_search_diagnostics_hybrid_and_reranked(tmp_path):
    from qdrant_client import QdrantClient

    from hgrag.indexing import index_chunks
    from hgrag.retrieval import search_hits
    from test_keywords import ConstantEmbedder, _chunks

    client = QdrantClient(path=str(tmp_path / "q"))
    try:
        index_chunks(client, _chunks(["step size tuning", "step size theory", "Laplace and Cauchy targets"]),
                     ConstantEmbedder(), doc_id="d", collection="c")
        diag: dict = {}
        search_hits("Laplace targets", 3, embedder=ConstantEmbedder(), client=client, collection="c", keyword_weight=0.3,
                    diagnostics=diag)
        # the dense side returns the 3 chunks, the keyword side only the Laplace one: 1 of 3 candidates found by both
        assert diag["n_candidates"] == 3 and diag["n_both"] == 1 and diag["agreement"] == 1 / 3
        assert diag["top_score_kind"] == "dense" and abs(diag["top_score"] - 1.0) < 1e-5

        class Reranker:
            def predict(self, pairs, **kw):
                return [0.9 if "Laplace" in text else 0.01 for _q, text in pairs]

        diag = {}
        search_hits("Laplace targets", 2, embedder=ConstantEmbedder(), client=client, collection="c", keyword_weight=0.3,
                    rerank=True, reranker=Reranker(), candidates=3, diagnostics=diag)
        assert diag["top_score_kind"] == "rerank" and diag["top_score"] == 0.9
        diag = {}
        search_hits("Laplace", 2, embedder=ConstantEmbedder(), client=client, collection="c", keyword_weight=0.0, diagnostics=diag)
        assert "agreement" not in diag and diag["top_score_kind"] == "dense"
    finally:
        client.close()


def test_top_score_and_agreement_checks():
    off = RT.top_score({"top_score": 0.0004, "top_score_kind": "rerank"}, min_score=0.1)
    assert off.passed is False and "does not hold the answer" in off.rationale
    assert RT.top_score({"top_score": 0.97, "top_score_kind": "rerank"}, min_score=0.1).passed
    dense = RT.top_score({"top_score": 0.42, "top_score_kind": "dense"})
    assert dense.passed is None and dense.value == 0.42
    assert RT.top_score({}).passed is False
    diag = {"agreement": 0.05, "n_candidates": 20, "n_both": 1, "n_dense_only": 10, "n_keyword_only": 9}
    low = RT.search_agreement(diag, min_agreement=0.1)
    assert low.value == 0.05 and low.passed is False and "1/20 candidates" in low.rationale
    assert RT.search_agreement({}).value is None and RT.search_agreement({}).passed is None
    assert [a.name for a in RT.assess(diag)] == ["retrieval.top_score", "retrieval.search_agreement"]


def test_context_judge(monkeypatch, tmp_path):
    from hgrag import generation
    from test_generation import SECTIONS

    seen = {}

    def fake(cfg, system, user, **kw):
        seen.update(cfg=cfg, system=system, user=user)
        return ('{"relevance": {"1": 5, "2": 2}, "missing": "the 2023 figure", "sufficiency": 3, '
                '"rationale": "source 1 gives part of the answer"}'), {}

    monkeypatch.setattr(generation, "complete", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    msgs = generation.build_messages("How many branches?", SECTIONS)
    rel, suff = RT.judge_context("How many branches?", msgs, model="claude-haiku-5-5")
    assert rel.name == "retrieval.context_relevance" and rel.value == 0.5 and rel.passed and rel.source == "llm_judge"
    assert [s["relevance"] for s in rel.metadata["sources"]] == [5, 2]
    assert suff.value == 3 and suff.passed and suff.metadata["missing"] == "the 2023 figure"  # fails at 2 or less
    assert seen["system"] == RT.CONTEXT_JUDGE_SYSTEM and seen["user"].endswith("Question: How many branches?")
    assert "[1] aurora.pdf" in seen["user"] and seen["cfg"].thinking == "disabled"
    empty = RT.judge_context("q", generation.build_messages("q", []))
    assert empty[1].value == 1 and empty[1].passed is False
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert all("ANTHROPIC_API_KEY" in a.error for a in RT.judge_context("q", msgs))


def test_context_judge_sees_the_figures(monkeypatch, tmp_path):
    from PIL import Image

    from hgrag import generation
    from test_generation import SECTIONS

    png = tmp_path / "f.png"
    Image.new("RGB", (40, 30), "red").save(png)
    sections = [dict(SECTIONS[0], figures=[{"ref": "#/p/0", "caption": "Figure 1", "png_path": str(png)}])]
    seen = {}
    monkeypatch.setattr(generation, "complete", lambda cfg, system, user, **kw: seen.update(user=user) or (
        '{"relevance": {"1": 4}, "sufficiency": 5, "rationale": "ok"}', {}))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    RT.judge_context("q", generation.build_messages("q", sections))
    assert [p["type"] for p in seen["user"]] == ["text", "image", "text"]


# ---------------------------------------------------------------------------------------------- step 6: prompt
def test_prompt_checks(with_png):
    from hgrag import generation
    from test_generation import SECTIONS

    sections, _big = with_png
    msgs = generation.build_messages("q", sections, max_chars=100000, max_images=1)
    rec = {a.name: a for a in PR.assess(msgs, max_chars=100000, max_images=1)}
    assert rec["prompt.dropped_sections"].value == 0 and rec["prompt.dropped_sections"].passed
    assert rec["prompt.budget_use"].value == round(len(msgs["sources_text"]) / 100000, 3) and rec["prompt.budget_use"].passed
    img = rec["prompt.images"]
    assert img.value == 1 and img.passed is False
    assert img.metadata == {"figures": 3, "on_disk": 2, "missing": ["gone"], "over_cap": 1}
    tight = generation.build_messages("q", SECTIONS, max_chars=len(SECTIONS[0]["text"]) + 80)
    rec = {a.name: a for a in PR.assess(tight, max_chars=len(SECTIONS[0]["text"]) + 80, max_images=6)}
    assert rec["prompt.dropped_sections"].value == 1 and rec["prompt.dropped_sections"].passed is False
    assert rec["prompt.images"].passed and rec["prompt.images"].value == 0
    over = generation.build_messages("q", SECTIONS, max_chars=5)
    assert {a.name: a for a in PR.assess(over, max_chars=5, max_images=6)}["prompt.budget_use"].passed is False
    off = {a.name: a for a in PR.assess(msgs, max_chars=100000, max_images=1, send_images=False)}["prompt.images"]
    assert off.passed is None and off.value == 0 and "described in text only" in off.rationale


def test_answer_stream_reports_the_prompt_checks(monkeypatch):
    from hgrag import generation
    from test_generation import SECTIONS, FakeResponse, _anthropic_stream

    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: FakeResponse(_anthropic_stream(["ok"])))
    info: dict = {}
    "".join(generation.answer_stream("q", SECTIONS, generation.LLMConfig(provider="anthropic", api_key="k"), info=info))
    assert [c["name"] for c in info["prompt_checks"]] == ["prompt.dropped_sections", "prompt.budget_use", "prompt.images"]
