"""Step 1 - Docling parsing: text coverage against the PDF text layer, and the label profile."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .. import config
from . import Assessment

_WORD = re.compile(r"[^\W_]{2,}")
_HYPHEN = re.compile(r"(?<=[^\W\d_])[-­\u0002]\s+(?=[^\W\d_])")  # a word cut at the end of a line
MIN_PAGE_WORDS = 20  # fewer words in the text layer: scanned or near-empty page, not scored


def words(text: str) -> Counter[str]:
    """Bag of lowercase words of 2+ letters or digits, ligatures unfolded (NFKC) and line-end hyphens joined."""
    text = _HYPHEN.sub("", unicodedata.normalize("NFKC", text or ""))
    return Counter(w.lower() for w in _WORD.findall(text))


def pdf_page_texts(pdf_path: str | Path) -> dict[int, str]:
    """``page number (1-based) -> text`` of the PDF text layer (pdfium, the backend Docling also uses)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    out: dict[int, str] = {}
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            out[i + 1] = textpage.get_text_range()
            textpage.close()
            page.close()
    finally:
        pdf.close()
    return out


def docling_page_texts(doc: Any) -> dict[int, list[str]]:
    """``page number -> texts`` Docling extracted on it: every text item (furniture included, formulas by their
    original PDF text, not the LaTeX that replaced it) and the table cells."""
    pages: dict[int, list[str]] = {}
    for item in doc.texts:
        text = getattr(item, "orig", None) or item.text or ""
        provs = item.prov or []
        if len(provs) == 1:
            pages.setdefault(provs[0].page_no, []).append(text)
        elif provs:  # an item over several pages: its charspans cut the text
            parts = [(p.page_no, text[p.charspan[0]:p.charspan[1]]) for p in provs]
            if sum(len(t) for _, t in parts) < len(text) / 2:
                parts = [(provs[0].page_no, text)]
            for page_no, part in parts:
                pages.setdefault(page_no, []).append(part)
    for table in doc.tables:
        if table.prov:
            pages.setdefault(table.prov[0].page_no, []).extend(c.text for c in table.data.table_cells)
    return pages


def text_coverage(doc: Any, pdf_pages: dict[int, str], *, min_coverage: float | None = None,
                  min_page_coverage: float | None = None) -> list[Assessment]:
    """Share of the words of the PDF text layer that Docling extracted, over the document and page by page.
    Catches the pages or blocks Docling lost (a page without items, a column, a table read as a picture)."""
    min_coverage = config.ASSESS_MIN_COVERAGE if min_coverage is None else min_coverage
    min_page_coverage = config.ASSESS_MIN_PAGE_COVERAGE if min_page_coverage is None else min_page_coverage
    extracted = {p: words("\n".join(t)) for p, t in docling_page_texts(doc).items()}
    scored, unscored = [], []
    for page_no in sorted(pdf_pages):
        layer = words(pdf_pages[page_no])
        total = sum(layer.values())
        if total < MIN_PAGE_WORDS:
            unscored.append(page_no)
            continue
        got = extracted.get(page_no, Counter())
        scored.append((page_no, total, sum(min(n, got[w]) for w, n in layer.items())))
    total = sum(t for _, t, _ in scored)
    if not total:
        why = "no text layer (scanned PDF?): coverage not computable"
        return [Assessment("parse.text_coverage", None, rationale=why, metadata={"unscored_pages": unscored}),
                Assessment("parse.low_coverage_pages", None, rationale=why)]
    coverage = sum(f for _, _, f in scored) / total
    low = [(p, round(f / t, 2)) for p, t, f in scored if f / t < min_page_coverage]
    pages = ", ".join(f"{p} ({c})" for p, c in low) or "none"
    meta = {"words_pdf": total, "words_found": sum(f for _, _, f in scored), "unscored_pages": unscored}
    return [
        Assessment("parse.text_coverage", round(coverage, 3), passed=coverage >= min_coverage, metadata=meta,
                   rationale=f"{meta['words_found']}/{total} words of the PDF text layer found by Docling "
                             f"(threshold {min_coverage}); {len(unscored)} page(s) with no text layer not scored"),
        Assessment("parse.low_coverage_pages", len(low), passed=not low, metadata={"pages": low},
                   rationale=f"pages below {min_page_coverage}: {pages}"),
    ]


def label_profile(doc: Any, figures: list[dict[str, Any]], doc_dir: str | Path | None = None, *,
                  max_headings_per_page: float | None = None) -> list[Assessment]:
    """Counts of the Docling labels that betray a parsing problem: no heading (no tree), too many headings per page
    (sentences or captions read as headings), empty formulas or tables, figures without their PNG."""
    from ..describe import png_file

    max_hpp = config.ASSESS_MAX_HEADINGS_PER_PAGE if max_headings_per_page is None else max_headings_per_page
    labels = Counter(getattr(t.label, "value", str(t.label)) for t in doc.texts)
    n_pages = max(1, len(doc.pages))
    headings = labels.get("section_header", 0)
    empty_formulas = [t.self_ref for t in doc.texts
                      if getattr(t.label, "value", "") == "formula" and not ((getattr(t, "orig", "") or t.text or "").strip())]
    empty_tables = [t.self_ref for t in doc.tables
                    if not t.data.num_rows or not t.data.num_cols or not any(c.text.strip() for c in t.data.table_cells)]
    no_png = [f.get("self_ref") or f"figure {f.get('figure_index')}" for f in figures
              if not f.get("png_path") or (doc_dir is not None and not png_file(Path(doc_dir), f["png_path"]).exists())]
    hpp = headings / n_pages
    profile = {"pages": len(doc.pages), "tables": len(doc.tables), "pictures": len(doc.pictures), **dict(labels)}
    return [
        Assessment("parse.headings", headings, passed=headings > 0, metadata={"labels": profile},
                   rationale=f"{headings} section header(s) over {len(doc.pages)} page(s)"
                             + ("" if headings else ": no section tree can be built")),
        Assessment("parse.headings_per_page", round(hpp, 2), passed=hpp <= max_hpp,
                   rationale=f"{hpp:.2f} section headers per page (threshold {max_hpp})"),
        Assessment("parse.empty_formulas", len(empty_formulas), passed=not empty_formulas,
                   metadata={"items": empty_formulas},
                   rationale=f"{len(empty_formulas)} formula(s) without text out of {labels.get('formula', 0)}"),
        Assessment("parse.empty_tables", len(empty_tables), passed=not empty_tables, metadata={"items": empty_tables},
                   rationale=f"{len(empty_tables)} table(s) without cells or text out of {len(doc.tables)}"),
        Assessment("parse.figures_without_png", len(no_png), passed=not no_png, metadata={"items": no_png},
                   rationale=f"{len(no_png)} figure(s) without their PNG out of {len(figures)}"),
    ]


def assess(doc: Any, figures: list[dict[str, Any]], doc_dir: str | Path | None = None, *,
           pdf_path: str | Path | None = None) -> list[Assessment]:
    out: list[Assessment] = []
    pdf = Path(str(pdf_path).replace("\\", "/")) if pdf_path else None
    if pdf is not None and pdf.exists():
        try:
            out += text_coverage(doc, pdf_page_texts(pdf))
        except Exception as exc:  # noqa: BLE001  (damaged PDF): the other checks still run
            out.append(Assessment("parse.text_coverage", None, error=f"text layer not readable: {exc}"))
    else:
        out.append(Assessment("parse.text_coverage", None, error=f"source PDF not found: {pdf_path}"))
    return out + label_profile(doc, figures, doc_dir)
