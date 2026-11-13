"""PDF bookmarks (outline) as a heading signal: headings run into their paragraph.

Some styles (Annals of Applied Statistics, IMS, many journals) set a section heading at the start of its first
paragraph: "2.2. Transmural Trauma Care Model (TTCM). Our data of interest...". Docling's layout model sees one
text region and labels it ``text``: the heading is lost, and a 41-page paper ends up with a handful of sections.
Docling has no option for this (its heading-hierarchy step only sets levels and never splits an item), but most
PDFs made with LaTeX carry bookmarks that list the real headings with their page. ``with_outline_headings`` splits
every paragraph that starts with the title of a bookmark of its page into a heading (the bookmark's level) and the
rest of the paragraph, then the paragraphs that hold a numbered bookmark title after the end of a sentence (several
run-in sections read as one text region). The Docling parsing itself is unchanged: the bookmarks are read from the PDF.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from docling_core.types.doc import (ContentLayer, DocItemLabel, DoclingDocument, GroupItem, RefItem, SectionHeaderItem,
                                    TextItem)

from . import config

logger = logging.getLogger(__name__)

OUTLINE_FILE = "outline.json"
MANIFEST_KEY = "outline_headings"  # ref -> level of the headings confirmed by a bookmark, put in the loaded manifest

# The numbering before a heading: "2.2.", "4", "A.1", "IV.", "B.", "Appendix A:" (the bookmarks drop it)
_NUMBER_RE = re.compile(
    r"^\s*(?:(?i:appendix)\s+[A-Z0-9]{1,3}[.:]?|(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2})*\.?|[IVXLC]{1,6}\.|[A-Z]\.)\s+")
_TEXT_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}
PAGE_TOLERANCE = 1  # a bookmark may point to the page before its heading (the anchor sits above the line)


def read_outline(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Bookmarks of a PDF in document order: ``{"title", "level" (1 = top), "page_no" (1-based or None)}``."""
    import pypdfium2 as pdfium  # installed with Docling

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        out = []
        for item in pdf.get_toc(max_depth=10):
            title = (item.get_title() or "").strip()
            dest = item.get_dest()
            index = dest.get_index() if dest is not None else None
            if title:
                out.append({"title": title, "level": item.level + 1, "page_no": None if index is None else index + 1})
        return out
    finally:
        pdf.close()


def save_outline(pdf_path: str | Path, doc_dir: str | Path) -> list[dict[str, Any]]:
    """Read the bookmarks of ``pdf_path`` and write them to ``outline.json``; never fails the parsing."""
    try:
        outline = read_outline(pdf_path)
    except Exception as exc:  # noqa: BLE001 - a broken outline must not stop the parsing
        logger.warning("%s: bookmarks not read: %s", Path(pdf_path).name, exc)
        outline = []
    (Path(doc_dir) / OUTLINE_FILE).write_text(json.dumps(outline, indent=1, ensure_ascii=False), encoding="utf-8")
    return outline


def load_outline(doc_dir: str | Path, source_pdf: str | Path | None = None) -> list[dict[str, Any]]:
    """``outline.json`` of a parsed document; for a document parsed before it existed, read from the source PDF
    (and saved) when that file is still there, else no bookmark."""
    path = Path(doc_dir) / OUTLINE_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pdf = Path(str(source_pdf).replace("\\", "/")) if source_pdf else None
    if pdf is None or not pdf.is_file():
        return []
    try:
        return save_outline(pdf, doc_dir)
    except OSError:  # read-only output folder: use the bookmarks without saving them
        return read_outline(pdf)


def section_outline(outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The bookmarks of the sections. When the first bookmark is the only one at the top level, it is the title of the
    document (its sections hang under it): it is dropped and the others move up one level, so that the sections are at
    level 1 as in the documents without that bookmark."""
    if len(outline) > 1 and outline[0]["level"] == 1 and all(b["level"] > 1 for b in outline[1:]):
        return [{**b, "level": b["level"] - 1} for b in outline[1:]]
    return list(outline)


def _key(text: str) -> str:
    """Letters and digits only: Docling glues words ("inCEA") and the bookmarks keep the spaces."""
    return re.sub(r"[\W_]+", "", text.lower())


def _match_end(text: str, title_key: str) -> int | None:
    """Index in ``text`` just after the title whose key is ``title_key``, when ``text`` starts with it."""
    seen = 0
    for i, ch in enumerate(text):
        if not (ch.isalnum()):
            continue
        if seen >= len(title_key) or ch.lower() != title_key[seen]:
            return None
        seen += 1
        if seen == len(title_key):
            return i + 1
    return None


def split_heading(text: str, title: str, *, numbered_only: bool) -> tuple[str, str] | None:
    """``(heading, rest)`` when ``text`` starts with ``title`` (after an optional number) followed by the end of the
    text or by ``.``/``:``; the heading keeps its number ("2.2. Transmural Trauma Care Model (TTCM)").
    ``numbered_only``: a paragraph must carry a number before the title, unless it is the title alone."""
    m = _NUMBER_RE.match(text)
    title_key = _key(title)
    if not title_key:
        return None
    # the bookmark may keep the number ("1 Executive summary") or drop it ("Transmural Trauma Care Model (TTCM)")
    for start in dict.fromkeys([0, m.end() if m else 0]):
        end = _match_end(text[start:], title_key)
        if end is not None:
            end += start
            break
    else:
        return None
    while end < len(text) and text[end] in ")]\"'’”":  # "Care Model (TTCM). Our data": the title ends with ")"
        end += 1
    rest = text[end:]
    if rest.strip() and not re.match(r"^\s*[.:]\s+\S", rest):
        return None  # "Introduction of the method..." is not the heading "Introduction"
    if numbered_only and m is None and rest.strip():
        return None
    return text[:end].strip(), re.sub(r"^\s*[.:]\s*", "", rest).strip()


# The number a heading must carry to be read inside a paragraph: "2.", "2.1.", "A.1"
_INNER_NUMBER = r"(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2})*\.?"


def inner_heading(text: str, title: str) -> tuple[int, int, int] | None:
    """``(start, end, start of the rest)`` of the numbered heading ``title`` found inside ``text``, after the end of a
    sentence and followed by ``.``/``:``: "...by later teams. 2.1. Monitoring wells. Water levels were...". The layout
    model sometimes reads several run-in sections as one text region. The number is required: an unnumbered title
    inside a paragraph is too often a plain sentence."""
    words = re.findall(r"[^\W_]+", _NUMBER_RE.sub("", title, count=1))
    if not words:
        return None
    pattern = (r"(?<=[.!?])\s+(" + _INNER_NUMBER + r"\s+" + r"[\W_]+".join(re.escape(w) for w in words)
               + r"[)\]\"'’”]*)\s*[.:]\s+(?=\S)")
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return (m.start(1), m.end(1), m.end()) if m else None


def _split_inner_headings(doc: DoclingDocument, pending: list[dict[str, Any]], confirmed: dict[str, int]) -> int:
    """Split the paragraphs that hold bookmarks still not found: each heading and the text that follows it become
    items placed after the paragraph, which keeps the text before the first heading. Returns the headings added."""
    inserted = 0
    for item, _depth in list(doc.iterate_items()):
        if not pending:
            break
        if not (isinstance(item, TextItem) and item.label in _TEXT_LABELS) or not item.prov:
            continue
        page, text = item.prov[0].page_no, item.text
        found = []
        for bm in pending:
            if bm["page_no"] is not None and abs(bm["page_no"] - page) > PAGE_TOLERANCE:
                continue
            span = inner_heading(text, bm["title"])
            if span is not None:
                found.append((span, bm))
        found.sort(key=lambda f: f[0][0])
        kept: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
        for span, bm in found:
            if not kept or span[0] >= kept[-1][0][2]:  # two bookmarks on the same words: the first one
                kept.append((span, bm))
        last: Any = item
        for i, ((start, end, rest_start), bm) in enumerate(kept):
            stop = kept[i + 1][0][0] if i + 1 < len(kept) else len(text)
            level = int(bm["level"])
            last = doc.insert_heading(sibling=last, text=text[start:end].strip(), level=level, prov=item.prov[0],
                                      after=True)
            confirmed[last.self_ref] = level
            pending.remove(bm)
            inserted += 1
            rest = text[rest_start:stop].strip()
            if rest:
                last = doc.insert_text(sibling=last, label=DocItemLabel.TEXT, text=rest, prov=item.prov[0], after=True)
        if kept:
            item.text = text[:kept[0][0][0]].strip()
    return inserted


def with_outline_headings(doc: DoclingDocument, outline: list[dict[str, Any]]) -> tuple[DoclingDocument, dict[str, int]]:
    """A copy of ``doc`` where each bookmark found at the start of a paragraph or of a Docling heading of its page
    becomes a heading at the bookmark's level, followed by the rest of the text (same ref as before, so the figures,
    tables, captions and descriptions keep their refs; the new headings are appended to ``texts``). A text item that
    is only the title is moved out of the body. Returns the copy and ``ref -> level`` of the headings confirmed by a
    bookmark, which the hierarchy keeps whatever the numbering says (``HGRAG_OUTLINE_HEADINGS``)."""
    if not config.OUTLINE_HEADINGS or not outline:
        return doc, {}
    doc = doc.model_copy(deep=True)
    pending = section_outline(outline)
    confirmed: dict[str, int] = {}
    inserted = 0
    for item, _depth in list(doc.iterate_items()):
        is_heading = isinstance(item, SectionHeaderItem)
        if not (is_heading or (isinstance(item, TextItem) and item.label in _TEXT_LABELS)) or not item.prov:
            continue
        page = item.prov[0].page_no
        for bm in pending:
            if bm["page_no"] is not None and abs(bm["page_no"] - page) > PAGE_TOLERANCE:
                continue
            split = split_heading(item.text, bm["title"], numbered_only=not is_heading)
            if split is None:
                continue
            heading, rest = split
            pending.remove(bm)
            level = int(bm["level"])
            if is_heading and not rest:
                item.level = level
                confirmed[item.self_ref] = level
            elif is_heading:  # a long Docling heading: the bookmark gives its real title, the rest is a paragraph
                doc.insert_text(sibling=item, label=DocItemLabel.TEXT, text=rest, prov=item.prov[0], after=True)
                item.text, item.orig, item.level = heading, heading, level
                confirmed[item.self_ref] = level
                inserted += 1
            else:
                new = doc.insert_heading(sibling=item, text=heading, level=level, prov=item.prov[0], after=False)
                confirmed[new.self_ref] = level
                inserted += 1
                if rest:
                    item.text = rest
                else:
                    item.content_layer = ContentLayer.FURNITURE  # the heading alone: replaced by the new heading
            break
    inserted += _split_inner_headings(doc, pending, confirmed)  # after the starts: a paragraph holding several sections
    lifted = _lift_out_of_groups(doc, confirmed)
    if confirmed:
        logger.info("%d heading(s) confirmed by the PDF bookmarks, %d split from their paragraph; %d bookmark(s) "
                    "not found%s", len(confirmed), inserted, len(pending),
                    f"; {lifted} group(s) dissolved" if lifted else "")
    return doc, confirmed


def _lift_out_of_groups(doc: DoclingDocument, refs: Iterable[str]) -> int:
    """Dissolve the groups that hold a confirmed heading: Docling reads "1. Introduction. Text..." as a numbered list,
    and the chunker then serialises the whole list, headings included, as one chunk (dropped for its labels). The
    items of such a group move up to its parent, at its place. Returns the number of groups dissolved."""
    dissolved = 0
    for ref in refs:
        item = RefItem(cref=ref).resolve(doc)
        while item.parent is not None and item.parent.cref != doc.body.self_ref:
            group = item.parent.resolve(doc)
            if not isinstance(group, GroupItem) or group.parent is None:
                break
            holder = group.parent.resolve(doc)
            at = next(i for i, child in enumerate(holder.children) if child.cref == group.self_ref)
            for child in group.children:
                child.resolve(doc).parent = holder.get_ref()
            holder.children[at:at + 1] = list(group.children)
            group.children = []
            dissolved += 1
    return dissolved
