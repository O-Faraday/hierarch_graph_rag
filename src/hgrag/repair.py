"""Repairs of the Docling output, applied when a parsed document is loaded (``pipeline._load``), before the tree and the
chunks. The Docling parsing on disk is unchanged; every item keeps its ref (figures, descriptions, captions).

* A table that runs over a page break comes out of Docling as one table per page, the header row repeated on each:
  the pieces are joined into the first one (``HGRAG_MERGE_SPLIT_TABLES``).
* Docling's reading order may put a centred heading at the top of a page after the column below it (the title of
  a one-column page was read last): the heading that sits above everything else on its page, across the middle of
  the page, is moved back to the start of that page (``HGRAG_TOP_HEADING_FIRST``).
"""

from __future__ import annotations

import logging
from typing import Any

from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, SectionHeaderItem, TableItem, TextItem

from . import config

logger = logging.getLogger(__name__)

MANIFEST_KEY = "merged_tables"  # first table ref -> refs of the pieces joined to it, put in the loaded manifest


def repair_document(doc: DoclingDocument) -> tuple[DoclingDocument, dict[str, list[str]]]:
    """A repaired copy of ``doc`` and ``first table ref -> refs of the pieces joined to it``."""
    if not (config.MERGE_SPLIT_TABLES or config.TOP_HEADING_FIRST):
        return doc, {}
    doc = doc.model_copy(deep=True)
    merged = merge_split_tables(doc) if config.MERGE_SPLIT_TABLES else {}
    if config.TOP_HEADING_FIRST:
        top_headings_first(doc)
    return doc, merged


def _header(table: TableItem) -> list[str] | None:
    """Texts of the first row when it is a column header row."""
    grid = table.data.grid
    if not grid or not all(cell.column_header for cell in grid[0]):
        return None
    return [cell.text.strip() for cell in grid[0]]


def _continues(first: TableItem, piece: TableItem) -> int | None:
    """Rows of ``piece`` to skip when it continues ``first`` on the next page (its repeated header), else None."""
    if piece.captions or piece.data.num_cols != first.data.num_cols or not first.prov or not piece.prov:
        return None
    if piece.prov[0].page_no != first.prov[-1].page_no + 1:
        return None
    head, again = _header(first), _header(piece)
    if again is not None:
        return 1 if again == head else None
    return 0  # no header on the new page: the rows simply go on


def merge_split_tables(doc: DoclingDocument) -> dict[str, list[str]]:
    """Join, in place, the consecutive tables of the body that continue each other over a page break (only page
    furniture between them). The pieces leave the body; their rows and provenance go to the first table."""
    merged: dict[str, list[str]] = {}
    children = doc.body.children
    first: TableItem | None = None
    kept = []
    for ref in children:
        item = ref.resolve(doc)
        if isinstance(item, TableItem):
            skip = _continues(first, item) if first is not None else None
            if skip is not None:
                offset = first.data.num_rows - skip
                for cell in item.data.table_cells:
                    if cell.start_row_offset_idx < skip:
                        continue
                    first.data.table_cells.append(cell.model_copy(update={
                        "start_row_offset_idx": cell.start_row_offset_idx + offset,
                        "end_row_offset_idx": cell.end_row_offset_idx + offset}))
                first.data.num_rows += item.data.num_rows - skip
                first.prov.extend(item.prov)
                merged.setdefault(first.self_ref, []).append(item.self_ref)
                continue  # the piece leaves the body
            first = item
        elif getattr(item, "content_layer", ContentLayer.BODY) != ContentLayer.BODY:
            pass  # a page header or footer between two pieces
        else:
            first = None
        kept.append(ref)
    if merged:
        doc.body.children = kept
        logger.info("tables split by a page break joined: %s", merged)
    return merged


def _box(item: Any, doc: DoclingDocument) -> Any:
    """Box of ``item`` with the origin at the top left of its page (smaller ``t`` = higher)."""
    prov = item.prov[0]
    return prov.bbox.to_top_left_origin(page_height=doc.pages[prov.page_no].size.height)


def top_headings_first(doc: DoclingDocument) -> list[str]:
    """Put back at the start of its page the heading that sits above everything else on that page.

    Docling's rule-based reading order links an element to the one above it only when they overlap horizontally,
    and sorts the elements that have nothing above them from left to right. A centred title above a narrow
    left-aligned heading ("Purpose") therefore starts its own chain, placed after the left column: the title of a
    one-column page was read last. The heading is moved only when it is the highest item of its page and crosses
    the middle of the page (a centred or full-width heading, not the heading of a right-hand column)."""
    by_page: dict[int, list[Any]] = {}
    for ref in doc.body.children:
        item = ref.resolve(doc)
        if getattr(item, "prov", None) and getattr(item, "content_layer", ContentLayer.BODY) == ContentLayer.BODY:
            by_page.setdefault(item.prov[0].page_no, []).append((ref, item))
    moved = []
    for page_no, items in by_page.items():
        if len(items) < 2 or page_no not in doc.pages:
            continue
        ref, item = min(items, key=lambda pair: _box(pair[1], doc).t)
        heading = isinstance(item, SectionHeaderItem) or (isinstance(item, TextItem) and item.label == DocItemLabel.TITLE)
        box, middle = _box(item, doc), doc.pages[page_no].size.width / 2
        if not heading or ref.cref == items[0][0].cref or not box.l < middle < box.r:
            continue
        children = doc.body.children
        children.remove(ref)
        children.insert(children.index(items[0][0]), ref)
        moved.append(ref.cref)
        logger.info("heading %r (top of page %d) moved to the start of its page in the reading order",
                    item.text[:60], page_no)
    return moved


def merge_descriptions(descriptions: dict[str, str], merged: dict[str, list[str]] | None) -> dict[str, str]:
    """The description of a joined table: those of its pieces, in order."""
    if not merged:
        return descriptions
    out = dict(descriptions)
    for first, pieces in merged.items():
        parts = [descriptions.get(ref) for ref in [first, *pieces]]
        if any(parts):
            out[first] = "\n".join(p for p in parts if p)
    return out
