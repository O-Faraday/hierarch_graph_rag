"""Repairs of the Docling output, applied when a parsed document is loaded (``pipeline._load``), before the tree and the
chunks. The Docling parsing on disk is unchanged; every item keeps its ref (figures, descriptions, captions).

* A table that runs over a page break comes out of Docling as one table per page, the header row repeated on each:
  the pieces are joined into the first one (``HGRAG_MERGE_SPLIT_TABLES``).
* Docling's reading order may put the title of the first page at the end of the page (a centred title above a
  single column): the heading that sits above everything else on page 1 is moved back to the start
  (``HGRAG_TITLE_FIRST``).
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
    if not (config.MERGE_SPLIT_TABLES or config.TITLE_FIRST):
        return doc, {}
    doc = doc.model_copy(deep=True)
    merged = merge_split_tables(doc) if config.MERGE_SPLIT_TABLES else {}
    if config.TITLE_FIRST:
        title_first(doc)
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


def _top(item: Any, doc: DoclingDocument) -> float:
    """Distance of the top of ``item`` from the top of its page (smaller = higher)."""
    prov = item.prov[0]
    height = doc.pages[prov.page_no].size.height
    return prov.bbox.to_top_left_origin(page_height=height).t


def title_first(doc: DoclingDocument) -> str | None:
    """Move back to the start of the body the heading that is the highest item of page 1 when the reading order put
    it after other items of that page. Returns its ref when moved."""
    page_one = []
    for ref in doc.body.children:
        item = ref.resolve(doc)
        if (getattr(item, "prov", None) and item.prov[0].page_no == 1
                and getattr(item, "content_layer", ContentLayer.BODY) == ContentLayer.BODY):
            page_one.append((ref, item))
    if len(page_one) < 2 or 1 not in doc.pages:
        return None
    ref, item = min(page_one, key=lambda pair: _top(pair[1], doc))
    is_heading = isinstance(item, SectionHeaderItem) or (isinstance(item, TextItem) and item.label == DocItemLabel.TITLE)
    if not is_heading or ref.cref == page_one[0][0].cref:
        return None
    children = doc.body.children
    children.remove(ref)
    children.insert(children.index(page_one[0][0]), ref)
    logger.info("title %r moved to the start of the reading order", item.text[:60])
    return ref.cref


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
