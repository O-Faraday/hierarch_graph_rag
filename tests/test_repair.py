"""Repairs of the Docling output: tables cut by a page break, a page-1 title put at the end of the reading order."""

from __future__ import annotations

from docling_core.types.doc import (ContentLayer, DocItemLabel, DoclingDocument, ProvenanceItem, TableCell, TableData)
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size

from hgrag import config, repair
from hgrag.hierarchy import build_hierarchy


def _prov(page: int, top: float = 700, bottom: float = 650) -> ProvenanceItem:
    box = BoundingBox(l=50, t=top, r=500, b=bottom, coord_origin=CoordOrigin.BOTTOMLEFT)
    return ProvenanceItem(page_no=page, bbox=box, charspan=(0, 1))


def _table(rows: list[list[str]], header: bool = True) -> TableData:
    cells = [TableCell(text=text, start_row_offset_idx=r, end_row_offset_idx=r + 1, start_col_offset_idx=c,
                       end_col_offset_idx=c + 1, column_header=header and r == 0)
             for r, row in enumerate(rows) for c, text in enumerate(row)]
    return TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0]))


HEAD = ["Vehicle", "Fuel"]


def _register(second_header=HEAD, second_page: int = 2, between: str | None = None) -> DoclingDocument:
    d = DoclingDocument(name="x")
    for p in (1, 2, 3):
        d.add_page(page_no=p, size=Size(width=600, height=800))
    d.add_heading(text="2 Register", level=1, prov=_prov(1))
    d.add_table(data=_table([HEAD, ["FV-301", "8.0"], ["FV-302", "21.1"]]), prov=_prov(1))
    d.add_text(label=DocItemLabel.PAGE_FOOTER, text="page 1", prov=_prov(1, 40, 30), content_layer=ContentLayer.FURNITURE)
    if between:
        d.add_text(label=DocItemLabel.TEXT, text=between, prov=_prov(1, 60, 50))
    rows = ([second_header] if second_header else []) + [["FV-358", "8.9"], ["FV-359", "5.5"]]
    d.add_table(data=_table(rows, header=bool(second_header)), prov=_prov(second_page))
    return d


def _tables_in_tree(doc, merged=None):
    root = build_hierarchy(doc, [], doc_title="x")
    return [e for n in root.walk() for e in n.elements if e.kind == "table"]


def test_a_table_cut_by_a_page_break_is_joined_without_its_repeated_header():
    doc, merged = repair.repair_document(_register())
    assert merged == {"#/tables/0": ["#/tables/1"]}
    (table,) = _tables_in_tree(doc)
    assert table.extra["num_rows"] == 5
    assert table.text.count("Vehicle") == 1 and "FV-301" in table.text and "FV-359" in table.text
    assert [p.page_no for p in doc.tables[0].prov] == [1, 2]
    assert len(_tables_in_tree(_register())) == 2  # the original is untouched


def test_a_piece_without_header_also_continues_the_table():
    doc, merged = repair.repair_document(_register(second_header=None))
    assert merged and _tables_in_tree(doc)[0].extra["num_rows"] == 5


def test_other_tables_stay_apart():
    for kwargs in ({"second_header": ["Plate", "Litres"]}, {"second_page": 3}, {"between": "Text between the tables."}):
        doc, merged = repair.repair_document(_register(**kwargs))
        assert merged == {} and len(_tables_in_tree(doc)) == 2, kwargs


def test_descriptions_of_the_pieces_are_joined():
    merged = {"#/tables/0": ["#/tables/1"]}
    out = repair.merge_descriptions({"#/tables/0": "Rows FV-301 to FV-329.", "#/tables/1": "Rows FV-330 to FV-360."},
                                    merged)
    assert out["#/tables/0"] == "Rows FV-301 to FV-329.\nRows FV-330 to FV-360."
    assert repair.merge_descriptions({"#/tables/5": "x"}, {}) == {"#/tables/5": "x"}


def _box(page: int, left: float, right: float, top: float, bottom: float) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=left, t=top, r=right, b=bottom,
                                                         coord_origin=CoordOrigin.BOTTOMLEFT), charspan=(0, 1))


def _policy(title_last: bool = True, page: int = 1) -> DoclingDocument:
    """The remote work policy as Docling ordered it: the centred title (x 172-423), the highest item of the page, after
    the column that starts with the narrow left-aligned heading "Purpose" (x 68-106)."""
    d = DoclingDocument(name="x")
    for p in range(1, page + 1):
        d.add_page(page_no=p, size=Size(width=595, height=842))
    if page > 1:
        d.add_text(label=DocItemLabel.TEXT, text="End of the previous page.", prov=_box(1, 68, 509, 700, 676))
    title = ("Remote Work Policy", _box(page, 172, 423, 766, 742))
    if not title_last:
        d.add_heading(text=title[0], level=1, prov=title[1])
    d.add_heading(text="Purpose", level=1, prov=_box(page, 68, 106, 721, 712))
    d.add_text(label=DocItemLabel.TEXT, text="This policy sets the rules.", prov=_box(page, 68, 509, 700, 676))
    if title_last:
        d.add_heading(text=title[0], level=1, prov=title[1])
    return d


def _titles(doc):
    return [n.title for n in build_hierarchy(doc, [], doc_title="x").walk() if n.level]


def test_a_centred_heading_at_the_top_of_a_page_is_read_first():
    doc, _ = repair.repair_document(_policy())
    assert _titles(doc)[:2] == ["Remote Work Policy", "Purpose"]
    doc, _ = repair.repair_document(_policy(page=2))  # any page, not only the first
    assert _titles(doc)[:2] == ["Remote Work Policy", "Purpose"]
    assert repair.top_headings_first(_policy(title_last=False)) == []  # already first: nothing moves


def test_the_heading_of_a_right_hand_column_stays_in_place():
    d = DoclingDocument(name="x")
    d.add_page(page_no=1, size=Size(width=595, height=842))
    d.add_heading(text="Left column", level=1, prov=_box(1, 50, 280, 760, 745))
    d.add_text(label=DocItemLabel.TEXT, text="Left text.", prov=_box(1, 50, 280, 740, 600))
    d.add_heading(text="Right column", level=1, prov=_box(1, 315, 545, 765, 750))  # a little higher, right column
    assert repair.top_headings_first(d) == []


def test_repairs_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "MERGE_SPLIT_TABLES", False)
    monkeypatch.setattr(config, "TOP_HEADING_FIRST", False)
    doc = _register()
    assert repair.repair_document(doc) == (doc, {})
