"""Step 2 - heading tree built from the DoclingDocument.

Titles/section headers form the branches (``SectionNode``). The elements found under a
heading (paragraphs, tables, figures, lists, formulas, code...) are the leaves attached to
that section (``SectionNode.elements``).
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

ROOT_REF = "#/root"
logger = logging.getLogger(__name__)
FIX_HEADING_LEVELS = os.environ.get("HGRAG_FIX_HEADING_LEVELS", "1").lower() in ("1", "true", "yes")

_PARAGRAPH_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}
_KIND_BY_LABEL = {
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.FORMULA: "formula",
    DocItemLabel.CODE: "code",
    DocItemLabel.FOOTNOTE: "footnote",
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.REFERENCE: "reference",
}


@dataclass
class Element:
    ref: str
    kind: str  # paragraph | table | figure | list_item | formula | code | ...
    label: str
    page_no: int | None = None
    text: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "label": self.label,
            "page_no": self.page_no,
            "text": self.text,
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class SectionNode:
    ref: str
    title: str
    level: int  # 0 for the synthetic root, then 1..n
    page_no: int | None = None
    path: str = ""
    children: list["SectionNode"] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self) -> Iterator["SectionNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "title": self.title,
            "level": self.level,
            "page_no": self.page_no,
            "path": self.path,
            "is_leaf": self.is_leaf,
            "elements": [e.to_dict() for e in self.elements],
            "children": [c.to_dict() for c in self.children],
        }


def _page_of(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


# ---------------------------------------------------------------------------------------- heading levels
_ROMAN_RE = re.compile(r"^\s*([IVXLCDM]+)[.)]\s+\S")
_LETTER_RE = re.compile(r"^\s*([A-Z])[.)]\s+\S")
_BACK_RE = re.compile(
    r"^\s*(?:(?:[A-Z]|\d+|[IVXLCDM]+)[.)]?\s+)?(?:references?|bibliography|acknowledg(?:e)?ments?|appendix(?:\s+[A-Z0-9]+)?|"
    r"appendices|supplementary\s+(?:materials?|information)|supporting\s+information|"
    r"data\s+availability(?:\s+statement)?)\s*[:.]?\s*$", re.I)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(token: str) -> int | None:
    total = 0
    for i, ch in enumerate(token):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        total += -v if i + 1 < len(token) and _ROMAN_VALUES[token[i + 1]] > v else v
    return total or None


def heading_levels(headings: list[tuple[str, str, int]]) -> dict[str, int]:
    """Correct the levels Docling gives to headings, from the numbering and the document layout.

    ``headings`` are ``(ref, text, level)`` in reading order. Returns ``ref -> level`` for every heading.

    * Roman numerals ``I. II. III. IV. V.`` are taken as top-level when they form a sequence (so ``V.`` is the
      fifth part, not the letter V after ``D.``), and the capital letters ``A. B. C.`` that follow are their
      sub-sections.
    * Back matter (References, Acknowledgments, Appendix, Supplementary materials...) goes back to the top level when
      the document has several top-level headings.
    """
    out = {ref: int(lvl) for ref, _t, lvl in headings}
    # 1) the roman sequence: v == previous + 1 (the first one may be I or II)
    roman: dict[int, int] = {}
    last = 0
    for i, (_r, text, _l) in enumerate(headings):
        m = _ROMAN_RE.match(text)
        v = roman_to_int(m.group(1)) if m else None
        if v is not None and (v == last + 1 or (last == 0 and v <= 2)):
            roman[i] = v
            last = v
    if len(roman) >= 2:
        last_letter = ""
        seen_roman = False
        for i, (ref, text, lvl) in enumerate(headings):
            if i in roman:
                out[ref], last_letter, seen_roman = 1, "", True
                continue
            if not seen_roman:
                continue
            if _BACK_RE.match(text):
                out[ref] = 1
                continue
            m = _LETTER_RE.match(text)
            if m and (m.group(1) == "A" if not last_letter else ord(m.group(1)) == ord(last_letter) + 1):
                out[ref], last_letter = 2, m.group(1)
            else:
                out[ref] = max(int(lvl), 2)  # anything else after a numbered part is below it
    # 2) back matter is top-level whenever the document has several top-level headings
    if sum(1 for v in out.values() if v == 1) >= 2:
        for ref, text, _l in headings:
            if _BACK_RE.match(text):
                out[ref] = 1
    return out


def _attached_refs(item: Any) -> dict[str, list[str]]:
    """Refs of the caption and footnote items attached to a table/figure (they get no leaf of their own)."""
    return {"caption_refs": [c.cref for c in item.captions], "footnote_refs": [f.cref for f in item.footnotes]}


def _referenced_crefs(doc: DoclingDocument) -> set[str]:
    """Captions and footnotes already attached to a table/figure (not standalone leaves)."""
    refs: set[str] = set()
    for item in [*doc.tables, *doc.pictures]:
        refs.update(c.cref for c in item.captions)
        refs.update(f.cref for f in item.footnotes)
    return refs


def build_hierarchy(
    doc: DoclingDocument,
    figures: list[dict[str, Any]] | None = None,
    doc_title: str | None = None,
    table_descriptions: dict[str, str] | None = None,
) -> SectionNode:
    """Build the heading tree. ``figures`` are the records produced by ``export_figures``;
    ``table_descriptions`` maps a table ref to its generated description (optional)."""
    figures_by_ref = {f["self_ref"]: f for f in (figures or [])}
    attached = _referenced_crefs(doc)

    root = SectionNode(ref=ROOT_REF, title=doc_title or doc.name or "document", level=0, path="")
    root.path = root.title
    stack: list[SectionNode] = [root]

    levels = {it.self_ref: int(it.level) for it, _d in doc.iterate_items() if isinstance(it, SectionHeaderItem)}
    if FIX_HEADING_LEVELS:
        fixed = heading_levels([(it.self_ref, it.text, it.level) for it, _d in doc.iterate_items()
                                if isinstance(it, SectionHeaderItem)])
        changed = {r: (levels[r], v) for r, v in fixed.items() if levels[r] != v}
        if changed:
            logger.info("heading levels corrected for %d of %d headings (Docling level -> new): %s", len(changed),
                        len(levels), dict(list(changed.items())[:8]))
        levels = fixed

    for item, _depth in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) or (
            isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
        ):
            level = levels[item.self_ref] if isinstance(item, SectionHeaderItem) else 1
            level = max(int(level), 1)
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            node = SectionNode(
                ref=item.self_ref,
                title=item.text.strip(),
                level=level,
                page_no=_page_of(item),
                path=f"{parent.path} > {item.text.strip()}",
            )
            parent.children.append(node)
            stack.append(node)
            continue

        current = stack[-1]

        if isinstance(item, PictureItem):
            fig = figures_by_ref.get(item.self_ref)
            extra: dict[str, Any] = {"caption": item.caption_text(doc) or None, **_attached_refs(item)}
            if fig:
                extra.update(
                    figure_index=fig["figure_index"],
                    png_path=fig["png_path"],
                    description=(fig.get("description") or {}).get("text"),
                )
            current.elements.append(
                Element(item.self_ref, "figure", "picture", _page_of(item), extra.get("description"), extra)
            )
        elif isinstance(item, TableItem):
            current.elements.append(
                Element(
                    item.self_ref,
                    "table",
                    "table",
                    _page_of(item),
                    item.export_to_markdown(doc),
                    {
                        "caption": item.caption_text(doc) or None,
                        **_attached_refs(item),
                        "num_rows": item.data.num_rows,
                        "num_cols": item.data.num_cols,
                        **({"description": table_descriptions[item.self_ref]}
                           if table_descriptions and item.self_ref in table_descriptions else {}),
                    },
                )
            )
        elif isinstance(item, TextItem):
            if item.self_ref in attached:
                continue
            if item.label in _PARAGRAPH_LABELS:
                kind = "paragraph"
            elif item.label in _KIND_BY_LABEL:
                kind = _KIND_BY_LABEL[item.label]
            else:
                kind = "other"
            current.elements.append(
                Element(item.self_ref, kind, item.label.value, _page_of(item), item.text)
            )
    return root


def section_index(root: SectionNode) -> dict[str, SectionNode]:
    """Map every element ref to the section that holds it."""
    index: dict[str, SectionNode] = {}
    for node in root.walk():
        for el in node.elements:
            index[el.ref] = node
            for ref in [*(el.extra or {}).get("caption_refs", []), *(el.extra or {}).get("footnote_refs", [])]:
                index.setdefault(ref, node)  # the caption of a figure/table belongs to the section of its figure/table
    return index


def render_tree(root: SectionNode) -> str:
    """Indented text view of the heading tree with the element counts of each section."""
    lines = []
    for node in root.walk():
        kinds = Counter(e.kind for e in node.elements)
        counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "empty"
        page = f" p.{node.page_no}" if node.page_no else ""
        lines.append(f"{'  ' * node.level}{'' if node.level == 0 else '- '}{node.title}{page}  [{counts}]")
    return "\n".join(lines) + "\n"


def tree_stats(root: SectionNode) -> dict[str, Any]:
    """Numbers describing the tree, for tracing and verification."""
    sections = [s for s in root.walk() if s.level > 0]
    kinds: Counter[str] = Counter(e.kind for s in root.walk() for e in s.elements)
    levels: Counter[int] = Counter(s.level for s in sections)
    return {
        "n_sections": len(sections),
        "n_leaf_sections": sum(1 for s in sections if s.is_leaf),
        "max_depth": max((s.level for s in sections), default=0),
        "n_empty_sections": sum(1 for s in sections if s.is_leaf and not s.elements),  # the document title is one
        "root_elements": len(root.elements),
        "n_elements": sum(kinds.values()),
        "elements_by_kind": dict(kinds),
        "sections_by_level": {str(k): v for k, v in sorted(levels.items())},
    }
