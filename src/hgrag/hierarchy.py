"""Step 2 - heading tree built from the DoclingDocument.

Titles/section headers form the branches (``SectionNode``). The elements found under a
heading (paragraphs, tables, figures, lists, formulas, code...) are the leaves attached to
that section (``SectionNode.elements``).
"""

from __future__ import annotations

import logging
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

from . import config

ROOT_REF = "#/root"
logger = logging.getLogger(__name__)

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
    r"data\s+availability(?:\s+statement)?|competing\s+interests?|conflicts?\s+of\s+interests?|declarations?|"
    r"funding|author\s+contributions?)\s*[:.]?\s*$", re.I)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# "3", "3.2", "3.2.1." / "A.1" (appendix A, sub-section 1) / "EC.2.1" (a prefix naming an e-companion or supplement)
_DECIMAL_RE = re.compile(r"^\s*(?:([A-Z]{1,3})\.)?(\d{1,2}(?:\.\d{1,2})*)\.?\s+\S")
# appendix part: "A QUERY EXAMPLES", "A. Omitted proof", "Appendix B: Proofs"
_APPENDIX_RE = re.compile(r"^\s*(?i:appendix\s+)?([A-Z])(?:[.:)]\s*|\s+)[A-Z0-9]")
_FRONT_END_RE = re.compile(r"^\W*(?:abstract|summary|introduction|\d{1,2}\.?\s+introduction)\b", re.I)
# not section headings: captions, run-in labels of proofs and cases, sub-figure labels
_CAPTION_RE = re.compile(r"^\s*(?:algorithm|figure|fig\.|table|listing)\s*\d", re.I)
_RUN_IN_RE = re.compile(r"^\s*(?:proof|case|claim|step|remark|example|lemma|theorem|proposition|corollary|definition|"
                        r"assumption|observation)\b", re.I)
_SUBLABEL_RE = re.compile(r"^\s*\([a-z0-9]{1,3}\)\s")
_NAME_MARKS = "*∗†‡§¶,0123456789"
MAX_HEADING_WORDS = 12


def roman_to_int(token: str) -> int | None:
    total = 0
    for i, ch in enumerate(token):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        total += -v if i + 1 < len(token) and _ROMAN_VALUES[token[i + 1]] > v else v
    return total or None


def numbering_depth(text: str) -> int | None:
    """Depth given by a decimal section number: ``3`` -> 1, ``3.2`` -> 2, ``A.1`` -> 2 (appendix A),
    ``EC.2.1`` -> 2 (a multi-letter prefix names a supplement, not a level); None when the heading is not numbered."""
    m = _DECIMAL_RE.match(text)
    if not m:
        return None
    prefix, numbers = m.groups()
    depth = numbers.count(".") + 1
    return depth + 1 if prefix and len(prefix) == 1 else depth


def _levels_from_numbering(headings: list[tuple[str, str, int]]) -> dict[str, int]:
    """Levels of a numbered document: the depth of each number; appendix letters ``A B C`` after the back matter and
    the back matter itself are top-level; an unnumbered heading goes one level below the last numbered (or back
    matter) heading, and keeps Docling's level before the first one (title, abstract)."""
    out: dict[str, int] = {}
    anchor: int | None = None
    in_back, last_letter = False, ""
    # letters numbering sub-sections elsewhere ("O.1" makes "O Online Appendix" a part even out of the A B C sequence)
    prefixes = {m.group(1) for _r, t, _l in headings if (m := _DECIMAL_RE.match(t)) and m.group(1) and len(m.group(1)) == 1}
    for ref, text, lvl in headings:
        depth = numbering_depth(text)
        if depth is None and in_back:
            m = _APPENDIX_RE.match(text)
            if m and (m.group(1) in prefixes
                      or (ord(m.group(1)) == ord(last_letter) + 1 if last_letter else m.group(1) == "A")):
                depth, last_letter = 1, m.group(1)
        if depth is None and _BACK_RE.match(text):
            depth, in_back = 1, True
        if depth is not None:
            out[ref] = anchor = depth
        else:
            out[ref] = int(lvl) if anchor is None else anchor + 1
    return out


def heading_levels(headings: list[tuple[str, str, int]]) -> dict[str, int]:
    """Correct the levels Docling gives to headings, from the numbering and the document layout.

    ``headings`` are ``(ref, text, level)`` in reading order. Returns ``ref -> level`` for every heading.

    * Roman numerals ``I. II. III. IV. V.`` are taken as top-level when they form a sequence (so ``V.`` is the
      fifth part, not the letter V after ``D.``), and the capital letters ``A. B. C.`` that follow are their
      sub-sections.
    * Otherwise, in a document with 3+ decimal-numbered headings, the numbering sets the levels
      (``_levels_from_numbering``): Docling often puts a ``4.2`` under ``4.1`` or an appendix under References.
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
    elif sum(1 for _r, text, _l in headings if numbering_depth(text)) >= 3:
        out.update(_levels_from_numbering(headings))
    # 2) back matter is top-level whenever the document has several top-level headings
    if sum(1 for v in out.values() if v == 1) >= 2:
        for ref, text, _l in headings:
            if _BACK_RE.match(text):
                out[ref] = 1
    return out


def _is_person_name(text: str) -> bool:
    """2 to 4 capitalised words or initials (``Jan Vondrák``, ``J.-P. Serre``), footnote marks allowed."""
    words = [w.strip(_NAME_MARKS) for w in text.split()]
    if not 2 <= len(words) <= 4:
        return False
    for w in words:
        parts = [p for p in w.replace("'", "-").replace("’", "-").split("-") if p]
        if not parts or not all(re.fullmatch(r"[A-Z]\.?", p) or (p[0].isupper() and p[1:].isalpha() and p[1:].islower())
                                for p in parts):
            return False
    return True


def fake_headings(headings: list[tuple[str, str, int]]) -> dict[str, str]:
    """Docling headings that are not section headings, ``ref -> reason``; ``headings`` are ``(ref, text, level)`` in
    reading order. Numbered and back-matter headings are always kept, and so is the first heading (the title).

    * ``caption``: ``Algorithm 1 ...``, ``Figure 2``, ``Table 3`` (a float caption read as a heading);
    * ``run-in label``: ``Proof of Lemma 3.``, ``Case 2 λ > 0.``, ``Steady case:`` (the first words of a paragraph);
    * ``sub-figure label``: ``(a) Weakly shill-proof auctions``;
    * ``sentence fragment``: starts with a lower-case letter (``maximize prefix reuse.``);
    * ``author name``: a person's name before the abstract / introduction / first numbered heading;
    * ``sentence``: more than ``MAX_HEADING_WORDS`` words after the front matter (a heading merged with its paragraph).
    """
    front_end = next((i for i, (_r, t, _l) in enumerate(headings) if numbering_depth(t) or _FRONT_END_RE.match(t)),
                     None)
    out: dict[str, str] = {}
    for i, (ref, text, _l) in enumerate(headings):
        t = text.strip()
        if i == 0 or not t or numbering_depth(t) or _BACK_RE.match(t):
            continue
        front = front_end is not None and i < front_end
        if _CAPTION_RE.match(t):
            out[ref] = "caption"
        elif _RUN_IN_RE.match(t) or t.endswith(":"):
            out[ref] = "run-in label"
        elif _SUBLABEL_RE.match(t):
            out[ref] = "sub-figure label"
        elif t[0].islower():
            out[ref] = "sentence fragment"
        elif front and _is_person_name(t):
            out[ref] = "author name"
        elif not front and len(t.split()) > MAX_HEADING_WORDS:
            out[ref] = "sentence"
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

    headings = [(it.self_ref, it.text, int(it.level)) for it, _d in doc.iterate_items()
                if isinstance(it, SectionHeaderItem)]
    demoted = fake_headings(headings) if config.DEMOTE_FAKE_HEADINGS else {}
    if demoted:
        logger.info("%d of %d headings turned into paragraphs: %s", len(demoted), len(headings),
                    dict(Counter(demoted.values())))
        headings = [h for h in headings if h[0] not in demoted]
    levels = {ref: lvl for ref, _t, lvl in headings}
    if config.FIX_HEADING_LEVELS:
        fixed = heading_levels(headings)
        changed = {r: (levels[r], v) for r, v in fixed.items() if levels[r] != v}
        if changed:
            logger.info("heading levels corrected for %d of %d headings (Docling level -> new): %s", len(changed),
                        len(levels), dict(list(changed.items())[:8]))
        levels = fixed

    for item, _depth in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) and item.self_ref in demoted:
            stack[-1].elements.append(Element(item.self_ref, "paragraph", item.label.value, _page_of(item), item.text,
                                              {"demoted_heading": demoted[item.self_ref]}))
            continue
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
