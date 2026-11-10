"""The structure of a document as one readable JSON tree: title, sections, sub-sections, typed content, pages.

``hierarchy.json`` is the working format of the pipeline (Docling refs, labels, ``extra``). ``structure.json`` is the
same tree for people and for other tools: every section holds its content in reading order and its sub-sections;
every block is typed (paragraph, list, table with its description, figure with its image and description, formula
with its LaTeX and reading, code, footnote) and carries its page. It is written next to ``hierarchy.json`` and can be
rebuilt from it at any time; ``hgrag structure`` gathers the documents of a folder into one list (the corpus).

    {"doc_id", "title", "source_pdf", "pages", "content": [blocks before the first section],
     "sections": [{"id": "s2.1", "number": "2.1", "title", "level", "pages": {"start", "end"},
                   "content": [{"type": "paragraph", "text", "page"}, {"type": "table", "caption", "markdown",
                                "description", "page"}, {"type": "figure", "caption", "image", "description",
                                "page"}, ...],
                   "sections": [...]}]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .hierarchy import _is_standard, numbering_depth
from .ids import pdf_name

FILE = "structure.json"
_NUMBER_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*\.?|[A-Z](?:\.\d{1,2})*[.)]|[IVXLC]{1,6}[.)])\s+(?=\S)")
_APPENDIX_RE = re.compile(r"^\s*appendix\s+([A-Z0-9]{1,3})\b", re.I)
_TEXT_KINDS = {"paragraph": "paragraph", "footnote": "footnote", "caption": "caption", "code": "code",
               "reference": "reference"}


def heading_number(title: str) -> str | None:
    """The number of a heading as written: "2.1", "IV", "B", "A" for "Appendix A: ..."; None if unnumbered."""
    m = _APPENDIX_RE.match(title) or _NUMBER_RE.match(title)
    return m.group(1).rstrip(".)") if m else None


def _block(el: dict[str, Any], formulas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    extra = el.get("extra") or {}
    kind, page, ref = el["kind"], el.get("page_no"), el.get("ref")
    if kind == "table":
        block = {"type": "table", "caption": extra.get("caption"), "markdown": el.get("text"),
                 "description": extra.get("description"), "rows": extra.get("num_rows"), "cols": extra.get("num_cols")}
    elif kind == "figure":
        image = extra.get("png_path")
        block = {"type": "figure", "caption": extra.get("caption"), "image": image.replace("\\", "/") if image else None,
                 "description": extra.get("description")}
    elif kind == "formula":
        rec = formulas.get(ref) or {}
        reading = rec.get("reading")
        block = {"type": "formula", "text": el.get("text"), "latex": rec.get("latex"),
                 "reading": reading.get("text") if isinstance(reading, dict) else reading}
    else:
        block = {"type": _TEXT_KINDS.get(kind, "text"), "text": el.get("text")}
    pages = extra.get("pages")  # a table or figure over several pages: all of them
    return {**{k: v for k, v in block.items() if v is not None}, "page": page, **({"pages": pages} if pages else {}),
            "ref": ref}


def _content(elements: list[dict[str, Any]], formulas: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Blocks in reading order; consecutive list items become one list."""
    out: list[dict[str, Any]] = []
    for el in elements:
        if el["kind"] == "list_item":
            last = out[-1] if out else None
            if last and last["type"] == "list":
                last["items"].append(el.get("text"))
                last["refs"].append(el.get("ref"))
                last["pages"] = sorted({*last["pages"], el.get("page_no")} - {None})
                continue
            out.append({"type": "list", "items": [el.get("text")], "pages": [p for p in [el.get("page_no")] if p],
                        "refs": [el.get("ref")]})
            continue
        out.append(_block(el, formulas))
    return out


def _pages(node: dict[str, Any]) -> list[int]:
    pages = [node.get("page_no")] + [p for e in node.get("elements", [])
                                      for p in [e.get("page_no"), *(e.get("extra") or {}).get("pages", [])]]
    for child in node.get("children", []):
        pages += _pages(child)
    return [p for p in pages if p]


def _section(node: dict[str, Any], sid: str, formulas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pages = _pages(node)
    return {"id": sid, "number": heading_number(node["title"]), "title": node["title"], "level": node["level"],
            "pages": {"start": min(pages), "end": max(pages)} if pages else None,
            "content": _content(node.get("elements", []), formulas),
            "sections": [_section(c, f"{sid}.{i}", formulas) for i, c in enumerate(node.get("children", []), 1)],
            "ref": node.get("ref")}


def _is_title(node: dict[str, Any]) -> bool:
    """The first top-level heading is the document title when it sits on page 1, unnumbered and not a standard
    section name (Abstract, Introduction...)."""
    title = node.get("title") or ""
    return node.get("page_no") in (1, None) and not numbering_depth(title) and not heading_number(title) \
        and not _is_standard(title)


def to_structure(tree: dict[str, Any], *, doc_id: str, source_pdf: str | None = None, n_pages: int | None = None,
                 formulas: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The readable tree of one document from its ``hierarchy.json`` (``formulas``: ``formulas.json`` by ref)."""
    formulas = formulas or {}
    top = list(tree.get("children", []))
    content = list(tree.get("elements", []))
    title = tree.get("title")
    if top and _is_title(top[0]):  # the title heading: its text is the title, its content the front matter
        head = top.pop(0)
        title = head["title"]
        content += head.get("elements", [])
        top = list(head.get("children", [])) + top
    return {"doc_id": doc_id, "title": title, "source_pdf": pdf_name(source_pdf) if source_pdf else None,
            "pages": n_pages, "content": _content(content, formulas),
            "sections": [_section(node, f"s{i}", formulas) for i, node in enumerate(top, 1)]}


def _formulas(doc_dir: Path) -> dict[str, dict[str, Any]]:
    path = doc_dir / "formulas.json"
    if not path.exists():
        return {}
    return {r["self_ref"]: r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("self_ref")}


def build(doc_dir: str | Path, tree: dict[str, Any] | None = None) -> dict[str, Any]:
    """The structure of a parsed document, from ``tree`` or its ``hierarchy.json``."""
    doc_dir = Path(doc_dir)
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    tree = tree if tree is not None else json.loads((doc_dir / "hierarchy.json").read_text(encoding="utf-8"))
    return to_structure(tree, doc_id=manifest.get("doc_id") or doc_dir.name, source_pdf=manifest.get("source_pdf"),
                        n_pages=manifest.get("n_pages"), formulas=_formulas(doc_dir))


def save(doc_dir: str | Path, tree: dict[str, Any] | None = None) -> Path:
    """Write ``structure.json`` next to ``hierarchy.json``."""
    path = Path(doc_dir) / FILE
    path.write_text(json.dumps(build(doc_dir, tree), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def corpus(output_dir: str | Path) -> list[dict[str, Any]]:
    """The structures of every parsed document of ``output_dir``, rebuilt from their ``hierarchy.json``."""
    dirs = sorted(p.parent for p in Path(output_dir).glob("*/hierarchy.json") if (p.parent / "manifest.json").exists())
    return sorted((build(d) for d in dirs), key=lambda s: s.get("source_pdf") or s["doc_id"])
