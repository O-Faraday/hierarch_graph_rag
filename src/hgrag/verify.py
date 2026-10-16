"""Check the pipeline output against ``expected.json`` (the control PDFs of ``scripts/make_control_pdfs.py``).

For every expected PDF found in ``pdf_dir`` the checks are: parsing files present, figure PNGs on disk,
table count, every expected heading present at the right level in ``hierarchy.json``, chunks indexed, and
each "needle" query retrieving the right section first.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config
from .ids import compute_doc_id


def _norm(title: str) -> str:
    """Lowercase, without the leading section number: Docling may or may not keep it."""
    return re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", title).strip().lower()


def _check(pdf: str, name: str, ok: bool, expected: Any, actual: Any) -> dict[str, Any]:
    return {"pdf": pdf, "check": name, "ok": bool(ok), "expected": expected, "actual": actual}


def _walk(tree: dict[str, Any]):
    yield tree
    for child in tree.get("children", []):
        yield from _walk(child)


def verify_pdf(
    pdf_path: Path, expected: dict[str, Any], *, output_dir: Path, embedder: Any = None,
    client: Any = None, collection: str = config.COLLECTION_NAME, k: int = 3,
) -> list[dict[str, Any]]:
    name = pdf_path.name
    doc_id = compute_doc_id(pdf_path)
    doc_dir = output_dir / doc_id
    out: list[dict[str, Any]] = []

    files = ["document.json", "figures.json", "manifest.json", "hierarchy.json"]
    missing = [f for f in files if not (doc_dir / f).exists()]
    out.append(_check(name, "parsing files on disk", not missing, files, f"missing: {missing}" if missing else "all present"))
    if missing:
        return out

    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    figures = json.loads((doc_dir / "figures.json").read_text(encoding="utf-8"))
    pngs = [f for f in figures if f.get("png_path") and (doc_dir / f["png_path"]).exists()
            and (doc_dir / f["png_path"]).stat().st_size > 0]
    out.append(_check(name, "figures extracted as PNG", len(pngs) == expected["n_figures"] == len(figures),
                      expected["n_figures"], f"{len(pngs)} PNG / {len(figures)} records"))

    tree = json.loads((doc_dir / "hierarchy.json").read_text(encoding="utf-8"))
    nodes = list(_walk(tree))
    n_tables = sum(1 for n in nodes for e in n["elements"] if e["kind"] == "table")
    n_fig_el = sum(1 for n in nodes for e in n["elements"] if e["kind"] == "figure")
    out.append(_check(name, "tables in the tree", n_tables == expected["n_tables"], expected["n_tables"], n_tables))
    out.append(_check(name, "figures attached to sections", n_fig_el == expected["n_figures"],
                      expected["n_figures"], n_fig_el))

    found = {_norm(n["title"]): n["level"] for n in nodes if n["level"] > 0}
    bad = []
    for title, level in expected["headings"]:
        got = found.get(_norm(title))
        if got != level:
            bad.append(f"{title!r}: expected level {level}, got {got}")
    out.append(_check(name, "headings and levels", not bad, f"{len(expected['headings'])} headings",
                      "all found" if not bad else bad))

    # nesting: each expected heading must sit under the closest previous heading of a lower level
    parent_of = {_norm(c["title"]): (_norm(n["title"]) if n["level"] > 0 else None)
                 for n in nodes for c in n.get("children", [])}
    wrong, stack = [], []
    for title, level in expected["headings"]:
        while stack and stack[-1][1] >= level:
            stack.pop()
        want = _norm(stack[-1][0]) if stack else None
        if parent_of.get(_norm(title)) != want:
            wrong.append(f"{title!r}: parent {parent_of.get(_norm(title))!r}, expected {want!r}")
        stack.append((title, level))
    out.append(_check(name, "tree nesting", not wrong, "children under the right parent", "ok" if not wrong else wrong))

    if embedder is None or client is None:
        return out
    from .retrieval import search_hits

    n_points = client.count(collection, exact=True).count if client.collection_exists(collection) else 0
    out.append(_check(name, "chunks indexed", n_points > 0, "> 0 points in the collection", n_points))
    for needle in expected["needles"]:
        hits = [h for h in search_hits(needle["query"], k, embedder=embedder, client=client, collection=collection)
                if h["doc_id"] == doc_id]
        titles = [_norm(h["section_path"].split(" > ")[-1]) for h in hits]
        want = _norm(needle["section"])
        rank = titles.index(want) + 1 if want in titles else None
        out.append(_check(name, f"query -> {needle['section']}", rank == 1, f"rank 1: {needle['query']!r}",
                          f"rank {rank}" if rank else f"not in top {k}: {titles}"))
    return out


def verify(
    pdf_dir: str | Path, *, output_dir: str | Path = config.OUTPUT_DIR, embedder: Any = None, client: Any = None,
    collection: str = config.COLLECTION_NAME,
) -> dict[str, Any]:
    pdf_dir, output_dir = Path(pdf_dir), Path(output_dir)
    expected_path = pdf_dir / "expected.json"
    if not expected_path.exists():
        raise FileNotFoundError(f"{expected_path} not found: run `uv run python scripts/make_control_pdfs.py`")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for name, exp in expected.items():
        pdf = pdf_dir / name
        if not pdf.exists():
            checks.append(_check(name, "pdf present", False, name, "missing"))
            continue
        checks += verify_pdf(pdf, exp, output_dir=output_dir, embedder=embedder, client=client, collection=collection)
    n_ok = sum(c["ok"] for c in checks)
    return {"ok": n_ok == len(checks), "n_checks": len(checks), "n_ok": n_ok, "checks": checks}


def format_report(report: dict[str, Any]) -> str:
    lines = []
    last = None
    for c in report["checks"]:
        if c["pdf"] != last:
            lines.append(f"\n{c['pdf']}")
            last = c["pdf"]
        mark = "OK  " if c["ok"] else "FAIL"
        extra = "" if c["ok"] else f"\n        expected: {c['expected']}\n        actual:   {c['actual']}"
        lines.append(f"  [{mark}] {c['check']}{extra}")
    lines.append(f"\n{report['n_ok']}/{report['n_checks']} checks passed")
    return "\n".join(lines)
