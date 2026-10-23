"""Step 3 - heading tree: numbering of the headings, references and acknowledgements kept out of the index, and an
LLM judge on the outline."""

from __future__ import annotations

import re
from typing import Any, Iterator

from .. import config
from ..hierarchy import roman_to_int
from . import Assessment

# "2", "2.1", "2.1.3", "A.1", "IV" at the start of a heading, followed by the title. Bare letters ("A Proofs") are
# left out: a title can start with the article "A".
_NUMBER = re.compile(r"^\s*(?:(?P<arabic>\d{1,2}(?:\.\d{1,2})*)|(?P<letter>[A-Z](?:\.\d{1,2})+)|(?P<roman>[IVX]{1,5}))"
                     r"\.?\s+(?=\S)")
_CITATION_START = re.compile(r"^\s*(?:\[\d{1,3}\]|\d{1,3}\.\s+[A-Z][\w'\-]+,)")  # "[12] Smith" / "12. Smith,"
_AUTHOR_START = re.compile(r"^\s*(?:[A-Z][\w'\-]+,\s+(?:[A-Z]\.\s*){1,3}|[A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Z][\w'\-]+,)")
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_VENUE = re.compile(r"\bet al\.|\barXiv\b|\bdoi\b|\bProceedings\b|\bProc\.|\bJournal\b|\bConference\b|\bpp\.\s*\d|"
                    r"\bvol\.\s*\d|\bIn\s+[A-Z]|\bpreprint\b|\bPress\b|\bTrans\.", re.I)
_ACK = re.compile(r"^\s*(?:acknowledg|(?:we|the authors?)\s+(?:would\s+like\s+to\s+|wish\s+to\s+)?(?:thank|acknowledge|"
                  r"are\s+grateful)|this\s+(?:work|research|project|study)\s+(?:was|is|has\s+been)\s+(?:\w+\s+)?"
                  r"(?:supported|funded))", re.I)


def _walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def heading_number(title: str) -> tuple[str, tuple[int, ...]] | None:
    """``(series, components)`` of a numbered heading: ("arabic", (2, 1)) for "2.1 Data", ("A", (1,)) for
    "A.1 Proof", ("roman", (4,)) for "IV. Results"; None if the heading has no number."""
    m = _NUMBER.match(title or "")
    if not m:
        return None
    if m.group("arabic"):
        return "arabic", tuple(int(x) for x in m.group("arabic").split("."))
    if m.group("letter"):
        letter, *rest = m.group("letter").split(".")
        return letter, tuple(int(x) for x in rest)
    value = roman_to_int(m.group("roman"))
    return ("roman", (value,)) if value else None


def numbering(tree: dict[str, Any]) -> Assessment:
    """The numbered headings, in reading order, follow each other: after 2.1 comes 2.2, 2.1.1 or 3 (or 3.1 when
    the heading 3 was lost). A skip (2.1 -> 2.3), a step back or a repeat (3 -> 2, 2.2 -> 2.2) means a heading
    lost, read twice or put out of order by the parsing."""
    seen: dict[tuple[str, tuple[int, ...]], int] = {}  # (series, prefix) -> last value seen under that prefix
    numbered, problems = 0, []
    for node in _walk(tree):
        if not node.get("level"):
            continue
        parsed = heading_number(node.get("title", ""))
        if not parsed:
            continue
        numbered += 1
        series, comps = parsed
        prefix, value = comps[:-1], comps[-1]
        last = seen.get((series, prefix), 0)
        label = node["title"][:60]
        if value <= last:
            problems.append(f"'{label}' after {'.'.join(map(str, prefix + (last,)))}")
        elif value > last + 1:
            problems.append(f"'{label}' skips {'.'.join(map(str, prefix + (last + 1,)))}")
        seen[(series, prefix)] = value
        for key in [k for k in seen if k[0] == series and len(k[1]) > len(prefix) and k[1][:len(prefix)] == prefix
                    and k[1][len(prefix):len(prefix) + 1] != (value,)]:
            del seen[key]  # a new section resets the numbering of its sub-sections
    if not numbered:
        return Assessment("hierarchy.numbering", None, rationale="no numbered heading")
    return Assessment("hierarchy.numbering", len(problems), passed=not problems, metadata={"problems": problems},
                      rationale=f"{numbered} numbered headings, {len(problems)} out of sequence"
                                + (": " + "; ".join(problems[:6]) if problems else ""))


def reference_like(text: str) -> bool:
    """A bibliography entry: "[12] ..." / "12. Smith, ..." or an author list, with a year and a venue marker."""
    text = (text or "").strip()
    if not text or len(text.split()) > 120:
        return False
    if _CITATION_START.match(text):
        return True
    return bool(_AUTHOR_START.match(text) and _YEAR.search(text) and _VENUE.search(text))


def references_excluded(tree: dict[str, Any], pattern: str | None = None) -> Assessment:
    """The bibliography and the acknowledgements are kept out of the index (``HGRAG_EXCLUDED_SECTIONS``, matched on
    the headings of a section). Fails when a section that is indexed holds a bibliography (3+ entries, at least
    half of its paragraphs and list items) or an acknowledgement paragraph: a lost "References" heading, or one
    Docling did not read as a heading."""
    pattern = config.EXCLUDED_SECTIONS if pattern is None else pattern
    excluded = re.compile(pattern, re.I) if pattern else None
    leaks, kept_out = [], []

    def is_excluded(path: list[str]) -> bool:
        return bool(excluded) and any(excluded.match(h.strip()) for h in path)

    def visit(node: dict[str, Any], path: list[str]) -> None:
        here = path + ([node["title"]] if node.get("level") else [])
        if is_excluded(here):
            if node.get("level") and not is_excluded(path):
                kept_out.append(node["title"])
            return
        # items Docling labelled "reference" are never chunked (HGRAG_CHUNK_LABELS): only paragraphs and list items leak
        texts = [e for e in node.get("elements", []) if e.get("kind") in ("paragraph", "list_item")]
        n_refs = sum(1 for e in texts if reference_like(e.get("text") or ""))
        if n_refs >= 3 and n_refs >= len(texts) / 2:
            leaks.append({"section": " > ".join(here) or "(root)", "kind": "bibliography", "elements": n_refs})
        acks = [e for e in texts if _ACK.match(e.get("text") or "")]
        if acks:
            leaks.append({"section": " > ".join(here) or "(root)", "kind": "acknowledgements", "elements": len(acks)})
        for child in node.get("children", []):
            visit(child, here)

    visit(tree, [])
    detail = "; ".join(f"{x['kind']} in '{x['section'][-80:]}' ({x['elements']})" for x in leaks)
    return Assessment("hierarchy.references_excluded", len(leaks), passed=not leaks,
                      metadata={"excluded_sections": kept_out, "leaks": leaks},
                      rationale=(f"sections kept out of the index: {', '.join(kept_out) or 'none'}"
                                 + (f"; indexed by mistake: {detail}" if leaks else "")))


# ------------------------------------------------------------------------------------------- LLM judge
OUTLINE_JUDGE_SYSTEM = """You check the outline of a document rebuilt from a PDF by a parser. Each line is a heading, \
indented by its level, with its page and the number of elements (paragraphs, tables, figures...) filed under it. \
The text of each section is filed under the heading it follows, so an error in the outline puts text in the wrong \
section.

List in "problems" every heading that is out of place: a heading at the wrong level (a sub-section shown as a \
section or the reverse), a line that is not a heading (a sentence, a figure or table caption, an author name or \
affiliation, a running header), headings out of order, or a heading that is obviously missing (numbering that \
jumps, an abstract or a references section that is absent while the body suggests it). At most 8 problems, the \
most harmful first.
These are normal, not problems: the document title as the first top-level line; a section with few or no elements \
of its own followed by sub-sections (an introductory paragraph); unnumbered front or back matter (Abstract, \
Acknowledgements, References, Appendix) at the top level.
Score 1-5: 5 = a sound outline; 4 = one minor problem; 3 = a few problems that move some text into the wrong section; \
2 = many problems; 1 = no usable structure.

Reply with ONLY a JSON object: {"problems": [{"heading": "...", "problem": "..."}], "score": <1-5>, \
"rationale": "<one sentence>"}"""
MAX_OUTLINE_LINES = 300


def outline(tree: dict[str, Any]) -> str:
    lines = []
    for node in _walk(tree):
        level = node.get("level") or 0
        if level:
            lines.append(f"{'  ' * (level - 1)}{node['title'][:120]}  (p.{node.get('page_no')}, "
                         f"{len(node.get('elements', []))} el.)")
    if len(lines) > MAX_OUTLINE_LINES:
        lines = lines[:MAX_OUTLINE_LINES] + [f"... ({len(lines) - MAX_OUTLINE_LINES} more headings)"]
    return "\n".join(lines)


def judge_outline(tree: dict[str, Any], *, model: str | None = None, api_key: str | None = None) -> Assessment:
    from .. import generation
    from .description import JudgeReplyError, parse_reply

    model = model or config.ASSESS_JUDGE_MODEL
    cfg = generation.LLMConfig(provider="anthropic", model=model, max_tokens=2500, send_images=False, api_key=api_key,
                               thinking=generation.judge_thinking(model))
    if not cfg.api_key:
        return Assessment("hierarchy.outline", None, source="llm_judge", source_id=model,
                          error="no API key for the judge (ANTHROPIC_API_KEY)")
    text = outline(tree)
    if not text:
        return Assessment("hierarchy.outline", 1, source="llm_judge", source_id=model, passed=False,
                          rationale="no heading at all: the whole document is one section")
    last: Exception | None = None
    for attempt in range(2):
        try:
            reply, _usage = generation.complete(cfg, OUTLINE_JUDGE_SYSTEM, f"OUTLINE:\n{text}"
                                                + ("\n\nReply with the JSON object only." if attempt else ""))
            data = parse_reply(reply, ("score",))
            break
        except (JudgeReplyError, generation.GenerationError) as exc:
            last = exc
    else:
        return Assessment("hierarchy.outline", None, source="llm_judge", source_id=model, error=str(last))
    return Assessment("hierarchy.outline", data["score"], source="llm_judge", source_id=model,
                      passed=data["score"] >= config.ASSESS_JUDGE_OK, rationale=str(data.get("rationale", "")),
                      metadata={"problems": data.get("problems") or []})


def assess(tree: dict[str, Any], *, judge: bool = False, judge_model: str | None = None,
           api_key: str | None = None) -> list[Assessment]:
    out = [numbering(tree), references_excluded(tree)]
    if judge:
        out.append(judge_outline(tree, model=judge_model, api_key=api_key))
    return out
