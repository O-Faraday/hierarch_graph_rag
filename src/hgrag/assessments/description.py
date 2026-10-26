"""Step 2 - descriptions written by the models: figures, tables, formulas (LaTeX and plain-English reading).

Deterministic: each description or reading is present, long enough and not a refusal; each LaTeX transcription is
well formed. LLM judge (vision, on a sample): a figure description against its image; a formula transcription and
its reading against the crop of the formula.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .. import config, tracing
from . import Assessment, sampled

REFUSAL = re.compile(
    r"\b(?:I(?:'m| am)? (?:cannot|can't|can not|unable to|not able to)"
    r"|(?:unable|impossible) to (?:see|read|view|determine|identify|describe)"
    r"|no (?:image|figure|table|formula) (?:is |was )?(?:provided|attached|visible|shown)"
    r"|as an AI)\b", re.I)


def text_problem(text: str | None, min_words: int) -> str | None:
    """Why a generated text is unusable, or None."""
    text = (text or "").strip()
    if not text:
        return "missing"
    if REFUSAL.search(text):
        return "refusal"
    n = len(text.split())
    return f"too short ({n} words)" if n < min_words else None


def _share_check(name: str, kind: str, results: list[tuple[str, str | None]]) -> Assessment:
    """One record per document and kind: share of the items without problem, the failing items listed."""
    if not results:
        return Assessment(name, None, rationale=f"no {kind}")
    bad = [(ref, why) for ref, why in results if why]
    detail = "; ".join(f"{ref}: {why}" for ref, why in bad[:10]) + (" ..." if len(bad) > 10 else "")
    return Assessment(name, round(1 - len(bad) / len(results), 3), passed=not bad,
                      metadata={"failed": [{"item": r, "problem": w} for r, w in bad]},
                      rationale=f"{len(results) - len(bad)}/{len(results)} {kind} usable" + (f" ({detail})" if bad else ""))


def description_checks(figures: list[dict[str, Any]], tables: list[dict[str, Any]], formulas: list[dict[str, Any]],
                       *, min_words: int | None = None) -> list[Assessment]:
    """Present, long enough, not a refusal: figure and table descriptions, formula readings. A figure without PNG is
    not described by design and is left out; a description error recorded by the describe step counts as missing."""
    min_words = config.ASSESS_MIN_DESCRIPTION_WORDS if min_words is None else min_words

    def text(rec: dict[str, Any], key: str) -> str | None:
        value = rec.get(key)
        return value.get("text") if isinstance(value, dict) else value

    fig = [(f.get("self_ref") or f"figure {f.get('figure_index')}", text_problem(text(f, "description"), min_words))
           for f in figures if f.get("png_path")]
    tab = [(t.get("self_ref") or f"table {t.get('table_index')}", text_problem(text(t, "description"), min_words))
           for t in tables]
    form = [(r["self_ref"], text_problem(text(r, "reading"), min_words)) for r in formulas if r.get("latex")]
    return [_share_check("describe.figure_descriptions", "figure descriptions", fig),
            _share_check("describe.table_descriptions", "table descriptions", tab),
            _share_check("describe.formula_readings", "formula readings", form)]


_ENV = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")
_LEFT_RIGHT = re.compile(r"\\(left|right)(?![a-zA-Z])")
_LOOP = re.compile(r"(\\[a-zA-Z]+\s*)\1{11,}")  # the same command 12+ times in a row: the model looped until max_tokens
_LESS_THAN = ("<", r"\lt", r"\nless", r"\ll")


def latex_problems(latex: str | None, original: str | None = None) -> list[str]:
    """Structural defects of a LaTeX transcription: unbalanced braces, \\left without \\right, unclosed
    environments, Markdown fence or unpaired $, a command repeated in a loop, and, against the PDF text of the
    formula (``original``), a '<' lost (on the ragbench papers the transcription was cut right before it)."""
    latex = (latex or "").strip()
    if not latex:
        return ["no LaTeX"]
    out = []
    body = re.sub(r"\\\\|\\[{}$]", "", latex)  # escaped braces, dollars and line breaks are not delimiters
    if _LOOP.search(latex):
        out.append("command repeated in a loop (output cut)")
    if original and "<" in original and not any(s in latex for s in _LESS_THAN):
        out.append("'<' of the PDF text missing (output cut?)")
    depth = 0
    for ch in body:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            break
    if depth:
        out.append("unbalanced braces")
    sides = [m.group(1) for m in _LEFT_RIGHT.finditer(latex)]
    if sides.count("left") != sides.count("right"):
        out.append(f"\\left x{sides.count('left')} / \\right x{sides.count('right')}")
    stack: list[str] = []
    for kind, env in _ENV.findall(latex):
        if kind == "begin":
            stack.append(env.strip())
        elif not stack or stack.pop() != env.strip():
            out.append(f"\\end{{{env}}} without its \\begin")
            break
    else:
        if stack:
            out.append(f"\\begin{{{stack[-1]}}} not closed")
    if "```" in latex or body.count("$") % 2:
        out.append("Markdown fence or unpaired $ in the LaTeX")
    return out


def latex_checks(formula_refs: list[str], formulas: list[dict[str, Any]]) -> Assessment:
    """Every formula of the document has a well-formed LaTeX transcription. Paired $...$ are not a defect: they mark a
    sentence with inline math that Docling labelled formula (listed in ``metadata["prose"]``)."""
    recs = {r["self_ref"]: r for r in formulas}
    results = []
    for ref in formula_refs:
        rec = recs.get(ref) or {}
        results.append((ref, "; ".join(latex_problems(rec.get("latex"), rec.get("original_text"))) or None))
    out = _share_check("describe.latex_wellformed", "formula transcriptions", results)
    out.metadata["prose"] = [r for r, rec in recs.items() if "$" in re.sub(r"\\\$", "", rec.get("latex") or "")]
    return out


# ------------------------------------------------------------------------------------------- LLM judges
FIGURE_JUDGE_SYSTEM = """You check a machine-generated description of a figure against the figure itself (the image). \
The description will be indexed for search, so what matters is that every statement is true to the image.

List in "unsupported" every statement of the description that the image contradicts or does not show (wrong or \
invented value, axis, label, series, trend). An approximate reading of a value is fine.
List in "missing" the main content of the figure that the description leaves out (at most 3 items).
Score 1-5: 5 = every statement true to the image and the main content covered; 4 = true but a minor omission or a \
loose approximation; 3 = one unsupported statement or a major omission; 2 = several unsupported statements; \
1 = mostly wrong or about another figure.

Reply with ONLY a JSON object: {"unsupported": ["..."], "missing": ["..."], "score": <1-5>, "rationale": "<one sentence>"}"""

FORMULA_JUDGE_SYSTEM = """You check two machine-generated texts against the image of a formula cropped from a PDF: \
its LaTeX transcription, and a plain-English reading of that formula written for a search index.

latex_score 1-5: 5 = the LaTeX renders the same formula as the image; 4 = a cosmetic difference (spacing, font \
command, equation number); 3 = one wrong symbol, index or exponent; 2 = several wrong parts; 1 = another formula \
or garbage.
reading_score 1-5: 5 = the reading states what the formula states, with nothing added; 4 = correct but vague; \
3 = one wrong relation, condition or limit; 2 = mostly wrong; 1 = about something else. The reading was written from \
the formula AND the text around it, which you also receive: names of symbols, conditions and purposes that this text \
gives are not errors. Count as errors only the statements that neither the formula nor this text support.

Reply with ONLY a JSON object: {"latex_errors": ["..."], "latex_score": <1-5>, "reading_errors": ["..."], \
"reading_score": <1-5>, "rationale": "<one sentence>"}"""


class JudgeReplyError(ValueError):
    pass


def parse_reply(text: str, scores: tuple[str, ...]) -> dict[str, Any]:
    """The first JSON object of the reply (a judge sometimes writes a note after it), with its scores checked."""
    start = (text or "").find("{")
    if start < 0:
        raise JudgeReplyError(f"no JSON in the judge reply: {(text or '')[:200]!r}")
    try:
        try:
            data, _end = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            data = {}
        for key in (*scores, "rationale"):  # malformed JSON (a quote in a heading, a brace closed too early)
            if key not in data and (m := re.search(rf'"{key}"\s*:\s*(\d|"(?:[^"\\]|\\.)*")', text)):
                data[key] = json.loads(m.group(1))
        for key in scores:
            data[key] = int(data[key])
            if not 1 <= data[key] <= 5:
                raise JudgeReplyError(f"{key} out of range: {data[key]}")
    except (ValueError, KeyError, TypeError) as exc:
        raise JudgeReplyError(f"invalid judge reply ({exc}): {text[:200]!r}") from exc
    return data


def judge_image(cfg: Any, png: Path, system: str, prompt: str, scores: tuple[str, ...]) -> dict[str, Any]:
    """One judge call with an image; one retry if the reply is not the expected JSON."""
    from ..describe import _ask_image

    last: Exception | None = None
    for attempt in range(2):
        text = _ask_image(cfg, png, prompt + ("\n\nReply with the JSON object only." if attempt else ""), system)
        try:
            return parse_reply(text, scores)
        except JudgeReplyError as exc:
            last = exc
    raise last  # type: ignore[misc]


JUDGE_MAX_TOKENS = 1500


def judge_config(model: str | None = None, api_key: str | None = None) -> Any:
    """Vision judge; thinking off on Claude Haiku (``generation.judge_thinking``)."""
    from .. import generation, vlm_bench

    spec = model or config.ASSESS_JUDGE_MODEL
    cfg = vlm_bench.model_config(spec if ":" in spec else f"anthropic:{spec}", api_key, max_tokens=JUDGE_MAX_TOKENS)
    cfg.thinking = generation.judge_thinking(cfg.model) if cfg.provider == "anthropic" else None
    return cfg


def _figure_judgement(cfg: Any, doc_dir: Path, rec: dict[str, Any]) -> list[Assessment]:
    from ..describe import png_file

    ref = rec.get("self_ref") or f"figure {rec.get('figure_index')}"
    desc = (rec.get("description") or {}).get("text") or ""
    try:
        data = judge_image(cfg, png_file(doc_dir, rec["png_path"]), FIGURE_JUDGE_SYSTEM,
                           f"DESCRIPTION TO CHECK:\n{desc}", ("score",))
    except Exception as exc:  # noqa: BLE001
        return [Assessment("describe.figure_fidelity", None, source="llm_judge", target=ref, source_id=cfg.model,
                           error=str(exc))]
    return [Assessment("describe.figure_fidelity", data["score"], source="llm_judge", target=ref, source_id=cfg.model,
                       passed=data["score"] >= config.ASSESS_JUDGE_OK, rationale=str(data.get("rationale", "")),
                       metadata={"unsupported": data.get("unsupported") or [], "missing": data.get("missing") or []})]


def _formula_judgement(cfg: Any, doc_dir: Path, rec: dict[str, Any], context: tuple[str, str] = ("", "")) -> list[Assessment]:
    from ..describe import png_file

    ref = rec["self_ref"]
    reading = (rec.get("reading") or {}).get("text") or "(none)"
    before, after = context
    prompt = (f"TEXT BEFORE THE FORMULA:\n{before or '(none)'}\n\nTEXT AFTER THE FORMULA:\n{after or '(none)'}\n\n"
              f"LATEX TRANSCRIPTION:\n{rec['latex']}\n\nREADING:\n{reading}")
    try:
        data = judge_image(cfg, png_file(doc_dir, rec["png_path"]), FORMULA_JUDGE_SYSTEM, prompt,
                           ("latex_score", "reading_score"))
    except Exception as exc:  # noqa: BLE001
        return [Assessment(f"describe.formula_{k}", None, source="llm_judge", target=ref, source_id=cfg.model,
                           error=str(exc)) for k in ("latex", "reading")]
    why = str(data.get("rationale", ""))
    return [Assessment(f"describe.formula_{k}", data[f"{k}_score"], source="llm_judge", target=ref, source_id=cfg.model,
                       passed=data[f"{k}_score"] >= config.ASSESS_JUDGE_OK, rationale=why,
                       metadata={"errors": data.get(f"{k}_errors") or []})
            for k in ("latex", "reading") if k == "latex" or rec.get("reading")]


def judge_items(figures: list[dict[str, Any]], formulas: list[dict[str, Any]], doc_dir: str | Path, *, doc_key: str,
                sample: float, max_judged: int, model: str | None = None, api_key: str | None = None,
                workers: int = 4, contexts: dict[str, tuple[str, str]] | None = None) -> list[Assessment]:
    """LLM judges on a deterministic sample (``sample`` share, at most ``max_judged`` per kind) of the described
    figures and the transcribed formulas that have their crop. ``contexts``: ``self_ref -> (text before, text
    after)`` of the formulas, the text their reading was written from (``describe.formula_context``)."""
    from ..describe import png_file

    doc_dir = Path(doc_dir)

    def has_png(rec: dict[str, Any]) -> bool:
        path = png_file(doc_dir, rec.get("png_path"))
        return path is not None and path.exists()

    figs = [f for f in figures if (f.get("description") or {}).get("text") and has_png(f)
            and sampled(f"{doc_key}{f.get('self_ref')}", sample)][:max_judged]
    forms = [r for r in formulas if r.get("latex") and has_png(r) and sampled(f"{doc_key}{r['self_ref']}", sample)][:max_judged]
    if not figs and not forms:
        return []
    cfg = judge_config(model, api_key)
    if not cfg.api_key:
        return [Assessment("describe.judge", None, source="llm_judge", source_id=cfg.model,
                           error="no API key for the judge (ANTHROPIC_API_KEY)")]
    contexts = contexts or {}
    jobs = [lambda f=f: _figure_judgement(cfg, doc_dir, f) for f in figs]
    jobs += [lambda r=r: _formula_judgement(cfg, doc_dir, r, contexts.get(r["self_ref"], ("", ""))) for r in forms]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = [f.result() for f in [tracing.submit(pool, job) for job in jobs]]
    return [a for recs in results for a in recs]


def assess(doc: Any, figures: list[dict[str, Any]], tables: list[dict[str, Any]], formulas: list[dict[str, Any]],
           doc_dir: str | Path, *, judge: bool = False, sample: float = 0.2, max_judged: int = 10,
           judge_model: str | None = None, api_key: str | None = None, workers: int = 4) -> list[Assessment]:
    from ..describe import formula_context, formula_items

    out = description_checks(figures, tables, formulas)
    items = formula_items(doc)
    refs = [i.self_ref for i in items]
    out.append(latex_checks(refs, formulas) if formulas or config.FORMULA_MODE == "vlm"
               else Assessment("describe.latex_wellformed", None, rationale="formulas not transcribed (HGRAG_FORMULA_MODE)"))
    if judge:
        out += judge_items(figures, formulas, doc_dir, doc_key=Path(doc_dir).name, sample=sample, max_judged=max_judged,
                           model=judge_model, api_key=api_key, workers=workers, contexts=formula_context(doc, items))
    return out
