"""Second opinion on the low judge scores of a ragbench run: is the context good, and if so, is the answer really wrong?

The first judge (``evaluation.judge_answer``) only sees the question, the reference answer and the generated answer: it
cannot tell a retrieval failure from a generation failure, nor a wrong reference from a wrong answer. For every
question scored below ``threshold``:

1. the **context judge** reads the whole PDF and the context the generator received (sections and figures) and scores
   how far that context holds what is needed to answer (``context_score`` 1-5);
2. if ``context_score >= REFINE_CONTEXT_OK``, the **generation judge** grades the answer again with the PDF and the
   context at hand, and gives ``refine_score`` and a ``refine_cause`` (generation, reference, judge, none);
   otherwise ``refine_score`` keeps the first score and the cause is ``context``.

``score`` is never overwritten: the summary reports both. The context is rebuilt from the run's own output (same
sections, same rendering as for the generator), so an old run can be refined without regenerating anything.
The PDF, the figures and the context come first in the request with a cache breakpoint: the second call of a
question reads them from the prompt cache.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from . import config, generation, ragbench
from .evaluation import JudgeError, _parse_judgement
from .generation import GenerationError, LLMConfig
from .ids import compute_doc_id

CAUSES = ("generation", "reference", "judge", "none")
REFINE_FIELDS = ("context_score", "context_rationale", "refine_score", "refine_cause", "refine_rationale", "refine_usage",
                 "refine_error", "refine_sections")
# The Messages API takes 32 MB per request, base64 included (+1/3): keep the raw PDF under 23 MB.
MAX_PDF_BYTES = 23_000_000
FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1")

SYSTEM = """You are an expert reviewer of a retrieval-augmented question answering system built on scientific papers.
You receive the full source PDF, then the CONTEXT that the system retrieved from its own parsing of that PDF and gave \
to the answering model (numbered sources; figures are attached as images labelled with their source number), then a \
TASK. The PDF is the ground truth for what the paper says, including the values shown in its figures and tables.
Reply with ONLY a JSON object, no other text."""

CONTEXT_TASK = """TASK: judge the CONTEXT, not any answer.

QUESTION:
{question}

REFERENCE ANSWER (from the benchmark, may be imprecise):
{expected}

Find where the PDF answers the question. Then score how far the CONTEXT holds that information, as text, as a table or \
in an attached figure:
5 = everything needed is in the context and faithful to the PDF.
4 = the essential is there; a minor detail is missing or can only be read approximately (e.g. a value read off a chart).
3 = part of the needed information is there, part is missing.
2 = only related material (the right paper or section) without the information needed.
1 = nothing relevant, or the context contradicts the PDF on the point asked (parsing error, wrong figure description).

In the rationale, say where the PDF answers (section, page, figure or table) and what the context lacks or distorts.
{{"rationale": "<two or three sentences>", "score": <integer 1-5>}}"""

GENERATION_TASK = """TASK: the context was judged sufficient. Grade the GENERATED ANSWER again.

QUESTION:
{question}

REFERENCE ANSWER (from the benchmark, may be imprecise):
{expected}

GENERATED ANSWER:
{answer}

FIRST JUDGE (it compared the answer with the reference only): {score}/5 - {rationale}

Grade the answer against what the PDF says:
- if the reference answer is imprecise or wrong according to the PDF, follow the PDF;
- a value read off a chart within about 5 % of the PDF value, or a precise value consistent with a range of the \
reference, is correct;
- extra facts are fine when the PDF supports them; a statement that the PDF or the context contradicts is an error.

Scale:
5 = fully correct and complete. 4 = correct, minor omission or imprecision. 3 = partially correct.
2 = mostly wrong, with a small relevant element. 1 = wrong, contradicting the PDF, or no real answer.

Cause of the gap with the first score:
"reference" = the reference answer is wrong or imprecise according to the PDF;
"judge" = the first judge was too strict with a correct answer;
"generation" = the answer misreads, omits or contradicts the context;
"none" = the first score stands.
{{"rationale": "<two or three sentences>", "score": <integer 1-5>, "cause": "<reference|judge|generation|none>"}}"""


def refine_config(model: str | None = None, api_key: str | None = None) -> LLMConfig:
    model = model or config.REFINE_MODEL
    return LLMConfig(provider="anthropic", model=model, max_tokens=8000, send_images=True, api_key=api_key, timeout=300.0,
                     fallbacks="default" if model in FALLBACK_MODELS else None)


def candidates(rows: list[dict[str, Any]], threshold: int, force: bool = False) -> list[dict[str, Any]]:
    """Questions to judge again: an answer and a score below ``threshold``, not refined yet (unless ``force``)."""
    return [r for r in rows if r.get("answer") is not None and r.get("score") is not None and r["score"] < threshold
            and (force or r.get("refine_score") is None)]


# ------------------------------------------------------------------------------------------ context
def _find_by_path(tree: dict[str, Any], path: str) -> dict[str, Any] | None:
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.get("path") == path:
            return node
        stack.extend(node.get("children", []))
    return None


def rebuild_sections(row: dict[str, Any], *, output_dir: Path, pdf_dir: Path) -> list[dict[str, Any]]:
    """The sections the generator received, rebuilt from the run's outputs (exact ids if recorded, else by section path)."""
    from .retrieval import load_hierarchy, reconstruct_section

    trees: dict[str, Any] = {}
    doc_ids: dict[str, str | None] = {}
    out = []
    for h in row.get("retrieved") or []:
        paper = h.get("paper")
        pdf = pdf_dir / f"{ragbench.safe_name(paper)}.pdf" if paper else None
        doc_id = h.get("doc_id")
        if not doc_id:
            if paper not in doc_ids:
                doc_ids[paper] = compute_doc_id(pdf) if pdf and pdf.is_file() else None
            doc_id = doc_ids[paper]
        if not doc_id:
            continue
        if doc_id not in trees:
            try:
                trees[doc_id] = load_hierarchy(doc_id, output_dir)
            except FileNotFoundError:
                trees[doc_id] = None
        tree = trees[doc_id]
        if tree is None:
            continue
        ref = h.get("section_ref")
        if not ref:
            node = _find_by_path(tree, ragbench._path(h["section_path"]))
            if node is None:
                continue
            ref = node["ref"]
        try:
            sec = reconstruct_section(doc_id, ref, output_dir=output_dir, tree=tree)
        except KeyError:
            continue
        out.append({**sec, "score": h.get("score"), "source_pdf": str(pdf) if pdf else None})
    return out


def _pdf_block(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    data = path.read_bytes()
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"{path.name}: {len(data) / 1e6:.0f} MB, over the {MAX_PDF_BYTES / 1e6:.0f} MB the API accepts")
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                           "data": base64.standard_b64encode(data).decode("ascii")}}


def _shared_parts(cfg: LLMConfig, pdf: Path, msgs: dict[str, Any]) -> list[dict[str, Any]]:
    """PDF, figures, then the context with the cache breakpoint: identical for the two judges of a question."""
    parts = [_pdf_block(pdf)]
    for img in msgs["images"]:
        media, data = generation._encode_image(img["path"], cfg.max_image_edge)
        parts.append({"type": "text", "text": f"Figure attached to source [{img['n']}]: {img['caption']}"})
        parts.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
    context = msgs["sources_text"] if msgs["sources"] else "Sources: (none could be rebuilt for this question)"
    parts.append({"type": "text", "text": "CONTEXT\n\n" + context, "cache_control": {"type": "ephemeral"}})
    return parts


# ------------------------------------------------------------------------------------------- judges
def _ask(cfg: LLMConfig, parts: list[dict[str, Any]], task: str, *, with_cause: bool) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(2):
        text, usage = generation.complete(cfg, SYSTEM, parts + [{"type": "text", "text": task + (
            "\n\nReply with the JSON object only." if attempt else "")}])
        try:
            j = _parse_judgement(text)
            if with_cause:
                m = re.search(r'"cause"\s*:\s*"(\w+)"', text)
                if not m or m.group(1) not in CAUSES:
                    raise JudgeError(f"no valid cause in the reply: {text[:200]!r}")
                j["cause"] = m.group(1)
            return {**j, "usage": usage}
        except JudgeError as exc:
            last = exc
    raise last  # type: ignore[misc]


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for k, v in usage.items():
        if isinstance(v, int):
            total[k] = total.get(k, 0) + v


def refine_row(row: dict[str, Any], cfg: LLMConfig, *, output_dir: Path, pdf_dir: Path,
               context_ok: int | None = None) -> dict[str, Any]:
    """The new fields of one question: context judge, then (if the context is good) generation judge."""
    context_ok = config.REFINE_CONTEXT_OK if context_ok is None else context_ok
    sections = rebuild_sections(row, output_dir=output_dir, pdf_dir=pdf_dir)
    msgs = generation.build_messages(row["question"], sections, cfg.max_context_chars, cfg.max_images)
    parts = _shared_parts(cfg, pdf_dir / f"{ragbench.safe_name(row['paper'])}.pdf", msgs)
    usage: dict[str, int] = {}
    c = _ask(cfg, parts, CONTEXT_TASK.format(question=row["question"], expected=row.get("expected_answer", "")),
             with_cause=False)
    _add_usage(usage, c["usage"])
    out: dict[str, Any] = {"context_score": c["score"], "context_rationale": c["rationale"], "refine_usage": usage,
                           "refine_sections": f"{len(msgs['sources'])}/{len(row.get('retrieved') or [])}"}
    if c["score"] < context_ok:
        return {**out, "refine_score": row["score"], "refine_cause": "context",
                "refine_rationale": f"context judged insufficient ({c['score']}/5): the first score is kept"}
    g = _ask(cfg, parts, GENERATION_TASK.format(question=row["question"], expected=row.get("expected_answer", ""),
                                                answer=row["answer"] or "(empty)", score=row["score"],
                                                rationale=row.get("rationale", "")), with_cause=True)
    _add_usage(usage, g["usage"])
    return {**out, "refine_score": g["score"], "refine_cause": g["cause"], "refine_rationale": g["rationale"]}


# ---------------------------------------------------------------------------------------------- run
def _write_rows(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    f = run_dir / "results.jsonl"
    tmp = f.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    os.replace(tmp, f)


def refine_run(run_dir: str | Path, *, threshold: int | None = None, model: str | None = None, api_key: str | None = None,
               cfg: LLMConfig | None = None, workdir: str | Path | None = None, force: bool = False, workers: int = 4,
               context_ok: int | None = None,
               on_progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Judge again the questions of a run scored below ``threshold``; results.jsonl, summary.json, questions.md updated.

    ``results.jsonl`` is rewritten after each question, so an interrupted run resumes where it stopped."""
    run_dir = Path(run_dir)
    workdir = Path(workdir or ragbench.RAGBENCH_DIR)
    f = run_dir / "results.jsonl"
    if not f.is_file():
        raise ragbench.RagBenchError(f"{f} not found")
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    threshold = config.REFINE_THRESHOLD if threshold is None else threshold
    context_ok = config.REFINE_CONTEXT_OK if context_ok is None else context_ok
    cfg = cfg or refine_config(model, api_key)
    todo = candidates(rows, threshold, force)
    if todo and not cfg.api_key:
        raise ragbench.RagBenchError("ANTHROPIC_API_KEY is not set (needed by the refine judges).")
    emit = on_progress or (lambda _n, _t, _m: None)
    for r in todo:
        for k in REFINE_FIELDS:
            r.pop(k, None)
    n_errors = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(refine_row, r, cfg, output_dir=workdir / "output", pdf_dir=workdir / "pdfs",
                               context_ok=context_ok): r for r in todo}
        for n, fut in enumerate(as_completed(futures), start=1):
            r = futures[fut]
            try:
                r.update(fut.result())
                msg = f"{n}/{len(todo)} · {r['id'][:8]} · context {r['context_score']}/5 · score {r['score']} → {r['refine_score']} ({r['refine_cause']})"
            except (GenerationError, JudgeError, OSError, ValueError, KeyError) as exc:
                r["refine_error"] = f"{type(exc).__name__}: {exc}"
                n_errors += 1
                msg = f"{n}/{len(todo)} · {r['id'][:8]} · error: {exc}"
            _write_rows(run_dir, rows)  # main thread only: no lock needed
            emit(n, len(todo), msg)
    info = {"model": cfg.model, "threshold": threshold, "context_ok": context_ok, "n_candidates": len(todo),
            "n_errors": n_errors, "seconds": round(time.perf_counter() - t0, 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    sf = run_dir / "summary.json"
    if sf.exists():
        summary = json.loads(sf.read_text(encoding="utf-8"))
        summary["refine"] = info
        sf.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    ragbench.write_qa_report(run_dir, workdir / "pdfs")
    return {**info, "rows": todo}
