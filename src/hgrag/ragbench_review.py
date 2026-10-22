"""Review of ragbench runs (used by the Streamlit pages): runs grouped by sample, rule-based diagnosis of every
question, per-run keep/exclude decisions and a list of improvement leads. No model, no API call.

The keep/exclude decisions live in ``<run>/review.json`` ({query id: {"keep": bool, "note": str}}): they only change the
"kept selection" metrics of that run, never the CLI summary nor the other runs. A question listed in
``data/ragbench_disputed.json`` starts as excluded.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import ragbench

REQUIRED = ("id", "paper", "source", "type", "question", "section_rank", "doc_rank")
REVIEW_FILE = "review.json"

# category -> (label, improvement lead); the order of CATEGORIES is the order of the rules in diagnose()
CATEGORIES: dict[str, tuple[str, str]] = {
    "error": ("Error", "The generation or the judge failed (see the error): rerun the question; make the judge reply parsing more robust."),
    "disputed": ("Disputed reference", "The reference answer was found wrong in the PDF: keep it out of the bench."),
    "ok": ("Correct", "Nothing to fix."),
    "doc_missed": ("Paper not retrieved", "The right paper is absent: hybrid (keyword / sparse) search for acronyms and rare terms, a stronger embedder."),
    "section_missed": ("Section not retrieved", "The paper is found, not the section: check how the reference heading was chunked, "
                       "hybrid search, more reranker candidates."),
    "refusal": ("Refusal", "The model declined although sources were given: loosen the prompt (partial answer with citations), "
                "check that the right section is in the context."),
    "figure": ("Figure reading", "The answer depends on a chart without written values: have the vision model extract approximate values "
               "in the figure descriptions; accept a tolerance in the reference."),
    "contradiction": ("Contradiction", "The answer contradicts the reference with the section in context: check the reference in the PDF "
                      "(candidate for exclusion), otherwise a reading error (table, sign, direction)."),
    "ranked_low": ("Section ranked low", "The right section is retrieved beyond rank 3: reranker, fewer distractor chunks, heading prefix."),
    "partial": ("Partial answer", "The section is there but the answer misses or adds facts: compare with the reference section, "
                "check the context budget and the truncation."),
}

_REFUSAL = re.compile(r"\b(cannot|can't|can not) (answer|determine)|do(es)? not (explain|address|describe|mention|discuss|report)", re.I)
_FIGURE = re.compile(r"\b(chart|plot|graph|figure|curve|bar|axis|approximate(ly)?|read from|no (data )?(labels|values))\b", re.I)


# --------------------------------------------------------------------------------------------- runs
def list_benches(runs_dir: str | Path | None = None) -> dict[str, list[Path]]:
    """{sample key: [run dirs, latest first]} for the runs that have a summary.json and a results.jsonl."""
    runs_dir = Path(runs_dir or ragbench.RAGBENCH_DIR / "runs")
    out: dict[str, list[Path]] = defaultdict(list)
    if not runs_dir.is_dir():
        return {}
    for d in sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        try:
            key = json.loads((d / "summary.json").read_text(encoding="utf-8"))["config"]["key"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (d / "results.jsonl").is_file():
            out[key].append(d)
    return dict(out)


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))


def load_results(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """(rows, problems): a line that is not JSON or lacks a required field is reported instead of breaking the page."""
    rows, problems = [], []
    lines = (Path(run_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {n}: not JSON ({exc})")
            continue
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"line {n} ({r.get('id', '?')}): missing {', '.join(missing)}")
            continue
        rows.append(r)
    return rows, problems


# ---------------------------------------------------------------------------------------- diagnosis
def diagnose(row: dict[str, Any], disputed: dict[str, str] | None = None) -> dict[str, str]:
    """One category per question, first matching rule wins (see CATEGORIES for the order)."""
    answer = (row.get("answer") or "")
    score = row.get("score")
    if row.get("error") or (score is None and row.get("answer") is not None):
        cat = "error"
    elif row["id"] in (disputed or {}):
        cat = "disputed"
    elif (score >= 4) if score is not None else (row.get("section_rank") or 99) <= 3:  # without --generate: retrieval only
        cat = "ok"
    elif row.get("doc_rank") is None:
        cat = "doc_missed"
    elif row.get("section_rank") is None:
        cat = "section_missed"
    elif _REFUSAL.search(answer[:300]):
        cat = "refusal"
    elif "image" in row.get("source", "") and _FIGURE.search(answer):
        cat = "figure"
    elif "contradict" in (row.get("rationale") or "").lower():
        cat = "contradiction"
    elif row["section_rank"] > 3:
        cat = "ranked_low"
    else:
        cat = "partial"
    label, fix = CATEGORIES[cat]
    return {"category": cat, "label": label, "fix": fix}


def improvements(rows: list[dict[str, Any]], disputed: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Failure categories (without 'ok'), most frequent first, with the lead and the sources concerned."""
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[diagnose(r, disputed)["category"]].append(r)
    out = []
    for cat, items in by_cat.items():
        if cat == "ok":
            continue
        label, fix = CATEGORIES[cat]
        sources = Counter(r["source"] for r in items)
        out.append({"category": cat, "label": label, "n": len(items), "share": round(len(items) / len(rows), 3),
                    "sources": ", ".join(f"{s} {c}" for s, c in sources.most_common()), "fix": fix,
                    "ids": [r["id"] for r in items]})
    return sorted(out, key=lambda x: (-x["n"], list(CATEGORIES).index(x["category"])))


# ------------------------------------------------------------------------------------------- review
def load_review(run_dir: str | Path, rows: list[dict[str, Any]], disputed: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """{query id: {"keep", "note"}} for every row: saved decisions first, else excluded if disputed, kept otherwise."""
    disputed = ragbench.load_disputed() if disputed is None else disputed
    f = Path(run_dir) / REVIEW_FILE
    saved = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    return {r["id"]: saved.get(r["id"]) or {"keep": r["id"] not in disputed, "note": disputed.get(r["id"], "")} for r in rows}


def save_review(run_dir: str | Path, review: dict[str, dict[str, Any]]) -> Path:
    """Atomic write (temporary file then replace), UTF-8."""
    f = Path(run_dir) / REVIEW_FILE
    fd, tmp = tempfile.mkstemp(dir=f.parent, prefix=".review-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(review, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, f)
    return f


def excluded(review: dict[str, dict[str, Any]]) -> dict[str, str]:
    """{query id: note} of the excluded questions, in the shape ragbench.summarize() takes as ``disputed``."""
    return {qid: v.get("note") or "excluded" for qid, v in review.items() if not v.get("keep", True)}


def corrected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows with ``score`` replaced by the second opinion (``refine_score``) where there is one."""
    return [{**r, "score": r["refine_score"]} if r.get("refine_score") is not None else r for r in rows]


def compare(rows: list[dict[str, Any]], review: dict[str, dict[str, Any]], k: int, *, corrected: bool = False) -> dict[str, Any]:
    """{"all": metrics of every question, "kept": metrics without the excluded ones} (ragbench.summarize).

    ``corrected``: score metrics on the refined score where it exists, else the first judge's."""
    s = ragbench.summarize(corrected_rows(rows) if corrected else rows, k, excluded(review))
    kept = s.pop("excluding_disputed", None)
    if kept is None:
        kept = {key: v for key, v in s.items() if key not in ("k_sections", "n_errors")}
        kept["excluded"] = []
    return {"all": s, "kept": kept}


def rank_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    cnt = Counter(r.get("section_rank") for r in rows)
    return {"1": cnt[1], "2-3": cnt[2] + cnt[3], "4-10": sum(cnt[i] for i in range(4, 11)), "absent": cnt[None]}


def pdf_path(row: dict[str, Any]) -> Path:
    return ragbench.RAGBENCH_DIR / "pdfs" / (ragbench.safe_name(row["paper"]) + ".pdf")


# ------------------------------------------------------------------------------------------- refine
REFINE_CAUSES = {"reference": "Reference wrong or imprecise", "judge": "First judge too strict",
                 "generation": "Generation error", "context": "Context insufficient", "none": "First score stands"}


def refine_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """First judge against the second opinion (hgrag.refine), on the questions that were judged again.

    ``matrix``: {first score: {refined score: count}}; ``rows``: one line per refined question, largest gain first."""
    refined = [r for r in rows if r.get("refine_score") is not None and r.get("score") is not None]
    out: dict[str, Any] = {"n": len(refined), "n_errors": sum(1 for r in rows if r.get("refine_error"))}
    if not refined:
        return out
    before = [r["score"] for r in refined]
    after = [r["refine_score"] for r in refined]
    out.update(mean_before=round(sum(before) / len(before), 2), mean_after=round(sum(after) / len(after), 2),
               n_raised_to_4_plus=sum(1 for b, a in zip(before, after) if b < 4 <= a),
               causes={c: n for c, n in Counter(r.get("refine_cause") or "none" for r in refined).most_common()})
    matrix: dict[int, dict[int, int]] = {s: {t: 0 for t in range(1, 6)} for s in range(1, 6)}
    for b, a in zip(before, after):
        matrix[b][a] += 1
    out["matrix"] = {s: v for s, v in matrix.items() if any(v.values())}
    out["rows"] = sorted(({"id": r["id"], "source": r["source"], "type": r["type"], "score": r["score"],
                           "refined": r["refine_score"], "gain": r["refine_score"] - r["score"],
                           "context": r.get("context_score"), "cause": r.get("refine_cause"),
                           "rationale": r.get("refine_rationale") or "", "question": r["question"]} for r in refined),
                         key=lambda x: (-x["gain"], x["score"]))
    return out
