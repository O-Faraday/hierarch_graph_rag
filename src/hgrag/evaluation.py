"""Benchmark of the whole chain on ``eval_dataset.jsonl``: retrieval rank, generated answer, LLM judge.

Docling is the slow part and its output does not depend on anything downstream, so the benchmark never runs it:
the Docling outputs of the control PDFs are *frozen* once (``hgrag freeze`` -> ``data/pdf_control/parsed``) and
the benchmark rebuilds the hierarchy, the chunks and a private Qdrant store from them in seconds. Changing the
chunking (``HGRAG_CHUNK_LABELS``, chunk size, model) and re-running the benchmark is then a matter of a minute.

The judge is a second LLM call (``claude-haiku-5-5`` by default) comparing the answer with the ground truth.
"""

from __future__ import annotations

import json
import re
import os
import shutil
import stat
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config, generation, pipeline, tracking
from .generation import GenerationError, LLMConfig
from .ids import compute_doc_id, pdf_name
from .verify import _norm

EVAL_COLLECTION = "hgrag_eval"

JUDGE_SYSTEM = """You are a strict but fair grader of answers produced by a retrieval-augmented system.
You receive a QUESTION, the GROUND TRUTH answer and the CANDIDATE answer. Judge only whether the candidate \
conveys the same information as the ground truth, not its style.

Scale:
5 = fully correct and complete: every fact of the ground truth is present and correct.
4 = correct, with a minor omission or imprecision that does not change the meaning.
3 = partially correct: some key facts are right, others missing or wrong.
2 = mostly wrong or off-topic, with a small relevant element.
1 = wrong, contradicting the ground truth, invented, or no real answer.

Rules:
- Numbers, units and names must match the ground truth. A contradiction is never above 2.
- Extra details that are correct are fine. Citation markers like [1] and formatting are irrelevant.
- If the ground truth says the documents do not contain the answer, a candidate that clearly says it cannot \
answer from the documents scores 5, and any invented answer scores 1.

Reply with ONLY a JSON object, no other text:
{"rationale": "<two or three sentences explaining the comparison>", "score": <integer 1-5>}"""


class JudgeError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ data
def load_dataset(path: str | Path, ids: list[str] | None = None, kinds: list[str] | None = None,
                 limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if ids:
        unknown = set(ids) - {r["id"] for r in rows}
        if unknown:
            raise ValueError(f"unknown question ids: {sorted(unknown)}")
        rows = [r for r in rows if r["id"] in ids]
    if kinds:
        rows = [r for r in rows if r["source_kind"] in kinds]
    return rows[:limit] if limit else rows


def _expected_pdfs(pdf_dir: Path) -> list[Path]:
    expected = json.loads((pdf_dir / "expected.json").read_text(encoding="utf-8"))
    return [pdf_dir / name for name in expected]


def _force_remove(func, path, _exc) -> None:
    """rmtree error hook: clear the read-only flag (Windows) and retry."""
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except OSError:
        pass


def _clean_dir(path: Path) -> None:
    """Remove a directory tree, also when files are read-only; fail loudly if it cannot be emptied."""
    if not path.exists():
        return
    shutil.rmtree(path, onerror=_force_remove)
    if path.exists():
        raise PermissionError(f"cannot remove {path}: close the programs using it (editor, Explorer, OneDrive, "
                              "antivirus) or delete the folder by hand, then retry")


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy file contents only (no permission bits), so read-only sources give writable copies."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        target = dst / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)


def freeze_parsed(pdf_dir: str | Path = config.CONTROL_PDF_DIR, output_dir: str | Path = config.OUTPUT_DIR,
                  dest: str | Path | None = None) -> list[str]:
    """Copy the Docling outputs (document.json, figures, manifest) of the control PDFs next to them."""
    pdf_dir, output_dir = Path(pdf_dir), Path(output_dir)
    dest = Path(dest or pdf_dir / "parsed")
    done = []
    for pdf in _expected_pdfs(pdf_dir):
        doc_id = compute_doc_id(pdf)
        src = output_dir / doc_id
        missing = [f for f in ("document.json", "figures.json", "manifest.json") if not (src / f).exists()]
        if missing:
            raise FileNotFoundError(f"{pdf.name}: {missing} missing in {src}. Run the parse step on the control PDFs first.")
        target = dest / doc_id
        _clean_dir(target)
        (target / "figures").mkdir(parents=True)
        for png in (src / "figures").glob("*.png"):
            shutil.copyfile(png, target / "figures" / png.name)
        shutil.copyfile(src / "document.json", target / "document.json")
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(source_pdf=pdf.name, doc_dir="")  # machine-independent
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        figures = json.loads((src / "figures.json").read_text(encoding="utf-8"))  # descriptions are kept if present
        (target / "figures.json").write_text(json.dumps(figures, indent=2, ensure_ascii=False), encoding="utf-8")
        done.append(pdf.name)
    return done


def build_store(pdf_dir: str | Path = config.CONTROL_PDF_DIR, parsed_dir: str | Path | None = None,
                workdir: str | Path | None = None, collection: str = EVAL_COLLECTION) -> dict[str, Any]:
    """Hierarchy + chunks + embeddings from the frozen parsing, in a private store (the app's is untouched)."""
    pdf_dir = Path(pdf_dir)
    parsed_dir = Path(parsed_dir or pdf_dir / "parsed")
    workdir = Path(workdir or config.EVAL_DIR / "store")
    _clean_dir(workdir)
    out = workdir / "output"
    dirs = []
    for pdf in _expected_pdfs(pdf_dir):
        doc_id = compute_doc_id(pdf)
        src = parsed_dir / doc_id
        if not (src / "document.json").exists():
            raise FileNotFoundError(f"no frozen parsing for {pdf.name} in {parsed_dir}: run `uv run hgrag freeze` "
                                    "after a parse of the control PDFs")
        _copy_tree(src, out / doc_id)
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(source_pdf=str(pdf), doc_dir=str(out / doc_id))
        (out / doc_id / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pipeline.build_and_save_hierarchy(out / doc_id)
        dirs.append(str(out / doc_id))
    summary = pipeline.index_documents(dirs, qdrant_path=workdir / "qdrant", collection=collection)
    return {"output_dir": out, "qdrant_path": workdir / "qdrant", "collection": collection, "index": summary}


def store_kwargs(store: dict[str, Any]) -> dict[str, Any]:
    """The part of a ``build_store`` result that ``run_eval`` takes."""
    return {k: store[k] for k in ("output_dir", "qdrant_path", "collection")}


# ----------------------------------------------------------------------------------------- judge
def _parse_judgement(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise JudgeError(f"no JSON in the judge reply: {text[:200]!r}")
    try:
        data = json.loads(m.group(0))
        score = int(data["score"])
        rationale = str(data["rationale"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        # the rationale often contains unescaped double quotes ("No"): read the fields with regexes instead
        sc = re.search(r'"score"\s*:\s*(\d)', text)
        ra = re.search(r'"rationale"\s*:\s*"(.*)"\s*,\s*"score"', text, re.S)
        if not sc:
            raise JudgeError(f"invalid judge JSON ({exc}): {text[:200]!r}") from exc
        score, rationale = int(sc.group(1)), (ra.group(1).strip() if ra else "")
    if not 1 <= score <= 5:
        raise JudgeError(f"score out of range: {score}")
    return {"score": score, "rationale": rationale}


def judge_answer(cfg: LLMConfig, question: str, ground_truth: str, answer: str) -> dict[str, Any]:
    """Score 1-5 and rationale of ``answer`` against ``ground_truth`` (one retry if the reply is not valid JSON)."""
    user = (f"QUESTION:\n{question}\n\nGROUND TRUTH:\n{ground_truth}\n\nCANDIDATE ANSWER:\n{answer or '(empty)'}")
    last: Exception | None = None
    for attempt in range(2):
        text, usage = generation.complete(cfg, JUDGE_SYSTEM, user + ("\n\nReply with the JSON object only."
                                                                       if attempt else ""), max_tokens=600)
        try:
            return {**_parse_judgement(text), "judge_usage": usage}
        except JudgeError as exc:
            last = exc
    raise last  # type: ignore[misc]


def judge_config(model: str | None = None, api_key: str | None = None) -> LLMConfig:
    return LLMConfig(provider="anthropic", model=model or config.JUDGE_MODEL, max_tokens=600, send_images=False,
                     api_key=api_key)


# ------------------------------------------------------------------------------------------ run
def section_rank(sections: list[dict[str, Any]], pdf: str | None, section: str | None) -> int | None:
    """1-based rank of the expected section among the retrieved ones (grouped by section), else None."""
    if not pdf or not section:
        return None
    for i, s in enumerate(sections, start=1):
        if pdf_name(s.get("source_pdf") or "") == pdf and _norm(s["section_path"].split(" > ")[-1]) == _norm(section):
            return i
    return None


def evaluate_row(row: dict[str, Any], *, client: Any, embedder: Any, output_dir: Path, collection: str, k: int,
                 include_children: bool, gen_cfg: LLMConfig | None, judge_cfg: LLMConfig | None) -> dict[str, Any]:
    from .retrieval import reconstruct_hit, search_hits

    t0 = time.perf_counter()
    res: dict[str, Any] = {k_: row[k_] for k_ in ("id", "source_kind", "pdf", "section", "question", "expected_answer")}
    hits = search_hits(row["question"], k, embedder=embedder, client=client, collection=collection)
    res.update(rank=section_rank(hits, row["pdf"], row["section"]),
               retrieved=[{"rank": i, "section_path": h["section_path"], "score": round(h["score"], 4),
                           "pdf": pdf_name(h.get("source_pdf") or "")} for i, h in enumerate(hits, start=1)])
    res["hit_at_1"] = (res["rank"] == 1) if row["pdf"] else None  # unanswerable questions have no expected section
    res["hit_at_k"] = (res["rank"] is not None) if row["pdf"] else None
    if gen_cfg is not None:
        try:
            sections = [reconstruct_hit(h, include_children=include_children, output_dir=output_dir) for h in hits]
            info: dict[str, Any] = {}
            res["answer"] = "".join(generation.answer_stream(row["question"], sections, gen_cfg, info=info)).strip()
            res.update(usage=info.get("usage"), n_images=info.get("n_images"),
                       sources=[x["section_path"] for x in info.get("sources", [])], first_token_s=info.get("first_token_s"))
        except GenerationError as exc:
            res["error"] = f"generation: {exc}"
    if judge_cfg is not None and res.get("answer") is not None:
        try:
            j = judge_answer(judge_cfg, row["question"], row["expected_answer"], res["answer"])
            res.update(score=j["score"], rationale=j["rationale"])
        except (GenerationError, JudgeError) as exc:
            res["error"] = f"judge: {exc}"
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def summarize(results: list[dict[str, Any]], k: int) -> dict[str, Any]:
    answerable = [r for r in results if r["pdf"]]
    scored = [r for r in results if r.get("score") is not None]
    unans = [r for r in scored if not r["pdf"]]
    by_kind: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0})
    for r in results:
        b = by_kind[r["source_kind"]]
        b["n"] += 1
    for kind, b in by_kind.items():
        rs = [r for r in results if r["source_kind"] == kind]
        sc = [r["score"] for r in rs if r.get("score") is not None]
        if sc:
            b["mean_score"] = round(statistics.fmean(sc), 2)
        if any(r["pdf"] for r in rs):
            b["hit_at_1"] = round(sum(bool(r["hit_at_1"]) for r in rs) / len(rs), 3)
    mrr = [1 / r["rank"] if r["rank"] else 0.0 for r in answerable]
    out: dict[str, Any] = {
        "n_questions": len(results), "k": k, "n_answerable": len(answerable),
        "hit_at_1": round(sum(bool(r["hit_at_1"]) for r in answerable) / len(answerable), 3) if answerable else None,
        f"hit_at_{k}": round(sum(bool(r["hit_at_k"]) for r in answerable) / len(answerable), 3) if answerable else None,
        "mrr": round(statistics.fmean(mrr), 3) if mrr else None,
        "n_errors": sum(1 for r in results if r.get("error")),
        "by_kind": dict(by_kind),
    }
    if scored:
        sc = [r["score"] for r in scored]
        out.update(mean_score=round(statistics.fmean(sc), 2), mean_score_norm=round((statistics.fmean(sc) - 1) / 4, 3),
                   pct_score_4_plus=round(sum(s >= 4 for s in sc) / len(sc), 3), n_scored=len(sc))
        if unans:
            out["abstention_accuracy"] = round(sum(r["score"] >= 4 for r in unans) / len(unans), 3)
    toks = [r["usage"] for r in results if r.get("usage")]
    if toks:
        out["input_tokens"] = sum(t.get("input_tokens", 0) for t in toks)
        out["output_tokens"] = sum(t.get("output_tokens", 0) for t in toks)
    return out


def format_summary(s: dict[str, Any]) -> str:
    k = s["k"]
    lines = [f"questions: {s['n_questions']} ({s['n_answerable']} answerable) | errors: {s['n_errors']}",
             f"retrieval: hit@1 {s['hit_at_1']} | hit@{k} {s[f'hit_at_{k}']} | MRR {s['mrr']}"]
    if "mean_score" in s:
        lines.append(f"judge: mean score {s['mean_score']}/5 ({s['mean_score_norm']:.0%}) | score >= 4: "
                     f"{s['pct_score_4_plus']:.0%}" + (f" | abstention {s['abstention_accuracy']:.0%}"
                                                     if "abstention_accuracy" in s else ""))
    lines.append("by kind: " + "; ".join(
        f"{kind} n={b['n']}" + (f" score={b['mean_score']}" if "mean_score" in b else "")
        + (f" hit@1={b['hit_at_1']}" if "hit_at_1" in b else "") for kind, b in sorted(s["by_kind"].items())))
    return "\n".join(lines)


def run_eval(
    rows: list[dict[str, Any]], *, output_dir: Path, qdrant_path: Path, collection: str, embedder: Any, k: int = 5,
    include_children: bool = False, gen_cfg: LLMConfig | None = None, judge_cfg: LLMConfig | None = None,
    run_key: str | None = None, dest: Path | None = None, on_row: Callable[[dict[str, Any]], None] | None = None,
    config_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .indexing import open_store

    dest = Path(dest or config.EVAL_DIR / (run_key or tracking.new_run_key()))
    dest.mkdir(parents=True, exist_ok=True)
    client = open_store(qdrant_path)
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            res = evaluate_row(row, client=client, embedder=embedder, output_dir=output_dir, collection=collection, k=k,
                               include_children=include_children, gen_cfg=gen_cfg, judge_cfg=judge_cfg)
            results.append(res)
            _track_row(run_key, res)
            if on_row:
                on_row(res)
    finally:
        client.close()
    summary = summarize(results, k)
    summary["config"] = {"chunk_labels": list(config.CHUNK_LABELS), "embed_model": config.EMBED_MODEL_ID, "k": k,
                         "include_children": include_children,
                         "generator": gen_cfg.describe() if gen_cfg else None,
                         "judge": judge_cfg.describe() if judge_cfg else None, **(config_info or {})}
    (dest / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (dest / "report.md").write_text(_markdown(results, summary), encoding="utf-8")
    if run_key:
        flat = {k_: v for k_, v in summary.items() if isinstance(v, (int, float)) and v is not None}
        tracking.finish_execution(tracking.EVAL, run_key, **flat)
    summary["dest"] = str(dest)
    return summary


def _track_row(run_key: str | None, res: dict[str, Any]) -> None:
    with tracking.track_step(tracking.EVAL, "question", run_key, item=res["id"], kind=res["source_kind"]) as tr:
        tr.metrics(**{k: v for k, v in {
            "rank": res.get("rank") or 0, "hit_at_1": float(bool(res.get("hit_at_1"))), "score": res.get("score"),
            "seconds": res["seconds"], "input_tokens": (res.get("usage") or {}).get("input_tokens"),
            "output_tokens": (res.get("usage") or {}).get("output_tokens"), "n_images": res.get("n_images")}.items()
            if v is not None})
        tr.tags(question=res["question"][:200], error=res.get("error") or "")
        tr.dict(res, "result.json")


def _markdown(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    out = ["# Evaluation report", "", "```", format_summary(summary), "```", "",
           "| id | kind | rank | score | question | rationale |", "|---|---|---|---|---|---|"]
    for r in results:
        out.append(f"| {r['id']} | {r['source_kind']} | {r.get('rank') or '-'} | {r.get('score', '-')} | "
                   f"{r['question']} | {(r.get('rationale') or r.get('error') or '').replace('|', '/')} |")
    out += ["", "## Answers", ""]
    for r in results:
        out += [f"### {r['id']} — {r['question']}", f"- expected: {r['expected_answer']}",
                f"- answer: {r.get('answer', '(none)')}", ""]
    return "\n".join(out)
