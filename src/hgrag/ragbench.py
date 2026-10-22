"""Validate the pipeline on Open RAG Bench (Vectara): arXiv PDFs with tables and figures, questions whose gold
evidence is a section of the paper. A sample of N papers is downloaded, parsed, described (optional), indexed
in a private store, and every question of these papers is run through the retrieval.

    uv run hgrag ragbench --docs 10            # 10 papers + their questions (about 30 queries)
    uv run hgrag ragbench --docs 100 --negatives 20 --generate

The gold sections come from the benchmark's own parsing, not from ours, so a retrieved section is compared with
the gold section by **text overlap** (word 5-grams), never by index:

* ``doc_hit``: the right paper is among the retrieved sections;
* ``section_hit``: a matching chunk of a retrieved section lies inside the gold text (≥ 50 % of its 5-grams);
* ``context_recall``: share of the gold text found in the sections rebuilt for the LLM (the hierarchical benefit).
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import requests

from . import config, generation, tracking
from .keywords import has_keywords
from .evaluation import JudgeError, _clean_dir, judge_answer
from .generation import GenerationError, LLMConfig
from .ids import compute_doc_id, pdf_stem

DATASET = "vectara/open_ragbench"
BASE = "pdf/arxiv"  # the README says official/pdf/arxiv, the repo has pdf/arxiv
RAGBENCH_DIR = Path(config.PROJECT_ROOT / "data" / "ragbench")
SOURCES = ("text", "text-image", "text-table", "text-table-image")
NGRAM = 5
CHUNK_ON_GOLD = 0.5
CONTEXT_OK = 0.5
GOLD_EXCERPT = 1500  # chars of the reference section kept in results.jsonl for diagnosis


class RagBenchError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ source
class HFSource:
    """Small files of the dataset, fetched from the Hub and cached on disk."""

    def __init__(self, cache: Path, token: str | None = None):
        self.cache = Path(cache)
        self.token = token or config_token()

    def json(self, name: str) -> Any:
        path = self.cache / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{BASE}/{name}"
            resp = requests.get(url, timeout=120, headers={"authorization": f"Bearer {self.token}"} if self.token else {})
            if resp.status_code >= 400:
                raise RagBenchError(f"cannot fetch {url}: HTTP {resp.status_code}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
        return json.loads(path.read_text(encoding="utf-8"))


def config_token() -> str | None:
    import os

    return os.environ.get("HF_TOKEN")


def safe_name(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


# ---------------------------------------------------------------------------------------- sampling
def pick_sample(queries: dict[str, Any], qrels: dict[str, Any], pdf_urls: dict[str, str], n_docs: int, *, seed: int = 0,
                negatives: int = 0, sources: list[str] | None = None, max_queries: int | None = None) -> dict[str, Any]:
    """``n_docs`` papers that have questions (optionally only of the given sources) + ``negatives`` papers without any."""
    wanted = set(sources) if sources else None
    by_doc: dict[str, list[str]] = defaultdict(list)
    for qid, rel in qrels.items():
        if qid in queries and rel["doc_id"] in pdf_urls and (not wanted or queries[qid]["source"] in wanted):
            by_doc[rel["doc_id"]].append(qid)
    pool = sorted(by_doc)
    if not pool:
        raise RagBenchError("no paper matches (check --sources)")
    rng = random.Random(seed)
    golden = sorted(rng.sample(pool, min(n_docs, len(pool))))
    all_golden = {rel["doc_id"] for rel in qrels.values()}
    neg_pool = sorted(d for d in pdf_urls if d not in all_golden)
    negs = sorted(rng.sample(neg_pool, min(negatives, len(neg_pool))))
    qids = [q for d in golden for q in sorted(by_doc[d])]
    if max_queries:
        qids = qids[:max_queries]
    return {"golden": golden, "negatives": negs, "query_ids": qids, "seed": seed}


def sample_key(sample: dict[str, Any], n_docs: int, sources: list[str] | None) -> str:
    return f"n{n_docs}_s{sample['seed']}_neg{len(sample['negatives'])}" + (f"_{'+'.join(sorted(sources))}" if sources else "")


# ------------------------------------------------------------------------------------------ files
def download_pdf(url: str, dest: Path, *, retries: int = 3, pause: float = 3.0) -> None:
    """arXiv asks for about one request every 3 s; a failed download raises after ``retries`` attempts."""
    if dest.exists() and dest.read_bytes()[:5] == b"%PDF-":
        return
    last = ""
    for attempt in range(retries):
        try:
            resp = requests.get(url.replace("http://", "https://"), timeout=120,
                                headers={"user-agent": "hgrag-ragbench (research; pipeline validation)"})
            if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                time.sleep(pause)
                return
            last = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(pause * (attempt + 1))
    raise RagBenchError(f"cannot download {url}: {last}")


def fetch_pdfs(ids: list[str], pdf_urls: dict[str, str], pdf_dir: Path, *, pause: float = 3.0,
               on_progress: Callable[[str], None] | None = None) -> tuple[dict[str, Path], dict[str, str]]:
    ok: dict[str, Path] = {}
    failed: dict[str, str] = {}
    for i, pid in enumerate(ids, start=1):
        dest = pdf_dir / f"{safe_name(pid)}.pdf"
        try:
            download_pdf(pdf_urls[pid], dest, pause=pause)
            ok[pid] = dest
        except RagBenchError as exc:
            failed[pid] = str(exc)
        if on_progress:
            on_progress(f"pdf {i}/{len(ids)}: {pid}" + (" FAILED" if pid in failed else ""))
    return ok, failed


# ------------------------------------------------------------------------------------------ text
def words(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def shingles(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def containment(part: set, whole: set) -> float:
    """Share of ``part`` found in ``whole``."""
    return len(part & whole) / len(part) if part else 0.0


def detect_base(qrels: dict[str, Any], corpus_sections: dict[str, int], qids: list[str]) -> int:
    """Section index base (the README does not say): 1 if an index equals the number of sections, else 0."""
    for qid in qids:
        rel = qrels[qid]
        n = corpus_sections.get(rel["doc_id"])
        if n is not None and int(rel["section_id"]) >= n:
            return 1
    return 0


def gold_text(corpus_doc: dict[str, Any], section_id: Any, base: int = 0) -> str:
    idx = int(section_id) - base
    sections = corpus_doc.get("sections") or []
    if not 0 <= idx < len(sections):
        raise RagBenchError(f"section {section_id} (base {base}) out of range for {corpus_doc.get('id')}")
    return sections[idx].get("text") or ""


# --------------------------------------------------------------------------------------- the store
def _cached(doc_dir: Path) -> bool:
    """A parse is reused only if it exists and was made with the current Docling options."""
    mf = doc_dir / "manifest.json"
    if not ((doc_dir / "document.json").exists() and mf.exists()):
        return False
    from .parsing import parse_options

    try:
        return json.loads(mf.read_text(encoding="utf-8")).get("parse_options") == parse_options()
    except (OSError, ValueError):
        return False


def qdrant_dirname(model_id: str | None = None) -> str:
    """One Qdrant folder per embedding model (vector sizes differ); the historical MiniLM store keeps its name."""
    model_id = model_id or config.EMBED_MODEL_ID
    if model_id == "sentence-transformers/all-MiniLM-L6-v2":
        return "qdrant"
    return "qdrant-" + re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def build_store(pdfs: dict[str, Path], workdir: Path, *, output_dir: Path | None = None, collection: str,
                describe: bool = False, parse: Callable[..., Any] | None = None,
                on_progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Parse (cached per document), build the hierarchy and index the sample in its own Qdrant store."""
    from . import pipeline

    output_dir = Path(output_dir or RAGBENCH_DIR / "output")
    stats = {"parsed": 0, "reused": 0, "failed": {}}
    converter = None
    dirs: dict[str, str] = {}
    for i, (pid, pdf) in enumerate(pdfs.items(), start=1):
        doc_dir = output_dir / compute_doc_id(pdf)
        try:
            if _cached(doc_dir):
                stats["reused"] += 1
            else:
                if doc_dir.exists():  # parsed with other Docling options: start again
                    _clean_dir(doc_dir)
                if parse is None:
                    from .parsing import build_converter, parse_pdf

                    converter = converter or build_converter()
                    parse = lambda p, **kw: parse_pdf(p, output_root=output_dir, converter=converter, **kw)  # noqa: E731
                parse(pdf, describe_pictures=False)
                stats["parsed"] += 1
            if describe:  # a failure here (no token, API down) must not drop the document
                from .describe import describe_document

                try:
                    describe_document(doc_dir)
                except Exception as exc:  # noqa: BLE001
                    stats.setdefault("describe_failed", {})[pid] = f"{type(exc).__name__}: {exc}"
            pipeline.build_and_save_hierarchy(doc_dir)
            dirs[pid] = str(doc_dir)
        except Exception as exc:  # noqa: BLE001  (one bad PDF must not stop the benchmark)
            hint = (" -> a Docling model is not in the local cache and HF_HUB_OFFLINE=1: download it once with HF_HUB_OFFLINE=0 "
                    "(README, 'Premier téléchargement des modèles')" if "traffic has been disabled" in str(exc) else "")
            stats["failed"][pid] = f"{type(exc).__name__}: {exc}{hint}"
        if on_progress:
            on_progress(f"parse {i}/{len(pdfs)}: {pid}" + (" FAILED" if pid in stats["failed"] else ""))
    if not dirs:
        raise RagBenchError(f"no document could be parsed: {stats['failed']}")
    t0 = time.perf_counter()
    qdrant_path = workdir / qdrant_dirname()
    index = pipeline.index_documents(list(dirs.values()), qdrant_path=qdrant_path, collection=collection)
    return {"output_dir": output_dir, "qdrant_path": qdrant_path, "collection": collection, "dirs": dirs,
            "stats": stats, "index": index, "index_seconds": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------------------------------- eval
def _norm_title(text: str) -> str:
    return re.sub(r"\W+", "", (text or "").lower())


def evaluate_query(qid: str, q: dict[str, Any], answer: str, gold: str, paper_id: str, *, client: Any, embedder: Any,
                   output_dir: Path, collection: str, k: int, stem_to_paper: dict[str, str],
                   gen_cfg: LLMConfig | None = None, judge_cfg: LLMConfig | None = None,
                   rerank: bool | None = None, reranker: Any = None, candidates: int | None = None) -> dict[str, Any]:
    from .retrieval import reconstruct_hit, search_hits

    t0 = time.perf_counter()
    gold_sh = shingles(gold)
    gold_title = _norm_title(gold.lstrip().split("\n", 1)[0].lstrip("#")) if gold.lstrip().startswith("#") else ""
    hits = search_hits(q["query"], k, embedder=embedder, client=client, collection=collection, rerank=rerank,
                       reranker=reranker, candidates=candidates)
    doc_rank = sec_rank = None
    retrieved = []
    covered: set = set()
    sections = []
    for i, h in enumerate(hits, start=1):
        pid = stem_to_paper.get(pdf_stem(h.get("source_pdf") or ""))
        same = pid == paper_id
        by_text = same and any(containment(shingles(c["text"] or ""), gold_sh) >= CHUNK_ON_GOLD for c in h["hits"])
        # LaTeX written differently by the two parsers breaks the word overlap: the section heading is the second proof
        by_title = same and bool(gold_title) and _norm_title(h["section_path"].split(" > ")[-1]) == gold_title
        on_gold = by_text or by_title
        doc_rank = doc_rank or (i if same else None)
        sec_rank = sec_rank or (i if on_gold else None)
        rec = reconstruct_hit(h, output_dir=output_dir)
        sections.append(rec)
        sec_sh = shingles(rec["text"]) if same else set()
        covered |= sec_sh & gold_sh
        best = max(h["hits"], key=lambda c: c.get("score") or 0, default={})
        retrieved.append({"rank": i, "paper": pid, "doc_id": h["doc_id"], "section_ref": h["section_ref"],
                          "section_path": h["section_path"], "score": round(h["score"], 4),
                          "preview": (best.get("text") or "")[:300],
                          "doc": same, "on_gold": on_gold, "on_gold_by": "text" if by_text else ("title" if by_title else None),
                          "recall_so_far": round(len(covered) / len(gold_sh), 3) if gold_sh else None})
    first_recall = retrieved[0]["recall_so_far"] if retrieved else None
    res: dict[str, Any] = {
        "id": qid, "paper": paper_id, "source": q["source"], "type": q["type"], "question": q["query"],
        "expected_answer": answer, "gold_words": len(words(gold)), "gold_text": gold[:GOLD_EXCERPT], "doc_rank": doc_rank, "section_rank": sec_rank,
        "context_recall": retrieved[-1]["recall_so_far"] if retrieved else None, "context_recall_at_1": first_recall,
        "retrieved": retrieved}
    if gen_cfg is not None:
        try:
            info: dict[str, Any] = {}
            res["answer"] = "".join(generation.answer_stream(q["query"], sections, gen_cfg, info=info)).strip()
            if not res["answer"] or (info.get("usage") or {}).get("stop_reason") == "max_tokens":
                # empty or cut answer: the model spent its budget before/while answering; retry once with twice the budget
                bigger = dataclasses.replace(gen_cfg, max_tokens=gen_cfg.max_tokens * 2)
                info = {}
                res["answer"] = "".join(generation.answer_stream(q["query"], sections, bigger, info=info)).strip()
                res["retried"] = True
            res.update(usage=info.get("usage"), n_images=info.get("n_images"),
                       truncated=(info.get("usage") or {}).get("stop_reason") == "max_tokens")
            if judge_cfg is not None:
                j = judge_answer(judge_cfg, q["query"], answer, res["answer"])
                res.update(score=j["score"], rationale=j["rationale"])
        except (GenerationError, JudgeError) as exc:
            res["error"] = str(exc)
    res["seconds"] = round(time.perf_counter() - t0, 2)
    return res


def _paths(res: dict[str, Any]) -> list[tuple[Any, str]]:
    return [(h.get("paper"), h["section_path"] if isinstance(h["section_path"], str) else " > ".join(h["section_path"]))
            for h in res.get("retrieved") or []]


def _retrieved_paths(run_dir: Path) -> dict[str, list[tuple[Any, str]]]:
    """{query id: retrieved (paper, section path) in order} of an existing run (unreadable lines skipped)."""
    f = run_dir / "results.jsonl"
    if not f.is_file():
        raise RagBenchError(f"reference run: {f} not found")
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out[r["id"]] = _paths(r)
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _rate(rows: list[dict[str, Any]], key: str, k: int | None = None) -> float | None:
    if not rows:
        return None
    if k is None:
        return round(sum(1 for r in rows if r[key] == 1) / len(rows), 3)
    return round(sum(1 for r in rows if r[key] is not None and r[key] <= k) / len(rows), 3)


def load_disputed(path: str | Path | None = None) -> dict[str, str]:
    """{query id: reason} of the questions whose reference answer is wrong (checked in the PDF); {} if no file."""
    p = Path(path or config.RAGBENCH_DISPUTED)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except ValueError as exc:
        raise RagBenchError(f"{p} is not valid JSON: {exc}") from exc


def summarize(results: list[dict[str, Any]], k_sections: int, disputed: dict[str, str] | None = None) -> dict[str, Any]:
    """Metrics on every question; with ``disputed``, also ``excluding_disputed`` (the same without those questions)."""
    out = _summarize(results, k_sections)
    excluded = [r for r in results if r["id"] in (disputed or {})]
    if excluded:
        out["excluding_disputed"] = {**_summarize([r for r in results if r["id"] not in disputed], k_sections),
                                     "excluded": [r["id"] for r in excluded]}
    return out


def _summarize(results: list[dict[str, Any]], k_sections: int) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rec = [r["context_recall"] for r in rows if r["context_recall"] is not None]
        out = {"n": len(rows), "doc_hit_at_1": _rate(rows, "doc_rank", 1), f"doc_hit_at_{k_sections}": _rate(rows, "doc_rank", k_sections),
               "section_hit_at_1": _rate(rows, "section_rank", 1), f"section_hit_at_{k_sections}": _rate(rows, "section_rank", k_sections),
               "mrr": round(statistics.fmean(1 / r["section_rank"] if r["section_rank"] else 0.0 for r in rows), 3) if rows else None,
               "context_recall": round(statistics.fmean(rec), 3) if rec else None,
               "pct_context_recall_50": round(sum(x >= CONTEXT_OK for x in rec) / len(rec), 3) if rec else None}
        scored = [r["score"] for r in rows if r.get("score") is not None]
        if scored:
            out.update(mean_score=round(statistics.fmean(scored), 2), pct_score_4_plus=round(sum(s >= 4 for s in scored) / len(scored), 3))
        if any(r.get("refine_score") is not None for r in rows):  # second opinion (hgrag.refine): score kept where not refined
            refined = [r["refine_score"] if r.get("refine_score") is not None else r["score"] for r in rows if r.get("score") is not None]
            out.update(mean_refined_score=round(statistics.fmean(refined), 2),
                       pct_refined_4_plus=round(sum(s >= 4 for s in refined) / len(refined), 3))
        return out

    out = block(results)
    out["k_sections"] = k_sections
    out["n_errors"] = sum(1 for r in results if r.get("error"))
    out["by_source"] = {s: block([r for r in results if r["source"] == s]) for s in SOURCES if any(r["source"] == s for r in results)}
    out["by_type"] = {t: block([r for r in results if r["type"] == t]) for t in sorted({r["type"] for r in results})}
    causes = Counter(r["refine_cause"] for r in results if r.get("refine_cause"))
    if causes:
        out["refine_causes"] = dict(causes.most_common())
    return out


def format_summary(s: dict[str, Any]) -> str:
    k = s["k_sections"]
    lines = [f"{s['n']} questions · doc hit@1 {s['doc_hit_at_1']} · doc hit@{k} {s[f'doc_hit_at_{k}']} · "
             f"section hit@1 {s['section_hit_at_1']} · hit@{k} {s[f'section_hit_at_{k}']} · MRR {s['mrr']} · "
             f"context recall {s['context_recall']} ({s['pct_context_recall_50']} ≥ 50 %)"]
    if "mean_score" in s:
        lines.append(f"answers: mean score {s['mean_score']}/5, {s['pct_score_4_plus']} ≥ 4")
    if "mean_refined_score" in s:
        lines.append(f"refined (second opinion on the low scores): mean {s['mean_refined_score']}/5, {s['pct_refined_4_plus']} ≥ 4 · "
                     "causes " + ", ".join(f"{c} {n}" for c, n in s.get("refine_causes", {}).items()))
    if (x := s.get("excluding_disputed")):
        lines.append(f"without {len(x['excluded'])} disputed reference(s): {x['n']} questions · section hit@1 {x['section_hit_at_1']} · "
                     f"hit@{k} {x[f'section_hit_at_{k}']} · MRR {x['mrr']}"
                     + (f" · mean score {x['mean_score']}/5, {x['pct_score_4_plus']} ≥ 4" if "mean_score" in x else ""))
    for name, group in (("source", s["by_source"]), ("type", s["by_type"])):
        for key, b in group.items():
            lines.append(f"  {name} {key:<17} n={b['n']:<4} section hit@1 {b['section_hit_at_1']}  hit@{k} {b[f'section_hit_at_{k}']}  "
                         f"context recall {b['context_recall']}")
    return "\n".join(lines)


def _path(p: Any) -> str:
    return p if isinstance(p, str) else " > ".join(p or [])


def _sort_key(r: dict[str, Any]) -> tuple:
    score = r.get("score")
    rec = r.get("context_recall")
    return (-1 if score is None else score, -1 if rec is None else rec)


def qa_report(results: list[dict[str, Any]], pdf_dir: Path, *, top: int = 3, disputed: dict[str, str] | None = None) -> str:
    """One Markdown document: every question with expected answer, generated answer and judge comment.

    Ordered by increasing judge score (then by increasing context recall; without --generate the recall alone orders them).
    """
    rows = sorted(results, key=_sort_key)
    disputed = disputed or {}
    lines = [f"# Questions, expected and generated answers ({len(rows)}), lowest score first", ""]
    for n, r in enumerate(rows, start=1):
        score = r.get("score")
        lines += [f"## {n}. {r['id']} - score {'n/a' if score is None else score} - recall {r.get('context_recall')} - "
                  f"section rank {r.get('section_rank')}" + (" - DISPUTED REFERENCE" if r["id"] in disputed else ""), "",
                  f"{r['source']} / {r['type']} - PDF `{pdf_dir / (safe_name(r['paper']) + '.pdf')}`", ""]
        if r["id"] in disputed:
            lines += [f"**Disputed reference:** {disputed[r['id']]}", ""]
        lines += [
                  f"**Question:** {r['question']}", "", f"**Expected answer:** {r.get('expected_answer', '')}", ""]
        if r.get("answer") is not None:
            lines += [f"**Answer:** {r['answer']}", ""]
        if r.get("rationale"):
            lines += [f"**Comment (judge):** {r['rationale']}", ""]
        if r.get("context_score") is not None:
            lines += [f"**Context judge:** {r['context_score']}/5 - {r.get('context_rationale', '')}", ""]
        if r.get("refine_score") is not None:
            lines += [f"**Refined score:** {r['refine_score']} (cause: {r.get('refine_cause')}) - {r.get('refine_rationale', '')}", ""]
        if r.get("refine_error"):
            lines += [f"**Refine error:** {r['refine_error']}", ""]
        if r.get("error"):
            lines += [f"**Error:** {r['error']}", ""]
        if r.get("gold_text"):
            lines += ["<details><summary>Reference section (Vectara parse, excerpt)</summary>", "",
                      "> " + r["gold_text"].replace("\n", "\n> "), "", "</details>", ""]
        for h in (r.get("retrieved") or [])[:top]:
            lines += [f"- #{h['rank']} {_path(h['section_path'])} (score {h['score']}, same paper: {h['doc']}, "
                      f"on gold: {h['on_gold']}) {(h.get('preview') or '')[:200]!r}"]
        lines.append("")
    return "\n".join(lines)


def write_qa_report(run_dir: str | Path, pdf_dir: str | Path | None = None) -> Path:
    """(Re)build ``questions.md`` from the ``results.jsonl`` of an existing run, and the summary's disputed block."""
    run_dir = Path(run_dir)
    f = run_dir / "results.jsonl"
    if not f.exists():
        raise RagBenchError(f"{f} not found")
    results = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    disputed = load_disputed()
    out = run_dir / "questions.md"
    out.write_text(qa_report(results, Path(pdf_dir or RAGBENCH_DIR / "pdfs"), disputed=disputed), encoding="utf-8")
    sf = run_dir / "summary.json"
    if sf.exists():
        old = json.loads(sf.read_text(encoding="utf-8"))
        summary = {**summarize(results, old.get("k_sections", 10), disputed),
                   **{k: old[k] for k in ("config", "failed", "refine") if k in old}}
        sf.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "report.md").write_text("# Open RAG Bench\n\n```\n" + format_summary(summary) + "\n```\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------- run
def run(n_docs: int = 10, *, seed: int = 0, negatives: int = 0, sources: list[str] | None = None, k: int = 5,
        max_queries: int | None = None, workdir: str | Path | None = None, describe: bool = False,
        generate: bool = False, judge_model: str | None = None, prepare_only: bool = False, source: Any = None,
        embedder: Any = None, parse: Callable[..., Any] | None = None, pause: float = 3.0, section_base: int | None = None,
        rerank: bool | None = None, reranker: Any = None, candidates: int | None = None, gen_model: str | None = None,
        reference_run: str | Path | None = None, track: bool = True, questions: list[str] | None = None,
        on_progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """``gen_model``: generation model (default HGRAG_LLM_MODEL), the judge stays HGRAG_JUDGE_MODEL / ``judge_model``.
    ``reference_run``: a previous run dir of the same sample; every question records ``context_same`` (same retrieved
    sections, in the same order), so a change of model can be compared on an unchanged context."""
    workdir = Path(workdir or RAGBENCH_DIR)
    reference = _retrieved_paths(Path(reference_run)) if reference_run else None
    src = source or HFSource(workdir / "meta")
    emit = on_progress or (lambda _m: None)
    queries, qrels, answers, pdf_urls = (src.json(n) for n in ("queries.json", "qrels.json", "answers.json", "pdf_urls.json"))
    sample = pick_sample(queries, qrels, pdf_urls, n_docs, seed=seed, negatives=negatives, sources=sources,
                         max_queries=max_queries)
    key = sample_key(sample, n_docs, sources)
    store_dir = workdir / key
    emit(f"sample {key}: {len(sample['golden'])} papers, {len(sample['negatives'])} negatives, {len(sample['query_ids'])} questions")
    pdfs, failed_pdf = fetch_pdfs(sample["golden"] + sample["negatives"], pdf_urls, workdir / "pdfs", pause=pause, on_progress=emit)
    store = build_store(pdfs, store_dir, output_dir=workdir / "output", collection=f"ragbench_{key}"[:60], describe=describe,
                        parse=parse, on_progress=emit)
    if prepare_only:
        return {"key": key, "store": store, "failed_pdf": failed_pdf}

    corpus = {}
    for pid in sample["golden"]:
        if pid in store["dirs"]:
            corpus[pid] = src.json(f"corpus/{pid}.json")
    qids = [q for q in sample["query_ids"] if qrels[q]["doc_id"] in corpus
            and (not questions or any(q.startswith(p) for p in questions))]  # ids or id prefixes (the 8 shown in the app)
    base = section_base if section_base is not None else detect_base(
        qrels, {p: len(c.get("sections") or []) for p, c in corpus.items()}, qids)
    stem_to_paper = {safe_name(pid): pid for pid in pdfs}
    gen_cfg = judge_cfg = None
    if generate:
        gen_cfg = LLMConfig(provider="anthropic", model=gen_model or "")
        from .evaluation import judge_config

        judge_cfg = judge_config(judge_model)
        if not gen_cfg.api_key:
            raise RagBenchError("ANTHROPIC_API_KEY is not set (needed by --generate).")
    if embedder is None:
        from .indexing import load_embedder

        embedder = load_embedder()
    use_rerank = config.RERANK if rerank is None else rerank
    if use_rerank and reranker is None:
        from .rerank import load_reranker

        emit(f"loading the reranker {config.RERANK_MODEL}")
        reranker = load_reranker(config.RERANK_MODEL)
    from .indexing import open_store

    run_key = tracking.new_run_key() if track else None
    dest = workdir / "runs" / (run_key or time.strftime("%Y%m%dT%H%M%S"))
    dest.mkdir(parents=True, exist_ok=True)
    client = open_store(store["qdrant_path"])
    store["hybrid"] = has_keywords(client, store["collection"])  # stores indexed before the hybrid search: dense only
    results: list[dict[str, Any]] = []
    try:
        for i, qid in enumerate(qids, start=1):
            rel = qrels[qid]
            try:
                gold = gold_text(corpus[rel["doc_id"]], rel["section_id"], base)
            except RagBenchError as exc:
                results.append({"id": qid, "paper": rel["doc_id"], "source": queries[qid]["source"], "type": queries[qid]["type"],
                                "question": queries[qid]["query"], "doc_rank": None, "section_rank": None, "context_recall": None,
                                "context_recall_at_1": None, "error": str(exc)})
                continue
            res = evaluate_query(qid, queries[qid], answers.get(qid, ""), gold, rel["doc_id"], client=client, embedder=embedder,
                                 output_dir=store["output_dir"], collection=store["collection"], k=k, stem_to_paper=stem_to_paper,
                                 gen_cfg=gen_cfg, judge_cfg=judge_cfg, rerank=use_rerank, reranker=reranker, candidates=candidates)
            if reference is not None:
                res["context_same"] = reference.get(qid) == _paths(res)
            results.append(res)
            emit(f"query {i}/{len(qids)}: doc rank {res['doc_rank']} · section rank {res['section_rank']} · recall {res['context_recall']}")
    finally:
        client.close()
    disputed = load_disputed()
    summary = summarize(results, k_sections=k, disputed=disputed)
    summary["config"] = {"docs": n_docs, "seed": seed, "negatives": negatives, "sources": sources, "k_chunks": k, "section_base": base,
                         "described": describe, "generate": generate, "embed_model": config.EMBED_MODEL_ID,
                         "keyword_weight": config.KEYWORD_WEIGHT if store["hybrid"] else 0.0,
                         "rerank": use_rerank, "rerank_model": config.RERANK_MODEL if use_rerank else None,
                         "rerank_candidates": (candidates or config.RERANK_CANDIDATES) if use_rerank else None,
                         "gen_model": gen_cfg.model if gen_cfg else None, "judge_model": judge_cfg.model if judge_cfg else None,
                         "reference_run": str(reference_run) if reference_run else None, "questions": questions,
                         "context_changed": sum(not r.get("context_same", True) for r in results) if reference is not None else None,
                         "chunk_labels": list(config.CHUNK_LABELS), "n_pdf_failed": len(failed_pdf), "n_describe_failed": len(store["stats"].get("describe_failed", {})), "n_parse_failed": len(store["stats"]["failed"]),
                         "parsed": store["stats"]["parsed"], "reused": store["stats"]["reused"], "papers_indexed": len(store["dirs"]),
                         "points_in_collection": store["index"]["points_in_collection"], "key": key}
    summary["failed"] = {"download": failed_pdf, "parse": store["stats"]["failed"],
                         "describe": store["stats"].get("describe_failed", {})}
    (dest / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
    (dest / "questions.md").write_text(qa_report(results, workdir / "pdfs", disputed=disputed), encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (dest / "report.md").write_text("# Open RAG Bench\n\n```\n" + format_summary(summary) + "\n```\n", encoding="utf-8")
    if run_key:
        flat = {m: v for m, v in summary.items() if isinstance(v, (int, float)) and v is not None}
        with tracking.track_step(tracking.RAGBENCH, "summary", run_key, **{k_: str(v) for k_, v in summary["config"].items()}) as tr:
            tr.dict(summary, "summary.json")
        tracking.finish_execution(tracking.RAGBENCH, run_key, **flat)
    return {"key": key, "dir": dest, "summary": summary, "results": results, "run_key": run_key}
