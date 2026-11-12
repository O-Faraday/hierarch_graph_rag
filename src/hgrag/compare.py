"""Paired comparison of two ragbench runs of the same sample (``hgrag ragbench-compare``).

Two runs of the same questions are compared question by question: the mean difference of a metric, and its standard
error with the questions of one paper grouped (the questions of a paper are not independent: a paper that parses badly
lowers all of them). A difference within two standard errors cannot be told from the noise of the generation.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


class CompareError(RuntimeError):
    pass


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    f = Path(run_dir) / "results.jsonl"
    if not f.exists():
        raise CompareError(f"{f} not found")
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def _k_sections(run_dir: str | Path) -> int:
    f = Path(run_dir) / "summary.json"
    return int(json.loads(f.read_text(encoding="utf-8")).get("k_sections") or 10) if f.exists() else 10


def clustered_se(diffs: list[float], clusters: list[Any]) -> float | None:
    """Standard error of the mean of ``diffs``, the values of a cluster (a paper) being correlated.

    ``sqrt(G / (G - 1) * sum_c (sum_{i in c} (d_i - mean))**2) / n`` for G clusters; None below two clusters."""
    n = len(diffs)
    groups = len(set(clusters))
    if n == 0 or groups < 2:
        return None
    mean = sum(diffs) / n
    by_cluster: dict[Any, float] = {}
    for d, c in zip(diffs, clusters, strict=True):
        by_cluster[c] = by_cluster.get(c, 0.0) + (d - mean)
    return math.sqrt(groups / (groups - 1) * sum(v * v for v in by_cluster.values())) / n


def _metric(name: str, rows: list[tuple[dict[str, Any], dict[str, Any]]], get) -> dict[str, Any] | None:
    pairs = []
    for a, b in rows:
        va, vb = get(a), get(b)
        if va is not None and vb is not None:
            pairs.append((float(va), float(vb), a.get("paper")))
    if not pairs:
        return None
    diffs = [vb - va for va, vb, _ in pairs]
    se = clustered_se(diffs, [p for _, _, p in pairs])
    mean = sum(diffs) / len(diffs)
    return {"metric": name, "n": len(pairs), "a": round(statistics.fmean(va for va, _, _ in pairs), 3),
            "b": round(statistics.fmean(vb for _, vb, _ in pairs), 3), "diff": round(mean, 3),
            "se": None if se is None else round(se, 3), "better": sum(d > 1e-9 for d in diffs),
            "worse": sum(d < -1e-9 for d in diffs),
            "significant": se is not None and abs(mean) > 2 * se}


def compare(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    """Paired metrics of run B against run A, over the questions both runs answered (matched by id)."""
    a_rows = {r["id"]: r for r in load_results(run_a) if not r.get("error")}
    b_rows = {r["id"]: r for r in load_results(run_b) if not r.get("error")}
    ids = [i for i in a_rows if i in b_rows]
    if not ids:
        raise CompareError("the two runs share no question")
    pairs = [(a_rows[i], b_rows[i]) for i in ids]
    k = max(_k_sections(run_a), _k_sections(run_b))
    answerable = [(a, b) for a, b in pairs if not a.get("unanswerable")]

    def hit(limit: int | None):
        def get(r: dict[str, Any]) -> float | None:
            rank = r.get("section_rank")
            return float(rank is not None and (limit is None or rank <= limit)) if "section_rank" in r else None
        return get

    def mrr(r: dict[str, Any]) -> float | None:
        return (1 / r["section_rank"] if r.get("section_rank") else 0.0) if "section_rank" in r else None

    metrics = [
        _metric("answer F1 (all questions)", pairs, lambda r: r.get("answer_f1")),
        _metric("judge score (1-5)", pairs, lambda r: r.get("score")),
        _metric("section hit@1", answerable, hit(1)),
        _metric(f"section hit@{k}", answerable, hit(k)),
        _metric("MRR", answerable, mrr),
        _metric("context recall", answerable, lambda r: r.get("context_recall")),
    ]
    by_type = {}
    for t in sorted({a.get("type") for a, _ in pairs if a.get("type")}):
        sub = [(a, b) for a, b in pairs if a.get("type") == t]
        m = _metric("answer F1", sub, lambda r: r.get("answer_f1"))
        if m:
            by_type[t] = m
    return {"run_a": str(run_a), "run_b": str(run_b), "n_common": len(ids),
            "n_only_a": len(a_rows) - len(ids), "n_only_b": len(b_rows) - len(ids), "k_sections": k,
            "metrics": [m for m in metrics if m], "by_type": by_type}


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def format_comparison(c: dict[str, Any]) -> str:
    lines = [f"A = {c['run_a']}", f"B = {c['run_b']}",
             f"{c['n_common']} questions in common (only in A: {c['n_only_a']}, only in B: {c['n_only_b']}); "
             "difference B - A, standard error with the questions of a paper grouped", "",
             f"{'metric':<28}{'n':>5}{'A':>8}{'B':>8}{'B - A':>9}{'± SE':>8}  {'better/worse':<13} verdict"]

    def row(m: dict[str, Any], label: str) -> str:
        verdict = ("beyond the noise (> 2 SE)" if m["significant"]
                   else "within the noise" if m["se"] is not None else "no standard error")
        se = "" if m["se"] is None else f"{m['se']:.3f}"
        return (f"{label:<28}{m['n']:>5}{m['a']:>8.3f}{m['b']:>8.3f}{_signed(m['diff']):>9}"
                f"{se:>8}  {m['better']}/{m['worse']:<11} {verdict}")

    lines += [row(m, m["metric"]) for m in c["metrics"]]
    if c["by_type"]:
        lines += ["", "answer F1 by answer type:"] + [row(m, f"  {t}") for t, m in c["by_type"].items()]
    return "\n".join(lines)


def write_comparison(run_a: str | Path, run_b: str | Path) -> tuple[dict[str, Any], Path]:
    """Compare, and write ``comparison.md`` in the directory of run B."""
    c = compare(run_a, run_b)
    out = Path(run_b) / "comparison.md"
    out.write_text("# Comparison with " + Path(run_a).name + "\n\n```\n" + format_comparison(c) + "\n```\n", encoding="utf-8")
    return c, out
