"""Evaluation on Databricks: the questions of an evaluation dataset asked to the AI Search index, scored like
``hgrag eval`` (rank of the expected section, answer, LLM judge against the reference answer).

Same rows, same scoring and same files as the local benchmark (``results.jsonl``, ``summary.json``, ``report.md``), so
a Databricks run and a local run of the same dataset compare line by line. Only the search differs: AI Search and its
reranker instead of Qdrant and the cross-encoder; the sections are rebuilt from ``hierarchy.json`` on the volume.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import config, evaluation, tracking
from ..generation import LLMConfig


def run_eval(rows: list[dict[str, Any]], dest: str | Path, *, k: int = 10, rerank: bool | None = None,
             gen_cfg: LLMConfig | None = None, judge_cfg: LLMConfig | None = None,
             search: Callable[[str, int], list[dict[str, Any]]] | None = None, output_dir: str | Path | None = None,
             on_row: Callable[[dict[str, Any]], None] | None = None, run_key: str | None = None) -> dict[str, Any]:
    """Ask every row, write the run to ``dest`` (a folder of the volume) and return its summary.

    ``search(question, k)`` defaults to the AI Search flow of ``hgrag.dbx.retrieval`` (the tests pass a fake).
    With ``run_key``, the run is tracked in the ``hgrag_eval`` experiment: one step per question, the summary metrics on
    the parent run."""
    if search is None:
        from . import retrieval

        def search(question: str, k_: int) -> list[dict[str, Any]]:
            return retrieval.search_sections(question, k_, rerank=rerank)

    out_dir = Path(output_dir or config.dbx_output_dir())
    results = []
    for row in rows:
        res = evaluation.evaluate_row(row, client=None, embedder=None, output_dir=out_dir, collection="", k=k,
                                      include_children=False, gen_cfg=gen_cfg, judge_cfg=judge_cfg, search=search)
        results.append(res)
        evaluation._track_row(run_key, res)
        if on_row:
            on_row(res)
    summary = evaluation.summarize(results, k)
    summary["config"] = {"backend": "databricks", "k": k, "rerank": config.DBX_RERANK if rerank is None else rerank,
                         "embedding_endpoint": config.DBX_EMBED_ENDPOINT,
                         "generator": gen_cfg.describe() if gen_cfg else None,
                         "judge": judge_cfg.describe() if judge_cfg else None}
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
                                       encoding="utf-8")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (dest / "report.md").write_text(evaluation._markdown(results, summary), encoding="utf-8")
    if run_key:
        flat = {k_: v for k_, v in summary.items() if isinstance(v, (int, float)) and v is not None}
        tracking.finish_execution(tracking.EVAL, run_key, failed=bool(summary["n_errors"]), **flat)
    summary["dest"] = str(dest)
    return summary
