"""Local runner without Airflow: ``uv run hgrag run``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from . import config
from . import tracking


def _eval(args) -> int:
    import os

    from . import evaluation, generation
    from .indexing import load_embedder

    need_llm = not args.no_generate
    if need_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (put it in .env, or use --no-generate for retrieval metrics only).")
        return 2
    rows = evaluation.load_dataset(args.dataset, ids=args.ids.split(",") if args.ids else None,
                                   kinds=args.kind.split(",") if args.kind else None, limit=args.limit)
    if args.live:
        store = {"output_dir": config.OUTPUT_DIR, "qdrant_path": config.QDRANT_PATH, "collection": config.COLLECTION_NAME}
    else:
        workdir = config.EVAL_DIR / "store"
        if args.reuse_store and (workdir / "qdrant").exists():
            store = {"output_dir": workdir / "output", "qdrant_path": workdir / "qdrant",
                     "collection": evaluation.EVAL_COLLECTION}
        else:
            t0 = time.perf_counter()
            print("Building the benchmark store from the frozen parsing (no Docling)…", flush=True)
            store = evaluation.build_store(args.pdf_dir)
            print(f"  {store['index']['total_chunks']} chunks from {len(store['index']['documents'])} documents "
                  f"in {time.perf_counter() - t0:.0f} s (labels: {', '.join(config.CHUNK_LABELS)})", flush=True)
    gen_cfg = judge_cfg = None
    if need_llm:
        gen_cfg = generation.LLMConfig(provider="anthropic", model=args.model or "")
        if not args.no_judge:
            judge_cfg = evaluation.judge_config(args.judge_model)
    run_key = None if args.no_mlflow else tracking.new_run_key()

    def show(r):
        verdict = f"score {r['score']}/5" if r.get("score") is not None else "-"
        rank = f"rank {r['rank']}" if r.get("rank") else ("rank -" if r["pdf"] else "n/a")
        note = r.get("error") or (r.get("rationale") or "")
        print(f"{r['id']} [{r['source_kind']:<12}] {rank:<7} {verdict:<9} {note[:110]}", flush=True)

    summary = evaluation.run_eval(rows, **evaluation.store_kwargs(store), embedder=load_embedder(), k=args.k, include_children=args.children,
                                  gen_cfg=gen_cfg, judge_cfg=judge_cfg, run_key=run_key, on_row=show)
    print("\n" + evaluation.format_summary(summary))
    print(f"\nResults: {summary['dest']}" + (f" | MLflow experiment {tracking.EVAL}, run {run_key}" if run_key else ""))
    return 1 if summary["n_errors"] else 0


def _vlm_bench(args) -> int:
    from . import vlm_bench

    if args.list_models:
        for m in vlm_bench.list_models():
            print(f"hf:{m['id']:<55} {', '.join(m['providers'])}")
        return 0
    models = args.models or vlm_bench.DEFAULT_MODELS
    print(f"VLM benchmark: {len(models)} models on the control figures (judge: {'off' if args.no_judge else 'Claude'})")

    def progress(r, n=0, total=0) -> None:
        tail = f"error: {r['error']}" if "error" in r else (
            f"score {r.get('score', '-')}  recall {r.get('recall', float('nan')):.0%}  {r['seconds']} s" if "score" in r
            else f"{r['seconds']} s")
        print(f"  {r['model']:<45} {r['figure']:<12} {tail}", flush=True)

    try:
        res = vlm_bench.run_bench(models, pdf_dir=args.pdf_dir, judge=not args.no_judge, judge_model=args.judge_model,
                                  figures=args.figures.split(",") if args.figures else None, progress=progress,
                                  workers=args.workers)
    except (vlm_bench.BenchError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    print("\nSummary (best first)")
    for s in res["summary"]:
        print(f"  {s['model']:<45} score {s['score']}  recall {s['recall']}  invented/fig {s['invented_per_figure']}  "
              f"{s['seconds']} s  errors {s['errors']}")
    print(f"\nResults: {res['dir']}  (open the 'VLM comparison' page of the app to read them)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="hgrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="parse -> hierarchy -> index")
    run.add_argument("--pdf-dir", default=str(config.PDF_DIR))
    run.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    run.add_argument("--vlm", action="store_true", help="describe the figures after parsing with the HF vision model (HGRAG_FIGURE_VLM; default on when HF_TOKEN is set)")
    run.add_argument("--no-vlm", action="store_true", help=argparse.SUPPRESS)
    query = sub.add_parser("query", help="similarity search, then rebuild the matched sections")
    query.add_argument("text", help="the query")
    query.add_argument("-k", type=int, default=5, help="number of chunks to retrieve")
    query.add_argument("--children", action="store_true", help="also include sub-sections")
    query.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    query.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    ver = sub.add_parser("verify", help="check the output of the control PDFs against expected.json")
    ver.add_argument("--pdf-dir", default=str(config.CONTROL_PDF_DIR))
    bun = sub.add_parser("bundle", help="zip everything needed to analyse the runs (outputs, logs, MLflow, versions)")
    bun.add_argument("--dest", default=None, help="zip path (default data/bundles/hgrag-analysis-<time>.zip)")
    fr = sub.add_parser("freeze", help="freeze the Docling outputs of the control PDFs (fast benchmarks without Docling)")
    fr.add_argument("--pdf-dir", default=str(config.CONTROL_PDF_DIR))
    fr.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    ev = sub.add_parser("eval", help="benchmark: retrieval rank + generated answer + LLM judge on the evaluation dataset")
    ev.add_argument("--dataset", default=str(config.CONTROL_PDF_DIR / "eval_dataset.jsonl"))
    ev.add_argument("--pdf-dir", default=str(config.CONTROL_PDF_DIR))
    ev.add_argument("-k", type=int, default=5, help="chunks retrieved per question")
    ev.add_argument("--ids", help="comma-separated question ids, e.g. q01,q07")
    ev.add_argument("--kind", help="comma-separated source kinds, e.g. figure,table")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--children", action="store_true", help="include sub-sections when rebuilding")
    ev.add_argument("--no-generate", action="store_true", help="retrieval metrics only (no API call)")
    ev.add_argument("--no-judge", action="store_true", help="generate the answers but do not judge them")
    ev.add_argument("--reuse-store", action="store_true", help="do not rebuild the benchmark store from the frozen parsing")
    ev.add_argument("--live", action="store_true", help="evaluate the app's own store (data/output + data/qdrant)")
    ev.add_argument("--model", help="answer model (default HGRAG_LLM_MODEL or claude-haiku-5-5)")
    ev.add_argument("--judge-model", help="judge model (default HGRAG_JUDGE_MODEL or claude-haiku-5-5)")
    ev.add_argument("--no-mlflow", action="store_true")
    vb = sub.add_parser("vlm-bench", help="compare vision models (HF router / Claude) on the control figures")
    vb.add_argument("--models", nargs="+", help="e.g. hf:google/gemma-3-12b-it anthropic:claude-haiku-5-5")
    vb.add_argument("--figures", help="comma-separated figure ids (aurora_f1,botany_f1,botany_f2)")
    vb.add_argument("--pdf-dir", default=str(config.CONTROL_PDF_DIR))
    vb.add_argument("--no-judge", action="store_true", help="only collect the descriptions")
    vb.add_argument("--judge-model")
    vb.add_argument("--workers", type=int, default=4)
    vb.add_argument("--list-models", action="store_true", help="list the image-capable models served by the HF router")
    de = sub.add_parser("describe", help="describe the figures and tables of the parsed documents with the API models (no re-parse)")
    de.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    de.add_argument("--model", help="figure model, e.g. hf:Qwen/Qwen3-VL-30B-A3B-Instruct (default HGRAG_FIGURE_VLM)")
    de.add_argument("--force", action="store_true", help="redo the descriptions that already exist")
    rr = sub.add_parser("ragbench-report", help="rebuild questions.md (lowest score first) and the disputed-reference metrics of a ragbench run")
    rr.add_argument("run_dir", help="e.g. data/ragbench/runs/20261015T203656-d39c3b")
    rb = sub.add_parser("ragbench", help="validate the pipeline on a sample of Open RAG Bench (arXiv PDFs, tables, figures)")
    rb.add_argument("--docs", type=int, default=10, help="number of papers in the sample (10, 100, ...)")
    rb.add_argument("--seed", type=int, default=0, help="same seed = same sample")
    rb.add_argument("--negatives", type=int, default=0, help="extra papers without any question (distractors)")
    rb.add_argument("--sources", help="keep papers with questions of these kinds: text,text-image,text-table,text-table-image")
    rb.add_argument("--max-queries", type=int, help="cap the number of questions")
    rb.add_argument("-k", type=int, default=5, help="chunks retrieved per question")
    rb.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=config.RERANK,
                    help=f"cross-encoder reranking of the candidate chunks ({config.RERANK_MODEL}; HGRAG_RERANK)")
    rb.add_argument("--candidates", type=int, help=f"chunks fetched before reranking (default {config.RERANK_CANDIDATES})")
    rb.add_argument("--workdir", default=None, help="default data/ragbench")
    rb.add_argument("--describe", action=argparse.BooleanOptionalAction, default=config.DESCRIBE_PICTURES,
                    help="describe figures and tables first with the HF API (default: on when HF_TOKEN is set; --no-describe to skip)")
    rb.add_argument("--generate", action="store_true", help="also generate the answers (Claude) and judge them against the gold answer")
    rb.add_argument("--judge-model")
    rb.add_argument("--gen-model", help="generation model, e.g. claude-sonnet-5-5 (default HGRAG_LLM_MODEL); the judge is unchanged")
    rb.add_argument("--reference-run", help="previous run dir of the same sample: flags every question whose retrieved "
                    "sections differ (compare models on the same context)")
    rb.add_argument("--prepare-only", action="store_true", help="download, parse and index, without evaluating")
    rb.add_argument("--section-base", type=int, choices=(0, 1), help="base of the benchmark's section indices (default: detected)")
    rb.add_argument("--pause", type=float, default=3.0, help="seconds between arXiv downloads")
    rb.add_argument("--no-mlflow", action="store_true")
    sub.add_parser("mlflow-ui", help="start the MLflow UI on the project store (http://localhost:5000)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "mlflow-ui":
        import subprocess
        import sys

        tracking.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        print(f"MLflow store: {tracking.TRACKING_URI}\nStarting MLflow UI on http://localhost:5000 "
              "(Ctrl+C to stop). The server needs ~20-30 s to be ready; the terminal stays busy while it runs.", flush=True)
        raise SystemExit(subprocess.call(
            [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking.TRACKING_URI]
        ))

    if args.cmd == "freeze":
        from .evaluation import freeze_parsed

        done = freeze_parsed(args.pdf_dir, args.output_dir)
        print(f"frozen parsing of {len(done)} PDFs in {Path(args.pdf_dir) / 'parsed'}: {', '.join(done)}")
        return

    if args.cmd == "describe":
        from .describe import describe_document

        out = Path(args.output_dir)
        dirs = sorted(p.parent for p in out.glob("*/figures.json"))
        if not dirs:
            raise SystemExit(f"No parsed document in {out}: run the parse step first.")
        failed = 0
        for d in dirs:
            try:
                m = describe_document(d, spec=args.model, force=args.force)
            except Exception as exc:  # noqa: BLE001
                print(f"{d.name}: {exc}")
                failed += 1
                continue
            name = Path(m["source_pdf"].replace(chr(92), "/")).name
            bad = 0
            for kind, key in (("figures", "figure_vlm"), ("tables", "table_llm")):
                v = m.get(key)
                if v:
                    print(f"{name}: {v['n_described']}/{v['n_todo']} {kind} described in {v['seconds']} s "
                          f"({v['model']}), {v['n_errors']} error(s)")
                    bad += v["n_errors"]
            failed += bool(bad)
        print("Next: rebuild hierarchy + index (app: 'Rebuild hierarchy + index') so the descriptions are searchable.")
        raise SystemExit(1 if failed else 0)

    if args.cmd == "ragbench-report":
        from . import ragbench

        try:
            print(ragbench.write_qa_report(args.run_dir))
        except ragbench.RagBenchError as exc:
            print(f"error: {exc}")
            raise SystemExit(2)
        return

    if args.cmd == "ragbench":
        from . import ragbench

        try:
            res = ragbench.run(args.docs, seed=args.seed, negatives=args.negatives, k=args.k, max_queries=args.max_queries,
                               sources=args.sources.split(",") if args.sources else None, workdir=args.workdir,
                               describe=args.describe, generate=args.generate, judge_model=args.judge_model,
                               prepare_only=args.prepare_only, section_base=args.section_base, pause=args.pause,
                               rerank=args.rerank, candidates=args.candidates, gen_model=args.gen_model,
                               reference_run=args.reference_run, track=not args.no_mlflow, on_progress=lambda m: print(m, flush=True))
        except ragbench.RagBenchError as exc:
            print(f"error: {exc}")
            raise SystemExit(2)
        if args.prepare_only:
            print(f"prepared {res['key']} ({len(res['store']['dirs'])} papers indexed)")
            return
        print("\n" + ragbench.format_summary(res["summary"]))
        if args.reference_run:
            print(f"context changed against the reference run for {res['summary']['config']['context_changed']} question(s)")
        failed = res["summary"]["failed"]
        if failed["download"] or failed["parse"]:
            print(f"skipped: {len(failed['download'])} download(s), {len(failed['parse'])} parse failure(s) (see summary.json)")
        if failed.get("describe"):
            print(f"descriptions NOT done for {len(failed['describe'])} document(s): {next(iter(failed['describe'].values()))}")
        print(f"\nResults: {res['dir']}" + (f" | MLflow experiment {tracking.RAGBENCH}, run {res['run_key']}" if res["run_key"] else ""))
        return

    if args.cmd == "eval":
        raise SystemExit(_eval(args))

    if args.cmd == "vlm-bench":
        raise SystemExit(_vlm_bench(args))

    if args.cmd == "bundle":
        from .bundle import build_bundle

        path = build_bundle(args.dest)
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
        return

    if args.cmd == "verify":
        from .indexing import load_embedder, open_store
        from .verify import format_report, verify

        client = open_store()
        try:
            report = verify(args.pdf_dir, embedder=load_embedder(), client=client)
        finally:
            client.close()
        print(format_report(report))
        raise SystemExit(0 if report["ok"] else 1)

    if args.cmd == "query":
        from .retrieval import retrieve_sections

        results = retrieve_sections(
            args.text, args.k, include_children=args.children,
            run_key=None if args.no_mlflow else tracking.new_run_key(),
        )
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        for r in results if not args.json else []:
            print(f"\n{'=' * 78}\nscore={r['score']:.3f}  {r['section_path']}  "
                  f"(doc {r['doc_id']}, page {r['page_no']}, {r['n_elements']} elements)\n{'=' * 78}")
            print(r["text"])
        if not results:
            print("No result.")
        return

    # Heavy imports (Docling, torch) only for the `run` command: keeps `mlflow-ui` and `query` snappy.
    from .parsing import build_converter
    from .pipeline import build_and_save_hierarchy, index_documents, list_pdfs, parse_step

    pdfs = list_pdfs(args.pdf_dir)
    if not pdfs:
        raise SystemExit(f"No PDF found in {args.pdf_dir}")
    run_key = None if args.no_mlflow else tracking.new_run_key()
    describe = args.vlm and not args.no_vlm
    converter = build_converter(describe_pictures=describe)
    doc_dirs = []
    for pdf in pdfs:
        manifest = parse_step(pdf, converter=converter, run_key=run_key, describe_pictures=describe)
        print(json.dumps(build_and_save_hierarchy(manifest["doc_dir"], run_key=run_key)))
        doc_dirs.append(manifest["doc_dir"])
    print(json.dumps(index_documents(doc_dirs, run_key=run_key)))
    if run_key:
        print(f"MLflow run: {run_key} (uv run mlflow ui --backend-store-uri {tracking.TRACKING_URI})")
