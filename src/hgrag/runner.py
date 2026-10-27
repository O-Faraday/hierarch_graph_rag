"""Local runner of the indexing pipeline (no Airflow): same steps, same MLflow tracking.

The pipeline runs in its own process (``python -m hgrag.runner indexing``) so that the UI stays
responsive and heavy models are released at the end. Progress is written to
``data/runs/<run_key>/status.json`` (atomic writes) and the log to ``log.txt``; the Streamlit
page only reads those files. One indexing run at a time (Qdrant local mode is single-process).

The task graph mirrors the Airflow DAG ``hgrag_pipeline``, so Airflow can replace this runner
later without changing the steps (``hgrag.pipeline`` functions).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from . import config, tracing, tracking
from .ids import pdf_name

logger = logging.getLogger(__name__)

RUNS_DIR = Path(os.environ.get("HGRAG_RUNS_DIR", config.PROJECT_ROOT / "data" / "runs"))

INDEXING_GRAPH: dict[str, Any] = {
    "pipeline": tracking.INDEXING,
    "tasks": ["list_pdfs", "parse", "hierarchy", "index"],
    "edges": [["list_pdfs", "parse"], ["parse", "hierarchy"], ["hierarchy", "index"]],
}

PENDING, RUNNING, SUCCESS, FAILED, CANCELLED, SKIPPED = (
    "pending", "running", "success", "failed", "cancelled", "skipped")
STEPS = ("parse", "hierarchy", "index")


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------
def _now() -> float:
    return time.time()


def run_dir(run_key: str, runs_dir: Path | None = None) -> Path:
    return (runs_dir or RUNS_DIR) / run_key


class StatusWriter:
    """Single writer of ``status.json`` (the runner process). Writes are atomic."""

    def __init__(self, run_key: str, graph: dict[str, Any], params: dict[str, Any], runs_dir: Path | None = None):
        self.path = run_dir(run_key, runs_dir) / "status.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {
            "pipeline": graph["pipeline"], "run_key": run_key, "state": RUNNING, "pid": os.getpid(),
            "started_at": _now(), "ended_at": None, "params": params, "error": None,
            "graph": {"tasks": graph["tasks"], "edges": graph["edges"]},
            "tasks": {t: {"instances": []} for t in graph["tasks"]},
        }
        self.flush()

    def flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def instance(self, task: str, item: str | None = None) -> dict[str, Any]:
        """Create (or return) the instance of ``task`` for ``item`` and mark it running."""
        insts = self.data["tasks"][task]["instances"]
        for i in insts:
            if i["item"] == item:
                return i
        inst = {"item": item, "state": PENDING, "started_at": None, "ended_at": None, "seconds": None,
                "error": None, "info": None}
        insts.append(inst)
        return inst

    def start(self, task: str, item: str | None = None) -> dict[str, Any]:
        inst = self.instance(task, item)
        inst.update(state=RUNNING, started_at=_now())
        self.flush()
        return inst

    def finish(self, inst: dict[str, Any], *, error: str | None = None, info: Any = None) -> None:
        inst.update(state=FAILED if error else SUCCESS, ended_at=_now(), error=error, info=info)
        inst["seconds"] = inst["ended_at"] - (inst["started_at"] or inst["ended_at"])
        self.flush()

    def skip(self, task: str, reason: str) -> None:
        """Task not selected for this run: its inputs are read from disk (outputs of an earlier run)."""
        inst = self.instance(task)
        inst.update(state=SKIPPED, info={"reason": reason})
        self.flush()

    def end(self, state: str, error: str | None = None) -> None:
        self.data.update(state=state, ended_at=_now(), error=error)
        self.flush()


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    """State of a task aggregated over its (possibly mapped) instances."""
    insts = task["instances"]
    states = [i["state"] for i in insts]
    if not insts:
        state = PENDING
    elif FAILED in states:
        state = FAILED
    elif RUNNING in states:
        state = RUNNING
    elif all(s == SKIPPED for s in states):
        state = SKIPPED
    elif all(s in (SUCCESS, SKIPPED) for s in states):
        state = SUCCESS
    else:
        state = PENDING
    return {"state": state, "n": len(insts), "n_success": states.count(SUCCESS),
            "seconds": sum(i["seconds"] or 0 for i in insts), "mapped": len(insts) > 1 or any(
                i["item"] for i in insts)}


# ---------------------------------------------------------------------------
# Reading side (used by the UI)
# ---------------------------------------------------------------------------
def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def read_status(run_key: str, runs_dir: Path | None = None) -> dict[str, Any] | None:
    path = run_dir(run_key, runs_dir) / "status.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):  # caught while being replaced
        return None
    if data["state"] == RUNNING and not _pid_alive(data.get("pid")):
        data.update(state=FAILED, error=data.get("error") or "the runner process died (see log.txt)",
                    ended_at=data.get("ended_at") or _now())
    return data


def list_runs(limit: int = 20, runs_dir: Path | None = None) -> list[dict[str, Any]]:
    base = runs_dir or RUNS_DIR
    if not base.exists():
        return []
    out = []
    for d in sorted((p for p in base.iterdir() if (p / "status.json").exists()), reverse=True)[:limit]:
        st = read_status(d.name, base)
        if st:
            out.append(st)
    return out


def active_run(runs_dir: Path | None = None) -> dict[str, Any] | None:
    for st in list_runs(5, runs_dir):
        if st["state"] == RUNNING:
            return st
    return None


def read_log(run_key: str, tail: int = 200, runs_dir: Path | None = None) -> str:
    path = run_dir(run_key, runs_dir) / "log.txt"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:])


def start_indexing(pdf_dir: str | Path | None = None, *, describe_pictures: bool = config.DESCRIBE_PICTURES,
                   steps: tuple[str, ...] | list[str] = STEPS, pdfs: list[str] | None = None,
                   runs_dir: Path | None = None, assess_judges: bool | None = None) -> str:
    """Launch the indexing pipeline in a detached subprocess and return its ``run_key``.

    ``steps`` selects what to (re)run (the others reuse the outputs already on disk) and
    ``pdfs`` restricts the run to some file names (``None`` = every PDF of the folder).
    ``assess_judges`` sets HGRAG_ASSESS_JUDGES for this run only (``None`` = the configured value).
    """
    if active_run(runs_dir):
        raise RuntimeError("An indexing run is already in progress (Qdrant local mode is single-process).")
    run_key = tracking.new_run_key()
    d = run_dir(run_key, runs_dir)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "hgrag.runner", "indexing", "--run-key", run_key,
           "--pdf-dir", str(pdf_dir or config.PDF_DIR), "--runs-dir", str(runs_dir or RUNS_DIR)]
    cmd.append("--vlm" if describe_pictures else "--no-vlm")
    cmd += ["--steps", ",".join(steps)]
    for name in pdfs or []:
        cmd += ["--pdf", name]
    log = open(d / "log.txt", "ab")  # noqa: SIM115 - handed to the child process
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    if assess_judges is not None:
        env["HGRAG_ASSESS_JUDGES"] = "1" if assess_judges else "0"
    kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)  # noqa: S603
    log.close()
    # The child writes status.json itself; wait a moment so the UI finds it right away.
    for _ in range(50):
        if (d / "status.json").exists() or proc.poll() is not None:
            break
        time.sleep(0.1)
    return run_key


def cancel(run_key: str, runs_dir: Path | None = None) -> bool:
    st = read_status(run_key, runs_dir)
    if not st or st["state"] != RUNNING:
        return False
    try:
        import psutil

        proc = psutil.Process(st["pid"])
        for child in proc.children(recursive=True):
            child.terminate()
        proc.terminate()
    except Exception:  # noqa: BLE001
        return False
    path = run_dir(run_key, runs_dir) / "status.json"
    st.update(state=CANCELLED, ended_at=_now(), error="cancelled by the user")
    path.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    tracking.finish_execution(tracking.INDEXING, run_key, failed=True)
    return True


# ---------------------------------------------------------------------------
# Execution side (runner process)
# ---------------------------------------------------------------------------
def document_state(pdf_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """What exists on disk for each PDF of the folder (outputs of previous runs)."""
    from . import pipeline
    from .ids import compute_doc_id

    rows = []
    for pdf in pipeline.list_pdfs(pdf_dir or config.PDF_DIR):
        doc_id = compute_doc_id(pdf)
        d = config.OUTPUT_DIR / doc_id
        rows.append({
            "pdf": Path(pdf).name, "doc_id": doc_id,
            "parsed": (d / "manifest.json").exists() and (d / "document.json").exists(),
            "hierarchy": (d / "hierarchy.json").exists(),
            "figures": len(list((d / "figures").glob("*.png"))) if (d / "figures").exists() else 0,
        })
    return rows


def retry_plan(status: dict[str, Any]) -> tuple[list[str], list[str]]:
    """From a finished run: the PDFs that failed and the first step to redo for them."""
    failed: dict[str, str] = {}
    for step in STEPS:
        for inst in status["tasks"].get(step, {"instances": []})["instances"]:
            if inst["state"] == FAILED and inst["item"] and inst["item"] not in failed:
                failed[inst["item"]] = step
    if not failed:
        return [], list(STEPS)
    first = min(failed.values(), key=STEPS.index)
    return sorted(failed), list(STEPS[STEPS.index(first):])


def _doc_dir_for(pdf: str | Path) -> Path:
    from .ids import compute_doc_id

    return config.OUTPUT_DIR / compute_doc_id(pdf)


def run_indexing(
    run_key: str,
    pdf_dir: str | Path | None = None,
    *,
    describe_pictures: bool = config.DESCRIBE_PICTURES,
    steps: tuple[str, ...] | list[str] = STEPS,
    pdfs: list[str] | None = None,
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    """parse (per PDF) -> hierarchy (per PDF) -> index, with status file and MLflow tracking.

    ``steps`` chooses which steps run; a step that is not selected reuses the outputs of a
    previous run found in ``data/output`` (shown as *skipped*). ``pdfs`` limits the run to some
    file names. A failing PDF does not stop the others; the run fails only if nothing can be
    indexed. Re-indexing a document replaces its chunks (idempotent).
    """
    from . import pipeline  # attribute access at call time: easy to substitute in tests

    pdf_dir = pdf_dir or config.PDF_DIR
    steps = [s_ for s_ in STEPS if s_ in set(steps)]
    status = StatusWriter(run_key, INDEXING_GRAPH,
                          {"pdf_dir": str(pdf_dir), "describe_pictures": describe_pictures,
                           "steps": steps, "pdfs": pdfs or "all", "assess_judges": config.ASSESS_JUDGES}, runs_dir)
    # one MLflow trace per PDF: its parse, hierarchy and index steps are child spans (all share the run_key session)
    traces: dict[str, tracing.Span] = {}

    def pdf_trace(name: str) -> tracing.Span:
        if name not in traces:
            traces[name] = tracing.root("indexing", tracking.INDEXING, run_key, tags={"hgrag.pdf": name}, inputs={
                "pdf": name, "pdf_dir": str(pdf_dir), "steps": steps, "describe_pictures": describe_pictures})
        return traces[name]

    failed_any = False
    try:
        if not steps:
            raise ValueError("No step selected.")
        inst = status.start("list_pdfs")
        pdfs_all = pipeline.list_pdfs(pdf_dir)
        if pdfs:
            wanted = set(pdfs)
            missing = wanted - {Path(p).name for p in pdfs_all}
            if missing:
                raise FileNotFoundError(f"PDF not found in {pdf_dir}: {', '.join(sorted(missing))}")
            pdfs_all = [p for p in pdfs_all if Path(p).name in wanted]
        status.finish(inst, info={"n_pdfs": len(pdfs_all)})
        if not pdfs_all:
            raise FileNotFoundError(f"No PDF found in {pdf_dir}")

        # -- parse ---------------------------------------------------------------------------
        manifests: list[dict[str, Any]] = []
        if "parse" in steps:
            from .parsing import build_converter

            converter = build_converter(describe_pictures=describe_pictures)  # loaded once
            for pdf in pdfs_all:
                name = Path(pdf).name
                inst = status.start("parse", name)
                try:
                    manifest = pipeline.parse_step(pdf, converter=converter, run_key=run_key,
                                                   describe_pictures=describe_pictures, trace=pdf_trace(name))
                    status.finish(inst, info={"doc_id": manifest["doc_id"], **{k: manifest[k] for k in ("n_pages", "n_figures", "n_tables") if k in manifest}})
                    manifests.append(manifest)
                except Exception as exc:  # noqa: BLE001
                    logger.error("parse failed for %s:\n%s", name, traceback.format_exc())
                    status.finish(inst, error=f"{type(exc).__name__}: {exc}")
                    pdf_trace(name).end(error=exc)
                    failed_any = True
        else:
            status.skip("parse", "reusing the parsing already on disk")
            for pdf in pdfs_all:
                mpath = _doc_dir_for(pdf) / "manifest.json"
                if mpath.exists():
                    manifests.append(json.loads(mpath.read_text(encoding="utf-8")))
                else:
                    inst = status.start("hierarchy", Path(pdf).name)
                    status.finish(inst, error="not parsed yet: select the parse step")
                    failed_any = True

        # -- hierarchy -----------------------------------------------------------------------
        doc_dirs: list[str] = []
        if "hierarchy" in steps:
            for manifest in manifests:
                name = pdf_name(manifest["source_pdf"])
                inst = status.start("hierarchy", name)
                try:
                    summary = pipeline.build_and_save_hierarchy(manifest["doc_dir"], run_key=run_key,
                                                                trace=pdf_trace(name))
                    status.finish(inst, info={k: summary[k] for k in ("n_sections", "n_leaf_sections", "n_elements")})
                    doc_dirs.append(manifest["doc_dir"])
                except Exception as exc:  # noqa: BLE001
                    logger.error("hierarchy failed for %s:\n%s", name, traceback.format_exc())
                    status.finish(inst, error=f"{type(exc).__name__}: {exc}")
                    pdf_trace(name).end(error=exc)
                    failed_any = True
        else:
            status.skip("hierarchy", "reusing hierarchy.json already on disk")
            doc_dirs = [m["doc_dir"] for m in manifests]

        # -- index ---------------------------------------------------------------------------
        if not doc_dirs:
            raise RuntimeError("No document available: nothing to index.")
        if "index" in steps:
            inst = status.start("index")
            try:
                doc_traces = {m["doc_dir"]: pdf_trace(pdf_name(m["source_pdf"])) for m in manifests
                              if m["doc_dir"] in doc_dirs}
                result = pipeline.index_documents(doc_dirs, run_key=run_key, doc_traces=doc_traces)
                status.finish(inst, info={"total_chunks": result["total_chunks"], "n_documents": len(result.get("documents", {}))})
            except Exception as exc:  # noqa: BLE001
                logger.error("index failed:\n%s", traceback.format_exc())
                status.finish(inst, error=f"{type(exc).__name__}: {exc}")
                raise
        else:
            status.skip("index", "Qdrant left untouched")
            tracking.finish_execution(tracking.INDEXING, run_key)
        status.end(SUCCESS if not failed_any else FAILED,
                   "some documents failed (see tasks)" if failed_any else None)
        for name, trace in traces.items():  # a trace already ended (failed step) is left as it is
            trace.end({"run_key": run_key, "steps": steps})
    except Exception as exc:  # noqa: BLE001
        status.end(FAILED, f"{type(exc).__name__}: {exc}")
        tracking.finish_execution(tracking.INDEXING, run_key, failed=True)
        for trace in traces.values():
            trace.end(error=exc)
    return status.data


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hgrag.runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("indexing")
    p.add_argument("--run-key", default=None)
    p.add_argument("--pdf-dir", default=str(config.PDF_DIR))
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--vlm", action="store_true", help="describe figures with the HF vision model after parsing")
    p.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of parse,hierarchy,index")
    p.add_argument("--pdf", action="append", help="restrict to this PDF file name (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    final = run_indexing(args.run_key or tracking.new_run_key(), args.pdf_dir,
                         describe_pictures=(True if args.vlm else False if args.no_vlm else config.DESCRIBE_PICTURES),
                         steps=[x for x in args.steps.split(",") if x], pdfs=args.pdf,
                         runs_dir=Path(args.runs_dir))
    raise SystemExit(0 if final["state"] == SUCCESS else 1)


if __name__ == "__main__":
    main()
