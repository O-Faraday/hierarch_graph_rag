"""Page 1: run (or re-run) the indexing pipeline, follow its task graph, statuses, logs, MLflow."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import ui
from mlflow_view import MLFLOW_UI_URL, execution_steps

from hgrag import config, runner, tracking

STEP_LABELS = {"parse": "Parse (Docling)", "hierarchy": "Hierarchy", "index": "Index (Qdrant)"}


def _fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "—"


def _short(info) -> str:
    return ", ".join(f"{k}={v}" for k, v in info.items()) if info else ""


def launch(pdf_dir, vlm, steps, pdfs=None) -> None:
    try:
        st.session_state["selected_run"] = runner.start_indexing(
            pdf_dir, describe_pictures=vlm, steps=steps, pdfs=pdfs)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    st.rerun()


ui.inject_css()
ui.page_header(
    "Pipelines",
    "PDF → Docling parsing → heading tree → paragraph chunks → Qdrant. Each run is a separate process "
    f"tracked in MLflow (experiment <code>{tracking.INDEXING}</code>) · <a href='{MLFLOW_UI_URL}'>open MLflow</a>",
)

active = runner.active_run()
busy = active is not None
if busy:
    st.info(f"Run `{active['run_key']}` in progress: new runs are disabled until it ends.")

# ---- launch -------------------------------------------------------------------------------
def _use_control() -> None:
    st.session_state["pdf_dir"] = str(config.CONTROL_PDF_DIR)


st.session_state.setdefault("pdf_dir", str(config.PDF_DIR))  # state only: no `value=` next to `key=`
pdf_dir = st.text_input("PDF folder", key="pdf_dir")
st.button("Use the 3 control PDFs", on_click=_use_control, icon=":material/fact_check:",
          help="Known content: headings, figures, tables and one needle sentence each (see Control check below)")
docs = runner.document_state(pdf_dir)
vlm = st.checkbox(f"Describe figures and tables after parsing ({config.FIGURE_VLM.split(':', 1)[-1]})", value=config.DESCRIBE_PICTURES,
                  help="Models called through the Hugging Face API (needs HF_TOKEN in .env). A figure description "
                       "becomes a searchable chunk, a table description is added to the embedding of its chunks; "
                       "the PNG is still sent to the answer LLM.")
if vlm and not os.environ.get("HF_TOKEN") and config.FIGURE_VLM.startswith("hf:"):
    st.warning("`HF_TOKEN` is not set: figures will not be described (add it to `.env` and restart the app).")

st.markdown("##### Run")
p1, p2, p3 = st.columns(3)
if p1.button("Run whole pipeline", type="primary", icon=":material/play_arrow:", disabled=busy,
             width="stretch", help="Parse → hierarchy → index for every PDF"):
    launch(pdf_dir, vlm, list(runner.STEPS))
if p2.button("Rebuild hierarchy + index", icon=":material/refresh:", disabled=busy, width="stretch",
             help="Reuse the Docling parsing on disk (no re-parse), rebuild the tree and the chunks"):
    launch(pdf_dir, vlm, ["hierarchy", "index"])
if p3.button("Re-index only", icon=":material/refresh:", disabled=busy, width="stretch",
             help="Re-chunk and re-embed from the parsing on disk (e.g. after changing the model or the chunk size)"):
    launch(pdf_dir, vlm, ["index"])

with st.expander("Custom run: choose the documents and the steps"):
    with st.form("custom"):
        chosen = st.multiselect("Documents", [d["pdf"] for d in docs], default=[d["pdf"] for d in docs])
        steps = st.multiselect("Steps", list(runner.STEPS), default=list(runner.STEPS),
                               format_func=STEP_LABELS.get)
        c1, c2 = st.columns(2)
        go = c1.form_submit_button("Run selection", type="primary", icon=":material/play_arrow:", disabled=busy)
        stop = c2.form_submit_button("Cancel run", icon=":material/stop:", disabled=not busy)
    if go:
        if not chosen or not steps:
            st.error("Select at least one document and one step.")
        else:
            launch(pdf_dir, vlm, steps, None if len(chosen) == len(docs) else chosen)
    if stop and active:
        runner.cancel(active["run_key"])
        st.rerun()
if busy and st.button("Cancel run", key="cancel_top", icon=":material/stop:"):
    runner.cancel(active["run_key"])
    st.rerun()

if st.button("Describe figures and tables of the parsed documents", icon=":material/image_search:", disabled=busy,
             help="Runs the API models on the figures and tables already extracted (no Docling): then use "
                  "'Rebuild hierarchy + index' so the descriptions become searchable."):
    from hgrag.describe import describe_document

    done = []
    with st.spinner("Describing figures…"):
        for d in (d for d in docs if d["parsed"]):
            try:
                m = describe_document(Path(config.OUTPUT_DIR) / d["doc_id"])
                for kind, key in (("figures", "figure_vlm"), ("tables", "table_llm")):
                    v = m.get(key)
                    if v:
                        done.append(f"{d['pdf']}: {v['n_described']}/{v['n_todo']} {kind} described in {v['seconds']} s"
                                    + (f", {v['n_errors']} error(s): {v['errors'][0]}" if v["n_errors"] else ""))
            except Exception as exc:  # noqa: BLE001
                done.append(f"{d['pdf']}: {exc}")
    st.info("\n\n".join(done) or "No parsed document yet.")

with st.expander(f"Documents on disk ({len(docs)})", expanded=False):
    if docs:
        ok = {True: "done", False: "—"}
        st.dataframe(
            pd.DataFrame(docs).assign(parsed=lambda d: d.parsed.map(ok), hierarchy=lambda d: d.hierarchy.map(ok)),
            hide_index=True, width="stretch",
        )
    else:
        st.warning("No PDF in this folder.")

with st.expander("Inspect a document: what each step produced"):
    parsed = [d for d in docs if d["parsed"]]
    if not parsed:
        st.caption("Nothing parsed yet.")
    else:
        pick = st.selectbox("Document", parsed, format_func=lambda d: d["pdf"], key="inspect_doc")
        doc_dir = Path(config.OUTPUT_DIR) / pick["doc_id"]
        t1, t2, t3, t4 = st.tabs(["Parse", "Hierarchy", "Chunks", "Figures"])
        with t1:
            mf = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame({"field": [k for k, v in mf.items() if not isinstance(v, (dict, list))],
                                       "value": [str(v) for v in mf.values() if not isinstance(v, (dict, list))]}),
                         hide_index=True, width="stretch")  # strings only: mixed types break Arrow
            if mf.get("labels"):
                st.bar_chart(pd.Series(mf["labels"], name="items per Docling label"), height=200)
            if (doc_dir / "document.md").exists():
                with st.expander("Docling Markdown export"):
                    st.markdown((doc_dir / "document.md").read_text(encoding="utf-8"))
        with t2:
            tree = doc_dir / "hierarchy_tree.txt"
            st.code(tree.read_text(encoding="utf-8") if tree.exists() else "Hierarchy step not run for this document.",
                    language="text", wrap_lines=False)
        with t3:
            ch = doc_dir / "chunks.jsonl"
            if ch.exists():
                rows = [json.loads(x) for x in ch.read_text(encoding="utf-8").splitlines() if x]
                st.caption(f"{len(rows)} paragraph chunks")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("Index step not run for this document.")
        with t4:
            figs = doc_dir / "figures.json"
            records = json.loads(figs.read_text(encoding="utf-8")) if figs.exists() else []
            if not records:
                st.caption("No figure in this document.")
            for rec in records:
                png = doc_dir / rec["png_path"] if rec.get("png_path") else None
                c1, c2 = st.columns([1, 2])
                if png and png.exists():
                    c1.image(str(png), width=260)
                c2.json({k: rec[k] for k in ("figure_index", "page_no", "png_bytes", "description", "classification",
                                              "image", "caption") if k in rec}, expanded=False)

with st.expander("Export for analysis"):
    st.caption("One zip: Docling outputs of every document (JSON, Markdown, figures, tree, chunks), the last 10 run "
               "logs, the MLflow database, library versions, the control expectations and a structural check.")
    if st.button("Prepare the analysis zip", icon=":material/inventory_2:"):
        from hgrag.bundle import build_bundle

        with st.spinner("Collecting files…"):
            st.session_state["bundle"] = build_bundle()
    ready = st.session_state.get("bundle")
    if ready and Path(ready).exists():
        st.download_button(f"Download {Path(ready).name} ({Path(ready).stat().st_size / 1e6:.1f} MB)",
                           data=Path(ready).read_bytes(), file_name=Path(ready).name, mime="application/zip",
                           icon=":material/download:", type="primary")
        st.caption(f"Also saved on disk: `{ready}`")

with st.expander("Control check: does the pipeline give the expected result?"):
    st.caption("Compares the output of the 3 control PDFs with their expected structure: headings and levels, "
               "figures, tables, chunks, and the section returned for each needle query.")
    if st.button("Run the check", disabled=busy, icon=":material/rule:"):
        from hgrag import verify
        from hgrag.indexing import open_store
        from models import embedder

        try:
            client = open_store()
            try:
                report = verify.verify(config.CONTROL_PDF_DIR, embedder=embedder(), client=client)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            st.error(f"{exc}")
        else:
            (st.success if report["ok"] else st.error)(
                f"{report['n_ok']} of {report['n_checks']} checks passed", icon=":material/check_circle:" if report["ok"] else ":material/error:")
            st.dataframe(pd.DataFrame(report["checks"]).assign(
                ok=lambda d: d.ok.map({True: "passed", False: "FAILED"}),
                expected=lambda d: d.expected.astype(str), actual=lambda d: d.actual.astype(str)),
                hide_index=True, width="stretch")

st.divider()
auto = st.toggle("Auto-refresh every 2 s", value=True)


@st.fragment(run_every=2 if auto else None)
def monitor() -> None:
    runs = runner.list_runs(15)
    if not runs:
        st.info("No run yet: use one of the buttons above.")
        return
    keys = [r["run_key"] for r in runs]
    wanted = st.session_state.get("selected_run")
    run_key = st.selectbox(
        "Run", keys, index=keys.index(wanted) if wanted in keys else 0,
        format_func=lambda k: f"{k} — {next(r['state'] for r in runs if r['run_key'] == k)}",
    )
    status = next(r for r in runs if r["run_key"] == run_key)

    p = status["params"]
    st.html(ui.run_header(status, time.time()))
    if status.get("error"):
        st.error(status["error"])
    st.html(ui.run_board(status))

    rows = []
    for tid in status["graph"]["tasks"]:
        for inst in status["tasks"][tid]["instances"] or [
            {"item": None, "state": "pending", "seconds": None, "info": None, "error": None}
        ]:
            rows.append({
                "task": tid, "item": inst["item"] or "", "state": inst["state"],
                "seconds": round(inst["seconds"], 1) if inst["seconds"] else None,
                "output": _short(inst["info"]), "error": inst["error"] or "",
            })
    table = pd.DataFrame(rows)
    st.markdown("##### Tasks")
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="multi-row", key=f"inst_{run_key}")

    # ---- re-run from this run -------------------------------------------------------------
    finished = status["state"] != runner.RUNNING
    b1, b2 = st.columns(2)
    failed_pdfs, failed_steps = runner.retry_plan(status)
    if b1.button(f"Retry failed ({', '.join(failed_pdfs)})" if failed_pdfs else "Retry failed (none)",
                 icon=":material/replay:", disabled=busy or not failed_pdfs, width="stretch", key="retry"):
        launch(p["pdf_dir"], p.get("describe_pictures", config.DESCRIBE_PICTURES), failed_steps, failed_pdfs)
    picked = table.iloc[event.selection.rows] if event.selection.rows else table.iloc[0:0]
    if b2.button(f"Re-run selected rows ({len(picked)})", icon=":material/replay:", disabled=busy or picked.empty,
                 width="stretch", key="rerun_sel",
                 help="Re-runs the selected documents from the earliest selected step onward"):
        first = min(picked.task, key=lambda t: runner.STEPS.index(t) if t in runner.STEPS else 0)
        items = sorted({i for i in picked["item"] if i})
        launch(p["pdf_dir"], p.get("describe_pictures", config.DESCRIBE_PICTURES), list(runner.STEPS[runner.STEPS.index(first):])
               if first in runner.STEPS else list(runner.STEPS), items or None)
    if not finished:
        st.caption("Re-run buttons are enabled when the run has ended.")

    with st.expander("MLflow steps of this run", expanded=True):
        steps_ml = execution_steps(tracking.INDEXING, run_key)
        if steps_ml:
            st.dataframe(pd.DataFrame(steps_ml).drop(columns=["run_id"]), hide_index=True, width="stretch")
        else:
            st.caption("No MLflow step recorded (yet) for this run.")

    with st.expander("Log"):
        st.code(runner.read_log(run_key) or "(empty)", language="log", height=320)


monitor()
