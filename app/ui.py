"""Visual layer: CSS and HTML components (theme-agnostic: colours derive from the current text colour).

The centrepiece is the *run board*: documents as rows, pipeline steps as columns, one cell per
(document, step) with its state, duration and error. It replaces a generic box-and-arrow graph
because the question while a run is going is "which document is at which step?".
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import streamlit as st

from hgrag import runner

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..700&display=swap');

:root {
  /* pastel palette: running = blue, success = green, failed = red */
  --hg-run: #7FB2F5; --hg-ok: #86D0A6; --hg-bad: #F29A94; --hg-skip: #B9B2EA; --hg-wait: #BFC8D6;
  --hg-ink: #1F2A37;
  --hg-brand: #5B93E6;
  --hg-line: color-mix(in srgb, currentColor 15%, transparent);
  --hg-faint: color-mix(in srgb, currentColor 6%, transparent);
  --hg-muted: color-mix(in srgb, currentColor 60%, transparent);
  --hg-display: 'Bricolage Grotesque', 'Segoe UI Variable Display', 'Segoe UI', system-ui, sans-serif;
}
.block-container { max-width: 1180px; padding-top: 2.6rem; }
h1, h2, h3 { font-family: var(--hg-display); letter-spacing: -0.02em; font-weight: 600; }
h1 { font-size: 2.3rem; line-height: 1.1; margin-bottom: .15rem; }
.hg-lede { color: var(--hg-muted); max-width: 62ch; margin: 0 0 1.4rem; line-height: 1.5; }
.hg-lede a { color: var(--hg-brand); }

/* tabs and buttons */
[data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--hg-line); }
[data-baseweb="tab"] { padding-left: 0 !important; padding-right: 0 !important; font-weight: 500; }
.stButton > button { border-radius: .55rem; font-weight: 500; }
.stButton > button[kind="primary"] { box-shadow: none; }

/* the run header */
.hg-head { display: flex; flex-wrap: wrap; align-items: center; gap: .9rem 1.6rem; margin: .4rem 0 1rem; }
.hg-fact { display: flex; flex-direction: column; gap: .1rem; }
.hg-fact b { font-weight: 600; font-variant-numeric: tabular-nums; }
.hg-fact span { color: var(--hg-muted); font-size: .8rem; }
.hg-pill { display: inline-flex; align-items: center; gap: .45rem; padding: .28rem .7rem .28rem .55rem;
  border-radius: 99px; font-weight: 600; font-size: .86rem; border: 1px solid currentColor; }
.hg-pill i { width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }
.hg-pill { border: none; color: var(--hg-ink); background: var(--c, var(--hg-wait)); }
.hg-pill i { background: var(--hg-ink); opacity: .55; }
.hg-pill.running { --c: var(--hg-run); } .hg-pill.running i { animation: hg-pulse 1.4s ease-in-out infinite; }
.hg-pill.success { --c: var(--hg-ok); } .hg-pill.failed, .hg-pill.cancelled { --c: var(--hg-bad); }

/* the run board */
.hg-board { display: grid; border: 1px solid var(--hg-line); border-radius: .8rem; overflow: hidden; margin: .2rem 0 .4rem; }
.hg-colhead { padding: .8rem 1rem .7rem; border-bottom: 1px solid var(--hg-line); background: var(--hg-faint); }
.hg-colhead strong { font-family: var(--hg-display); font-size: 1.02rem; font-weight: 600; display: block; }
.hg-colhead small { color: var(--hg-muted); font-size: .78rem; font-variant-numeric: tabular-nums; }
.hg-rail { height: 3px; border-radius: 2px; background: var(--hg-line); margin-top: .5rem; overflow: hidden; }
.hg-rail > div { height: 100%; background: var(--hg-ok); transition: width .5s ease; }
.hg-rail.live > div { background: var(--hg-run); }
.hg-doc { padding: .85rem 1rem; border-bottom: 1px solid var(--hg-line); font-weight: 500;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hg-doc small { display: block; color: var(--hg-muted); font-weight: 400; font-size: .74rem; }
.hg-cell { padding: .7rem 1rem; border-bottom: 1px solid var(--hg-line); border-left: 1px solid var(--hg-line);
  display: flex; align-items: center; gap: .7rem; min-width: 0; position: relative; }
.hg-cell .t { display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }
.hg-cell .t b { font-weight: 600; font-size: .9rem; }
.hg-cell .t span { color: var(--hg-muted); font-size: .78rem; font-variant-numeric: tabular-nums;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hg-cell { --c: var(--hg-wait); }
.hg-ic { flex: none; width: 1.35rem; height: 1.35rem; border-radius: 50%; display: grid; place-items: center;
  font-size: .8rem; font-weight: 700; line-height: 1; box-sizing: border-box; color: var(--hg-ink); background: var(--c); }
.hg-ic.success::after { content: "\\2713"; } .hg-ic.failed::after, .hg-ic.cancelled::after { content: "\\2715"; }
.hg-ic.running { background: transparent; border: 3px solid color-mix(in srgb, var(--c) 35%, transparent);
  border-top-color: var(--c); animation: hg-spin 0.9s linear infinite; }
.hg-ic.pending { background: transparent; border: 2px dashed color-mix(in srgb, currentColor 40%, transparent); }
.hg-ic.skipped::after { content: ""; width: .6rem; height: 2px; background: var(--hg-ink); opacity: .6; border-radius: 1px; }
@keyframes hg-spin { to { transform: rotate(360deg); } }
.hg-cell.success { --c: var(--hg-ok); background: color-mix(in srgb, var(--hg-ok) 20%, transparent); }
.hg-cell.failed, .hg-cell.cancelled { --c: var(--hg-bad); background: color-mix(in srgb, var(--hg-bad) 22%, transparent); }
.hg-cell.running { --c: var(--hg-run); background: color-mix(in srgb, var(--hg-run) 22%, transparent); }
.hg-cell.skipped { --c: var(--hg-skip); background: color-mix(in srgb, var(--hg-skip) 16%, transparent); }
.hg-cell.pending { color: var(--hg-muted); }
.hg-cell.running::after { content: ""; position: absolute; left: 0; bottom: 0; height: 3px; width: 40%;
  background: linear-gradient(90deg, transparent, var(--hg-run), transparent); animation: hg-sweep 1.8s ease-in-out infinite; }
.hg-cell.merged { align-items: center; justify-content: flex-start; }
.hg-empty { grid-column: 1 / -1; padding: 1.3rem 1rem; color: var(--hg-muted); }
@keyframes hg-sweep { from { transform: translateX(-100%); } to { transform: translateX(260%); } }
@keyframes hg-pulse { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) { .hg-cell.running::after, .hg-pill.running i, .hg-ic.running { animation: none; } }

/* query page */
.hg-step { display: flex; align-items: baseline; gap: .7rem; margin: 1.5rem 0 .4rem; }
.hg-step em { font-style: normal; font-family: var(--hg-display); font-weight: 600; font-size: 1.5rem; color: var(--hg-brand); }
.hg-step h3 { margin: 0; font-size: 1.15rem; } .hg-step span { color: var(--hg-muted); font-size: .85rem; margin-left: auto; }
.hg-hit { display: flex; align-items: center; gap: .8rem; flex-wrap: wrap; }
.hg-hit .path { font-weight: 600; } .hg-hit .score { font-variant-numeric: tabular-nums; color: var(--hg-brand); font-weight: 600; }
.hg-bar { width: 6rem; height: 4px; border-radius: 2px; background: var(--hg-line); overflow: hidden; }
.hg-bar > div { height: 100%; background: var(--hg-brand); }
</style>
"""

_ICONS = {k: f'<span class="hg-ic {k}"></span>' for k in ("success", "failed", "running", "pending", "skipped")}
_ICONS["cancelled"] = _ICONS["failed"]
_LABEL = {"success": "Done", "failed": "Failed", "running": "Running", "pending": "Waiting",
          "skipped": "Reused", "cancelled": "Cancelled"}
STEP_TITLES = {"parse": "Parse", "hierarchy": "Hierarchy", "index": "Index"}
STEP_HINTS = {"parse": "Docling, figures", "hierarchy": "heading tree", "index": "chunks, embeddings"}


def inject_css() -> None:
    st.html(CSS)


def fmt_secs(s: float | None) -> str:
    if s is None:
        return ""
    return f"{s:.1f} s" if s < 90 else f"{int(s // 60)} min {int(s % 60):02d} s"


def fmt_clock(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "—"


def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _info_text(info: dict | None) -> str:
    if not info:
        return ""
    if "reason" in info:
        return info["reason"]
    labels = {"n_figures": "figures", "n_sections": "sections", "n_elements": "elements",
              "total_chunks": "chunks", "n_pages": "pages", "n_tables": "tables",
              "n_documents": "documents"}
    return ", ".join(f"{v} {labels[k]}" for k, v in info.items() if k in labels)


def _cell(state: str, title: str, detail: str, *, style: str = "", extra: str = "", tip: str = "") -> str:
    return (f'<div class="hg-cell {state} {extra}" style="{style}" title="{_esc(tip or detail)}">'
            f'{_ICONS[state]}<div class="t"><b>{_esc(title)}</b>'
            f'{f"<span>{_esc(detail)}</span>" if detail else ""}</div></div>')


def page_header(title: str, lede: str) -> None:
    st.html(f"<h1>{_esc(title)}</h1><p class='hg-lede'>{lede}</p>")


def run_header(status: dict[str, Any], now: float) -> str:
    end = status["ended_at"] or now
    steps = ", ".join(STEP_TITLES.get(s_, s_) for s_ in status["params"].get("steps", runner.STEPS)).lower()
    pdfs = status["params"].get("pdfs", "all")
    return (
        '<div class="hg-head">'
        f'<span class="hg-pill {status["state"]}"><i></i>{_esc(status["state"].capitalize())}</span>'
        f'<div class="hg-fact"><b>{fmt_clock(status["started_at"])}</b><span>started</span></div>'
        f'<div class="hg-fact"><b>{fmt_secs(end - status["started_at"])}</b><span>elapsed</span></div>'
        f'<div class="hg-fact"><b>{_esc(steps)}</b><span>steps run</span></div>'
        f'<div class="hg-fact"><b>{"all documents" if pdfs == "all" else _esc(", ".join(pdfs))}</b><span>scope</span></div>'
        "</div>"
    )


def run_board(status: dict[str, Any]) -> str:
    """Documents as rows, steps as columns; the index step is one cell merged over all rows."""
    tasks = status["tasks"]
    summ = {t: runner.task_summary(tasks[t]) for t in ("parse", "hierarchy", "index")}
    docs: list[str] = sorted({i["item"] for t in ("parse", "hierarchy") for i in tasks[t]["instances"] if i["item"]})
    n = max(len(docs), 1)
    live = status["state"] == runner.RUNNING

    out = ['<div class="hg-board" style="grid-template-columns: minmax(10rem,1.1fr) repeat(3, minmax(11rem,1fr));">']
    out.append('<div class="hg-colhead"><strong>Documents</strong><small>'
               f'{len(docs) or "no"} in this run</small></div>')
    for step in ("parse", "hierarchy", "index"):
        s = summ[step]
        total = s["n"] if step != "index" else 1
        done = s["n_success"] if s["state"] != runner.SKIPPED else 0
        pct = 100 * done / total if total else 0
        count = ("reused from disk" if s["state"] == runner.SKIPPED
                 else (f"{done} of {total}" if step != "index" else _LABEL.get(s["state"], "").lower()))
        out.append(f'<div class="hg-colhead" style="border-left:1px solid var(--hg-line)"><strong>{STEP_TITLES[step]}</strong>'
                   f'<small>{STEP_HINTS[step]}, {count}</small>'
                   f'<div class="hg-rail {"live" if s["state"] == runner.RUNNING else ""}"><div style="width:{pct:.0f}%"></div></div></div>')

    if not docs:
        waiting = "Waiting for the first document…" if live else "No document in this run."
        out.append(f'<div class="hg-empty">{waiting}</div>')
        out.append("</div>")
        return "".join(out)

    for r, doc in enumerate(docs):
        out.append(f'<div class="hg-doc" title="{_esc(doc)}">{_esc(doc)}</div>')
        for c, step in enumerate(("parse", "hierarchy"), start=2):
            if summ[step]["state"] == runner.SKIPPED:
                if r == 0:
                    out.append(_cell("skipped", "Reused", "output already on disk",
                                     style=f"grid-column:{c}; grid-row:2 / span {n}", extra="merged"))
                continue
            inst = next((i for i in tasks[step]["instances"] if i["item"] == doc), None)
            if inst is None:
                out.append(_cell("pending", "Waiting" if live else "Not run", ""))
            elif inst["state"] == runner.FAILED:
                out.append(_cell("failed", "Failed", inst["error"] or "", tip=inst["error"] or ""))
            else:
                detail = _info_text(inst["info"])
                if inst["seconds"]:
                    detail = (detail + ", " if detail else "") + fmt_secs(inst["seconds"])
                out.append(_cell(inst["state"], _LABEL.get(inst["state"], inst["state"]), detail))
    idx = tasks["index"]["instances"][0] if tasks["index"]["instances"] else None
    st_idx = summ["index"]["state"]
    if st_idx == runner.SKIPPED:
        out.append(_cell("skipped", "Left untouched", "Qdrant not modified", style=f"grid-column:4; grid-row:2 / span {n}", extra="merged"))
    else:
        detail = _info_text(idx["info"]) if idx else ""
        if idx and idx["seconds"]:
            detail = (detail + ", " if detail else "") + fmt_secs(idx["seconds"])
        if idx and idx["error"]:
            detail = idx["error"]
        out.append(_cell(st_idx if st_idx in _ICONS else "pending", _LABEL.get(st_idx, "Waiting"), detail,
                         style=f"grid-column:4; grid-row:2 / span {n}", extra="merged"))
    out.append("</div>")
    return "".join(out)


def step_header(n: int, title: str, meta: str = "") -> None:
    st.html(f'<div class="hg-step"><em>{n}</em><h3>{_esc(title)}</h3><span>{_esc(meta)}</span></div>')


def score_bar(score: float) -> str:
    pct = max(0.0, min(1.0, score)) * 100
    return f'<span class="hg-bar"><div style="width:{pct:.0f}%"></div></span><span class="score">{score:.3f}</span>'
