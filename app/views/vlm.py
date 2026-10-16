"""Page 3: compare vision-language models on the control figures (HF router + Claude, nothing local)."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import ui

from hgrag import config, vlm_bench

ui.inject_css()
ui.page_header(
    "VLM comparison",
    "Which model describes a figure best? Each model gets the control figures, a Claude judge checks every "
    "description against known facts (type, title, axes, values, trend) and flags what was invented. "
    "Models run through the Hugging Face router, so nothing is installed locally.",
)

try:
    truth = vlm_bench.load_truth()
except (FileNotFoundError, OSError) as exc:
    st.error(f"{exc}")
    st.stop()
by_id = {t["id"]: t for t in truth}

# ---- run -----------------------------------------------------------------------------------------
with st.expander("Run a comparison", expanded=not vlm_bench.list_runs()):
    known = list(dict.fromkeys(vlm_bench.DEFAULT_MODELS + st.session_state.get("vlm_extra", [])))
    chosen = st.multiselect("Models", known, default=vlm_bench.DEFAULT_MODELS,
                            help="hf:<org>/<model> goes through the HF router, anthropic:<model> through the Claude API.")
    c1, c2 = st.columns([3, 1])
    extra = c1.text_input("Add a model", placeholder="hf:Qwen/Qwen3-VL-30B-A3B-Instruct", key="vlm_add")
    if c2.button("Add", width="stretch", icon=":material/add:") and extra.strip():
        st.session_state.setdefault("vlm_extra", []).append(extra.strip())
        st.rerun()
    if st.button("List models served by the HF router", icon=":material/list:"):
        try:
            st.session_state["vlm_catalog"] = vlm_bench.list_models(st.session_state.get("vlm_hf") or None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{exc}")
    if st.session_state.get("vlm_catalog"):
        st.dataframe(pd.DataFrame([{"model": f"hf:{m['id']}", "providers": ", ".join(m["providers"])}
                                   for m in st.session_state["vlm_catalog"]]), hide_index=True, width="stretch")

    figs = st.multiselect("Figures", list(by_id), default=list(by_id))
    k1, k2 = st.columns(2)
    hf_key = k1.text_input("HF token (this session only)", type="password", key="vlm_hf",
                           placeholder="taken from HF_TOKEN if empty",
                           help="Fine-grained token with 'Make calls to Inference Providers'. Never written to disk.")
    an_key = k2.text_input("Anthropic key (judge and Claude, this session only)", type="password", key="vlm_an",
                           placeholder="taken from ANTHROPIC_API_KEY if empty")
    st.caption(f"`HF_TOKEN` is {'set' if os.environ.get('HF_TOKEN') else 'not set'} · "
               f"`ANTHROPIC_API_KEY` is {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'} in the environment. "
               "HF providers bill per call (a few cents for this benchmark).")
    prompt = st.text_area("Prompt sent with each image", vlm_bench.DESCRIBE_PROMPT, height=90)
    use_judge = st.checkbox("Judge the descriptions with Claude", value=True)
    if st.button("Run the comparison", type="primary", icon=":material/play_arrow:", disabled=not chosen or not figs):
        bar = st.progress(0.0, text="Starting…")

        def progress(r, n, total):
            bar.progress(n / total, text=f"{n}/{total} · {r['model']} · {r['figure']}"
                         + (" · error" if "error" in r else ""))

        try:
            res = vlm_bench.run_bench(chosen, figures=figs, judge=use_judge, prompt=prompt, progress=progress,
                                      hf_token=hf_key or None, anthropic_key=an_key or None)
        except vlm_bench.BenchError as exc:
            st.error(f"{exc}")
        else:
            st.session_state["vlm_run"] = res["run_key"]
            st.rerun()

# ---- results -------------------------------------------------------------------------------------
runs = vlm_bench.list_runs()
if not runs:
    st.info("No comparison yet: run one above (or `uv run hgrag vlm-bench`).")
    st.stop()
names = [r.name for r in runs]
wanted = st.session_state.get("vlm_run")
run = vlm_bench.load_run(runs[names.index(st.selectbox("Run", names, index=names.index(wanted) if wanted in names else 0))])
rows, summary = run["rows"], run["summary"]

ui.step_header(1, "Leaderboard", "mean over the figures, best first")
st.dataframe(
    pd.DataFrame(summary).rename(columns={"invented_per_figure": "invented / figure", "output_tokens": "tokens out"}),
    hide_index=True, width="stretch",
    column_config={
        "score": st.column_config.ProgressColumn("score (1-5)", min_value=1, max_value=5, format="%.2f"),
        "recall": st.column_config.ProgressColumn("facts found", min_value=0.0, max_value=1.0, format="percent"),
        "seconds": st.column_config.NumberColumn("seconds", format="%.1f"),
    },
)
judged = [s for s in summary if s["score"] is not None]
if judged:
    ch1, ch2 = st.columns(2)
    ch1.bar_chart(pd.DataFrame({"model": [s["model"] for s in judged], "facts found": [s["recall"] for s in judged]}),
                  x="model", y="facts found", height=240)
    ch2.bar_chart(pd.DataFrame({"model": [s["model"] for s in judged],
                                "invented / figure": [s["invented_per_figure"] for s in judged]}),
                  x="model", y="invented / figure", height=240)

ui.step_header(2, "Figure by figure", "pick a figure to read every description next to the image")
fig_ids = list(dict.fromkeys(r["figure"] for r in rows))
fig_id = st.radio("Figure", fig_ids, horizontal=True, label_visibility="collapsed")
cells = [r for r in rows if r["figure"] == fig_id]
left, right = st.columns([1, 2])
with left:
    png = cells[0]["png_path"]
    if Path(png).exists():
        st.image(png, width="stretch")
    if fig_id in by_id:
        st.caption(by_id[fig_id]["full_description"])
with right:
    for r in sorted(cells, key=lambda r: -(r.get("score") or 0)):
        with st.container(border=True):
            head = f"**{r['model']}**"
            if "error" in r:
                st.markdown(head)
                st.error(r["error"])
                continue
            if "score" in r:
                head += f" · score {r['score']}/5 · {r['recall']:.0%} facts · {len(r['invented'])} invented"
            st.markdown(head + f" · {r['seconds']} s")
            st.write(r["description"] or "(empty)")
            if "facts" in r:
                labels = {f["id"]: f["text"] for f in by_id.get(fig_id, {}).get("facts", [])}
                st.caption("  ".join(f"{'✅' if ok else '❌'} {labels.get(k, k)}" for k, ok in r["facts"].items()))
                for bad in r["invented"]:
                    st.warning(bad, icon=":material/report:")

with st.expander("Raw results (JSONL)"):
    st.json(rows, expanded=False)
st.caption(f"Saved in `{run['dir']}`.")
