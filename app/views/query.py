"""Page 2: run a query and display the output of each retrieval step as it completes."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import ui
from models import embedder, reranker
from mlflow_view import MLFLOW_UI_URL

from hgrag import config, generation, runner, tracking
from hgrag.retrieval import retrieve_steps


ui.inject_css()
ui.page_header(
    "Query",
    f"Collection <code>{config.COLLECTION_NAME}</code> · model <code>{config.EMBED_MODEL_ID}</code> · "
    f"reranker <code>{config.RERANK_MODEL}</code>. "
    "Qdrant local mode allows one process: queries pause while an indexing run is going.",
)

if (running := runner.active_run()):
    st.warning(
        f"An indexing run is in progress (`{running['run_key']}`): Qdrant local mode accepts a single "
        "process, so queries are paused. Follow it on the **Pipelines** page."
    )
    st.stop()

with st.expander("Answer generation (LLM)", expanded=False):
    g1, g2 = st.columns(2)
    provider = g1.selectbox("Provider", generation.PROVIDERS, index=generation.PROVIDERS.index(generation.LLMConfig().provider),
                            format_func={"anthropic": "Claude API", "openai": "OpenAI-compatible (Ollama, LM Studio…)"}.get,
                            help="Claude needs ANTHROPIC_API_KEY. The OpenAI-compatible option targets a local server.")
    llm_model = g2.text_input("Model", value=generation.LLMConfig(provider=provider).model, key=f"llm_model_{provider}")
    g3, g4 = st.columns(2)
    llm_url = g3.text_input("Base URL", value=generation.LLMConfig(provider=provider).base_url, key=f"llm_url_{provider}")
    llm_key = g4.text_input("API key (this session only)", type="password",
                            placeholder="taken from the environment if empty",
                            help="Never written to disk nor to MLflow.")
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    st.caption(f"`{env_var}` is {'set' if os.environ.get(env_var) else 'not set'} in the environment.")

with st.form("query"):
    query = st.text_input("Query", placeholder="e.g. three new branches portfolio")
    c1, c2, c3 = st.columns([1, 1, 2])
    k = c1.number_input("k (chunks)", 1, 50, 5)
    children = c2.checkbox("Include sub-sections")
    track = c3.checkbox("Track in MLflow (experiment hgrag_retrieval)", value=True)
    rerank = st.checkbox(f"Rerank the {config.RERANK_CANDIDATES} closest chunks with a cross-encoder", value=config.RERANK,
                         help="The vector search fetches more chunks, a cross-encoder reads (question, chunk) together and "
                              "keeps the best k. Slower (seconds on CPU), usually better ranked.")
    generate = st.checkbox("Generate the answer from the rebuilt sections", value=True)
    go = st.form_submit_button("Search", type="primary")

if go:
    if not query.strip():
        st.warning("Enter a query.")
        st.stop()
    run_key = tracking.new_run_key() if track else None
    final: list[dict] = []
    try:
        model = embedder()
        stream = retrieve_steps(query, int(k), include_children=children, run_key=run_key, embedder=model,
                                rerank=rerank, reranker=reranker() if rerank else None)
        slots: dict[str, st.delta_generator.DeltaGenerator] = {}
        for event in stream:
            step, data, secs = event["step"], event["data"], event["seconds"]
            if step == "embed":
                with st.status(f"Embed the query · {secs * 1000:.0f} ms", state="complete", expanded=False):
                    st.write(f"Model `{data['model']}` → vector of dimension **{data['dim']}** (normalized).")
                    st.bar_chart(pd.Series(data["vector"][:64], name="first 64 components"), height=140)
            elif step == "vector_search":
                with st.status(f"Vector search · {len(data)} chunks{' (candidates)' if rerank else ''} · {secs * 1000:.0f} ms",
                               state="complete", expanded=True):
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(
                            df[["score", "section_path", "page_no", "text", "doc_id", "chunk_index"]],
                            hide_index=True, width="stretch",
                            column_config={"score": st.column_config.ProgressColumn(
                                "score", min_value=0.0, max_value=1.0, format="%.3f")},
                        )
                    else:
                        st.info("No chunk found.")
            elif step == "rerank":
                with st.status(f"Rerank · best {len(data)} chunks · {secs * 1000:.0f} ms", state="complete", expanded=True):
                    st.caption("Cross-encoder score (0-1) next to the vector similarity the chunk had.")
                    st.dataframe(
                        pd.DataFrame(data)[["score", "vector_score", "section_path", "page_no", "text"]],
                        hide_index=True, width="stretch",
                        column_config={"score": st.column_config.ProgressColumn("rerank", min_value=0.0, max_value=1.0, format="%.3f"),
                                       "vector_score": st.column_config.NumberColumn("vector", format="%.3f")},
                    )
            elif step == "group":
                with st.status(f"Group by section · {len(data)} sections", state="complete",
                               expanded=True):
                    st.dataframe(
                        pd.DataFrame([{"score": s["score"], "section_path": s["section_path"],
                                       "doc_id": s["doc_id"], "chunks": len(s["hits"])} for s in data]),
                        hide_index=True, width="stretch",
                    )
                if data:
                    ui.step_header(4, "Reconstructed sections", "all elements of each matching section")
                    for s in data:
                        slots[(s["doc_id"], s["section_ref"])] = st.empty()
                        slots[(s["doc_id"], s["section_ref"])].info(f"Rebuilding {s['section_path']}…")
            elif step == "reconstruct":
                with slots[(data["doc_id"], data["section_ref"])].expander(
                    f"{data['section_path']} · score {data['score']:.3f} · {data['n_elements']} elements "
                    f"· {secs * 1000:.0f} ms", expanded=True,
                ):
                    st.markdown(data["text"])
                    for fig in data["figures"]:
                        if fig["png_path"] and Path(fig["png_path"]).exists():
                            st.image(fig["png_path"], caption=fig["caption"] or fig["png_path"], width=420)
            elif step == "done":
                final = data
    except FileNotFoundError as exc:
        st.error(f"{exc} Run the indexing first.")
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
    else:
        answer_info: dict = {}
        if not final:
            st.info("No section to display.")
        else:
            st.success(f"{len(final)} sections rebuilt, best first.", icon=":material/check_circle:")
            if generate:
                ui.step_header(5, "Answer", "generated from the sections above")
                cfg = generation.LLMConfig(provider=provider, model=llm_model, base_url=llm_url,
                                           api_key=llm_key or None)
                try:
                    with st.container(border=True):
                        st.write_stream(generation.answer_stream(query, final, cfg, run_key=run_key, info=answer_info))
                except generation.GenerationError as exc:
                    st.error(f"{exc}", icon=":material/error:")
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                if answer_info.get("sources"):
                    st.caption("Sources: " + " · ".join(f"[{x['n']}] {x['section_path'].split(' > ', 1)[-1]} "
                                                        f"({x['score']:.2f})" for x in answer_info["sources"]))
                if answer_info.get("dropped"):
                    st.warning(f"{len(answer_info['dropped'])} lower-ranked sections did not fit the prompt budget "
                               "(HGRAG_LLM_MAX_CONTEXT_CHARS).")
                if answer_info.get("usage"):
                    u = answer_info["usage"]
                    st.caption(f"{answer_info['model']} · {u.get('input_tokens', '?')} tokens in, "
                               f"{u.get('output_tokens', '?')} out · {answer_info.get('n_images', 0)} figure image(s) sent · "
                               f"first token after {answer_info['first_token_s'] or 0:.1f} s")
            with st.expander("Final output (JSON)"):
                st.json({"answer": answer_info.get("answer"), "sections": final}, expanded=False)
        if run_key:
            st.caption(f"MLflow run `{run_key}` — [open MLflow]({MLFLOW_UI_URL}) (experiment `hgrag_retrieval`).")
