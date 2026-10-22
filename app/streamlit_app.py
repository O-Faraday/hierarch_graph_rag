"""Streamlit front-end: pipeline runs + retrieval query.

    uv run streamlit run app/streamlit_app.py
"""

import streamlit as st

import hgrag.config  # noqa: F401  (loads .env before any model library is imported)

st.set_page_config(page_title="Hierarchical RAG", page_icon=":material/account_tree:", layout="wide")

pages = st.navigation(
    {
        "Pipeline": [
            st.Page("views/pipeline.py", title="Pipelines", icon=":material/schema:", default=True),
            st.Page("views/query.py", title="Query", icon=":material/search:"),
            st.Page("views/vlm.py", title="VLM comparison", icon=":material/compare:"),
        ],
        "Ragbench": [
            st.Page("views/rb_results.py", title="Results", icon=":material/table_chart:"),
            st.Page("views/rb_detail.py", title="Question detail", icon=":material/quiz:"),
            st.Page("views/rb_summary.py", title="Summary", icon=":material/insights:"),
            st.Page("views/rb_compare.py", title="Compare runs", icon=":material/compare_arrows:"),
        ],
    }
)
pages.run()
