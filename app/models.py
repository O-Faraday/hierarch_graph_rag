"""Shared cached resources."""

import streamlit as st

from hgrag import config


@st.cache_resource(show_spinner="Loading the embedding model…")
def embedder():
    from hgrag.indexing import load_embedder

    return load_embedder(config.EMBED_MODEL_ID)
