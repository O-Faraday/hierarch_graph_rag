"""Versions of the two applications (``rag``: question -> answer; ``indexing``: PDFs -> Qdrant) for MLflow.

A version is the configuration that changes what the application produces: models, chunking, prompt, search
settings. ``hgrag register`` logs each application as an MLflow model (``registry``) tagged with the hash of this
configuration; every trace made later with the same configuration is linked to that model (trace metadata
``mlflow.modelId``), so the MLflow UI compares versions on their traces and their checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from . import config, tracking

logger = logging.getLogger(__name__)

KINDS = ("rag", "indexing")
EXPERIMENTS = {"rag": tracking.RETRIEVAL, "indexing": tracking.INDEXING}
MODEL_NAMES = {"rag": "hgrag-rag", "indexing": "hgrag-indexing"}
HASH_TAG = "hgrag.config_hash"

_IDS: dict[tuple[str, str, str], str | None] = {}


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def config_snapshot(kind: str) -> dict[str, str]:
    """The settings that define a version of ``kind``, as strings (MLflow model params)."""
    if kind == "rag":
        from . import generation

        snap: dict[str, Any] = {
            "embed_model": config.EMBED_MODEL_ID, "collection": config.COLLECTION_NAME,
            "keyword_weight": config.KEYWORD_WEIGHT, "rerank": config.RERANK, "rerank_model": config.RERANK_MODEL,
            "rerank_candidates": config.RERANK_CANDIDATES,
            "llm_model": os.environ.get("HGRAG_LLM_MODEL") or generation.DEFAULT_MODELS["anthropic"],
            "max_context_chars": generation.LLMConfig(provider="anthropic", api_key="-").max_context_chars,
            "max_images": generation.LLMConfig(provider="anthropic", api_key="-").max_images,
            "system_prompt_sha": _sha(generation.SYSTEM_PROMPT),
        }
    elif kind == "indexing":
        from .parsing import parse_options

        snap = {
            **parse_options(), "figure_vlm": config.FIGURE_VLM, "table_model": config.TABLE_MODEL,
            "formula_mode": config.FORMULA_MODE, "formula_vlm": config.FORMULA_VLM or config.FIGURE_VLM,
            "formula_reading": config.FORMULA_READING, "embed_model": config.EMBED_MODEL_ID,
            "collection": config.COLLECTION_NAME, "chunk_labels": ",".join(config.CHUNK_LABELS),
            "chunk_max_tokens": config.CHUNK_MAX_TOKENS, "min_chunk_words": config.MIN_CHUNK_WORDS,
            "min_prose_words": config.MIN_PROSE_WORDS, "heading_context": config.HEADING_CONTEXT,
            "excluded_sections": config.EXCLUDED_SECTIONS, "fix_heading_levels": config.FIX_HEADING_LEVELS,
            "demote_fake_headings": config.DEMOTE_FAKE_HEADINGS,
        }
    else:
        raise ValueError(f"unknown application {kind!r} (choose from {KINDS})")
    return {k: str(v) for k, v in sorted(snap.items())}


def config_hash(kind: str) -> str:
    return _sha(json.dumps(config_snapshot(kind), sort_keys=True))


def model_id(kind: str) -> str | None:
    """Id of the MLflow model registered for the current configuration of ``kind`` (``hgrag register``), or None.
    Looked up once per process; a failure (no MLflow, no model) gives None."""
    try:
        h = config_hash(kind)
    except Exception as exc:  # noqa: BLE001  (a setting that cannot be read must not stop the trace)
        logger.warning("configuration of %s not readable: %s", kind, exc)
        return None
    key = (tracking.TRACKING_URI, kind, h)
    if key not in _IDS:
        try:
            import mlflow

            exp_id = tracking._setup(EXPERIMENTS[kind])
            found = mlflow.search_logged_models(experiment_ids=[exp_id], filter_string=f"tags.`{HASH_TAG}` = '{h}'",
                                                max_results=1, output_format="list")
            _IDS[key] = found[0].model_id if found else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model lookup failed for %s: %s", kind, exc)
            _IDS[key] = None
    return _IDS[key]
