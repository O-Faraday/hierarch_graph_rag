"""Steps 3-4 - paragraph chunking, embeddings and the single Qdrant vector store."""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.base import BaseChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from . import config
from .hierarchy import SectionNode, section_index

logger = logging.getLogger(__name__)

_PARAGRAPH_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}  # kept for backward compatibility


def chunk_label_set(names: Sequence[str] | None = None) -> set[DocItemLabel]:
    """Docling labels whose chunks are indexed (``HGRAG_CHUNK_LABELS``)."""
    out: set[DocItemLabel] = set()
    for name in names if names is not None else config.CHUNK_LABELS:
        try:
            out.add(DocItemLabel(name))
        except ValueError:
            raise ValueError(f"unknown Docling label {name!r} in HGRAG_CHUNK_LABELS; "
                             f"valid: {sorted(l.value for l in DocItemLabel)}") from None
    return out


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


def load_embedder(model_id: str = config.EMBED_MODEL_ID):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def make_chunker(
    model_id: str = config.EMBED_MODEL_ID,
    *,
    max_tokens: int | None = None,
    merge_peers: bool = False,
) -> HybridChunker:
    """HybridChunker whose tokenizer is aligned on the embedding model.

    ``merge_peers=False`` keeps one chunk per element, so a paragraph chunk never gets
    merged with a list item or other element type.
    """
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"tokenizer": AutoTokenizer.from_pretrained(model_id)}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return HybridChunker(tokenizer=HuggingFaceTokenizer(**kwargs), merge_peers=merge_peers)


PROSE_LABELS = {"text", "paragraph", "list_item", "footnote"}


def chunk_paragraphs(
    doc: DoclingDocument,
    chunker: BaseChunker,
    root: SectionNode,
    doc_id: str,
    min_words: int = config.MIN_CHUNK_WORDS,
    stats: dict[str, Any] | None = None,
    labels: Sequence[str] | set[DocItemLabel] | None = None,
    table_descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Decompose the textual elements (``labels``, default ``HGRAG_CHUNK_LABELS``) into chunks.

    Chunks made of other element types (figures...) are dropped, as are "singletons": chunks
    with fewer than ``min_words`` words (``min_words=0`` disables the filter).
    """
    allowed = chunk_label_set() if labels is None else (
        set(labels) if all(isinstance(l, DocItemLabel) for l in labels) else chunk_label_set(list(labels)))
    by_ref = section_index(root)
    titles = {t.text.strip() for t in doc.texts if getattr(t.label, "value", str(t.label)) == "title"}
    excluded = re.compile(config.EXCLUDED_SECTIONS, re.I) if config.EXCLUDED_SECTIONS else None
    out: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for chunk in chunker.chunk(dl_doc=doc):
        # HybridChunker exposes generic ``DocItem`` objects (not ``TextItem``), so only the
        # label can be relied on to recognise paragraphs.
        resolved = list(chunk.meta.doc_items)
        labels_ = {getattr(r.label, "value", str(r.label)) for r in resolved}
        if not resolved or not all(r.label in allowed for r in resolved):
            dropped.update(labels_ or {"<no items>"})
            continue
        n_words = len(chunk.text.split())
        if n_words < min_words or (min_words and n_words < config.MIN_PROSE_WORDS and labels_ <= PROSE_LABELS):
            dropped["too_short"] += 1
            continue
        first = resolved[0]
        chain = [h for h in (chunk.meta.headings or [])]
        if excluded and any(excluded.match(h.strip()) for h in chain):
            dropped["excluded_section"] += 1
            continue
        chain = [h for h in chain if h.strip() not in titles]
        context = chain[-config.HEADING_CONTEXT:] if config.HEADING_CONTEXT > 0 else []
        heading_prefix = "\n".join(context)
        embed_text = f"{heading_prefix}\n{chunk.text}" if heading_prefix else chunk.text
        description = next((table_descriptions[r.self_ref] for r in resolved
                            if table_descriptions and r.self_ref in table_descriptions), None)
        if description:  # generated description first: a long table is truncated by the encoder, not the description
            embed_text = (f"{heading_prefix}\n{description}\n{chunk.text}" if heading_prefix
                          else f"{description}\n{chunk.text}")
        section = by_ref.get(first.self_ref)
        page_no = first.prov[0].page_no if first.prov else None
        out.append(
            {
                "chunk_index": len(out),
                "doc_id": doc_id,
                "text": chunk.text,
                "embed_text": embed_text,
                **({"table_description": description} if description else {}),
                "kinds": sorted(labels_),
                "headings": list(chunk.meta.headings or []),
                "element_refs": [r.self_ref for r in resolved],
                "page_no": page_no,
                "section_ref": section.ref if section else None,
                "section_title": section.title if section else None,
                "section_path": section.path if section else None,
            }
        )
    if stats is not None:
        stats.update(n_kept=len(out), dropped=dict(dropped), n_dropped=sum(dropped.values()))
    logger.info(
        "doc %s: kept %d chunks, dropped chunks by label: %s",
        doc_id, len(out), dict(dropped) or "none",
    )
    if not out:
        logger.warning("doc %s: no chunk found (dropped: %s)", doc_id, dict(dropped))
    return out


def figure_chunks(root: SectionNode, doc_id: str, start_index: int = 0) -> list[dict[str, Any]]:
    """One chunk per figure that has a description: caption + description, searchable like any paragraph."""
    out: list[dict[str, Any]] = []
    for node in root.walk():
        headings = [h for h in node.path.split(" > ")[1:]]  # the first part is the document title (synthetic root)
        for el in node.elements:
            desc = ((el.extra or {}).get("description") or "").strip()
            if el.kind != "figure" or not desc:
                continue
            caption = (el.extra or {}).get("caption")
            text = f"{caption}\n{desc}" if caption else desc
            out.append({
                "chunk_index": start_index + len(out), "doc_id": doc_id, "text": text,
                "embed_text": "\n".join([*headings, text]), "kinds": ["figure"], "headings": headings,
                "element_refs": [el.ref], "page_no": el.page_no, "section_ref": node.ref,
                "section_title": node.title, "section_path": node.path,
            })
    return out


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return
    existing = client.get_collection(name).config.params.vectors
    size = getattr(existing, "size", None)  # a single unnamed vector; anything else is not ours
    if size is not None and size != dim:
        raise ValueError(
            f"collection {name!r} holds {size}-dimension vectors but the embedding model gives {dim}: "
            f"index into another collection (HGRAG_COLLECTION) or Qdrant folder, or delete this one first")


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hgrag:{doc_id}:{chunk_index}"))


def index_chunks(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    embedder: Embedder,
    *,
    doc_id: str,
    source_pdf: str | None = None,
    collection: str = config.COLLECTION_NAME,
    batch_size: int = 32,
    model_id: str = config.EMBED_MODEL_ID,
) -> int:
    """Embed chunks and (re)write them in the collection. Idempotent per ``doc_id``.
    ``model_id`` only selects the passage prefix the model expects (``config.embed_prefixes``)."""
    if not chunks:
        return 0
    _, doc_prefix = config.embed_prefixes(model_id)
    vectors = embedder.encode(
        [doc_prefix + c["embed_text"] for c in chunks], batch_size=batch_size, normalize_embeddings=True
    )
    ensure_collection(client, collection, int(vectors.shape[1]))
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )
    points = [
        PointStruct(
            id=_point_id(doc_id, c["chunk_index"]),
            vector=vec.tolist(),
            payload={
                **{k: v for k, v in c.items() if k != "embed_text"},
                "source_pdf": source_pdf,
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def open_store(path: str | Path = config.QDRANT_PATH) -> QdrantClient:
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))
