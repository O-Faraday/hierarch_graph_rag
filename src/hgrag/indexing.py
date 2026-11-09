"""Steps 3-4 - paragraph chunking, embeddings and the single Qdrant vector store."""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.base import BaseChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument

from . import config, keywords
from .hierarchy import SectionNode, section_index, without_long_headings

if TYPE_CHECKING:  # Qdrant is only installed with the local backend: the chunking above runs without it
    from qdrant_client import QdrantClient

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

    # Qwen3-Embedding pools the last token: left padding, as its model card recommends
    kwargs = {"tokenizer_kwargs": {"padding_side": "left"}} if "qwen3-embedding" in model_id.lower() else {}
    try:
        return SentenceTransformer(model_id, **kwargs)
    except OSError as exc:
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
            raise OSError(f"embedding model {model_id!r} is not in the local Hugging Face cache and HF_HUB_OFFLINE=1: "
                          f"download it once with HF_HUB_OFFLINE=0, or choose a cached model with HGRAG_EMBED_MODEL "
                          f"(e.g. sentence-transformers/all-MiniLM-L6-v2). Cause: {exc}") from exc
        raise


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
MERGE_LABELS = PROSE_LABELS | {"code"}  # the lines of a code block cut by Docling are text or code items
MAX_MERGED_WORDS = 120  # a run of short lines is cut into chunks of about this size


@dataclass
class _Piece:
    """A chunk of the chunker, before the filters: its text, its Docling items, headings and labels."""

    text: str
    items: list[Any]
    headings: list[str]
    labels: set[str]


def _too_short(n_words: int, labels: set[str], min_words: int) -> bool:
    return n_words < min_words or bool(min_words and n_words < config.MIN_PROSE_WORDS and labels <= PROSE_LABELS)


def _merge_short_pieces(pieces: list[_Piece], by_ref: dict[str, SectionNode], min_words: int) -> list[_Piece]:
    """Join the consecutive short pieces of one section (``HGRAG_MERGE_SHORT_LINES``): Docling cuts a code block or a
    list of values into lines of a few words, each dropped as too short, and the command it holds was never indexed.
    Joined, the lines form one chunk; what is still short after joining is dropped as before."""
    out: list[_Piece] = []
    run: list[_Piece] = []

    def section(piece: _Piece) -> str | None:
        node = by_ref.get(piece.items[0].self_ref)
        return node.ref if node else None

    def flush() -> None:
        if len(run) == 1:
            out.append(run[0])
        elif run:
            out.append(_Piece("\n".join(p.text for p in run), [i for p in run for i in p.items], run[0].headings,
                              set().union(*(p.labels for p in run))))
        run.clear()

    for piece in pieces:
        short = _too_short(len(piece.text.split()), piece.labels, min_words) and piece.labels <= MERGE_LABELS
        if not short:
            flush()
            out.append(piece)
            continue
        words = sum(len(p.text.split()) for p in run)
        if run and (section(run[0]) != section(piece) or words >= MAX_MERGED_WORDS):
            flush()
        run.append(piece)
    flush()
    return out


def chunk_paragraphs(
    doc: DoclingDocument,
    chunker: BaseChunker,
    root: SectionNode,
    doc_id: str,
    min_words: int = config.MIN_CHUNK_WORDS,
    stats: dict[str, Any] | None = None,
    labels: Sequence[str] | set[DocItemLabel] | None = None,
    table_descriptions: dict[str, str] | None = None,
    formula_readings: dict[str, str] | None = None,
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
    pieces = []
    for chunk in chunker.chunk(dl_doc=without_long_headings(doc)):
        # HybridChunker exposes generic ``DocItem`` objects (not ``TextItem``), so only the
        # label can be relied on to recognise paragraphs.
        resolved = list(chunk.meta.doc_items)
        labels_ = {getattr(r.label, "value", str(r.label)) for r in resolved}
        if not resolved or not all(r.label in allowed for r in resolved):
            dropped.update(labels_ or {"<no items>"})
            continue
        pieces.append(_Piece(chunk.text, resolved, list(chunk.meta.headings or []), labels_))
    if config.MERGE_SHORT_LINES:
        pieces = _merge_short_pieces(pieces, by_ref, min_words)
    for piece in pieces:
        text, resolved, labels_ = piece.text, piece.items, piece.labels
        n_words = len(text.split())
        if _too_short(n_words, labels_, min_words):
            dropped["too_short"] += 1
            continue
        first = resolved[0]
        chain = list(piece.headings)
        if excluded and any(excluded.match(h.strip()) for h in chain):
            dropped["excluded_section"] += 1
            continue
        chain = [h for h in chain if h.strip() not in titles]
        context = chain[-config.HEADING_CONTEXT:] if config.HEADING_CONTEXT > 0 else []
        heading_prefix = "\n".join(context)
        embed_text = f"{heading_prefix}\n{text}" if heading_prefix else text
        description = next((table_descriptions[r.self_ref] for r in resolved
                            if table_descriptions and r.self_ref in table_descriptions), None)
        reading = next((formula_readings[r.self_ref] for r in resolved
                        if formula_readings and r.self_ref in formula_readings), None)
        if description or reading:  # generated text first: a long table is truncated by the encoder, not the description
            embed_text = "\n".join([*([heading_prefix] if heading_prefix else []),
                                    *(x for x in (description, reading) if x), text])
        section = by_ref.get(first.self_ref)
        page_no = first.prov[0].page_no if first.prov else None
        out.append(
            {
                "chunk_index": len(out),
                "doc_id": doc_id,
                "text": text,
                "embed_text": embed_text,
                **({"table_description": description} if description else {}),
                **({"formula_reading": reading} if reading else {}),
                "kinds": sorted(labels_),
                "headings": list(piece.headings),
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
    """New collections get the dense vector (unnamed) and the BM25 keyword vector (``keywords.SPARSE_NAME``, IDF
    applied by Qdrant); a collection made before the hybrid search keeps its single dense vector."""
    from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            sparse_vectors_config={keywords.SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF)},
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
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

    _, doc_prefix = config.embed_prefixes(model_id)
    vectors = embedder.encode(
        [doc_prefix + c["embed_text"] for c in chunks], batch_size=batch_size, normalize_embeddings=True
    )
    ensure_collection(client, collection, int(vectors.shape[1]))
    hybrid = keywords.has_keywords(client, collection)
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )
    points = [
        PointStruct(
            id=_point_id(doc_id, c["chunk_index"]),
            vector=_vectors(vec.tolist(), keywords.document_vector(c["embed_text"]) if hybrid else None),
            payload={
                **{k: v for k, v in c.items() if k != "embed_text"},
                "source_pdf": source_pdf,
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def _vectors(dense: list[float], sparse: Any) -> Any:
    return {"": dense, keywords.SPARSE_NAME: sparse} if sparse is not None else dense


def open_store(path: str | Path = config.QDRANT_PATH) -> QdrantClient:
    from qdrant_client import QdrantClient

    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))
