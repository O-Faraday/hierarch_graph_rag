"""Step 1 - PDF parsing with Docling.

Heading levels are rebuilt with ``HeadingHierarchyOptions(enabled=True)`` (bookmarks,
numbering, visual style). Figures are exported as PNG files and a metadata record is
produced for each one.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, PictureItem

from . import config
from .ids import compute_doc_id  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)


def parse_options() -> dict[str, Any]:
    """Docling options that change the content of a parsed document (stored in the manifest, used as cache key)."""
    return {"table_mode": config.TABLE_MODE, "table_cell_matching": config.TABLE_CELL_MATCHING,
            "formula_enrichment": config.FORMULA_ENRICHMENT, "code_enrichment": config.CODE_ENRICHMENT,
            "picture_classification": config.PICTURE_CLASSIFICATION, "table_images": config.TABLE_IMAGES}


def build_converter(
    describe_pictures: bool = False,  # unused here: figures are described after parsing (hgrag.figure_vlm)
    images_scale: float = config.IMAGES_SCALE,
) -> DocumentConverter:
    opts = PdfPipelineOptions()
    # Hierarchical headings. The style signal needs the parsed pages to be kept.
    opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    opts.generate_parsed_pages = True
    # Figure images (PNG) are produced from these.
    opts.images_scale = images_scale
    opts.generate_picture_images = True
    # Tables: structure model in its most precise mode, text cells matched to the PDF text layer.
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode(config.TABLE_MODE)
    opts.table_structure_options.do_cell_matching = config.TABLE_CELL_MATCHING
    # Enrichments (extra models run by Docling on the formula / code regions).
    opts.do_formula_enrichment = config.FORMULA_ENRICHMENT
    opts.do_code_enrichment = config.CODE_ENRICHMENT
    opts.do_picture_classification = config.PICTURE_CLASSIFICATION
    opts.generate_table_images = config.TABLE_IMAGES
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _bbox_dict(bbox: Any) -> dict[str, Any] | None:
    if bbox is None:
        return None
    return {
        "l": bbox.l,
        "t": bbox.t,
        "r": bbox.r,
        "b": bbox.b,
        "coord_origin": getattr(bbox.coord_origin, "value", str(bbox.coord_origin)),
    }


def picture_metadata(
    pic: PictureItem,
    doc: DoclingDocument,
    *,
    index: int,
    doc_id: str,
    png_path: Path | None,
    rel_png_path: str | None,
) -> dict[str, Any]:
    """All metadata attached to one figure (see PictureItem / PictureMeta in docling-core)."""
    prov = [
        {"page_no": p.page_no, "bbox": _bbox_dict(p.bbox), "charspan": list(p.charspan)}
        for p in pic.prov
    ]
    image = pic.image
    meta = pic.meta
    description = getattr(meta, "description", None) if meta is not None else None
    classification = getattr(meta, "classification", None) if meta is not None else None

    record: dict[str, Any] = {
        "doc_id": doc_id,
        "figure_index": index,
        "self_ref": pic.self_ref,
        "parent_ref": pic.parent.cref if pic.parent else None,
        "children_refs": [c.cref for c in pic.children],
        "label": getattr(pic.label, "value", str(pic.label)),
        "content_layer": getattr(pic.content_layer, "value", str(pic.content_layer)),
        "page_no": prov[0]["page_no"] if prov else None,
        "prov": prov,
        "caption": pic.caption_text(doc) or None,
        "caption_refs": [c.cref for c in pic.captions],
        "footnote_refs": [f.cref for f in pic.footnotes],
        "reference_refs": [r.cref for r in pic.references],
        "png_path": rel_png_path,
        "image": (
            {
                "mimetype": image.mimetype,
                "dpi": image.dpi,
                "width": image.size.width,
                "height": image.size.height,
            }
            if image is not None
            else None
        ),
        "description": (
            {
                "text": description.text,
                "created_by": description.created_by,
                "confidence": description.confidence,
            }
            if description is not None
            else None
        ),
        "classification": classification.model_dump(mode="json") if classification is not None else None,
        "meta": meta.model_dump(mode="json") if meta is not None else None,
        "source": pic.source.model_dump(mode="json") if pic.source else None,
    }
    if png_path is not None and png_path.exists():
        record["png_bytes"] = png_path.stat().st_size
    return record


def export_figures(doc: DoclingDocument, doc_id: str, doc_dir: Path) -> list[dict[str, Any]]:
    """Save each figure as PNG under ``doc_dir/figures`` and return their metadata."""
    fig_dir = doc_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    index = 0
    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        index += 1
        png_path: Path | None = None
        rel: str | None = None
        img = item.get_image(doc)
        if img is not None:
            png_path = fig_dir / f"figure-{index:03d}.png"
            img.save(png_path, format="PNG")
            rel = png_path.relative_to(doc_dir).as_posix()  # portable (no backslashes on Windows)
        records.append(
            picture_metadata(item, doc, index=index, doc_id=doc_id, png_path=png_path, rel_png_path=rel)
        )
    return records


def doc_stats(doc: DoclingDocument, figures: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts used to trace the parsing: pages, items per label, figures and their descriptions."""
    labels: Counter[str] = Counter()
    for item, _level in doc.iterate_items():
        labels[getattr(item.label, "value", str(item.label))] += 1
    return {
        "n_pages": len(doc.pages),
        "n_tables": len(doc.tables),
        "n_pictures": len(doc.pictures),
        "n_headings": labels.get("section_header", 0) + labels.get("title", 0),
        "n_formulas": labels.get("formula", 0),
        "n_paragraphs": labels.get("text", 0) + labels.get("paragraph", 0),
        "n_list_items": labels.get("list_item", 0),
        "n_figures_described": sum(1 for f in figures if f.get("description")),
        "n_figures_png": sum(1 for f in figures if f.get("png_path")),
        "labels": dict(sorted(labels.items())),
    }


def parse_pdf(
    pdf_path: str | Path,
    output_root: str | Path = config.OUTPUT_DIR,
    *,
    converter: DocumentConverter | None = None,
    describe_pictures: bool = config.DESCRIBE_PICTURES,
    trace: Any = None,
) -> dict[str, Any]:
    """Parse one PDF and write ``document.json``, ``figures/*.png`` and ``figures.json``. ``trace``: the span of the
    parse step (``tracing``), under which the conversion, the figure export and the descriptions are recorded."""
    from .tracing import Span

    trace = trace or Span()
    pdf_path = Path(pdf_path)
    doc_id = compute_doc_id(pdf_path)
    doc_dir = Path(output_root) / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    converter = converter or build_converter(describe_pictures=describe_pictures)
    t0 = time.perf_counter()
    with trace.child("docling", "PARSER", inputs={"pdf": str(pdf_path), "options": parse_options()}) as sp:
        doc = converter.convert(pdf_path).document
        sp.set({"stats": doc_stats(doc, []), "markdown": doc.export_to_markdown()})
    convert_seconds = time.perf_counter() - t0

    doc.save_as_json(doc_dir / "document.json")
    (doc_dir / "document.md").write_text(doc.export_to_markdown(), encoding="utf-8")  # for analysis
    with trace.child("export_figures", "PARSER", inputs={"n_pictures": len(doc.pictures)}) as sp:
        figures = export_figures(doc, doc_id, doc_dir)
        sp.set({"figures": figures})
    (doc_dir / "figures.json").write_text(json.dumps(figures, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "doc_dir": str(doc_dir),
        "n_figures": len(figures),
        "convert_seconds": round(convert_seconds, 2),
        "parse_options": parse_options(),
        **doc_stats(doc, figures),
    }
    (doc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if describe_pictures and (figures or manifest.get("n_tables") or manifest.get("n_formulas")):
        from .describe import describe_document

        try:
            manifest = describe_document(doc_dir, trace=trace)
        except Exception as exc:  # noqa: BLE001  (no token, API down...): the parsing itself is still valid
            logger.warning("figure/table description skipped for %s: %s", pdf_path.name, exc)
            manifest["figure_vlm"] = {"error": str(exc)}
            (doc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
