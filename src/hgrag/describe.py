"""Describe the figures and the tables of a parsed document with a model called through an API.

Runs after the Docling parsing (Docling is not involved) and writes the text next to the Docling outputs:

* figures: each PNG goes to the vision model (``HGRAG_FIGURE_VLM``, winner of ``hgrag vlm-bench``);
  the text is stored in ``figures.json`` (``description.text``) and becomes one chunk of its own;
* formulas: the crop of each formula goes to the same vision model, which returns LaTeX (``formulas.json``);
  ``document.json`` is updated, so the formula chunks and the reconstructed sections carry the LaTeX;
* tables: the table as Markdown goes to the model (``HGRAG_TABLE_MODEL``, text only: the numbers come from
  Docling's extraction, not from reading pixels); the text is stored in ``tables.json`` and is **added to the
  embedding text of the table chunks** (the raw rows stay the chunk text).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import config, generation, vlm_bench
from .generation import GenerationError

logger = logging.getLogger(__name__)


def png_file(doc_dir: Path, png_path: str | None) -> Path | None:
    if not png_path:
        return None
    return doc_dir / Path(*png_path.replace("\\", "/").split("/"))  # outputs written on Windows use backslashes


def describe_figures(figures: list[dict[str, Any]], doc_dir: str | Path, *, spec: str | None = None,
                     prompt: str | None = None, force: bool = False, workers: int = 4,
                     api_key: str | None = None) -> dict[str, Any]:
    """Fill ``description`` in the figure records (in place). Returns counts, model and timings."""
    doc_dir = Path(doc_dir)
    spec = spec or config.FIGURE_VLM
    prompt = prompt or config.PICTURE_PROMPT or vlm_bench.DESCRIBE_PROMPT
    cfg = vlm_bench.model_config(spec, api_key)
    if spec.startswith("hf:") and not cfg.api_key:
        raise GenerationError("HF_TOKEN is not set: add it to .env to describe the figures.")
    todo = [f for f in figures if png_file(doc_dir, f.get("png_path")) and (force or not f.get("description"))]
    errors: list[str] = []
    t0 = time.perf_counter()

    def work(rec: dict[str, Any]) -> None:
        try:
            text, _usage, _secs = vlm_bench.describe_figure(cfg, str(png_file(doc_dir, rec["png_path"])), prompt)
            if not text:
                raise GenerationError("empty description")
            rec["description"] = {"text": text, "created_by": spec, "confidence": None}
            rec.pop("description_error", None)
        except Exception as exc:  # noqa: BLE001
            rec["description_error"] = str(exc)
            errors.append(f"figure {rec['figure_index']}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    return {"model": spec, "n_todo": len(todo), "n_described": len(todo) - len(errors), "n_errors": len(errors),
            "errors": errors, "seconds": round(time.perf_counter() - t0, 2)}


TABLE_SYSTEM = "You describe tables extracted from PDF documents so that they can be indexed and searched."
TABLE_PROMPT = (
    "Describe this table in plain sentences for a search index: what it lists, its columns, the number of rows, "
    "and the notable values (labels with their numbers, highest and lowest, totals) written out in words and digits. "
    "Use only the data given, do not infer anything else. At most 120 words.\n\n{table}"
)
MAX_TABLE_CHARS = 8000


def table_hash(markdown: str) -> str:
    return hashlib.sha1(markdown.encode("utf-8")).hexdigest()[:12]


def table_records(doc: Any) -> list[dict[str, Any]]:
    """The tables of a DoclingDocument with their Markdown (what the hierarchy and the chunker also see)."""
    out = []
    for i, t in enumerate(doc.tables, start=1):
        md = t.export_to_markdown(doc)
        out.append({"table_index": i, "self_ref": t.self_ref, "page_no": t.prov[0].page_no if t.prov else None,
                    "caption": t.caption_text(doc) or None, "num_rows": t.data.num_rows, "num_cols": t.data.num_cols,
                    "sha": table_hash(md), "markdown": md})
    return out


def describe_tables(records: list[dict[str, Any]], previous: list[dict[str, Any]] | None = None, *,
                    spec: str | None = None, force: bool = False, workers: int = 4,
                    api_key: str | None = None) -> dict[str, Any]:
    """Fill ``description`` in the table records (in place). A table already described with the same content is kept."""
    spec = spec or config.TABLE_MODEL
    cfg = vlm_bench.model_config(spec, api_key, max_tokens=800)
    if spec.startswith("hf:") and not cfg.api_key:
        raise GenerationError("HF_TOKEN is not set: add it to .env to describe the tables.")
    kept = {(p["self_ref"], p.get("sha")): p["description"] for p in previous or [] if p.get("description")}
    todo = []
    for r in records:
        if not force and (r["self_ref"], r["sha"]) in kept:
            r["description"] = kept[(r["self_ref"], r["sha"])]
        else:
            todo.append(r)
    errors: list[str] = []
    t0 = time.perf_counter()

    def work(rec: dict[str, Any]) -> None:
        md = rec["markdown"][:MAX_TABLE_CHARS]
        table = (f"Caption: {rec['caption']}\n\n" if rec["caption"] else "") + md
        try:
            text, _usage = generation.complete(cfg, TABLE_SYSTEM, TABLE_PROMPT.format(table=table), max_tokens=800)
            text = vlm_bench.clean_description(text)
            if not text:
                raise GenerationError("empty description")
            rec["description"] = {"text": text, "created_by": spec, "confidence": None}
        except Exception as exc:  # noqa: BLE001
            rec["description"] = None
            rec["description_error"] = str(exc)
            errors.append(f"table {rec['table_index']}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    return {"model": spec, "n_todo": len(todo), "n_described": len(todo) - len(errors), "n_errors": len(errors),
            "errors": errors, "seconds": round(time.perf_counter() - t0, 2)}


def load_table_descriptions(doc_dir: str | Path) -> dict[str, str]:
    """``self_ref -> description`` of the tables, if ``tables.json`` exists (used by the hierarchy and the chunker)."""
    path = Path(doc_dir) / "tables.json"
    if not path.exists():
        return {}
    return {r["self_ref"]: r["description"]["text"] for r in json.loads(path.read_text(encoding="utf-8"))
            if r.get("description")}


FORMULA_SYSTEM = "You transcribe mathematical formulas from images into LaTeX."
FORMULA_PROMPT = ("Transcribe the formula in this image into LaTeX. Reply with the LaTeX only: no explanation, no code "
                  "fence, no $ delimiters. If the image is not a formula, reply with the text it contains.")
FORMULA_SCALE = 3.0  # rendering scale of the PDF page (1.0 = 72 dpi)
FORMULA_PAD = 4.0  # points added around the formula box


def clean_latex(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", text).strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
    return text


def formula_items(doc: Any) -> list[Any]:
    from docling_core.types.doc import DocItemLabel

    return [t for t in doc.texts if t.label == DocItemLabel.FORMULA and t.prov]


def crop_formulas(doc: Any, pdf_path: str | Path, items: list[Any], out_dir: Path) -> dict[str, Path]:
    """Render the PDF pages with pdfium and save one PNG per formula (key: self_ref)."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages: dict[int, Any] = {}
    paths: dict[str, Path] = {}
    try:
        for n, item in enumerate(items, start=1):
            prov = item.prov[0]
            if prov.page_no not in pages:
                pages[prov.page_no] = pdf[prov.page_no - 1].render(scale=FORMULA_SCALE).to_pil()
            page = pages[prov.page_no]
            height = doc.pages[prov.page_no].size.height
            box = prov.bbox.to_top_left_origin(page_height=height)
            k = FORMULA_SCALE
            crop = page.crop((max(0, int((box.l - FORMULA_PAD) * k)), max(0, int((box.t - FORMULA_PAD) * k)),
                              min(page.width, int((box.r + FORMULA_PAD) * k)), min(page.height, int((box.b + FORMULA_PAD) * k))))
            path = out_dir / f"formula_{n:04d}.png"
            crop.save(path)
            paths[item.self_ref] = path
    finally:
        pdf.close()
    return paths


def _ask_image(cfg: Any, png: Path, prompt: str, system: str) -> str:
    media, data = generation._encode_image(str(png), cfg.max_image_edge)
    image = ({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
             if cfg.provider == "anthropic" else {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
    parts = []
    for d in generation.stream_completion(cfg, system, [image, {"type": "text", "text": prompt}]):
        if "text" in d:
            parts.append(d["text"])
    return "".join(parts)


def transcribe_formulas(doc_dir: str | Path, doc: Any, pdf_path: str | Path, *, spec: str | None = None, force: bool = False,
                        workers: int = 4, api_key: str | None = None) -> dict[str, Any]:
    """Replace the text of each formula item by LaTeX read from its image. Updates ``doc`` in place; the caller saves it.
    Records (crop, model, LaTeX) are written to ``formulas.json``; formulas already done are kept unless ``force``."""
    doc_dir = Path(doc_dir)
    spec = spec or config.FORMULA_VLM or config.FIGURE_VLM
    cfg = vlm_bench.model_config(spec, api_key)
    if spec.startswith("hf:") and not cfg.api_key:
        raise GenerationError("HF_TOKEN is not set: add it to .env to transcribe the formulas.")
    items = formula_items(doc)
    path = doc_dir / "formulas.json"
    done = {r["self_ref"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    todo = [i for i in items if force or i.self_ref not in done or not done[i.self_ref].get("latex")]
    errors: list[str] = []
    t0 = time.perf_counter()
    if todo:
        crops = crop_formulas(doc, pdf_path, todo, doc_dir / "formulas")

        def work(item: Any) -> None:
            try:
                latex = clean_latex(_ask_image(cfg, crops[item.self_ref], FORMULA_PROMPT, FORMULA_SYSTEM))
                if not latex:
                    raise GenerationError("empty transcription")
                done[item.self_ref] = {"self_ref": item.self_ref, "page_no": item.prov[0].page_no, "latex": latex,
                                       "original_text": item.orig, "created_by": spec,
                                       "png_path": f"formulas/{crops[item.self_ref].name}"}
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{item.self_ref}: {exc}")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, todo))
    for item in items:
        rec = done.get(item.self_ref)
        if rec and rec.get("latex"):
            item.text = rec["latex"]
    path.write_text(json.dumps([done[i.self_ref] for i in items if i.self_ref in done], indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return {"model": spec, "n_formulas": len(items), "n_transcribed": sum(1 for i in items if done.get(i.self_ref, {}).get("latex")),
            "errors": errors, "seconds": round(time.perf_counter() - t0, 1)}


def describe_document(doc_dir: str | Path, *, spec: str | None = None, force: bool = False, workers: int = 4,
                      api_key: str | None = None, figures: bool = True, tables: bool = True,
                      formulas: bool | None = None) -> dict[str, Any]:
    """Describe the figures and the tables of one parsed document; rewrites ``figures.json``, ``tables.json`` and
    the manifest, returns the manifest. ``spec`` overrides the figure model only."""
    doc_dir = Path(doc_dir)
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if figures:
        recs = json.loads((doc_dir / "figures.json").read_text(encoding="utf-8"))
        stats = describe_figures(recs, doc_dir, spec=spec, force=force, workers=workers, api_key=api_key)
        (doc_dir / "figures.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["n_figures_described"] = sum(1 for f in recs if f.get("description"))
        manifest["figure_vlm"] = stats
        problems += stats["errors"]
    if tables:
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument.load_from_json(doc_dir / "document.json")
        recs = table_records(doc)
        prev_path = doc_dir / "tables.json"
        prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else None
        stats = describe_tables(recs, prev, force=force, workers=workers, api_key=api_key)
        (doc_dir / "tables.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["n_tables_described"] = sum(1 for r in recs if r.get("description"))
        manifest["table_llm"] = stats
        problems += stats["errors"]
    if (config.FORMULA_MODE == "vlm") if formulas is None else formulas:
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument.load_from_json(doc_dir / "document.json")
        pdf = Path(manifest.get("source_pdf") or "")
        if formula_items(doc) and pdf.exists():
            stats = transcribe_formulas(doc_dir, doc, pdf, force=force, workers=workers, api_key=api_key)
            doc.save_as_json(doc_dir / "document.json")
            (doc_dir / "document.md").write_text(doc.export_to_markdown(), encoding="utf-8")
            manifest["n_formulas_transcribed"] = stats["n_transcribed"]
            manifest["formula_vlm"] = stats
            problems += stats["errors"]
    (doc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if problems:
        logger.warning("%s: %d item(s) not described: %s", doc_dir.name, len(problems), problems)
    return manifest
