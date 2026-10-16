"""Control PDFs: the files match expected.json, and ``hgrag verify`` catches what it should (offline)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc import DoclingDocument, ProvenanceItem
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import TableCell, TableData
from PIL import Image
from pypdf import PdfReader

from hgrag import config, pipeline, verify
from hgrag.ids import compute_doc_id
from hgrag.indexing import open_store
from test_retrieval import BagOfWordsEmbedder

ROOT = Path(__file__).resolve().parents[1]
CONTROL = config.CONTROL_PDF_DIR
EXPECTED = json.loads((CONTROL / "expected.json").read_text(encoding="utf-8"))


def _outline(reader: PdfReader) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []

    def walk(items, level=1):
        for it in items:
            if isinstance(it, list):
                walk(it, level + 1)
            else:
                out.append((it.title, level))

    walk(reader.outline)
    return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_control_pdf_content_matches_expected(name):
    exp = EXPECTED[name]
    reader = PdfReader(CONTROL / name)
    assert _outline(reader) == [tuple(h) for h in exp["headings"]]  # bookmarks = heading hierarchy
    text = " ".join(p.extract_text() for p in reader.pages).lower()
    text = re.sub(r"\s+", " ", text)
    for needle in exp["needles"]:
        assert all(w in text for w in needle["query"].lower().split()), needle
    assert sum(len(p.images) for p in reader.pages) == exp["n_figures"]


def test_three_pdfs_and_committed_files_are_up_to_date(tmp_path):
    assert len(EXPECTED) == 3
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_control_pdfs.py"), str(tmp_path)],
                   check=True, capture_output=True)
    for name in EXPECTED:  # deterministic generation: committed PDFs = regenerated PDFs (stable doc_id)
        assert compute_doc_id(tmp_path / name) == compute_doc_id(CONTROL / name), name
    assert json.loads((tmp_path / "expected.json").read_text(encoding="utf-8")) == EXPECTED


# --------------------------------------------------------------------------- offline pipeline
def _prov(page=1):
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _fake_docling_output(pdf: Path, exp: dict, out_root: Path) -> Path:
    """What Docling would write for this PDF, rebuilt from expected.json (no model needed)."""
    doc_id = compute_doc_id(pdf)
    doc_dir = out_root / doc_id
    (doc_dir / "figures").mkdir(parents=True)
    needles = {n["section"]: n["query"] for n in exp["needles"]}
    d = DoclingDocument(name=pdf.stem)
    d.add_page(page_no=1, size=Size(width=600, height=800))
    d.add_title(text=pdf.stem.replace("_", " "), prov=_prov())
    last = None
    for title, level in exp["headings"]:
        last = d.add_heading(text=title, level=level, prov=_prov())
        body = needles.get(title)
        d.add_text(label="text", parent=last, prov=_prov(),
                   text=f"{body} as described here." if body else f"General background about {title.lower()} topics.")
    figures = []
    for i in range(1, exp["n_figures"] + 1):
        pic = d.add_picture(parent=last, prov=_prov(), caption=d.add_text(label="caption", text=f"Figure {i}"))
        Image.new("RGB", (40, 30), "white").save(doc_dir / "figures" / f"figure-{i:03d}.png")
        figures.append({"self_ref": pic.self_ref, "figure_index": i, "png_path": f"figures/figure-{i:03d}.png",
                        "description": {"text": "a chart"}})
    for _ in range(exp["n_tables"]):
        cells = [TableCell(text="a", start_row_offset_idx=r, end_row_offset_idx=r + 1,
                           start_col_offset_idx=0, end_col_offset_idx=1) for r in range(2)]
        d.add_table(data=TableData(num_rows=2, num_cols=1, table_cells=cells), parent=last, prov=_prov())
    d.save_as_json(doc_dir / "document.json")
    (doc_dir / "figures.json").write_text(json.dumps(figures), encoding="utf-8")
    (doc_dir / "manifest.json").write_text(json.dumps(
        {"doc_id": doc_id, "source_pdf": str(pdf), "doc_dir": str(doc_dir), "n_figures": len(figures)}), encoding="utf-8")
    return doc_dir


def _process(tmp_path, monkeypatch, run_key=None):
    import hgrag.indexing as indexing

    monkeypatch.setattr(indexing, "load_embedder", lambda *_: BagOfWordsEmbedder())
    monkeypatch.setattr(indexing, "make_chunker", lambda *a, **k: HierarchicalChunker())
    out = tmp_path / "output"
    dirs = []
    for name, exp in EXPECTED.items():
        dd = _fake_docling_output(CONTROL / name, exp, out)
        pipeline.build_and_save_hierarchy(dd, run_key=run_key)
        dirs.append(str(dd))
    res = pipeline.index_documents(dirs, qdrant_path=tmp_path / "qdrant", collection="c", run_key=run_key)
    return tmp_path, out, res


@pytest.fixture()
def processed(tmp_path, monkeypatch):
    return _process(tmp_path, monkeypatch)


def _run_verify(tmp_path, out):
    client = open_store(tmp_path / "qdrant")
    try:
        return verify.verify(CONTROL, output_dir=out, embedder=BagOfWordsEmbedder(), client=client, collection="c")
    finally:
        client.close()


def test_pipeline_traces_and_verify_passes(processed):
    tmp_path, out, res = processed
    assert res["total_chunks"] >= 20 and res["points_in_collection"] == res["total_chunks"]
    assert set(res["documents"]) == set(EXPECTED) and res["embedding_dim"] == 64
    for name, d in res["documents"].items():
        assert d["n_sections_covered"] >= len(EXPECTED[name]["headings"])
        assert (Path(d["doc_dir"]) / "chunks.jsonl").read_text(encoding="utf-8").count("\n") == d["n_chunks"]
        assert (Path(d["doc_dir"]) / "hierarchy_tree.txt").exists()
    tree = (Path(res["documents"]["02_alpine_botany_field_guide.pdf"]["doc_dir"]) / "hierarchy_tree.txt").read_text(encoding="utf-8")
    assert "    - 2.1.1 Edelweiss" in tree and "figure" in tree  # indentation = depth

    report = _run_verify(tmp_path, out)
    assert report["ok"], verify.format_report(report)
    assert report["n_checks"] == 3 * 7 + 6  # 7 structural checks per PDF + 6 needle queries


def test_verify_detects_problems(processed):
    tmp_path, out, res = processed
    bot = Path(res["documents"]["02_alpine_botany_field_guide.pdf"]["doc_dir"])
    (bot / "figures" / "figure-001.png").unlink()  # a figure lost
    tree = json.loads((bot / "hierarchy.json").read_text(encoding="utf-8"))
    tree["children"][2]["children"][0]["level"] = 3  # a heading at the wrong level
    (bot / "hierarchy.json").write_text(json.dumps(tree), encoding="utf-8")
    report = _run_verify(tmp_path, out)
    failed = {c["check"] for c in report["checks"] if not c["ok"]}
    assert not report["ok"]
    assert {"figures extracted as PNG", "headings and levels"} <= failed
    assert "FAIL" in verify.format_report(report)


def test_steps_leave_detailed_traces_in_mlflow(tmp_path, monkeypatch):
    import mlflow
    from hgrag import tracking

    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlflow")
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}")
    key = tracking.new_run_key()
    _process(tmp_path, monkeypatch, run_key=key)

    exp = mlflow.get_experiment_by_name(tracking.INDEXING)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list",
                              filter_string=f"tags.`hgrag.run_key` = '{key}' and tags.`hgrag.kind` = 'step'")
    hier = [r for r in runs if r.data.tags["hgrag.step"] == "hierarchy"]
    assert len(hier) == 3 and all(r.info.status == "FINISHED" for r in runs)
    botany = next(r for r in hier if r.data.metrics["max_depth"] == 3)
    m = botany.data.metrics
    assert m["sections_level_1"] == 4 and m["sections_level_3"] == 2 and m["elements.figure"] == 2
    names = {a.path for a in mlflow.artifacts.list_artifacts(run_id=botany.info.run_id)}
    assert {"hierarchy.json", "hierarchy_tree.txt", "tree_stats.json"} <= names

    idx = next(r for r in runs if r.data.tags["hgrag.step"] == "index")
    m = idx.data.metrics
    assert m["total_chunks"] == m["points_in_collection"] and m["embedding_dim"] == 64
    assert m["chunks.01_aurora_bank_annual_report.pdf"] == 9  # 7 paragraphs + the table + the described figure
    assert {"seconds_load_model", "seconds_chunk", "seconds_embed_store"} <= set(m)
    names = {a.path for a in mlflow.artifacts.list_artifacts(run_id=idx.info.run_id)}
    assert {"chunks", "index_summary.json"} <= names


def test_parse_step_traces_counts_and_artifacts(tmp_path, monkeypatch):
    """Parsing with a stand-in converter: manifest stats, figures PNG and MLflow trace."""
    import mlflow
    import hgrag.parsing as parsing
    from hgrag import tracking

    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlflow")
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}")
    name = "01_aurora_bank_annual_report.pdf"
    exp = EXPECTED[name]
    doc_dir = _fake_docling_output(CONTROL / name, exp, tmp_path / "scratch")
    fake_doc = DoclingDocument.load_from_json(doc_dir / "document.json")

    class Converter:
        def convert(self, _path):
            return type("R", (), {"document": fake_doc})

    real = parsing.parse_pdf
    monkeypatch.setattr(parsing, "parse_pdf", lambda *a, **k: real(*a, output_root=tmp_path / "out", **k))
    key = tracking.new_run_key()
    manifest = pipeline.parse_step(CONTROL / name, converter=Converter(), run_key=key, describe_pictures=False)

    assert manifest["n_pages"] == 1 and manifest["n_tables"] == exp["n_tables"] and manifest["n_headings"] >= 7
    assert manifest["labels"]["section_header"] == len(exp["headings"])
    assert (Path(manifest["doc_dir"]) / "document.md").exists()
    run = next(r for r in mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(tracking.INDEXING).experiment_id], output_format="list",
        filter_string="tags.`hgrag.step` = 'parse'"))
    assert run.data.metrics["n_tables"] == 1 and run.data.metrics["label.section_header"] == 7
    assert {"manifest.json", "figures.json", "document.md", "figures"} <= {
        a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)}


def test_bundle_collects_everything_for_analysis(tmp_path, monkeypatch):
    import json as _json
    import sqlite3
    import zipfile

    from hgrag import bundle, runner

    _, out, _ = _process(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    w = runner.StatusWriter("k1", runner.INDEXING_GRAPH, {}, runs)
    w.finish(w.start("list_pdfs"))
    (runs / "k1" / "log.txt").write_text("log line", encoding="utf-8")
    ml = tmp_path / "ml"
    ml.mkdir()
    con = sqlite3.connect(ml / "mlflow.db")
    con.execute("create table t(x)")
    con.commit()
    con.close()

    z = bundle.build_bundle(tmp_path / "b.zip", output_dir=out, runs_dir=runs, mlflow_dir=ml, control_dir=CONTROL)
    names = set(zipfile.ZipFile(z).namelist())
    doc_id = compute_doc_id(CONTROL / "02_alpine_botany_field_guide.pdf")
    assert {f"output/{doc_id}/document.json", f"output/{doc_id}/hierarchy_tree.txt", f"output/{doc_id}/chunks.jsonl",
            f"output/{doc_id}/figures/figure-001.png", "runs/k1/status.json", "runs/k1/log.txt", "mlflow/mlflow.db",
            "control/expected.json", "control/eval_dataset.jsonl", "control/verify_structure.txt",
            "bundle_info.json"} <= names
    info = _json.loads(zipfile.ZipFile(z).read("bundle_info.json"))
    assert "docling" in info["libraries"] and info["skipped"] == [] and info["files_per_section"]["output"] > 10

    empty = bundle.build_bundle(tmp_path / "e.zip", output_dir=tmp_path / "no", runs_dir=tmp_path / "no",
                                mlflow_dir=tmp_path / "no", control_dir=tmp_path / "no")
    skipped = _json.loads(zipfile.ZipFile(empty).read("bundle_info.json"))["skipped"]
    assert len(skipped) == 3  # nothing parsed, no run, no MLflow: reported, not an error
