"""Local runner: status file, task graph, failure isolation, MLflow tracking (no model, no Docling)."""

from __future__ import annotations

import sys
import types

import mlflow
import pytest

from hgrag import pipeline, runner, tracking


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlflow")
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}")
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    for n in ("a.pdf", "b.pdf"):
        (pdfs / n).write_bytes(b"%PDF")
    # Docling converter is never built here
    monkeypatch.setitem(sys.modules, "hgrag.parsing",
                        types.SimpleNamespace(build_converter=lambda **k: object()))
    return pdfs, tmp_path / "runs"


def _patch_steps(monkeypatch, fail_parse=()):
    def parse_step(pdf, *, converter=None, run_key=None, describe_pictures=True):
        name = pdf.split("/")[-1]
        with tracking.track_step(tracking.INDEXING, "parse", run_key, item=name) as tr:
            if name in fail_parse:
                raise ValueError("corrupt pdf")
            tr.metrics(n_figures=2)
        return {"doc_id": name[0], "source_pdf": pdf, "doc_dir": f"/tmp/{name[0]}", "n_figures": 2}

    def hierarchy(doc_dir, *, run_key=None):
        with tracking.track_step(tracking.INDEXING, "hierarchy", run_key, item=doc_dir):
            return {"n_sections": 3, "n_leaf_sections": 2, "n_elements": 9}

    def index(doc_dirs, *, run_key=None, **kw):
        with tracking.track_step(tracking.INDEXING, "index", run_key) as tr:
            tr.metrics(total_chunks=len(doc_dirs) * 5)
        tracking.finish_execution(tracking.INDEXING, run_key, total_chunks=len(doc_dirs) * 5)
        return {"total_chunks": len(doc_dirs) * 5, "chunks_per_doc": {}, "collection": "c"}

    monkeypatch.setattr(pipeline, "parse_step", parse_step)
    monkeypatch.setattr(pipeline, "build_and_save_hierarchy", hierarchy)
    monkeypatch.setattr(pipeline, "index_documents", index)


def test_success_run_status_and_mlflow(env, monkeypatch):
    pdfs, runs = env
    _patch_steps(monkeypatch)
    st = runner.run_indexing("k1", pdfs, runs_dir=runs)
    assert st["state"] == runner.SUCCESS
    summ = {t: runner.task_summary(v) for t, v in st["tasks"].items()}
    assert summ["parse"]["n"] == 2 and summ["parse"]["n_success"] == 2 and summ["parse"]["mapped"]
    assert summ["index"]["state"] == runner.SUCCESS and not summ["index"]["mapped"]
    assert runner.read_status("k1", runs)["state"] == runner.SUCCESS  # read from disk, as the UI does
    exp = mlflow.get_experiment_by_name(tracking.INDEXING)
    runs_ml = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list",
                                 filter_string="tags.`hgrag.run_key` = 'k1'")
    steps = sorted(r.data.tags["hgrag.step"] for r in runs_ml if r.data.tags["hgrag.kind"] == "step")
    assert steps == ["hierarchy", "hierarchy", "index", "parse", "parse"]


def test_one_failing_pdf_does_not_stop_the_others(env, monkeypatch):
    pdfs, runs = env
    _patch_steps(monkeypatch, fail_parse={"a.pdf"})
    st = runner.run_indexing("k2", pdfs, runs_dir=runs)
    parse = {i["item"]: i for i in st["tasks"]["parse"]["instances"]}
    assert parse["a.pdf"]["state"] == runner.FAILED and "corrupt" in parse["a.pdf"]["error"]
    assert parse["b.pdf"]["state"] == runner.SUCCESS
    assert [i["item"] for i in st["tasks"]["hierarchy"]["instances"]] == ["b.pdf"]
    assert st["tasks"]["index"]["instances"][0]["state"] == runner.SUCCESS
    assert st["state"] == runner.FAILED  # partial failure is visible


def test_no_pdf_fails_cleanly(env, monkeypatch, tmp_path):
    _, runs = env
    empty = tmp_path / "empty"
    empty.mkdir()
    _patch_steps(monkeypatch)
    st = runner.run_indexing("k3", empty, runs_dir=runs)
    assert st["state"] == runner.FAILED and "No PDF" in st["error"]


def test_dead_process_is_reported_failed(env):
    _, runs = env
    w = runner.StatusWriter("k4", runner.INDEXING_GRAPH, {"pdf_dir": "x"}, runs)
    w.data["pid"] = 2**22 + 12345  # no such process
    w.flush()
    st = runner.read_status("k4", runs)
    assert st["state"] == runner.FAILED and "died" in st["error"]
    assert runner.active_run(runs) is None


def test_partial_rerun_reuses_disk_and_limits_documents(env, monkeypatch, tmp_path):
    from hgrag import config

    pdfs, runs = env
    (pdfs / "a.pdf").write_bytes(b"%PDF a")
    (pdfs / "b.pdf").write_bytes(b"%PDF b")
    out = tmp_path / "out"
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    from hgrag.ids import compute_doc_id

    for n in ("a.pdf", "b.pdf"):  # outputs of a previous run
        d = out / compute_doc_id(pdfs / n)
        d.mkdir(parents=True)
        (d / "document.json").write_text("{}", encoding="utf-8")
        (d / "manifest.json").write_text(
            '{"doc_id": "%s", "source_pdf": "%s", "doc_dir": "%s", "n_figures": 0}'
            % (d.name, pdfs / n, str(d).replace("\\", "/")), encoding="utf-8")
    calls = {"parse": [], "hier": [], "index": []}

    def parse_step(*a, **k):
        calls["parse"].append(a)
        raise AssertionError("parse must not run")

    monkeypatch.setattr(pipeline, "parse_step", parse_step)
    monkeypatch.setattr(pipeline, "build_and_save_hierarchy",
                        lambda d, **k: calls["hier"].append(d) or {"n_sections": 1, "n_leaf_sections": 1, "n_elements": 1})
    monkeypatch.setattr(pipeline, "index_documents",
                        lambda dirs, **k: calls["index"].append(list(dirs)) or {"total_chunks": 1, "chunks_per_doc": {}})

    st = runner.run_indexing("k5", pdfs, steps=["hierarchy", "index"], pdfs=["b.pdf"], runs_dir=runs)
    assert st["state"] == runner.SUCCESS
    assert runner.task_summary(st["tasks"]["parse"])["state"] == runner.SKIPPED
    assert not calls["parse"] and len(calls["hier"]) == 1 and calls["hier"][0].endswith(compute_doc_id(pdfs / "b.pdf"))
    assert calls["index"] == [calls["hier"]]  # only the selected document is (re)indexed

    # index only: hierarchy skipped too, Qdrant step runs for the selected document
    st = runner.run_indexing("k6", pdfs, steps=["index"], pdfs=["a.pdf"], runs_dir=runs)
    assert runner.task_summary(st["tasks"]["hierarchy"])["state"] == runner.SKIPPED
    assert len(calls["index"]) == 2 and calls["index"][1][0].endswith(compute_doc_id(pdfs / "a.pdf"))

    rows = {r["pdf"]: r for r in runner.document_state(pdfs)}
    assert rows["a.pdf"]["parsed"] is True and rows["a.pdf"]["hierarchy"] is False


def test_missing_parse_output_is_reported_not_crashing(env, monkeypatch, tmp_path):
    from hgrag import config

    pdfs, runs = env
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "empty_out")
    _patch_steps(monkeypatch)
    st = runner.run_indexing("k7", pdfs, steps=["hierarchy", "index"], runs_dir=runs)
    assert st["state"] == runner.FAILED and "nothing to index" in st["error"]
    assert any("not parsed yet" in (i["error"] or "") for i in st["tasks"]["hierarchy"]["instances"])


def test_unknown_pdf_and_empty_steps(env, monkeypatch):
    pdfs, runs = env
    _patch_steps(monkeypatch)
    assert "PDF not found" in runner.run_indexing("k8", pdfs, pdfs=["nope.pdf"], runs_dir=runs)["error"]
    assert "No step" in runner.run_indexing("k9", pdfs, steps=[], runs_dir=runs)["error"]


def test_retry_plan_picks_failed_pdfs_and_earliest_step():
    w = {"tasks": {
        "parse": {"instances": [{"item": "a.pdf", "state": "success"}, {"item": "b.pdf", "state": "failed"}]},
        "hierarchy": {"instances": [{"item": "a.pdf", "state": "failed"}]},
        "index": {"instances": []}}}
    assert runner.retry_plan(w) == (["a.pdf", "b.pdf"], ["parse", "hierarchy", "index"])
    w["tasks"]["parse"]["instances"][1]["state"] = "success"
    assert runner.retry_plan(w) == (["a.pdf"], ["hierarchy", "index"])
    assert runner.retry_plan({"tasks": {}}) == ([], ["parse", "hierarchy", "index"])
