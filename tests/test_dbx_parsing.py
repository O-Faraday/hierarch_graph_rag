"""Databricks parsing step with a fake Spark session: nothing here runs on Databricks."""

import re
from types import SimpleNamespace

import pytest

from hgrag import config
from hgrag.dbx import parsing as dbx_parsing
from hgrag.ids import compute_doc_id


class FakeSession:
    """Keeps the ``documents`` table as a dict and understands the statements of the step."""

    def __init__(self):
        self.table: dict[str, dict] = {}
        self.queue: list[str] = []
        self.view: list[tuple] = []
        self.merges = 0
        self.statements: list[str] = []

    def createDataFrame(self, rows, schema):
        names = [c.split()[0] for c in schema.split(", ")]
        return SimpleNamespace(createOrReplaceTempView=lambda _name: setattr(
            self, "view", [dict(zip(names, r)) for r in rows]))

    def sql(self, statement, args=None):
        self.statements.append(statement)
        if statement.startswith("MERGE"):
            self.merges += 1
            for row in self.view:
                self.table[row["doc_id"]] = dict(row)
        elif statement.startswith("UPDATE"):
            for doc_id, row in self.table.items():
                if row["path"] == args["path"] and doc_id != args["doc_id"]:
                    row["status"] = "superseded"
        rows = []
        if "SELECT doc_id" in statement:
            rows = [SimpleNamespace(doc_id=d) for d, r in self.table.items()
                    if r["status"] == "parsed" and r["parse_options"] == args["options"]]
        elif "LEFT ANTI JOIN" in statement:
            known = {r["path"] for r in self.table.values()}
            rows = [SimpleNamespace(path=p.removeprefix("dbfs:")) for p in self.queue if p.removeprefix("dbfs:") not in known]
        elif "SELECT path" in statement:
            rows = [SimpleNamespace(path=r["path"]) for r in self.table.values() if r["status"] in ("failed", "partial")]
        return SimpleNamespace(collect=lambda: rows)


@pytest.fixture
def uc(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_VOLUME", "pdf_store")
    monkeypatch.setattr(config, "DBX_OUTPUT_DIR", str(tmp_path / "output"))


class FakeParse:
    def __init__(self, manifest=None, error=None):
        self.manifest = manifest if manifest is not None else {"n_pages": 2, "convert_seconds": 3.0}
        self.error = error
        self.calls = []

    def __call__(self, pdf, *, output_root, converter, describe_pictures):
        self.calls.append((pdf, output_root, describe_pictures))
        if self.error:
            raise self.error
        return self.manifest


def _pdf(tmp_path, name, content=b"%PDF one"):
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def _run(session, paths, parse):
    return dbx_parsing.parse_paths(session, paths, parse=parse, converter=object())


def test_a_replayed_batch_parses_and_writes_nothing_again(uc, tmp_path):
    session, parse, pdf = FakeSession(), FakeParse(), _pdf(tmp_path, "a.pdf")
    assert _run(session, [pdf], parse)["parsed"] == 1
    summary = _run(session, [pdf], parse)
    assert summary["skipped"] == 1 and summary["parsed"] == 0
    assert len(parse.calls) == 1 and session.merges == 1 and len(session.table) == 1


def test_same_content_under_two_names_is_one_document(uc, tmp_path):
    session, parse = FakeSession(), FakeParse()
    a, b = _pdf(tmp_path, "a.pdf"), _pdf(tmp_path, "copy of a.pdf")
    summary = _run(session, [a, b], parse)
    assert (summary["parsed"], summary["skipped"], len(parse.calls), len(session.table)) == (1, 1, 1, 1)


def test_a_changed_parsing_option_parses_again_without_a_second_row(uc, tmp_path, monkeypatch):
    session, parse, pdf = FakeSession(), FakeParse(), _pdf(tmp_path, "a.pdf")
    _run(session, [pdf], parse)
    monkeypatch.setattr(config, "TABLE_MODE", "fast")
    _run(session, [pdf], parse)
    assert len(parse.calls) == 2 and len(session.table) == 1 and '"fast"' in session.table[compute_doc_id(pdf)]["parse_options"]


def test_a_bad_pdf_does_not_stop_the_batch(uc, tmp_path):
    session = FakeSession()
    bad, good = _pdf(tmp_path, "bad.pdf", b"bad"), _pdf(tmp_path, "good.pdf", b"good")

    def parse(pdf, **kwargs):
        if pdf == bad:
            raise RuntimeError("corrupted")
        return {"n_pages": 1, "convert_seconds": 1.0}

    summary = _run(session, [bad, good], parse)
    assert summary["failed"] == 1 and summary["parsed"] == 1
    assert session.table[compute_doc_id(bad)]["status"] == "failed"
    assert "corrupted" in session.table[compute_doc_id(bad)]["error"]


def test_an_unreadable_file_gets_a_failed_row(uc, tmp_path):
    session = FakeSession()
    summary = _run(session, [str(tmp_path / "gone.pdf")], FakeParse())
    assert summary["failed"] == 1
    row = next(iter(session.table.values()))
    assert row["doc_id"].startswith("unreadable-") and row["status"] == "failed"


@pytest.mark.parametrize("manifest, expected", [
    ({"figure_vlm": {"error": "HF_TOKEN is not set"}}, "HF_TOKEN"),
    ({"table_llm": {"errors": ["t1", "t2"]}}, "2 item"),
    ({"formula_vlm": {"errors": ["f"]}}, "formula_vlm"),
    ({"conversion_status": "partial_success"}, "stopped"),
])
def test_an_incomplete_document_does_not_pass_for_parsed(uc, tmp_path, manifest, expected):
    session, pdf = FakeSession(), _pdf(tmp_path, "a.pdf")
    _run(session, [pdf], FakeParse(manifest))
    row = session.table[compute_doc_id(pdf)]
    assert row["status"] == "partial" and expected in row["error"]


def test_complete_descriptions_are_parsed(uc, tmp_path):
    session, pdf = FakeSession(), _pdf(tmp_path, "a.pdf")
    _run(session, [pdf], FakeParse({"figure_vlm": {"errors": []}, "conversion_status": "success"}))
    assert session.table[compute_doc_id(pdf)]["status"] == "parsed"


def test_retry_handles_failed_and_partial_but_not_parsed(uc, tmp_path):
    session = FakeSession()
    ok, partial = _pdf(tmp_path, "ok.pdf", b"ok"), _pdf(tmp_path, "partial.pdf", b"partial")

    def first(pdf, **kwargs):
        return {"figure_vlm": {"error": "API down"}} if pdf == partial else {}

    _run(session, [ok, partial], first)
    retried = FakeParse({})
    assert dbx_parsing.retry_incomplete(session, parse=retried, converter=object())["parsed"] == 1
    assert [c[0] for c in retried.calls] == [partial]
    assert session.table[compute_doc_id(partial)]["status"] == "parsed" and session.table[compute_doc_id(partial)]["error"] is None


def test_new_content_under_the_same_path_supersedes_the_old_version(uc, tmp_path):
    session, parse, pdf = FakeSession(), FakeParse(), _pdf(tmp_path, "a.pdf", b"v1")
    old = compute_doc_id(pdf)
    _run(session, [pdf], parse)
    _pdf(tmp_path, "a.pdf", b"v2")
    _run(session, [pdf], parse)
    assert session.table[old]["status"] == "superseded" and session.table[compute_doc_id(pdf)]["status"] == "parsed"


def test_parse_gets_the_volume_output_the_descriptions_and_a_python_path(uc, tmp_path):
    session, parse, pdf = FakeSession(), FakeParse(), _pdf(tmp_path, "a.pdf")
    _run(session, [f"dbfs:{pdf}"], parse)
    assert parse.calls == [(pdf, config.dbx_output_dir(), True)]


def test_summary_separates_conversion_and_descriptions(uc, tmp_path, monkeypatch):
    clock = iter([10.0, 18.0])
    monkeypatch.setattr(dbx_parsing.time, "perf_counter", lambda: next(clock))
    summary = _run(FakeSession(), [_pdf(tmp_path, "a.pdf")], FakeParse())
    assert (summary["pages"], summary["convert_seconds"], summary["describe_seconds"]) == (2, 3.0, 5.0)
    assert summary["convert_seconds_per_page"] == 1.5


def test_parse_pending_parses_the_queue_once(uc, tmp_path):
    session, parse = FakeSession(), FakeParse()
    pdf = _pdf(tmp_path, "a.pdf")
    session.queue = [f"dbfs:{pdf}"]
    assert dbx_parsing.parse_pending(session, parse=parse, converter=object())["parsed"] == 1
    assert dbx_parsing.parse_pending(session, parse=parse, converter=object())["parsed"] == 0
    assert parse.calls == [(pdf, config.dbx_output_dir(), True)]


def test_a_file_changed_after_its_row_is_pending_again(uc):
    session = FakeSession()
    dbx_parsing.parse_pending(session, parse=FakeParse(), converter=object())
    pending = next(s for s in session.statements if "LEFT ANTI JOIN" in s)
    assert "d.updated_at >= q.modificationTime" in pending


class _Chain:
    """Records the calls of the Structured Streaming builders."""

    def __init__(self):
        self.calls = {}
        self.writeStream = self

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.setdefault(name, []).append(args or kwargs)
            return self
        return call


def test_discovery_stops_when_the_new_files_are_done_and_runs_no_python_in_the_stream(uc, monkeypatch):
    checkpoint = "/Volumes/demo/checkpoints/unstructured_docs_docling/parse_queue"
    monkeypatch.setattr(config, "DBX_CHECKPOINT", checkpoint)
    chain = _Chain()
    spark = SimpleNamespace(readStream=chain, sql=lambda *_a, **_k: None)
    dbx_parsing.discover_pdfs(spark)
    assert chain.calls["trigger"] == [{"availableNow": True}] and chain.calls["awaitTermination"]
    assert ("checkpointLocation", checkpoint) in chain.calls["option"]
    assert ("pathGlobFilter", "*.pdf") in chain.calls["option"]
    assert ("cloudFiles.allowOverwrites", "false") in chain.calls["option"]
    assert chain.calls["load"] == [("/Volumes/demo/pdf_bench/pdf_store/pdf",)]
    assert chain.calls["select"] == [("path", "modificationTime")]
    assert chain.calls["toTable"] == [("`demo`.`pdf_bench`.`parse_queue`",)]
    assert "foreachBatch" not in chain.calls  # another Python process with Spark Connect: no configuration, no secret


def test_table_statement_is_created_with_the_configured_name(uc):
    session = FakeSession()
    dbx_parsing.ensure_table(session)
    assert re.search(r"CREATE TABLE IF NOT EXISTS `demo`.`pdf_bench`.`documents`", session.statements[0])


def test_models_are_downloaded_once_and_copied_from_the_local_disk(monkeypatch, tmp_path):
    from docling.utils import model_downloader

    calls = []

    def download(output_dir, **kwargs):
        calls.append(kwargs)
        (output_dir / "layout").mkdir()
        (output_dir / "layout" / "model.bin").write_text("x", encoding="utf-8")

    monkeypatch.setattr(model_downloader, "download_models", download)
    target = tmp_path / "models"
    dbx_parsing.prefetch_models(target)
    dbx_parsing.prefetch_models(target)
    assert len(calls) == 1 and (target / "layout" / "model.bin").exists()
    assert calls[0]["with_code_formula"] is False and calls[0]["with_picture_classifier"] is False
