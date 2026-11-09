"""Databricks indexing step with a fake Spark session and a fake AI Search client: nothing here runs on Databricks."""

import json
from types import SimpleNamespace

import pytest

from hgrag import config
from hgrag.dbx import indexing


@pytest.fixture
def uc(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_VOLUME", "pdf_store")
    monkeypatch.setattr(config, "DBX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DBX_SEARCH_ENDPOINT", "hgrag")


class FakeSession:
    """``pending``: rows of the anti join; the merged rows are kept in ``merged``."""

    def __init__(self, pending=()):
        self.pending = [SimpleNamespace(doc_id=d, path=p) for d, p in pending]
        self.merged, self.statements, self.view = [], [], []

    def createDataFrame(self, rows, _schema):
        return SimpleNamespace(createOrReplaceTempView=lambda _name: setattr(self, "view", rows))

    def sql(self, statement, args=None):
        self.statements.append((statement, args))
        if statement.startswith("MERGE"):
            self.merged.append((args["doc_id"], list(self.view)))
        rows = self.pending if "LEFT ANTI JOIN" in statement else []
        return SimpleNamespace(collect=lambda: rows)


def _chunk(index, **extra):
    return {"chunk_index": index, "text": f"text {index}", "embed_text": f"heading\ntext {index}", "kinds": ["text"],
            "headings": ["heading"], "element_refs": ["#/texts/1"], "page_no": 1, "section_ref": "#/s",
            "section_title": "heading", "section_path": "doc > heading", **extra}


def test_chunks_of_a_document_are_merged_with_stable_ids(uc, tmp_path):
    session = FakeSession([("d1", "/Volumes/x/a.pdf")])
    calls = []

    def build(doc_dir, chunker):
        calls.append(doc_dir)
        return [_chunk(0), _chunk(1, table_description="desc")]

    summary = indexing.chunk_pending(session, chunker=object(), build=build)
    assert (summary["documents"], summary["chunks"], summary["failed"]) == (1, 2, {})
    assert calls == [tmp_path / "d1"]
    doc_id, rows = session.merged[0]
    assert doc_id == "d1" and [r[0] for r in rows] == ["d1:0", "d1:1"]
    assert rows[0][3] == "/Volumes/x/a.pdf" and json.loads(rows[0][8]) == ["text"] and rows[1][13] == "desc"
    assert all(len(r) == len(indexing._COLUMNS) for r in rows)


def test_progress_of_the_chunks_counts_the_documents(uc, capsys):
    indexing.chunk_pending(FakeSession([("d1", "/Volumes/x/a.pdf"), ("d2", "/Volumes/x/b.pdf")]), chunker=object(),
                           build=lambda *_: [_chunk(0)])
    out = capsys.readouterr().out
    assert "chunks 1/2 a.pdf: started" in out and "chunks 2/2 b.pdf: 1 chunk(s), checks done" in out


def test_a_merge_removes_the_chunks_the_document_no_longer_has(uc):
    session = FakeSession([("d1", "a.pdf")])
    indexing.chunk_pending(session, chunker=object(), build=lambda *_: [])
    merge = next(s for s, _ in session.statements if s.startswith("MERGE"))
    assert "WHEN NOT MATCHED BY SOURCE AND t.doc_id = :doc_id THEN DELETE" in merge
    assert session.merged == [("d1", [])]


def test_a_failing_document_does_not_stop_the_others(uc):
    session = FakeSession([("bad", "bad.pdf"), ("good", "good.pdf")])

    def build(doc_dir, _chunker):
        if doc_dir.name == "bad":
            raise ValueError("no document.json")
        return [_chunk(0)]

    summary = indexing.chunk_pending(session, chunker=object(), build=build)
    assert summary["documents"] == 1 and "no document.json" in summary["failed"]["bad.pdf"]
    assert [d for d, _ in session.merged] == ["good"]


def test_pending_keeps_usable_documents_and_superseded_chunks_are_deleted(uc):
    session = FakeSession()
    indexing.chunk_pending(session, chunker=object(), build=lambda *_: [])
    statements = [s for s, _ in session.statements]
    pending = next(s for s in statements if "LEFT ANTI JOIN" in s)
    assert "c.indexed_at >= d.updated_at" in pending and "d.status IN ('parsed', 'partial')" in pending
    assert any(s.startswith("DELETE") and "status = 'superseded'" in s for s in statements)


def test_a_run_with_failed_documents_fails_at_the_end(uc):
    class Session:
        def sql(self, statement):
            assert "`demo`.`pdf_bench`.`documents`" in statement and "status = 'failed'" in statement
            return SimpleNamespace(collect=lambda: [SimpleNamespace(path="a.pdf")])

    with pytest.raises(RuntimeError, match=r"2 document\(s\) failed.*a\.pdf.*b\.pdf"):
        indexing.raise_on_failures(Session(), {"failed": {"b.pdf": "no document.json"}})


def test_a_run_without_failure_ends_normally(uc):
    session = SimpleNamespace(sql=lambda _statement: SimpleNamespace(collect=lambda: []))
    indexing.raise_on_failures(session, {"failed": {}})


def test_chunk_table_has_the_change_data_feed(uc):
    session = FakeSession()
    indexing.ensure_table(session)
    assert "delta.enableChangeDataFeed = true" in session.statements[0][0]


class FakeIndex:
    def __init__(self):
        self.synced = 0
        self.waited = []

    def sync(self):
        self.synced += 1

    def wait_until_ready(self, **kwargs):
        self.waited.append(kwargs)

    def describe(self):
        return {"status": {"detailed_state": "ONLINE_NO_PENDING_UPDATE", "indexed_row_count": 7}}

    def similarity_search(self, **kwargs):
        self.query = kwargs
        return {"manifest": {"columns": [{"name": "chunk_id"}, {"name": "text"}, {"name": "score"}]},
                "result": {"row_count": 1, "data_array": [["d1:0", "hello", 0.5]]}}


class FakeClient:
    def __init__(self, endpoint=True, index=False):
        self.endpoint, self.has_index = endpoint, index
        self.index = FakeIndex()
        self.created_endpoint, self.created_index = None, None

    def endpoint_exists(self, _name):
        return self.endpoint

    def create_endpoint_and_wait(self, name, **kwargs):
        self.created_endpoint = (name, kwargs)

    def index_exists(self, _endpoint, _name):
        return self.has_index

    def get_index(self, _endpoint, _name):
        return self.index

    def create_delta_sync_index(self, **kwargs):
        self.created_index = kwargs
        return self.index


def test_first_sync_creates_the_endpoint_and_a_managed_embedding_index(uc):
    client = FakeClient(endpoint=False)
    summary = indexing.sync_index(client)
    assert client.created_endpoint[0] == "hgrag"
    spec = client.created_index
    assert spec["index_name"] == "demo.pdf_bench.chunks_index" and spec["source_table_name"] == "demo.pdf_bench.chunks"
    assert spec["primary_key"] == "chunk_id" and spec["pipeline_type"] == "TRIGGERED"
    assert spec["embedding_source_column"] == "embed_text"
    assert spec["embedding_model_endpoint_name"] == "databricks-gte-large-en"
    assert "indexed_at" not in spec["columns_to_sync"]
    assert client.index.synced == 0 and client.index.waited[0]["wait_for_updates"] is True
    assert summary["indexed_rows"] == 7


def test_next_sync_reuses_the_endpoint_and_the_index(uc):
    client = FakeClient(index=True)
    indexing.sync_index(client)
    assert client.created_endpoint is None and client.created_index is None and client.index.synced == 1


def test_search_returns_named_columns_and_is_hybrid(uc):
    client = FakeClient(index=True)
    assert indexing.search("hello", 3, client=client) == [{"chunk_id": "d1:0", "text": "hello", "score": 0.5}]
    assert client.index.query["query_type"] == "HYBRID" and client.index.query["num_results"] == 3
    assert client.index.query["reranker"] is None


def test_endpoint_name_is_required(uc, monkeypatch):
    monkeypatch.setattr(config, "DBX_SEARCH_ENDPOINT", "")
    with pytest.raises(ValueError, match="HGRAG_DBX_SEARCH_ENDPOINT"):
        indexing.sync_index(FakeClient())
