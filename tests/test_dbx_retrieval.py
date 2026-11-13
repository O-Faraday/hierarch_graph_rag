"""Databricks question flow with a fake AI Search client and a fake LLM: nothing here runs on Databricks."""

import json
from types import SimpleNamespace

import pytest
from databricks.ai_search.exceptions import BadRequest, PermissionDenied

from hgrag import config, generation
from hgrag.dbx import indexing, retrieval
from hgrag.rerank import RerankError


def _tree(doc_id):
    return {"ref": "#/root", "level": 0, "title": doc_id, "path": doc_id, "elements": [], "children": [
        {"ref": "#/s1", "level": 1, "title": "Intro", "path": f"{doc_id} > Intro", "page_no": 1, "children": [],
         "elements": [{"kind": "paragraph", "ref": "#/t1", "text": "The matching chunk.", "extra": {}},
                      {"kind": "paragraph", "ref": "#/t2", "text": "A paragraph no chunk matched.", "extra": {}}]},
        {"ref": "#/s2", "level": 1, "title": "Next", "path": f"{doc_id} > Next", "page_no": 2, "children": [],
         "elements": [{"kind": "paragraph", "ref": "#/t3", "text": "Second section.", "extra": {}}]}]}


def _row(doc_id, section, index, score):
    return [f"{doc_id}:{index}", doc_id, index, f"/Volumes/x/{doc_id}.pdf", 1, section, "title", "path",
            '["text"]', '["Intro"]', '["#/t1"]', "The matching chunk.", "embed", None, None, score]


class Client:
    def __init__(self, rows):
        self.rows = rows
        self.query = None

    def get_index(self, _endpoint, _name):
        return SimpleNamespace(similarity_search=self.similarity_search)

    def similarity_search(self, **kwargs):
        self.query = kwargs
        columns = [{"name": c} for c in [*indexing._COLUMNS, "score"]]
        return {"manifest": {"columns": columns}, "result": {"data_array": self.rows}}


@pytest.fixture
def uc(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_VOLUME", "pdf_store")
    monkeypatch.setattr(config, "DBX_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DBX_SEARCH_ENDPOINT", "hgrag")
    for doc_id in ("d1", "d2"):
        (tmp_path / doc_id).mkdir()
        (tmp_path / doc_id / "hierarchy.json").write_text(json.dumps(_tree(doc_id)), encoding="utf-8")


@pytest.fixture
def llm(monkeypatch):
    prompts = []

    def stream(_cfg, _system, user):
        prompts.append(user)
        yield {"text": "Answer "}
        yield {"text": "[1]"}
        yield {"usage": {"input_tokens": 10, "output_tokens": 2, "stop_reason": "end_turn"}}

    monkeypatch.setattr(generation, "stream_completion", stream)
    return prompts


CFG = generation.LLMConfig(provider="anthropic", api_key="test", send_images=False)


class FakeReranker:
    """Scores the texts by the given order: the last chunk of AI Search comes first."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, batch_size=0):
        self.pairs = list(pairs)
        return [float(i) for i in range(len(pairs))]


NOT_ENABLED = BadRequest("[INVALID_PARAMETER_VALUE] Reranking is not yet enabled for this workspace.", status_code=400)


class RefusingClient(Client):
    """Refuses a query with the AI Search reranker, like a workspace without it (or with ``error``)."""

    def __init__(self, rows, error=NOT_ENABLED):
        super().__init__(rows)
        self.error = error

    def similarity_search(self, **kwargs):
        if kwargs.get("reranker") is not None:
            raise self.error
        return super().similarity_search(**kwargs)


def test_the_ai_search_reranker_comes_first_and_keeps_its_order(uc, capsys):
    rows = [_row("d1", "#/s2", 0, 0.1), _row("d1", "#/s1", 1, 0.9)]
    client, external = Client(rows), FakeReranker()
    sections = retrieval.search_sections("q", 3, rerank=True, reranker=external, client=client)
    assert client.query["reranker"].columns_to_rerank[-1] == "text" and client.query["num_results"] == 3
    assert external.pairs == [] and [s["section_ref"] for s in sections] == ["#/s2", "#/s1"]
    assert "'hgrag_reranker': 'ai_search'" in capsys.readouterr().out


def test_a_workspace_without_it_falls_back_on_the_external_reranker(uc, capsys):
    rows = [_row("d1", "#/s1", 0, 0.9), _row("d1", "#/s1", 1, 0.8), _row("d1", "#/s2", 2, 0.7)]
    client, reranker = RefusingClient(rows), FakeReranker()
    sections = retrieval.search_sections("q", 2, rerank=True, reranker=reranker, client=client)
    assert client.query["num_results"] == config.RERANK_CANDIDATES and client.query["reranker"] is None
    assert len(reranker.pairs) == 3 and reranker.pairs[0][0] == "q"
    assert [s["section_ref"] for s in sections] == ["#/s2", "#/s1"]
    assert [h["chunk_index"] for s in sections for h in s["hits"]] == [2, 1]
    assert sections[0]["score"] == 2.0 and sections[0]["hits"][0]["vector_score"] == 0.7
    assert "'hgrag_reranker': 'huggingface'" in capsys.readouterr().out


@pytest.mark.parametrize("error", [BadRequest("columns_to_rerank: unknown column", status_code=400),
                                   BadRequest("Index is not enabled", status_code=400),
                                   PermissionDenied("Reranking is not enabled for this user", status_code=403),
                                   ValueError("Reranking is not enabled")])
def test_any_other_failure_stays_visible_and_sends_nothing_outside(uc, error):
    reranker = FakeReranker()
    with pytest.raises(type(error)):
        retrieval.search_sections("q", 3, rerank=True, reranker=reranker, client=RefusingClient([], error))
    assert reranker.pairs == []


def test_without_reranking_only_k_chunks_are_fetched_and_no_reranker_is_called(uc, monkeypatch, capsys):
    monkeypatch.setattr(retrieval, "HFRouterReranker", lambda: pytest.fail("reranker called"))
    client = Client([])
    retrieval.search_sections("q", 3, rerank=False, client=client)
    assert client.query["num_results"] == 3 and client.query["reranker"] is None
    assert "'hgrag_reranker': 'none'" in capsys.readouterr().out


def test_a_failure_of_the_external_reranker_is_not_hidden(uc):
    class Failing:
        def predict(self, _pairs, batch_size=0):
            raise RerankError("HTTP 402")

    with pytest.raises(RerankError, match="402"):
        retrieval.search_sections("q", 3, rerank=True, reranker=Failing(),
                                  client=RefusingClient([_row("d1", "#/s1", 0, 0.5)]))


def test_grouping_keeps_the_order_of_the_service_without_reranking(uc):
    rows = [_row("d1", "#/s2", 0, 0.1), _row("d1", "#/s1", 1, 0.9), _row("d1", "#/s2", 2, 0.5)]
    sections = retrieval.search_sections("q", 3, rerank=False, client=Client(rows))
    assert [s["section_ref"] for s in sections] == ["#/s2", "#/s1"]
    assert len(sections[0]["hits"]) == 2 and sections[0]["hits"][0]["kinds"] == ["text"]
    assert sections[0]["hits"][0]["vector_score"] == 0.1 and sections[0]["source_pdf"] == "/Volumes/x/d1.pdf"


def test_the_llm_gets_the_whole_section_and_the_answer_cites_it(uc, llm):
    result = retrieval.ask("q", client=Client([_row("d1", "#/s1", 0, 0.5)]), cfg=CFG, judges=False)
    assert result["answer"] == "Answer [1]" and result["usage"]["output_tokens"] == 2
    assert result["sources"][0]["section_path"] == "d1 > Intro"
    assert "A paragraph no chunk matched." in llm[0] and "Second section." not in llm[0]


def test_sections_of_several_documents_are_rebuilt_from_their_own_tree(uc, llm):
    rows = [_row("d1", "#/s1", 0, 0.5), _row("d2", "#/s2", 0, 0.4)]
    result = retrieval.ask("q", client=Client(rows), cfg=CFG, judges=False)
    assert [s["section_path"] for s in result["sources"]] == ["d1 > Intro", "d2 > Next"]


def test_no_chunk_means_no_llm_call(uc, llm):
    result = retrieval.ask("q", client=Client([]), cfg=CFG, judges=False)
    assert result["answer"] == "" and result["sources"] == [] and llm == []


def test_an_empty_question_is_refused(uc):
    with pytest.raises(ValueError):
        retrieval.search_sections("  ", client=Client([]))


def test_k_sections_reads_the_candidates_and_keeps_the_first_sections(uc, monkeypatch):
    """As on the local backend: N sections from RERANK_CANDIDATES ranked chunks, not the sections of k chunks."""
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 40)
    rows = [_row("d1", "#/s1", 0, 0.9), _row("d1", "#/s2", 1, 0.8), _row("d2", "#/s1", 2, 0.7),
            _row("d1", "#/s1", 3, 0.6)]
    client = Client(rows)
    sections = retrieval.search_sections("q", 2, rerank=True, client=client, k_sections=2)
    assert client.query["num_results"] == 40
    assert [(s["doc_id"], s["section_ref"]) for s in sections] == [("d1", "#/s1"), ("d1", "#/s2")]
    assert len(sections[0]["hits"]) == 2  # the later chunk of a kept section is kept
    monkeypatch.setattr(config, "K_SECTIONS", 1)  # the setting is the default
    assert len(retrieval.search_sections("q", 2, rerank=False, client=Client(rows))) == 1
