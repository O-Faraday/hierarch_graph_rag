"""Deleting the index to stop the billing of AI Search, with a fake client: nothing here runs on Databricks."""

import json

import pytest

from hgrag import config
from hgrag.dbx import indexing

OURS = "demo.pdf_bench.chunks_index"


@pytest.fixture(autouse=True)
def uc(monkeypatch):
    monkeypatch.setattr(config, "DBX_CATALOG", "demo")
    monkeypatch.setattr(config, "DBX_SCHEMA", "pdf_bench")
    monkeypatch.setattr(config, "DBX_SEARCH_ENDPOINT", "shared")


class FakeClient:
    def __init__(self, indexes=(), endpoint=True, next_page_token=None):
        self.indexes, self.endpoint, self.next_page_token = list(indexes), endpoint, next_page_token
        self.deleted = []

    def endpoint_exists(self, _name):
        return self.endpoint

    def index_exists(self, _endpoint, name):
        return name in self.indexes

    def delete_index(self, endpoint, name):
        self.deleted.append((endpoint, name))
        self.indexes.remove(name)

    def list_indexes(self, _endpoint):
        listing = {"vector_indexes": [{"name": name} for name in self.indexes]} if self.indexes else {}
        return {**listing, "next_page_token": self.next_page_token} if self.next_page_token else listing

    def delete_endpoint(self, _name):
        raise AssertionError("the endpoint may host the indexes of others")


def test_deletes_only_the_index_of_the_kit(capsys):
    client = FakeClient([OURS])
    summary = indexing.stop_search(client)
    assert client.deleted == [("shared", OURS)]
    assert summary["deleted"] is True and summary["billing_stops"] is True
    assert json.loads(capsys.readouterr().out)["hgrag_stop_search"] == summary


def test_other_indexes_are_kept_and_reported():
    client = FakeClient([OURS, "other.team.index"])
    summary = indexing.stop_search(client)
    assert client.indexes == ["other.team.index"]
    assert summary["other_indexes"] == ["other.team.index"]
    assert summary["billing_stops"] is False


def test_a_second_page_of_indexes_does_not_promise_the_end_of_billing():
    assert indexing.stop_search(FakeClient([OURS], next_page_token="more"))["billing_stops"] is False


@pytest.mark.parametrize("client", [FakeClient(), FakeClient(endpoint=False)], ids=["no index", "no endpoint"])
def test_nothing_to_delete_is_not_an_error(client):
    summary = indexing.stop_search(client)
    assert summary["deleted"] is False and client.deleted == []
