"""New attempts of a model call on an overloaded provider, with a fake ``requests.post``: no network."""

import pytest
import requests

from hgrag import config, generation


class FakeResponse:
    def __init__(self, status, headers=None, lines=()):
        self.status_code, self.headers, self.lines, self.closed = status, headers or {}, lines, False

    def json(self):
        return {"error": {"message": "overloaded"}}

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)

    def close(self):
        self.closed = True


OK_LINES = [b'data: {"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]}', b"data: [DONE]"]


@pytest.fixture
def calls(monkeypatch):
    """Answers given in turn by the fake provider (a response or an exception); the waits are recorded."""
    state = {"answers": [], "posts": 0, "waits": []}

    def post(url, **kwargs):
        state["posts"] += 1
        answer = state["answers"].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(generation.requests, "post", post)
    monkeypatch.setattr(generation.time, "sleep", state["waits"].append)
    monkeypatch.setattr(config, "LLM_RETRIES", 3)
    monkeypatch.setattr(config, "LLM_RETRY_WAIT", 2.0)
    return state


def _cfg():
    return generation.LLMConfig(provider="openai", model="m", base_url="http://x", api_key="k")


def test_a_503_is_tried_again_and_the_answer_comes_once(calls):
    first = FakeResponse(503)
    calls["answers"] = [first, FakeResponse(503, {"retry-after": "7"}), FakeResponse(200, lines=OK_LINES)]
    text, _usage = generation.complete(_cfg(), "system", "question")
    assert text == "hello" and calls["posts"] == 3
    assert calls["waits"] == [2.0, 7.0]  # doubling waits, unless the provider says when to come back
    assert first.closed


def test_network_errors_are_tried_again(calls):
    calls["answers"] = [requests.ConnectionError("reset"), requests.Timeout("slow"), FakeResponse(200, lines=OK_LINES)]
    assert generation.complete(_cfg(), "system", "question")[0] == "hello"


def test_the_last_failure_is_reported_after_the_retries(calls):
    calls["answers"] = [FakeResponse(503)] * 4
    with pytest.raises(generation.GenerationError, match="HTTP 503"):
        generation.complete(_cfg(), "system", "question")
    assert calls["posts"] == 4 and calls["waits"] == [2.0, 4.0, 8.0]


def test_a_client_error_is_not_tried_again(calls):
    calls["answers"] = [FakeResponse(401)]
    with pytest.raises(generation.GenerationError, match="API key"):
        generation.complete(_cfg(), "system", "question")
    assert calls["posts"] == 1 and calls["waits"] == []


def test_no_retry_when_disabled(calls, monkeypatch):
    monkeypatch.setattr(config, "LLM_RETRIES", 0)
    calls["answers"] = [requests.ConnectionError("down")]
    with pytest.raises(generation.GenerationError, match="cannot reach"):
        generation.complete(_cfg(), "system", "question")
