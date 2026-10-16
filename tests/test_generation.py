"""Answer generation: prompt building, both streaming providers (HTTP faked), tracking and errors."""

from __future__ import annotations

import json

import mlflow
import pytest

from hgrag import generation, tracking
from hgrag.generation import GenerationError, LLMConfig

SECTIONS = [
    {"doc_id": "d1", "section_ref": "#/1", "section_path": "rep > 3 Branch network > 3.1 New branches", "page_no": 1,
     "score": 0.8, "source_pdf": "/x/aurora.pdf",
     "text": "### 3.1 New branches\n\nThe bank opened three new branches in Alsace.\n\n[Figure] Figure 1\n"
             "Description (machine-generated, unverified): about 200 each"},
    {"doc_id": "d1", "section_ref": "#/2", "section_path": "rep > 3.2 Closures", "page_no": 2, "score": 0.3,
     "source_pdf": "/x/aurora.pdf", "text": "### 3.2 Closures\n\n" + "Two branches were merged. " * 40},
]


class FakeResponse:
    def __init__(self, lines, status=200, body=None):
        self.status_code, self._lines, self._body = status, lines, body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=False):
        return iter(l.encode("utf-8") for l in self._lines)


def _anthropic_stream(parts, out_tokens=7):
    ev = [("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 321}}}),
          ("ping", {"type": "ping"})]
    ev += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "text_delta", "text": p}}) for p in parts]
    ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                              "usage": {"output_tokens": out_tokens}}), ("message_stop", {"type": "message_stop"})]
    lines = []
    for name, data in ev:
        lines += [f"event: {name}", f"data: {json.dumps(data)}", ""]
    return lines


def test_prompt_numbers_sources_and_respects_budget():
    m = generation.build_messages("How many branches?", SECTIONS, max_chars=10000)
    assert "[1] aurora.pdf > 3 Branch network > 3.1 New branches, page 1" in m["user"]
    assert "[2] aurora.pdf > 3.2 Closures, page 2" in m["user"] and m["user"].endswith("Question: How many branches?")
    assert "unverified" in m["system"] and "[1]" in m["system"] and "cannot answer" in m["system"]
    tight = generation.build_messages("q", SECTIONS, max_chars=len(SECTIONS[0]["text"]) + 80)
    assert [s["section_ref"] for s in tight["sources"]] == ["#/1"] and tight["dropped"] == [SECTIONS[1]["section_path"]]
    first_only = generation.build_messages("q", SECTIONS, max_chars=5)
    assert len(first_only["sources"]) == 1  # the best section is always kept


def test_anthropic_streaming_request_and_parsing(monkeypatch):
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return FakeResponse(_anthropic_stream(["Three ", "branches [1]."]))

    monkeypatch.setattr(generation.requests, "post", post)
    cfg = LLMConfig(provider="anthropic", api_key="sk-test")
    info: dict = {}
    text = "".join(generation.answer_stream("q", SECTIONS, cfg, info=info))
    assert text == "Three branches [1]."
    assert seen["url"] == "https://api.anthropic.com/v1/messages" and seen["stream"] is True
    assert seen["headers"]["x-api-key"] == "sk-test" and seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["json"]["stream"] and seen["json"]["system"] and seen["json"]["messages"][0]["role"] == "user"
    assert info["usage"] == {"input_tokens": 321, "output_tokens": 7, "stop_reason": "end_turn"}
    assert info["answer"] == text and len(info["sources"]) == 2 and info["first_token_s"] is not None


def test_openai_compatible_streaming(monkeypatch):
    chunks = [{"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
              {"choices": [{"delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 2}}]
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    seen = {}
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: seen.update(url=url, **kw) or FakeResponse(lines))
    info: dict = {}
    text = "".join(generation.answer_stream("q", SECTIONS, LLMConfig(provider="openai", model="m"), info=info))
    assert text == "Hello" and seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in seen["headers"]  # local servers need no key
    assert info["usage"]["output_tokens"] == 2 and info["usage"]["stop_reason"] == "stop"


def test_errors_are_readable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY is not set"):
        list(generation.answer_stream("q", SECTIONS, LLMConfig(provider="anthropic")))
    monkeypatch.setattr(generation.requests, "post",
                        lambda *a, **k: FakeResponse([], 401, {"error": {"message": "invalid x-api-key"}}))
    with pytest.raises(GenerationError, match="HTTP 401: invalid x-api-key.*Check the API key"):
        list(generation.answer_stream("q", SECTIONS, LLMConfig(provider="anthropic", api_key="bad")))
    err = ['event: error', 'data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}']
    monkeypatch.setattr(generation.requests, "post", lambda *a, **k: FakeResponse(err))
    with pytest.raises(GenerationError, match="Overloaded"):
        list(generation.answer_stream("q", SECTIONS, LLMConfig(provider="anthropic", api_key="k")))

    def boom(*a, **k):
        raise generation.requests.ConnectionError("refused")

    monkeypatch.setattr(generation.requests, "post", boom)
    with pytest.raises(GenerationError, match="local model server running"):
        list(generation.answer_stream("q", SECTIONS, LLMConfig(provider="openai")))
    with pytest.raises(GenerationError, match="unknown provider"):
        LLMConfig(provider="x")


def test_generate_step_is_tracked_without_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "ml")
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'ml' / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(generation.requests, "post", lambda *a, **k: FakeResponse(_anthropic_stream(["ok"])))
    key = tracking.new_run_key()
    "".join(generation.answer_stream("q", SECTIONS, LLMConfig(provider="anthropic", api_key="sk-secret"), run_key=key))
    exp = mlflow.get_experiment_by_name(tracking.RETRIEVAL)
    run = next(r for r in mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list",
                                             filter_string="tags.`hgrag.step` = 'generate'"))
    assert run.data.metrics["output_tokens"] == 7 and run.data.metrics["input_tokens"] == 321
    assert run.data.params["model"] == "claude-haiku-5-5" and "sk-secret" not in json.dumps(run.data.params)
    assert {a.path for a in mlflow.artifacts.list_artifacts(run_id=run.info.run_id)} == {"prompt.txt", "answer.txt"}


# ----------------------------------------------------------------------------- figures as images
@pytest.fixture()
def with_png(tmp_path):
    from PIL import Image

    big, small = tmp_path / "big.png", tmp_path / "small.png"
    Image.new("RGB", (3000, 1000), "white").save(big)
    Image.new("RGB", (40, 30), "red").save(small)
    sections = [dict(SECTIONS[0], figures=[{"ref": "#/p/0", "caption": "Figure 1. Branches per year", "png_path": str(small)},
                                           {"ref": "#/p/1", "caption": "gone", "png_path": str(tmp_path / "nope.png")}]),
                dict(SECTIONS[1], figures=[{"ref": "#/p/2", "caption": "", "png_path": str(big)}])]
    return sections, big


def test_figures_are_sent_as_images_before_the_text(monkeypatch, with_png):
    sections, _ = with_png
    seen = {}
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: seen.update(kw) or FakeResponse(_anthropic_stream(["ok"])))
    info: dict = {}
    "".join(generation.answer_stream("q", sections, LLMConfig(provider="anthropic", api_key="k"), info=info))
    content = seen["json"]["messages"][0]["content"]
    kinds = [c["type"] for c in content]
    assert kinds == ["text", "image", "text", "image", "text"]  # label, image, label, image, sources + question
    assert content[0]["text"] == "Figure from source [1]: Figure 1. Branches per year"
    assert content[2]["text"] == "Figure from source [2]"
    assert content[1]["source"]["type"] == "base64" and content[1]["source"]["media_type"] == "image/png"
    assert content[-1]["text"].startswith("Sources:") and content[-1]["text"].endswith("Question: q")
    assert info["n_images"] == 2 and info["images_missing"] == ["gone"]
    assert "temperature" not in seen["json"]  # not sent unless configured


def test_images_can_be_disabled_capped_and_downscaled(monkeypatch, with_png):
    import base64
    import io

    from PIL import Image

    sections, _ = with_png
    seen = {}
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: seen.update(kw) or FakeResponse(_anthropic_stream(["ok"])))
    "".join(generation.answer_stream("q", sections, LLMConfig(provider="anthropic", api_key="k", send_images=False)))
    assert isinstance(seen["json"]["messages"][0]["content"], str)  # plain text request

    "".join(generation.answer_stream("q", sections, LLMConfig(provider="anthropic", api_key="k", max_images=1)))
    assert sum(c["type"] == "image" for c in seen["json"]["messages"][0]["content"]) == 1

    "".join(generation.answer_stream("q", sections, LLMConfig(provider="anthropic", api_key="k", temperature=0.2)))
    assert seen["json"]["temperature"] == 0.2
    media, data = generation._encode_image(sections[1]["figures"][0]["png_path"], 1568)
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (1568, 523)


def test_openai_compatible_gets_image_url_only_when_enabled(monkeypatch, with_png):
    sections, _ = with_png
    seen = {}
    lines = ['data: {"choices": [{"delta": {"content": "x"}}]}', "data: [DONE]"]
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: seen.update(kw) or FakeResponse(lines))
    "".join(generation.answer_stream("q", sections, LLMConfig(provider="openai")))
    assert isinstance(seen["json"]["messages"][1]["content"], str)  # local text models: off by default
    "".join(generation.answer_stream("q", sections, LLMConfig(provider="openai", send_images=True)))
    parts = seen["json"]["messages"][1]["content"]
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
