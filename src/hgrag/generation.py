"""Answer generation: the reconstructed sections become the numbered sources of an LLM prompt.

Two providers, both over plain HTTP with streaming (``requests``; no extra dependency):

* ``anthropic``: Claude Messages API (``ANTHROPIC_API_KEY``), SSE events ``content_block_delta``.
* ``openai``: any OpenAI-compatible ``/chat/completions`` server (Ollama, LM Studio, vLLM...), for a local model.

The key is read from the environment (or passed by the caller) and is never written to disk or to MLflow.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from . import config, tracking
from .ids import pdf_name
from .tracking import track_step

PROVIDERS = ("anthropic", "openai")
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "llama3.1"}
DEFAULT_URLS = {"anthropic": "https://api.anthropic.com", "openai": "http://localhost:11434/v1"}
ANTHROPIC_VERSION = "2023-06-01"

SYSTEM_PROMPT = (
    "You answer questions about a corpus of documents using ONLY the numbered sources provided by the user. "
    "Each source is a section of a document, rebuilt with all its paragraphs, lists, tables and figures.\n"
    "Rules:\n"
    "- Answer the question directly first (yes or no when it is asked), then the supporting details.\n"
    "- Cite the sources you rely on as [1], [2]... right after the statement they support.\n"
    "- Read every source before concluding: the answer is often in a later source, an introduction, a conclusion "
    "or a table. If the sources answer only part of the question, or answer it in other words, give that answer "
    "with its citations and state briefly what is missing.\n"
    "- Say that you cannot answer from the documents only when no source contains anything relevant. Never use "
    "outside knowledge.\n"
    "- Figures are attached as images, each labelled with the number of its source: read them yourself when the "
    "question needs a value or a comparison from a chart or a diagram. When the chart has no written values, give "
    "your best reading, marked as approximate, rather than declining. A text description of a figure marked "
    "'machine-generated, unverified' is unreliable: prefer the image.\n"
    "- Be concise and keep the units and figures exactly as written in the sources."
)


class GenerationError(RuntimeError):
    """Readable error for the UI (missing key, HTTP error, server unreachable)."""


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.environ.get(
        "HGRAG_LLM_PROVIDER", "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"))
    model: str = ""
    base_url: str = ""
    api_key: str | None = None  # never logged nor stored
    max_tokens: int = int(os.environ.get("HGRAG_LLM_MAX_TOKENS", "2000"))
    temperature: float | None = field(default_factory=lambda: (
        float(os.environ["HGRAG_LLM_TEMPERATURE"]) if os.environ.get("HGRAG_LLM_TEMPERATURE") else None))  # omitted if None
    send_images: bool | None = None  # figures as images: default on for Claude, off for local servers
    max_images: int = int(os.environ.get("HGRAG_LLM_MAX_IMAGES", "6"))
    max_image_edge: int = 1568
    max_context_chars: int = int(os.environ.get("HGRAG_LLM_MAX_CONTEXT_CHARS", "24000"))
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise GenerationError(f"unknown provider {self.provider!r} (choose from {PROVIDERS})")
        self.model = self.model or os.environ.get("HGRAG_LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.base_url = (self.base_url or os.environ.get("HGRAG_LLM_BASE_URL") or DEFAULT_URLS[self.provider]).rstrip("/")
        if self.send_images is None:
            default = "1" if self.provider == "anthropic" else "0"
            self.send_images = os.environ.get("HGRAG_LLM_SEND_IMAGES", default).lower() in ("1", "true", "yes")
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY")

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "max_tokens": self.max_tokens, "temperature": self.temperature,
                "send_images": self.send_images, "max_images": self.max_images}


# ---------------------------------------------------------------------------------------- prompt
def build_messages(query: str, sections: list[dict[str, Any]], max_chars: int = 24000,
                   max_images: int = 6) -> dict[str, Any]:
    """Numbered sources, best score first, within a character budget (lowest scores are dropped first).

    ``images``: the figures (PNG on disk) of the kept sections, labelled with their source number.
    """
    blocks, used, dropped, total = [], [], [], 0
    for s in sections:
        name = pdf_name(s.get("source_pdf") or "") or s["doc_id"]
        page = f", page {s['page_no']}" if s.get("page_no") else ""
        block = f"[{len(used) + 1}] {name} > {s['section_path'].split(' > ', 1)[-1]}{page}\n{s['text']}"
        if used and total + len(block) > max_chars:
            dropped.append(s["section_path"])
            continue
        blocks.append(block)
        used.append(s)
        total += len(block)
    images, missing = [], []
    for n, s in enumerate(used, start=1):
        for fig in s.get("figures") or []:
            path = fig.get("png_path")
            if not path or not Path(path).is_file():
                missing.append(fig.get("caption") or fig.get("ref"))
            elif len(images) < max_images:
                images.append({"n": n, "caption": fig.get("caption") or "", "path": str(path)})
    sources_text = "Sources:\n\n" + "\n\n---\n\n".join(blocks)
    question = f"Question: {query}"
    return {"system": SYSTEM_PROMPT, "user": f"{sources_text}\n\n{question}", "sources_text": sources_text,
            "question": question, "sources": used, "dropped": dropped, "images": images, "images_missing": missing}


def _encode_image(path: str, max_edge: int) -> tuple[str, str]:
    """PNG, base64, long edge capped (the API downscales anyway above ~1568 px)."""
    import base64
    import io

    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if max(im.size) > max_edge:
            im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.convert("RGB" if im.mode not in ("RGB", "RGBA", "L") else im.mode).save(buf, format="PNG")
    return "image/png", base64.b64encode(buf.getvalue()).decode("ascii")


def _content_parts(cfg: LLMConfig, msgs: dict[str, Any], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """User message content: each figure (label, then image) first, then the sources and the question."""
    parts: list[dict[str, Any]] = []
    for img in images:
        media, data = _encode_image(img["path"], cfg.max_image_edge)
        label = f"Figure from source [{img['n']}]" + (f": {img['caption']}" if img["caption"] else "")
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
                     if cfg.provider == "anthropic" else
                     {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
    parts.append({"type": "text", "text": f"{msgs['sources_text']}\n\n{msgs['question']}"})
    return parts


# ------------------------------------------------------------------------------------- providers
def _raise_for(resp: requests.Response, cfg: LLMConfig) -> None:
    if resp.status_code < 400:
        return
    try:
        body = resp.json()
        detail = (body.get("error") or {}).get("message") or str(body)
    except Exception:  # noqa: BLE001
        detail = resp.text[:300]
    hint = " Check the API key." if resp.status_code in (401, 403) else ""
    raise GenerationError(f"{cfg.provider} returned HTTP {resp.status_code}: {detail}.{hint}")


def _sse(resp: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    event = ""
    for raw in resp.iter_lines(decode_unicode=False):
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                yield event, json.loads(data)


def _stream_anthropic(cfg: LLMConfig, system: str, user: Any) -> Iterator[dict[str, Any]]:
    if not cfg.api_key:
        raise GenerationError("ANTHROPIC_API_KEY is not set: export it, or paste a key in the Query page.")
    try:
        resp = requests.post(
            f"{cfg.base_url}/v1/messages", stream=True, timeout=cfg.timeout,
            headers={"x-api-key": cfg.api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"},
            json={"model": cfg.model, "max_tokens": cfg.max_tokens, "stream": True, "system": system,
                  "messages": [{"role": "user", "content": user}],
                  **({"temperature": cfg.temperature} if cfg.temperature is not None else {})},
        )
    except requests.RequestException as exc:
        raise GenerationError(f"cannot reach {cfg.base_url}: {exc}") from exc
    _raise_for(resp, cfg)
    usage: dict[str, int] = {}
    for event, data in _sse(resp):
        kind = data.get("type", event)
        if kind == "content_block_delta" and data["delta"].get("type") == "text_delta":
            yield {"text": data["delta"]["text"]}
        elif kind == "message_start":
            usage["input_tokens"] = data["message"].get("usage", {}).get("input_tokens", 0)
        elif kind == "message_delta":
            usage["output_tokens"] = data.get("usage", {}).get("output_tokens", 0)
            usage["stop_reason"] = data["delta"].get("stop_reason")
        elif kind == "error":
            raise GenerationError(f"anthropic stream error: {data['error'].get('message', data['error'])}")
    yield {"usage": usage}


def _stream_openai(cfg: LLMConfig, system: str, user: Any) -> Iterator[dict[str, Any]]:
    headers = {"content-type": "application/json", **({"authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {})}
    try:
        resp = requests.post(
            f"{cfg.base_url}/chat/completions", stream=True, timeout=cfg.timeout, headers=headers,
            json={"model": cfg.model, "max_tokens": cfg.max_tokens, "stream": True,
                  "stream_options": {"include_usage": True},
                  **({"temperature": cfg.temperature} if cfg.temperature is not None else {}),
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
    except requests.RequestException as exc:
        raise GenerationError(f"cannot reach {cfg.base_url}: is the local model server running? ({exc})") from exc
    _raise_for(resp, cfg)
    usage: dict[str, Any] = {}
    stop_reason = None
    for _event, data in _sse(resp):
        if data.get("usage"):  # sent in a last chunk, after finish_reason
            usage = {"input_tokens": data["usage"].get("prompt_tokens", 0),
                     "output_tokens": data["usage"].get("completion_tokens", 0)}
        for choice in data.get("choices", []):
            if (text := (choice.get("delta") or {}).get("content")):
                yield {"text": text}
            stop_reason = choice.get("finish_reason") or stop_reason
    yield {"usage": {**usage, "stop_reason": stop_reason}}


def stream_completion(cfg: LLMConfig, system: str, user: Any) -> Iterator[dict[str, Any]]:
    """Yield ``{"text": ...}`` deltas, then one ``{"usage": {...}}``. ``user`` is a string or a list of content parts."""
    return (_stream_anthropic if cfg.provider == "anthropic" else _stream_openai)(cfg, system, user)


def complete(cfg: LLMConfig, system: str, user: str, *, max_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
    """Non-streaming convenience (used by the judge): full text and usage."""
    saved = cfg.max_tokens
    cfg.max_tokens = max_tokens or saved
    try:
        parts, usage = [], {}
        for d in stream_completion(cfg, system, user):
            if "text" in d:
                parts.append(d["text"])
            else:
                usage = d["usage"]
        return "".join(parts), usage
    finally:
        cfg.max_tokens = saved


# ------------------------------------------------------------------------------------ the step
def answer_stream(
    query: str, sections: list[dict[str, Any]], cfg: LLMConfig | None = None, *, run_key: str | None = None,
    info: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream the answer text. ``info`` (optional dict) is filled with sources, usage and timings.

    Tracked as the ``generate`` step of the retrieval execution in MLflow (prompt and answer as artifacts).
    """
    cfg = cfg or LLMConfig()
    msgs = build_messages(query, sections, cfg.max_context_chars, cfg.max_images)
    images = msgs["images"] if cfg.send_images else []
    info = info if info is not None else {}
    info.update(sources=[{"n": i + 1, "section_path": s["section_path"], "doc_id": s["doc_id"], "score": s["score"]}
                         for i, s in enumerate(msgs["sources"])], dropped=msgs["dropped"], model=cfg.model,
                provider=cfg.provider, prompt_chars=len(msgs["system"]) + len(msgs["user"]),
                n_images=len(images), images=[{"n": i["n"], "caption": i["caption"]} for i in images],
                images_missing=msgs["images_missing"])
    with track_step(tracking.RETRIEVAL, "generate", run_key, **cfg.describe(), n_sources=len(msgs["sources"]),
                    n_dropped=len(msgs["dropped"]), n_images=len(images)) as tr:
        t0 = time.perf_counter()
        parts: list[str] = []
        first: float | None = None
        usage: dict[str, Any] = {}
        content = _content_parts(cfg, msgs, images) if images else msgs["user"]
        for delta in stream_completion(cfg, msgs["system"], content):
            if "text" in delta:
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(delta["text"])
                yield delta["text"]
            else:
                usage = delta["usage"]
        answer = "".join(parts)
        info.update(answer=answer, usage=usage, first_token_s=first, seconds=time.perf_counter() - t0)
        tr.metrics(n_images=len(images), prompt_chars=info["prompt_chars"], answer_chars=len(answer), first_token_s=first or 0.0,
                   input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0))
        tr.tags(stop_reason=usage.get("stop_reason"))
        tr.text(f"SYSTEM\n{msgs['system']}\n\nUSER\n{msgs['user']}\n\nIMAGES\n"
                + "\n".join(f"[{i['n']}] {i['caption']} ({Path(i['path']).name})" for i in images), "prompt.txt")
        tr.text(answer, "answer.txt")
