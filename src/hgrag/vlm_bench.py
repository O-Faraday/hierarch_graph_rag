"""Compare vision-language models on the control figures (description quality).

Each model describes each figure; a judge (Claude) checks the description against a checklist of facts and
lists what was invented. Models are called through the Hugging Face router (OpenAI-compatible,
``HF_TOKEN``) or the Anthropic API: nothing runs locally.

    uv run hgrag vlm-bench                       # default candidates
    uv run hgrag vlm-bench --models hf:google/gemma-3-12b-it anthropic:claude-haiku-5-5
    uv run hgrag vlm-bench --list-models         # image-capable models served by the HF router
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from . import config, evaluation, generation
from .generation import GenerationError, LLMConfig
from .ids import compute_doc_id

HF_URL = "https://router.huggingface.co/v1"
# Served by the router at the time of writing (granite-vision, SmolVLM and Qwen2.5-VL-3B are not).
DEFAULT_MODELS = [
    "hf:google/gemma-3-4b-it",
    "hf:google/gemma-3-12b-it",
    "hf:Qwen/Qwen3.5-9B",
    "hf:Qwen/Qwen3-VL-30B-A3B-Instruct",
    "hf:zai-org/GLM-4.6V-Flash",
    "anthropic:claude-haiku-5-5",
]
DESCRIBE_SYSTEM = "You describe figures extracted from PDF documents so that they can be indexed and searched."
DESCRIBE_PROMPT = (
    "Describe this figure. Give its type, its title, the axes (labels and units) and every data value you can "
    "read (approximate when needed), then the overall trend. Only report what is visible: do not guess. "
    "At most 120 words."
)
JUDGE_SYSTEM = """You grade a description of a figure. You receive the FULL TRUTH about the figure, a CHECKLIST \
of facts and the CANDIDATE description. Do not look at style.

For each checklist fact, answer true only if the candidate states it correctly (within the tolerance given), \
false if it is missing or wrong.
List in "invented" every claim of the candidate that contradicts the full truth or is not in it (wrong value, \
wrong axis, series or trend that does not exist). Rounding within the tolerance is not an invention.
Score 1-5: 5 = every fact correct and nothing invented; 3 = about half the facts or one invention; \
1 = mostly wrong or unrelated.

Reply with ONLY a JSON object: {"facts": {"<fact id>": true|false, ...}, "invented": ["..."], "score": <1-5>}"""


class BenchError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- data
def load_truth(pdf_dir: str | Path = config.CONTROL_PDF_DIR, parsed_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Figures of the control PDFs with their checklist and the path of the PNG (frozen parsing)."""
    pdf_dir = Path(pdf_dir)
    parsed_dir = Path(parsed_dir or pdf_dir / "parsed")
    rows = json.loads((pdf_dir / "figure_truth.json").read_text(encoding="utf-8"))
    for r in rows:
        doc_id = compute_doc_id(pdf_dir / r["pdf"])
        r["doc_id"] = doc_id
        r["png_path"] = str(parsed_dir / doc_id / "figures" / f"figure-{r['figure_index']:03d}.png")
        if not Path(r["png_path"]).exists():
            raise FileNotFoundError(f"{r['png_path']} is missing: run `uv run hgrag freeze` first")
    return rows


def model_config(spec: str, api_key: str | None = None, max_tokens: int = 1200, timeout: float = 120.0) -> LLMConfig:
    """``hf:org/model`` (HF router), ``anthropic:model`` or ``openai:model`` (HGRAG_LLM_BASE_URL)."""
    kind, _, name = spec.partition(":")
    if not name:
        raise BenchError(f"model spec {spec!r}: expected 'hf:org/model' or 'anthropic:model'")
    if kind == "hf":
        key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        return LLMConfig(provider="openai", model=name, base_url=HF_URL, api_key=key or "", max_tokens=max_tokens,
                         send_images=True, timeout=timeout)
    if kind in ("anthropic", "openai"):
        return LLMConfig(provider=kind, model=name, max_tokens=max_tokens, send_images=True, timeout=timeout)
    raise BenchError(f"unknown provider {kind!r} in {spec!r}")


def list_models(token: str | None = None) -> list[dict[str, Any]]:
    """Image-capable models served by the HF router, with their providers."""
    token = token or os.environ.get("HF_TOKEN")
    resp = requests.get(f"{HF_URL}/models", timeout=30, headers={"authorization": f"Bearer {token}"} if token else {})
    resp.raise_for_status()
    out = []
    for m in resp.json().get("data", []):
        if "image" in (m.get("architecture") or {}).get("input_modalities", []):
            out.append({"id": m["id"], "providers": [p.get("provider") for p in m.get("providers", [])]})
    return out


# ----------------------------------------------------------------------------------------- calls
_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_description(text: str) -> str:
    return _THINK.sub("", text).strip()


def describe_figure(cfg: LLMConfig, png_path: str, prompt: str = DESCRIBE_PROMPT) -> tuple[str, dict[str, Any], float]:
    if cfg.provider == "openai" and cfg.base_url == HF_URL and not cfg.api_key:
        raise GenerationError("HF_TOKEN is not set (token with the 'Make calls to Inference Providers' permission).")
    media, data = generation._encode_image(png_path, cfg.max_image_edge)
    image = ({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
             if cfg.provider == "anthropic" else {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
    t0 = time.perf_counter()
    parts, usage = [], {}
    for d in generation.stream_completion(cfg, DESCRIBE_SYSTEM, [image, {"type": "text", "text": prompt}]):
        if "text" in d:
            parts.append(d["text"])
        else:
            usage = d["usage"]
    return clean_description("".join(parts)), usage, time.perf_counter() - t0


def judge_description(cfg: LLMConfig, figure: dict[str, Any], description: str) -> dict[str, Any]:
    checklist = "\n".join(f"- {f['id']}: {f['text']}" for f in figure["facts"])
    user = (f"FULL TRUTH:\n{figure['full_description']}\n\nCHECKLIST:\n{checklist}\n\n"
            f"CANDIDATE:\n{description or '(empty)'}")
    last: Exception | None = None
    for attempt in range(2):
        text, usage = generation.complete(cfg, JUDGE_SYSTEM, user + ("\n\nReply with the JSON object only." if attempt else ""),
                                          max_tokens=600)
        try:
            return {**parse_judgement(text, [f["id"] for f in figure["facts"]]), "judge_usage": usage}
        except BenchError as exc:
            last = exc
    raise last  # type: ignore[misc]


def parse_judgement(text: str, fact_ids: list[str]) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise BenchError(f"no JSON in the judge reply: {text[:200]!r}")
    try:
        data = json.loads(m.group(0))
        facts = {k: bool(data["facts"].get(k, False)) for k in fact_ids}
        score = int(data["score"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BenchError(f"invalid judge reply: {exc}") from exc
    if not 1 <= score <= 5:
        raise BenchError(f"score out of range: {score}")
    invented = [str(x) for x in data.get("invented") or []]
    return {"facts": facts, "recall": sum(facts.values()) / len(fact_ids), "invented": invented, "score": score}


def run_one(spec: str, figure: dict[str, Any], *, judge_cfg: LLMConfig | None, prompt: str = DESCRIBE_PROMPT,
            api_key: str | None = None, max_tokens: int = 1200) -> dict[str, Any]:
    res: dict[str, Any] = {"model": spec, "figure": figure["id"], "pdf": figure["pdf"], "png_path": figure["png_path"]}
    try:
        desc, usage, secs = describe_figure(model_config(spec, api_key, max_tokens), figure["png_path"], prompt)
        res.update(description=desc, seconds=round(secs, 2), usage=usage)
        if judge_cfg is not None:
            res.update(judge_description(judge_cfg, figure, desc))
    except (GenerationError, BenchError, requests.RequestException) as exc:
        res["error"] = str(exc)
    return res


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per model: recall, mean score, inventions per figure, latency, tokens, errors."""
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        ok = [r for r in mine if "error" not in r]
        judged = [r for r in ok if "score" in r]

        def mean(key: str, src: list[dict[str, Any]]) -> float | None:
            return round(statistics.fmean(r[key] for r in src), 3) if src else None

        out.append({
            "model": model, "figures": len(mine), "errors": len(mine) - len(ok),
            "recall": mean("recall", judged), "score": mean("score", judged),
            "invented_per_figure": round(statistics.fmean(len(r["invented"]) for r in judged), 2) if judged else None,
            "seconds": mean("seconds", ok),
            "output_tokens": round(statistics.fmean((r.get("usage") or {}).get("output_tokens", 0) for r in ok))
            if ok else None,
        })
    return sorted(out, key=lambda s: (s["score"] is None, -(s["score"] or 0), -(s["recall"] or 0)))


def run_bench(models: list[str] | None = None, *, pdf_dir: str | Path = config.CONTROL_PDF_DIR,
              figures: list[str] | None = None, judge: bool = True, judge_model: str | None = None,
              prompt: str = DESCRIBE_PROMPT, hf_token: str | None = None, anthropic_key: str | None = None,
              workers: int = 4, max_tokens: int = 1200, out_dir: str | Path | None = None,
              progress: Any = None) -> dict[str, Any]:
    models = models or DEFAULT_MODELS
    truth = load_truth(pdf_dir)
    if figures:
        unknown = set(figures) - {t["id"] for t in truth}
        if unknown:
            raise BenchError(f"unknown figures {sorted(unknown)}")
        truth = [t for t in truth if t["id"] in figures]
    judge_cfg = evaluation.judge_config(judge_model, anthropic_key) if judge else None
    if judge_cfg is not None and not judge_cfg.api_key:
        raise BenchError("ANTHROPIC_API_KEY is not set (needed by the judge): add it to .env, or run with --no-judge.")
    jobs = [(m, t) for m in models for t in truth]
    key = lambda spec: hf_token if spec.startswith("hf:") else (anthropic_key if spec.startswith("anthropic:") else None)  # noqa: E731

    def work(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        return run_one(job[0], job[1], judge_cfg=judge_cfg, prompt=prompt, api_key=key(job[0]), max_tokens=max_tokens)

    done: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(futures):  # `progress` runs in the calling thread (safe for Streamlit)
            done[futures[fut]] = fut.result()
            if progress:
                progress(done[futures[fut]], len(done), len(jobs))
    rows = [done[i] for i in range(len(jobs))]
    run_key = time.strftime("%Y%m%dT%H%M%S")
    out = Path(out_dir or config.EVAL_DIR / "vlm") / run_key
    out.mkdir(parents=True, exist_ok=True)
    summary = {"run_key": run_key, "prompt": prompt, "judge_model": judge_cfg.model if judge_cfg else None,
               "models": summarize(rows)}
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"run_key": run_key, "dir": out, "rows": rows, "summary": summary["models"], "truth": truth}


def list_runs(root: str | Path | None = None) -> list[Path]:
    root = Path(root or config.EVAL_DIR / "vlm")
    return sorted((p for p in root.glob("*") if (p / "results.jsonl").exists()), reverse=True)


def load_run(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows = [json.loads(x) for x in (path / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"run_key": path.name, "dir": path, "rows": rows, "summary": summarize(rows)}
