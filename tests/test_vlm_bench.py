"""VLM comparison: ground truth, judge parsing, bench run with faked HTTP, and the Streamlit page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hgrag import config, generation, vlm_bench
from test_generation import FakeResponse, _anthropic_stream

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))


def _openai_stream(text):
    chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]},
              {"choices": [{"delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}]
    return [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]


def test_truth_matches_the_frozen_figures():
    truth = vlm_bench.load_truth()
    assert [t["id"] for t in truth] == ["aurora_f1", "botany_f1", "botany_f2"]
    for t in truth:
        assert Path(t["png_path"]).exists() and len(t["facts"]) >= 5
        assert len({f["id"] for f in t["facts"]}) == len(t["facts"])


def test_model_specs():
    hf = vlm_bench.model_config("hf:google/gemma-3-4b-it", api_key="hf_x")
    assert (hf.provider, hf.model, hf.base_url, hf.api_key) == ("openai", "google/gemma-3-4b-it", vlm_bench.HF_URL, "hf_x")
    assert vlm_bench.model_config("anthropic:claude-haiku-5-5", api_key="k").provider == "anthropic"
    with pytest.raises(vlm_bench.BenchError):
        vlm_bench.model_config("gemma")
    with pytest.raises(vlm_bench.BenchError):
        vlm_bench.model_config("foo:bar")


def test_parse_judgement_and_summary():
    ids = ["a", "b"]
    j = vlm_bench.parse_judgement('x {"facts": {"a": true}, "invented": ["wrong axis"], "score": 3} y', ids)
    assert j["facts"] == {"a": True, "b": False} and j["recall"] == 0.5 and j["invented"] == ["wrong axis"]
    for bad in ("no json", '{"facts": {}, "score": 9}', '{"facts": 1, "score": 3}'):
        with pytest.raises(vlm_bench.BenchError):
            vlm_bench.parse_judgement(bad, ids)
    rows = [{"model": "m1", "score": 5, "recall": 1.0, "invented": [], "seconds": 2, "usage": {"output_tokens": 10}},
            {"model": "m2", "score": 2, "recall": .4, "invented": ["x", "y"], "seconds": 4, "usage": {"output_tokens": 30}},
            {"model": "m3", "error": "HTTP 400"}]
    s = vlm_bench.summarize(rows)
    assert [x["model"] for x in s] == ["m1", "m2", "m3"] and s[2]["errors"] == 1 and s[1]["invented_per_figure"] == 2


def test_strips_thinking():
    assert vlm_bench.clean_description("<think>hmm</think>\nA bar chart.") == "A bar chart."


def test_run_bench_with_fake_http(tmp_path, monkeypatch):
    seen = []

    def post(url, **kw):
        seen.append((url, kw["json"]))
        if "anthropic" in url:  # the judge
            verdict = {"facts": {k: True for k in ("type", "title", "x_axis", "y_axis", "v2021", "v2024", "trend")},
                       "invented": [], "score": 5}
            return FakeResponse(_anthropic_stream([json.dumps(verdict)]))
        if kw["headers"].get("authorization") != "Bearer hf_secret":
            return FakeResponse([], status=401, body={"error": {"message": "bad token"}})
        assert any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/png;base64,")
                   for p in kw["json"]["messages"][1]["content"])
        return FakeResponse(_openai_stream("<think>x</think>Bar chart 'Branches per year', 2021 about 212."))

    monkeypatch.setattr(generation.requests, "post", post)
    done = []
    res = vlm_bench.run_bench(["hf:google/gemma-3-4b-it", "hf:org/other"], figures=["aurora_f1"], hf_token="hf_secret",
                              anthropic_key="sk-test", out_dir=tmp_path, progress=lambda r, n, t: done.append((n, t)))
    assert done[-1] == (2, 2)
    assert all(r["description"].startswith("Bar chart") and r["score"] == 5 for r in res["rows"])
    assert res["summary"][0]["recall"] == 1.0
    saved = vlm_bench.load_run(res["dir"])
    assert len(saved["rows"]) == 2 and (res["dir"] / "summary.json").exists()
    assert "sk-test" not in (res["dir"] / "results.jsonl").read_text() and "hf_secret" not in (res["dir"] / "summary.json").read_text()

    bad = vlm_bench.run_bench(["hf:google/gemma-3-4b-it"], figures=["aurora_f1"], hf_token="wrong", anthropic_key="k",
                              out_dir=tmp_path)
    assert "bad token" in bad["rows"][0]["error"] and bad["summary"][0]["errors"] == 1


def test_missing_keys_are_explained(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(vlm_bench.BenchError, match="ANTHROPIC_API_KEY"):
        vlm_bench.run_bench(["hf:a/b"], anthropic_key=None, out_dir=tmp_path)
    res = vlm_bench.run_bench(["hf:a/b"], judge=False, figures=["aurora_f1"], out_dir=tmp_path)
    assert "HF_TOKEN" in res["rows"][0]["error"]


def test_page_shows_a_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    truth = vlm_bench.load_truth()
    run = tmp_path / "vlm" / "20261015T120000"
    run.mkdir(parents=True)
    rows = []
    for t in truth:
        for model, score in (("hf:a/good", 5), ("hf:b/bad", 2)):
            rows.append({"model": model, "figure": t["id"], "pdf": t["pdf"], "png_path": t["png_path"],
                         "description": f"desc by {model}", "seconds": 3.0, "usage": {"output_tokens": 40},
                         "facts": {f["id"]: score == 5 for f in t["facts"]}, "recall": 1.0 if score == 5 else 0.0,
                         "invented": [] if score == 5 else ["made-up value"], "score": score})
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    at = AppTest.from_file(str(APP / "views" / "vlm.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "hf:a/good" in text and "hf:b/bad" in text and "made-up value" in " ".join(w.value for w in at.warning)
    first = at.dataframe[0].value
    assert list(first["model"]) == ["hf:a/good", "hf:b/bad"]


def test_page_without_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    at = AppTest.from_file(str(APP / "views" / "vlm.py"), default_timeout=30).run()
    assert not at.exception and any("No comparison yet" in i.value for i in at.info)
