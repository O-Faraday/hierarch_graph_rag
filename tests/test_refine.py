"""Second opinion on the low scores of a ragbench run (context judge, then generation judge), with faked API replies."""

from __future__ import annotations

import json

import pytest

from hgrag import cli, generation, ragbench, refine
from hgrag.generation import LLMConfig
from test_generation import FakeResponse, _anthropic_stream
from test_eval import offline  # noqa: F401
from test_ragbench import bench  # noqa: F401
from test_retrieval import BagOfWordsEmbedder


@pytest.fixture()
def scored_run(bench, monkeypatch):  # noqa: F811
    """A ragbench run of the control documents, with answers and first-judge scores 5, 2 and 3."""
    work, src, parse = bench
    monkeypatch.setattr(ragbench, "fetch_pdfs", lambda ids, urls, d, **kw: ({p: d / f"{p}.pdf" for p in ids if (d / f"{p}.pdf").exists()}, {}))
    res = ragbench.run(3, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False)
    rows = [json.loads(line) for line in (res["dir"] / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for r, score in zip(rows, (5, 2, 3)):
        r.update(answer=f"answer of {r['id']}", score=score, rationale="first judge")
    (res["dir"] / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (res["dir"] / "summary.json").write_text(json.dumps({**ragbench.summarize(rows, 5), "config": res["summary"]["config"]}), encoding="utf-8")
    return work, res["dir"], {r["id"]: r for r in rows}


def _fake_judges(monkeypatch, *, bad_context: str, cause_reply: str = '"cause": "reference"'):
    calls = []

    def complete(cfg, system, user, max_tokens=None):
        calls.append(user)
        task = user[-1]["text"]
        if task.startswith("TASK: judge the CONTEXT"):
            score = 2 if bad_context in task else 5
            return json.dumps({"rationale": "the PDF answers in 3.1", "score": score}), {"input_tokens": 100, "output_tokens": 10}
        return '{"rationale": "the reference is imprecise", "score": 5, ' + cause_reply + "}", {"input_tokens": 20, "output_tokens": 5,
                                                                                               "cache_read_input_tokens": 90}

    monkeypatch.setattr(generation, "complete", complete)
    return calls


def test_low_scores_get_a_context_then_a_generation_judgement(scored_run, monkeypatch):
    work, run_dir, rows = scored_run
    low = [q for q, r in rows.items() if r["score"] < 4]
    calls = _fake_judges(monkeypatch, bad_context=rows[low[1]]["question"])
    res = refine.refine_run(run_dir, threshold=4, cfg=LLMConfig(provider="anthropic", api_key="k"), workdir=work, workers=1)
    assert res["n_candidates"] == 2 and res["n_errors"] == 0
    out = {r["id"]: r for r in map(json.loads, (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines())}
    good = next(q for q, r in rows.items() if r["score"] == 5)
    assert "refine_score" not in out[good] and "context_score" not in out[good]  # above the threshold: untouched
    assert out[low[0]]["score"] == 2 and out[low[0]]["refine_score"] == 5 and out[low[0]]["refine_cause"] == "reference"
    assert out[low[0]]["refine_usage"] == {"input_tokens": 120, "output_tokens": 15, "cache_read_input_tokens": 90}
    assert out[low[1]]["context_score"] == 2 and out[low[1]]["refine_cause"] == "context" and out[low[1]]["refine_score"] == 3
    assert len(calls) == 3  # bad context: no generation judge
    parts = calls[0]
    assert parts[0]["type"] == "document" and parts[0]["source"]["media_type"] == "application/pdf"
    context = next(p for p in parts if p["type"] == "text" and p["text"].startswith("CONTEXT"))
    assert context["cache_control"] == {"type": "ephemeral"} and "[1]" in context["text"]
    assert calls[1][:-1] == parts[:-1]  # same prefix for both judges of a question (prompt cache)
    assert calls[1][-1]["text"].startswith("TASK: the context was judged sufficient")
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert s["mean_score"] == 3.33 and s["mean_refined_score"] == 4.33 and s["refine_causes"] == {"reference": 1, "context": 1}
    assert s["refine"]["threshold"] == 4 and s["config"]["papers_indexed"] == 3
    assert "refined (second opinion" in ragbench.format_summary(s)
    md = (run_dir / "questions.md").read_text(encoding="utf-8")
    assert "**Context judge:** 2/5" in md and "**Refined score:** 5 (cause: reference)" in md
    # resumed: nothing left to do, and no key needed for that
    again = refine.refine_run(run_dir, threshold=4, cfg=LLMConfig(provider="anthropic", api_key=""), workdir=work)
    assert again["n_candidates"] == 0


def test_context_is_rebuilt_by_section_path_for_old_runs(scored_run):
    work, _, rows = scored_run
    row = next(iter(rows.values()))
    exact = refine.rebuild_sections(row, output_dir=work / "output", pdf_dir=work / "pdfs")
    old = {**row, "retrieved": [{k: v for k, v in h.items() if k not in ("doc_id", "section_ref")} for h in row["retrieved"]]}
    by_path = refine.rebuild_sections(old, output_dir=work / "output", pdf_dir=work / "pdfs")
    assert exact and [s["section_ref"] for s in by_path] == [s["section_ref"] for s in exact]
    assert exact[0]["text"] and exact[0]["source_pdf"].endswith(".pdf")


def test_invalid_cause_is_an_error_on_the_question_only(scored_run, monkeypatch):
    work, run_dir, rows = scored_run
    _fake_judges(monkeypatch, bad_context="nothing matches", cause_reply='"cause": "luck"')
    res = refine.refine_run(run_dir, threshold=4, cfg=LLMConfig(provider="anthropic", api_key="k"), workdir=work, workers=2)
    assert res["n_errors"] == 2
    out = [json.loads(x) for x in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("no valid cause" in r["refine_error"] for r in out if r["score"] < 4)


def test_missing_key_is_refused_before_any_call(scored_run):
    work, run_dir, _ = scored_run
    with pytest.raises(ragbench.RagBenchError, match="ANTHROPIC_API_KEY"):
        refine.refine_run(run_dir, cfg=LLMConfig(provider="anthropic", api_key=""), workdir=work)


def test_sonnet_judges_ask_for_the_server_side_fallback(monkeypatch):
    seen = {}
    lines = _anthropic_stream(["ok"])
    lines[1] = "data: " + json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 5, "cache_read_input_tokens": 40}}})
    monkeypatch.setattr(generation.requests, "post", lambda url, **kw: seen.update(kw) or FakeResponse(lines))
    cfg = refine.refine_config("claude-sonnet-5-5", api_key="k")
    text, usage = generation.complete(cfg, "s", [{"type": "text", "text": "u"}])
    assert text == "ok" and usage["cache_read_input_tokens"] == 40
    assert seen["headers"]["anthropic-beta"] == generation.FALLBACK_BETA and seen["json"]["fallbacks"] == "default"
    assert refine.refine_config("claude-haiku-5-5", api_key="k").fallbacks is None


def test_cli_ragbench_refine(monkeypatch, tmp_path, capsys):
    seen = {}
    (tmp_path / "summary.json").write_text(json.dumps({**ragbench.summarize([], 5), "k_sections": 5}), encoding="utf-8")

    def fake(run_dir, **kw):
        seen.update(run_dir=run_dir, **kw)
        return {"n_candidates": 0, "model": "m", "seconds": 0.0, "n_errors": 0}

    monkeypatch.setattr(refine, "refine_run", fake)
    monkeypatch.setattr("sys.argv", ["hgrag", "ragbench-refine", str(tmp_path), "--threshold", "3", "--force"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0 and seen["threshold"] == 3 and seen["force"] is True
    assert "judged again by m" in capsys.readouterr().out
