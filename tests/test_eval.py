"""Benchmark on the frozen Docling outputs (real structure, no Docling, no model) and the LLM judge (HTTP faked)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

import hgrag.indexing as indexing
from hgrag import cli, config, evaluation, generation, tracking
from hgrag.evaluation import JudgeError
from test_generation import FakeResponse, _anthropic_stream
from test_retrieval import BagOfWordsEmbedder

DATASET = config.CONTROL_PDF_DIR / "eval_dataset.jsonl"


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(indexing, "load_embedder", lambda *_: BagOfWordsEmbedder())
    monkeypatch.setattr(indexing, "make_chunker", lambda *a, **k: HierarchicalChunker())
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "ml")
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{(tmp_path / 'ml' / 'mlflow.db').as_posix()}")
    return tmp_path


@pytest.fixture()
def store(offline):
    return evaluation.build_store(workdir=offline / "store")


def test_frozen_parsing_builds_a_store_with_lists_tables_and_code(store):
    docs = store["index"]["documents"]
    assert set(docs) == {"01_aurora_bank_annual_report.pdf", "02_alpine_botany_field_guide.pdf",
                         "03_network_operations_handbook.pdf"}
    net = Path(docs["03_network_operations_handbook.pdf"]["doc_dir"])
    chunks = [json.loads(l) for l in (net / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("certificate expiry dates" in c["text"] and "2.2 Health checks" in c["section_path"] for c in chunks)
    assert any("netctl failover status" in c["text"] for c in chunks)  # code is a chunk of its section
    assert any("On-call engineer" in c["text"] for c in chunks)  # and so is the table
    assert store["index"]["points_in_collection"] == store["index"]["total_chunks"]
    fig = json.loads((net.parent / next(d for d in net.parent.iterdir() if (d / "figures.json").exists() and
                                        d.name != net.name and d.name.startswith("275")).name / "figures.json")
                     .read_text(encoding="utf-8"))
    assert all("description" not in f for f in fig)  # the frozen fixtures were parsed without descriptions


def test_retrieval_only_benchmark_needs_no_api(store, offline):
    rows = evaluation.load_dataset(DATASET)
    seen = []
    s = evaluation.run_eval(rows, **evaluation.store_kwargs(store), embedder=BagOfWordsEmbedder(), k=5, run_key=tracking.new_run_key(),
                            on_row=seen.append)
    assert len(seen) == 26 and s["n_answerable"] == 24 and s["n_errors"] == 0 and "mean_score" not in s
    assert s["hit_at_5"] >= 0.8 and 0 < s["mrr"] <= 1
    assert {"paragraph", "table", "list", "code", "figure", "derived", "unanswerable"} == set(s["by_kind"])
    dest = Path(s["dest"])
    assert {"results.jsonl", "summary.json", "report.md"} <= {p.name for p in dest.iterdir()}
    first = json.loads((dest / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["retrieved"][0]["rank"] == 1 and "section_path" in first["retrieved"][0]


# ------------------------------------------------------------------------------------- judge + generation
def _fake_llm(monkeypatch, judge_replies=None):
    """Answerer: echoes the sources. Judge: 5 if the answer contains the ground truth's first words, else 1."""
    calls = {"judge": 0, "answer": 0, "payloads": []}
    replies = list(judge_replies or [])

    def post(url, **kw):
        body = kw["json"]
        calls["payloads"].append(body)
        content = body["messages"][0]["content"]
        text = content if isinstance(content, str) else content[-1]["text"]
        if body["system"].startswith("You are a strict"):
            calls["judge"] += 1
            if replies:
                return FakeResponse(_anthropic_stream([replies.pop(0)]))
            truth = re.search(r"GROUND TRUTH:\n(.*?)\n\nCANDIDATE", text, re.S).group(1).lower().split()[:3]
            cand = text.split("CANDIDATE ANSWER:\n", 1)[1].lower()
            ok = all(w.strip(".,") in cand for w in truth)
            return FakeResponse(_anthropic_stream([json.dumps({"rationale": "compared", "score": 5 if ok else 1})]))
        calls["answer"] += 1
        return FakeResponse(_anthropic_stream([text]))

    monkeypatch.setattr(generation.requests, "post", post)
    return calls


def test_judge_parses_json_retries_once_and_rejects_garbage(monkeypatch):
    cfg = evaluation.judge_config(api_key="k")
    assert cfg.model == "claude-haiku-5-5" and cfg.send_images is False
    calls = _fake_llm(monkeypatch, ['Sure! {"rationale": "close", "score": 4}'])  # prose around the JSON
    assert evaluation.judge_answer(cfg, "q", "gt", "a")["score"] == 4
    calls = _fake_llm(monkeypatch, ["not json", '{"rationale": "ok", "score": 5}'])
    assert evaluation.judge_answer(cfg, "q", "gt", "a")["score"] == 5 and calls["judge"] == 2
    assert "JSON object only" in calls["payloads"][1]["messages"][0]["content"]
    _fake_llm(monkeypatch, ["nope", "still nope"])
    with pytest.raises(JudgeError, match="no JSON"):
        evaluation.judge_answer(cfg, "q", "gt", "a")
    _fake_llm(monkeypatch, ['{"rationale": "x", "score": 9}'] * 2)
    with pytest.raises(JudgeError, match="out of range"):
        evaluation.judge_answer(cfg, "q", "gt", "a")


def test_full_benchmark_generates_judges_and_tracks(store, offline, monkeypatch):
    calls = _fake_llm(monkeypatch)
    key = tracking.new_run_key()
    gen = generation.LLMConfig(provider="anthropic", api_key="k")
    s = evaluation.run_eval(evaluation.load_dataset(DATASET, ids=["q01", "q08", "q21", "q25"]), **evaluation.store_kwargs(store),
                            embedder=BagOfWordsEmbedder(), k=5, gen_cfg=gen,
                            judge_cfg=evaluation.judge_config(api_key="k"), run_key=key)
    res = {json.loads(l)["id"]: json.loads(l) for l in (Path(s["dest"]) / "results.jsonl").read_text(encoding="utf-8").splitlines()}
    assert calls["answer"] == 4 and calls["judge"] == 4
    assert res["q01"]["score"] == 5 and "412 million" in res["q01"]["answer"]  # answer built from the sections
    assert res["q21"]["score"] == 5  # the list is now indexed: its items reach the answer
    assert res["q25"]["score"] == 1 and s["abstention_accuracy"] == 0.0  # the fake echoes sources: no abstention
    assert res["q08"]["n_images"] >= 1  # the figure of 3.1 travels as an image
    assert any(isinstance(p["messages"][0]["content"], list) and
               any(c["type"] == "image" for c in p["messages"][0]["content"]) for p in calls["payloads"])
    assert res["q08"]["score"] == 1  # the value is only in the image: the echoing fake cannot read it
    assert s["mean_score"] == pytest.approx((5 + 1 + 5 + 1) / 4) and s["input_tokens"] > 0

    import mlflow

    exp = mlflow.get_experiment_by_name(tracking.EVAL)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list",
                              filter_string=f"tags.`hgrag.run_key` = '{key}'")
    assert sum(r.data.tags["hgrag.kind"] == "step" for r in runs) == 4
    parent = next(r for r in runs if r.data.tags["hgrag.kind"] == "execution")
    assert parent.data.metrics["mean_score"] == s["mean_score"] and "hit_at_1" in parent.data.metrics


def test_generation_error_is_reported_per_question(store, offline, monkeypatch):
    def post(url, **kw):
        return FakeResponse([], 401, {"error": {"message": "invalid x-api-key"}})

    monkeypatch.setattr(generation.requests, "post", post)
    s = evaluation.run_eval(evaluation.load_dataset(DATASET, limit=2), **evaluation.store_kwargs(store), embedder=BagOfWordsEmbedder(),
                            gen_cfg=generation.LLMConfig(provider="anthropic", api_key="bad"),
                            judge_cfg=evaluation.judge_config(api_key="bad"))
    assert s["n_errors"] == 2 and "mean_score" not in s  # the run goes on, errors are listed


def test_cli_eval_and_freeze(offline, monkeypatch, capsys):
    args = argparse.Namespace(dataset=str(DATASET), pdf_dir=str(config.CONTROL_PDF_DIR), k=5, ids="q01,q02", kind=None,
                              limit=None, children=False, no_generate=True, no_judge=False, reuse_store=False,
                              live=False, model=None, judge_model=None, no_mlflow=True)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli._eval(args) == 0  # retrieval only: no key needed
    out = capsys.readouterr().out
    assert "Building the benchmark store from the frozen parsing" in out and "q01 [paragraph" in out and "hit@1" in out

    args.no_generate = False
    assert cli._eval(args) == 2 and "ANTHROPIC_API_KEY is not set" in capsys.readouterr().out

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _fake_llm(monkeypatch)
    args.reuse_store = True
    assert cli._eval(args) == 0
    out = capsys.readouterr().out
    assert "score 5/5" in out and "judge: mean score" in out and "Building the benchmark store" not in out


def test_dataset_filters():
    assert [r["id"] for r in evaluation.load_dataset(DATASET, kinds=["figure"])] == ["q08", "q13", "q16"]
    assert len(evaluation.load_dataset(DATASET, limit=5)) == 5
    with pytest.raises(ValueError, match="unknown question ids"):
        evaluation.load_dataset(DATASET, ids=["q99"])


def test_dotenv_is_loaded_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nHF_HUB_OFFLINE=1\nexport ANTHROPIC_API_KEY="sk-abc"  \nHGRAG_X=a b # note\nEXISTING=file\n',
                   encoding="utf-8")
    for k in ("HF_HUB_OFFLINE", "ANTHROPIC_API_KEY", "HGRAG_X"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EXISTING", "shell")
    config.load_dotenv(env)
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["ANTHROPIC_API_KEY"] == "sk-abc"
    assert os.environ["HGRAG_X"] == "a b" and os.environ["EXISTING"] == "shell"
    for k in ("HF_HUB_OFFLINE", "ANTHROPIC_API_KEY", "HGRAG_X"):
        monkeypatch.delenv(k, raising=False)


def test_clean_dir_and_copy_tree_handle_read_only_files(tmp_path):
    """Windows regression: read-only files must neither block the cleanup nor propagate to the copies."""
    import os
    import stat

    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    f = src / "sub" / "manifest.json"
    f.write_text("{}", encoding="utf-8")
    os.chmod(f, stat.S_IREAD)
    dst = tmp_path / "dst"
    evaluation._copy_tree(src, dst)
    copy = dst / "sub" / "manifest.json"
    copy.write_text("{\"a\": 1}", encoding="utf-8")  # writable copy
    os.chmod(copy, stat.S_IREAD)
    evaluation._clean_dir(dst)
    assert not dst.exists()
    os.chmod(f, stat.S_IWRITE | stat.S_IREAD)
