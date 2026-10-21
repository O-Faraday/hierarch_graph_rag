"""Open RAG Bench runner on the 3 control documents, with a fake dataset (no network, no Docling)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hgrag import cli, config, ragbench, tracking
from hgrag.parsing import parse_options
from test_eval import offline  # noqa: F401
from test_retrieval import BagOfWordsEmbedder

CONTROL = config.CONTROL_PDF_DIR
PAPERS = {"2401.0001": "01_aurora_bank_annual_report.pdf", "2401.0002": "02_alpine_botany_field_guide.pdf",
          "2401.0003": "03_network_operations_handbook.pdf"}


class FakeSource:
    """Dict-backed stand-in for the Hub files. Gold sections are copied from the real parsed control documents."""

    def __init__(self, gold):
        self.files = {
            "queries.json": {
                "q1": {"query": "three new branches in the Alsace region", "type": "extractive", "source": "text"},
                "q2": {"query": "failover to the secondary datacentre within ninety seconds", "type": "extractive", "source": "text-table"},
                "q3": {"query": "cushion plants colonise ground that moves with frost", "type": "abstractive", "source": "text-image"},
            },
            "qrels.json": {"q1": {"doc_id": "2401.0001", "section_id": 1}, "q2": {"doc_id": "2401.0003", "section_id": 1},
                           "q3": {"doc_id": "2401.0002", "section_id": 0}},
            "answers.json": {"q1": "Three new branches opened in Alsace.", "q2": "Within ninety seconds.", "q3": "Cushion plants."},
            "pdf_urls.json": {pid: f"http://arxiv.org/pdf/{pid}" for pid in PAPERS} | {"2401.0009": "http://arxiv.org/pdf/2401.0009"},
        }
        for pid, sections in gold.items():
            self.files[f"corpus/{pid}.json"] = {"id": pid, "sections": [{"text": t} for t in sections]}

    def json(self, name):
        return self.files[name]


def _section_text(doc_dir: Path, title: str) -> str:
    tree = json.loads((doc_dir / "hierarchy.json").read_text(encoding="utf-8"))
    stack = [tree]
    while stack:
        n = stack.pop()
        if n["title"] == title:
            return "\n".join(e.get("text") or "" for e in n["elements"])
        stack.extend(n["children"])
    raise KeyError(title)


@pytest.fixture()
def bench(tmp_path, monkeypatch, offline):  # noqa: F811
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker  # noqa: F401
    from hgrag import evaluation

    work = tmp_path / "rb"
    (work / "pdfs").mkdir(parents=True)
    for pid, name in PAPERS.items():  # "already downloaded" PDFs: the control PDFs under the paper ids
        shutil.copyfile(CONTROL / name, work / "pdfs" / f"{pid}.pdf")
    # hierarchy of the frozen parsing, to take gold sections from it (the benchmark would use its own parsing)
    ref = tmp_path / "ref"
    for pid, name in PAPERS.items():
        src = CONTROL / "parsed" / evaluation.compute_doc_id(CONTROL / name)
        shutil.copytree(src, ref / pid)
        m = json.loads((ref / pid / "manifest.json").read_text(encoding="utf-8"))
        (ref / pid / "manifest.json").write_text(json.dumps({**m, "source_pdf": f"{pid}.pdf", "doc_dir": ""}), encoding="utf-8")
        from hgrag import pipeline
        pipeline.build_and_save_hierarchy(ref / pid)
    gold = {"2401.0001": ["x", _section_text(ref / "2401.0001", "3.1 New branches")],
            "2401.0002": [_section_text(ref / "2401.0002", "2.2 Siliceous screes")],
            "2401.0003": ["x", _section_text(ref / "2401.0003", "3.1 Failover")]}

    def parse(pdf, describe_pictures=False):  # stands in for Docling: copy the frozen parsing under the file hash
        dst = work / "output" / evaluation.compute_doc_id(pdf)
        shutil.copytree(CONTROL / "parsed" / evaluation.compute_doc_id(CONTROL / PAPERS[Path(pdf).stem]), dst)
        m = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        (dst / "manifest.json").write_text(json.dumps({**m, "source_pdf": str(pdf), "doc_dir": str(dst),
                                                       "parse_options": parse_options()}), encoding="utf-8")

    return work, FakeSource(gold), parse


def test_sample_is_deterministic_and_filtered():
    src = FakeSource({})
    q, r, a, u = (src.json(n) for n in ("queries.json", "qrels.json", "answers.json", "pdf_urls.json"))
    s1 = ragbench.pick_sample(q, r, u, 2, seed=1)
    assert s1 == ragbench.pick_sample(q, r, u, 2, seed=1) and len(s1["golden"]) == 2
    only = ragbench.pick_sample(q, r, u, 5, sources=["text-table"])
    assert only["golden"] == ["2401.0003"] and only["query_ids"] == ["q2"]
    with_neg = ragbench.pick_sample(q, r, u, 3, negatives=5)
    assert with_neg["negatives"] == ["2401.0009"]  # the only paper without a question
    assert ragbench.pick_sample(q, r, u, 3, max_queries=2)["query_ids"] == ["q1", "q3"] or len(ragbench.pick_sample(q, r, u, 3, max_queries=2)["query_ids"]) == 2
    with pytest.raises(ragbench.RagBenchError):
        ragbench.pick_sample(q, r, u, 3, sources=["nope"])


def test_overlap_helpers():
    gold = "The branch network grew by three new branches in the Alsace region during the second quarter of the year."
    assert ragbench.containment(ragbench.shingles("three new branches in the Alsace region"), ragbench.shingles(gold)) == 1.0
    assert ragbench.containment(ragbench.shingles("completely different words in another sentence here"), ragbench.shingles(gold)) == 0.0
    assert ragbench.shingles("too short") == set()
    qrels = {"a": {"doc_id": "p", "section_id": 3}}
    assert ragbench.detect_base(qrels, {"p": 3}, ["a"]) == 1 and ragbench.detect_base(qrels, {"p": 9}, ["a"]) == 0
    doc = {"id": "p", "sections": [{"text": "zero"}, {"text": "one"}]}
    assert ragbench.gold_text(doc, 1) == "one" and ragbench.gold_text(doc, 2, base=1) == "one"
    with pytest.raises(ragbench.RagBenchError):
        ragbench.gold_text(doc, 5)


def test_download_pdf_validates_and_retries(tmp_path, monkeypatch):
    calls = []

    class R:
        def __init__(self, code, body):
            self.status_code, self.content = code, body

    seq = [R(503, b""), R(200, b"<html>not a pdf"), R(200, b"%PDF-1.7 ok")]
    monkeypatch.setattr(ragbench.requests, "get", lambda url, **kw: calls.append(url) or seq.pop(0))
    monkeypatch.setattr(ragbench.time, "sleep", lambda s: None)
    dest = tmp_path / "p.pdf"
    ragbench.download_pdf("http://arxiv.org/pdf/x", dest)
    assert dest.read_bytes().startswith(b"%PDF-") and calls == ["https://arxiv.org/pdf/x"] * 3
    ragbench.download_pdf("http://arxiv.org/pdf/x", dest)  # cached
    assert len(calls) == 3
    monkeypatch.setattr(ragbench.requests, "get", lambda url, **kw: R(404, b""))
    with pytest.raises(ragbench.RagBenchError, match="HTTP 404"):
        ragbench.download_pdf("http://arxiv.org/pdf/y", tmp_path / "q.pdf", retries=2)


def test_run_on_the_control_documents(bench, monkeypatch):
    work, src, parse = bench
    monkeypatch.setattr(ragbench, "fetch_pdfs", lambda ids, urls, d, **kw: (
        {p: d / f"{p}.pdf" for p in ids if (d / f"{p}.pdf").exists()}, {p: "HTTP 404" for p in ids if not (d / f"{p}.pdf").exists()}))
    msgs = []
    res = ragbench.run(3, negatives=1, k=5, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse,
                       track=False, on_progress=msgs.append)
    s = res["summary"]
    assert s["n"] == 3 and s["doc_hit_at_1"] == 1.0 and s["section_hit_at_1"] == 1.0 and s["mrr"] == 1.0, res["results"]
    assert s["context_recall"] >= 0.9 and s["by_source"]["text-table"]["n"] == 1 and s["by_type"]["abstractive"]["n"] == 1
    assert s["config"]["section_base"] == 0 and s["config"]["papers_indexed"] == 3
    assert "2401.0009" in s["failed"]["download"]  # the negative that is not on disk is skipped, not fatal
    assert (res["dir"] / "results.jsonl").exists() and "section hit@1" in (res["dir"] / "report.md").read_text(encoding="utf-8")
    assert any(m.startswith("query 3/3") for m in msgs)
    # a second run reuses the parsed documents
    again = ragbench.run(3, negatives=1, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False)
    assert again["summary"]["config"]["reused"] == 3 and again["summary"]["config"]["parsed"] == 0


def test_reference_run_flags_questions_whose_context_changed(bench, monkeypatch):
    work, src, parse = bench
    monkeypatch.setattr(ragbench, "fetch_pdfs", lambda ids, urls, d, **kw: ({p: d / f"{p}.pdf" for p in ids if (d / f"{p}.pdf").exists()}, {}))
    first = ragbench.run(3, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False)
    rows = [json.loads(line) for line in (first["dir"] / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["retrieved"] = rows[0]["retrieved"][::-1] + [{"paper": "x", "section_path": "other"}]  # q changed in the reference
    ref = work / "ref"
    ref.mkdir()
    (ref / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    again = ragbench.run(3, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False, reference_run=ref)
    same = {r["id"]: r["context_same"] for r in again["results"]}
    assert same[rows[0]["id"]] is False and sum(same.values()) == 2
    cfg = again["summary"]["config"]
    assert cfg["context_changed"] == 1 and cfg["reference_run"] == str(ref) and cfg["gen_model"] is None
    with pytest.raises(ragbench.RagBenchError):
        ragbench.run(3, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False,
                     reference_run=work / "missing")


def test_wrong_gold_gives_a_miss(bench, monkeypatch):
    work, src, parse = bench
    src.files["corpus/2401.0001.json"]["sections"][1]["text"] = "nothing here matches the question at all, really not a word"
    monkeypatch.setattr(ragbench, "fetch_pdfs", lambda ids, urls, d, **kw: ({p: d / f"{p}.pdf" for p in ids if (d / f"{p}.pdf").exists()}, {}))
    res = ragbench.run(3, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False)
    q1 = next(r for r in res["results"] if r["id"] == "q1")
    assert q1["doc_rank"] == 1 and q1["section_rank"] is None and q1["context_recall"] == 0.0
    assert res["summary"]["section_hit_at_1"] == pytest.approx(2 / 3, abs=0.01)


def test_cli_ragbench_parser(monkeypatch, capsys):
    seen = {}

    def fake_run(n, **kw):
        seen.update(n=n, **kw)
        return {"key": "k", "store": {"dirs": {"a": 1}}}

    monkeypatch.setattr(ragbench, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["hgrag", "ragbench", "--docs", "100", "--negatives", "20", "--sources", "text-table,text-image",
                                     "--prepare-only", "-k", "3"])
    cli.main()
    assert seen["n"] == 100 and seen["negatives"] == 20 and seen["sources"] == ["text-table", "text-image"] and seen["k"] == 3
    assert seen["prepare_only"] is True and "prepared k" in capsys.readouterr().out


def test_hub_files_url_and_cache(tmp_path, monkeypatch):
    urls = []

    class R:
        status_code = 200
        content = b'{"a": 1}'

    monkeypatch.setattr(ragbench.requests, "get", lambda url, **kw: urls.append(url) or R())
    src = ragbench.HFSource(tmp_path)
    assert src.json("queries.json") == {"a": 1} and src.json("queries.json") == {"a": 1}
    assert urls == ["https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv/queries.json"]  # cached after 1st call
    assert src.json("corpus/2410.14077v2.json") == {"a": 1}
    assert urls[-1].endswith("/pdf/arxiv/corpus/2410.14077v2.json")


def test_qa_report_sorted_by_increasing_score(tmp_path):
    rows = [
        {"id": "good", "paper": "2409.1", "source": "text", "type": "extractive", "question": "ok", "expected_answer": "A",
         "answer": "A!", "score": 5, "rationale": "perfect", "section_rank": 1, "context_recall": 0.9},
        {"id": "bad", "paper": "2409.1", "source": "text", "type": "extractive", "question": "Q?", "expected_answer": "B",
         "answer": "wrong", "score": 1, "rationale": "off topic", "section_rank": None, "context_recall": 0.1, "gold_text": "l1\nl2",
         "retrieved": [{"rank": 1, "section_path": ["Intro"], "score": 0.5, "doc": True, "on_gold": False, "preview": "txt"}]},
    ]
    md = ragbench.qa_report(rows, tmp_path)
    assert md.index("bad") < md.index("good")
    assert "**Expected answer:** B" in md and "**Answer:** wrong" in md and "**Comment (judge):** off topic" in md and "2409.1.pdf" in md
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert ragbench.write_qa_report(run, tmp_path).read_text(encoding="utf-8").index("bad") < 400


def test_disputed_references_get_their_own_metrics_and_are_flagged(tmp_path, monkeypatch):
    def row(qid, score, rank):
        return {"id": qid, "paper": "2409.1", "source": "text", "type": "extractive", "question": "q", "expected_answer": "a",
                "answer": "b", "score": score, "section_rank": rank, "doc_rank": 1, "context_recall": 0.5}

    rows = [row("ok", 5, 1), row("wrong-ref", 1, 1), row("miss", 3, None)]
    s = ragbench.summarize(rows, 10, {"wrong-ref": "table says otherwise", "not-in-run": "x"})
    assert s["n"] == 3 and s["mean_score"] == 3.0  # the raw metrics keep every question
    x = s["excluding_disputed"]
    assert x["n"] == 2 and x["excluded"] == ["wrong-ref"] and x["mean_score"] == 4.0 and x["section_hit_at_1"] == 0.5
    assert "without 1 disputed reference(s)" in ragbench.format_summary(s)
    assert "excluding_disputed" not in ragbench.summarize(rows, 10, {})

    disputed = tmp_path / "disputed.json"
    disputed.write_text(json.dumps({"wrong-ref": "table says otherwise"}), encoding="utf-8")
    monkeypatch.setattr(config, "RAGBENCH_DISPUTED", disputed)
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({**ragbench.summarize(rows, 10), "config": {"key": "k"}}), encoding="utf-8")
    md = ragbench.write_qa_report(run, tmp_path).read_text(encoding="utf-8")
    assert "DISPUTED REFERENCE" in md and "**Disputed reference:** table says otherwise" in md
    new = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert new["excluding_disputed"]["n"] == 2 and new["config"] == {"key": "k"}


def test_judge_reply_with_unescaped_quotes_is_still_read():
    from hgrag import evaluation

    j = evaluation._parse_judgement('{"rationale": "The candidate answers "No" and misses the one-third figure.", "score": 3}')
    assert j["score"] == 3 and "misses the one-third" in j["rationale"]


def test_qa_report_accepts_section_path_as_string(tmp_path):
    md = ragbench.qa_report([{"id": "x", "paper": "p", "source": "text", "type": "extractive", "question": "q", "section_rank": 1,
                              "context_recall": 1.0, "retrieved": [{"rank": 1, "section_path": "doc > A > B", "score": 0.5,
                                                                     "doc": True, "on_gold": True, "preview": "t"}]}], tmp_path)
    assert "#1 doc > A > B" in md


def test_empty_or_cut_answer_is_retried_with_double_budget(monkeypatch):
    from hgrag import generation

    calls = []

    def fake_stream(question, sections, cfg, info=None):
        calls.append(cfg.max_tokens)
        if len(calls) == 1:  # first call: the whole budget spent before any text
            info["usage"] = {"stop_reason": "max_tokens", "output_tokens": cfg.max_tokens}
            return iter(())
        info["usage"] = {"stop_reason": "end_turn"}
        return iter(["Final answer."])

    class FakeClient:
        pass

    monkeypatch.setattr(generation, "answer_stream", fake_stream)
    monkeypatch.setattr("hgrag.retrieval.search_hits", lambda *a, **k: [])
    cfg = generation.LLMConfig(provider="anthropic", model="m", api_key="k", max_tokens=100)
    res = ragbench.evaluate_query("q", {"query": "?", "source": "text", "type": "extractive"}, "A", "gold text here for the test",
                                  "p", client=FakeClient(), embedder=None, output_dir=Path("."), collection="c", k=3,
                                  stem_to_paper={}, gen_cfg=cfg, judge_cfg=None)
    assert calls == [100, 200] and res["answer"] == "Final answer." and res["retried"] and res["truncated"] is False


def test_docling_options_are_applied_and_invalidate_the_parse_cache(tmp_path, monkeypatch):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import TableFormerMode

    from hgrag import parsing

    opts = parsing.build_converter().format_to_options[InputFormat.PDF].pipeline_options
    assert opts.do_table_structure and opts.table_structure_options.mode == TableFormerMode.ACCURATE
    assert opts.table_structure_options.do_cell_matching
    assert not opts.do_code_enrichment  # CodeFormulaV2 runs on CPU (~3 min per item): opt in with HGRAG_CODE_ENRICHMENT=1
    assert not opts.do_formula_enrichment  # formulas are read by the vision model after the parsing (HGRAG_FORMULA_MODE=vlm)
    d = tmp_path / "doc"
    d.mkdir()
    (d / "document.json").write_text("{}")
    (d / "manifest.json").write_text(json.dumps({"parse_options": parsing.parse_options()}))
    assert ragbench._cached(d)
    monkeypatch.setattr(config, "FORMULA_ENRICHMENT", True)
    assert not ragbench._cached(d)  # parsed with other options: parse again
    (d / "manifest.json").write_text(json.dumps({}))  # parsed before the options were recorded
    monkeypatch.setattr(config, "FORMULA_ENRICHMENT", False)
    assert not ragbench._cached(d)


def test_offline_cache_error_gets_a_hint(tmp_path):
    def parse(pdf, describe_pictures=False):
        raise RuntimeError("outgoing traffic has been disabled. To enable repo look-ups set 'HF_HUB_OFFLINE=0'")

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    with pytest.raises(ragbench.RagBenchError, match="HF_HUB_OFFLINE=0"):
        ragbench.build_store({"a": pdf}, tmp_path / "w", output_dir=tmp_path / "o", collection="c", parse=parse)


def test_section_title_matches_gold_even_when_the_latex_differs():
    assert ragbench._norm_title("3.2.2 Response to a higher rival") == ragbench._norm_title("# 3.2.2 Response to a higher rival".lstrip("#"))
    assert ragbench._norm_title("III. RESULTS") == ragbench._norm_title(" iii. results ")
