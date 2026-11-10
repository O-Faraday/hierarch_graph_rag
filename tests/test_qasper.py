"""QASPER as a ragbench source: its JSON turned into the runner's files, the official answer F1, a run (no network)."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from hgrag import qasper, ragbench
from test_eval import _fake_llm, offline  # noqa: F401
from test_ragbench import bench  # noqa: F401
from test_retrieval import BagOfWordsEmbedder


def _qa(qid: str, question: str, *answers: dict) -> dict:
    return {"question": question, "question_id": qid,
            "answers": [{"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": None, "free_form_answer": "",
                                    "evidence": [], "highlighted_evidence": [], **a}} for a in answers]}


def _papers(evidence: dict[str, list[str]]) -> dict:
    """Three papers in the QASPER JSON format, their evidence taken from the control documents."""
    return {
        "2401.0001": {"title": "Aurora", "full_text": [], "qas": [
            _qa("q1", "three new branches in the Alsace region",
                {"extractive_spans": ["three new branches"], "evidence": evidence["2401.0001"]},
                {"extractive_spans": ["three"], "evidence": evidence["2401.0001"]}),
            _qa("q4", "Who is the chief executive of the bank?", {"unanswerable": True}, {"unanswerable": True})]},
        "2401.0003": {"title": "Netops", "full_text": [], "qas": [
            _qa("q2", "failover to the secondary datacentre within ninety seconds", {"yes_no": True, "evidence": evidence["2401.0003"]})]},
        "2401.0002": {"title": "Botany", "full_text": [], "qas": [
            _qa("q3", "cushion plants colonise ground that moves with frost",
                {"free_form_answer": "Cushion plants.",
                 "evidence": [*evidence["2401.0002"], "FLOAT SELECTED: Figure 2. Species per hiking day."]})]},
    }


def test_build_files_types_answers_and_evidence():
    files = qasper.build_files(_papers({"2401.0001": ["Alsace text."], "2401.0003": ["Failover text."],
                                        "2401.0002": ["Screes text."]}))
    q, rel, ans = files["queries.json"], files["qrels.json"], files["answers.json"]
    assert {k: (v["type"], v["source"], v["unanswerable"]) for k, v in q.items()} == {
        "q1": ("extractive", "text", False), "q4": ("none", "text", True), "q2": ("boolean", "text", False),
        "q3": ("abstractive", "figure-table", False)}
    assert ans["q1"] == "Any of these annotator answers is correct: three new branches | three"
    assert ans["q2"] == "Yes" and ans["q4"] == qasper.NO_ANSWER
    assert q["q1"]["references"] == ["three new branches", "three"]
    corpus = files["corpus/2401.0002.json"]["sections"]
    assert corpus[rel["q3"]["section_id"]]["text"] == "Screes text.\n\nFigure 2. Species per hiking day."
    assert files["corpus/2401.0001.json"]["sections"][rel["q4"]["section_id"]]["text"] == ""  # no evidence
    assert files["pdf_urls.json"]["2401.0002"] == "https://arxiv.org/pdf/2401.0002"
    # the question is aimed at its paper by the title, as the official setting gives the paper with it
    assert q["q4"]["query"] == 'In the paper "Aurora": Who is the chief executive of the bank?'
    assert q["q4"]["question_original"] == "Who is the chief executive of the bank?"
    assert qasper.scoped_question("  A  long\n title ", "Why?") == 'In the paper "A long title": Why?'
    assert qasper.scoped_question(None, "Why?") == "Why?"


def test_answer_f1_is_the_official_token_f1():
    assert qasper.token_f1("The three new branches.", "three new branches") == 1.0
    assert qasper.token_f1("an apple", "orange") == 0.0
    assert qasper.token_f1("three branches in Alsace", "three new branches") == pytest.approx(2 * (2 / 4) * (2 / 3) / (2 / 4 + 2 / 3))
    assert qasper.answer_f1("three", ["three new branches", "three"]) == 1.0  # best annotator


def test_source_downloads_the_split_once(tmp_path, monkeypatch):
    raw = json.dumps(_papers({"2401.0001": ["a"], "2401.0003": ["b"], "2401.0002": ["c"]})).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("qasper-dev-v0.3.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    calls = []

    class R:
        status_code, content = 200, buf.getvalue()

    monkeypatch.setattr(qasper.requests, "get", lambda url, **kw: calls.append(url) or R())
    src = qasper.QasperSource(tmp_path, "dev")
    assert set(src.json("queries.json")) == {"q1", "q2", "q3", "q4"}
    assert qasper.QasperSource(tmp_path, "dev").json("qrels.json")["q2"]["doc_id"] == "2401.0003"
    assert calls == [qasper.ARCHIVES["dev"][0]]  # the second source reads the cached JSON
    with pytest.raises(qasper.QasperError):
        qasper.QasperSource(tmp_path, "train")


class _FakeQasper:
    negatives_have_questions = True

    def __init__(self, files):
        self.files = files

    def json(self, name):
        return self.files[name]


def test_run_on_qasper_questions(bench, monkeypatch):  # noqa: F811
    work, orb, parse = bench
    evidence = {pid: [s["text"] for s in orb.files[f"corpus/{pid}.json"]["sections"] if s["text"] != "x"]
                for pid in ("2401.0001", "2401.0002", "2401.0003")}
    papers = {pid: {**p, "title": None} for pid, p in _papers(evidence).items()}  # titles would sway the toy embedder
    src = _FakeQasper(qasper.build_files(papers))
    monkeypatch.setattr(ragbench, "fetch_pdfs", lambda ids, urls, d, **kw: ({p: d / f"{p}.pdf" for p in ids}, {}))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _fake_llm(monkeypatch)
    res = ragbench.run(3, k=5, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False,
                       dataset="qasper", generate=True)
    s = res["summary"]
    assert s["n"] == 3 and s["section_hit_at_1"] == 1.0, res["results"]  # the unanswerable question is not counted
    assert s["unanswerable"]["n"] == 1 and "pct_abstained" in s["unanswerable"]
    assert s["by_source"]["figure-table"]["n"] == 1 and set(s["by_type"]) == {"extractive", "boolean", "abstractive"}
    assert 0 < s["answer_f1"] <= 1 and s["config"]["dataset"] == "qasper" and s["config"]["key"].startswith("qasper-dev_")
    assert (res["dir"] / "report.md").read_text(encoding="utf-8").startswith("# QASPER")
    unanswerable = next(r for r in res["results"] if r["id"] == "q4")
    assert unanswerable["unanswerable"] and "answer_f1_long" not in unanswerable  # no full-answer F1 without a reference
    assert all("short_answer" in r and r["answer_f1"] is not None for r in res["results"])  # every question, as official
    assert {"answer_f1_answerable", "answer_f1_long"} <= set(s)


def test_unknown_dataset_is_refused(tmp_path):
    with pytest.raises(ragbench.RagBenchError, match="unknown dataset"):
        ragbench.run(1, workdir=tmp_path, dataset="squad", track=False)


def test_paper_scope_searches_only_the_paper_of_the_question(bench, monkeypatch):  # noqa: F811
    work, orb, parse = bench
    evidence = {pid: [s["text"] for s in orb.files[f"corpus/{pid}.json"]["sections"] if s["text"] != "x"]
                for pid in ("2401.0001", "2401.0002", "2401.0003")}
    src = _FakeQasper(qasper.build_files(_papers(evidence)))  # titles kept: the paper scope asks the question as written
    monkeypatch.setattr(ragbench, "fetch_pdfs", lambda ids, urls, d, **kw: ({p: d / f"{p}.pdf" for p in ids}, {}))
    res = ragbench.run(3, k=5, workdir=work, source=src, embedder=BagOfWordsEmbedder(), parse=parse, track=False,
                       dataset="qasper", scope="paper")
    assert res["summary"]["config"]["scope"] == "paper" and res["summary"]["doc_hit_at_1"] == 1.0
    for r in res["results"]:
        assert {h["paper"] for h in r["retrieved"]} == {r["paper"]}, r["id"]
        assert not r["question"].startswith("In the paper")
    with pytest.raises(ragbench.RagBenchError, match="unknown scope"):
        ragbench.run(3, workdir=work, source=src, track=False, dataset="qasper", scope="everything")


def test_short_answer_is_asked_with_the_sources_and_cleaned(monkeypatch):
    from hgrag import generation
    from test_generation import FakeResponse, _anthropic_stream

    calls = []
    replies = ["", '  "Yes"  ']  # Haiku spent its budget thinking: empty, then the retry answers

    def post(url, **kw):
        calls.append(kw["json"])
        return FakeResponse(_anthropic_stream([replies.pop(0)]))

    monkeypatch.setattr(generation.requests, "post", post)
    cfg = generation.LLMConfig(provider="anthropic", model="claude-haiku-5-5", api_key="k")
    out = qasper.short_answer(cfg, "Do they release code?", "Yes, the code is on GitHub [1].",
                              [{"section_path": "p > 7 Conclusion", "text": "The code is available."}])
    assert out == "Yes"
    user = calls[0]["messages"][0]["content"]
    assert "QUESTION:\nDo they release code?" in user and "[p > 7 Conclusion]\nThe code is available." in user
    assert calls[0]["system"].startswith("You turn the answer")
    assert [c["max_tokens"] for c in calls] == list(qasper.SHORT_BUDGETS)
    assert all(c["thinking"] == {"type": "disabled"} for c in calls)  # thinking would eat the budget
    assert qasper.answer_f1(out, ["Yes", "Unanswerable"]) == 1.0


def test_short_answer_keeps_partial_and_hedged_answers():
    # 13 answers judged right were turned into "Unanswerable" because they started with a caveat
    assert "Unanswerable only when the ANSWER gives nothing" in qasper.SHORT_SYSTEM
    assert "even when it is hedged" in qasper.SHORT_SYSTEM
