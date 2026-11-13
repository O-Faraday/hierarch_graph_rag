"""The Databricks evaluation scores the rows like `hgrag eval`, with the search of the backend (a fake here)."""

import json

from hgrag.dbx import evaluate


def _tree(tmp_path):
    out = tmp_path / "output" / "doc1"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps({"source_pdf": "/Volumes/c/s/v/pdf/report.pdf"}), encoding="utf-8")
    tree = {"ref": "#/root", "title": "report", "elements": [], "children": [
        {"ref": "#/s1", "title": "1. Intro", "elements": [{"ref": "#/t1", "label": "text", "text": "Hello."}], "children": []},
        {"ref": "#/s2", "title": "2. Results", "elements": [{"ref": "#/t2", "label": "text", "text": "Delay is 4 days."}],
         "children": []}]}
    (out / "hierarchy.json").write_text(json.dumps(tree), encoding="utf-8")
    return tmp_path / "output"


def test_run_eval_ranks_the_expected_section_and_writes_the_local_files(tmp_path):
    output_dir = _tree(tmp_path)
    asked = []

    def search(question, k):
        asked.append((question, k))
        return [{"doc_id": "doc1", "section_ref": ref, "section_path": f"report > {title}", "score": s,
                 "source_pdf": "/Volumes/c/s/v/pdf/report.pdf", "hits": []}
                for ref, title, s in (("#/s1", "1. Intro", 0.9), ("#/s2", "2. Results", 0.5))]

    rows = [{"id": "q1", "source_kind": "paragraph", "pdf": "report.pdf", "section": "2. Results",
             "question": "What is the delay?", "expected_answer": "4 days"},
            {"id": "q2", "source_kind": "unanswerable", "pdf": None, "section": None,
             "question": "Who won?", "expected_answer": "not in the documents"}]
    seen = []
    summary = evaluate.run_eval(rows, tmp_path / "runs" / "r1", k=7, search=search, output_dir=output_dir,
                                on_row=seen.append)
    assert asked == [("What is the delay?", 7), ("Who won?", 7)]
    assert [r["rank"] for r in seen] == [2, None]
    assert summary["hit_at_1"] == 0.0 and summary["hit_at_7"] == 1.0 and summary["n_answerable"] == 1
    assert summary["config"]["backend"] == "databricks" and summary["config"]["generator"] is None
    dest = tmp_path / "runs" / "r1"
    assert {p.name for p in dest.iterdir()} == {"results.jsonl", "summary.json", "report.md"}
    assert summary["dest"] == str(dest)


def test_a_question_in_error_is_asked_once_more(tmp_path, monkeypatch):
    calls = []

    def evaluate_row(row, **_kwargs):
        calls.append(row["id"])
        failed = row["id"] == "q1" and calls.count("q1") == 1
        return {"id": row["id"], "source_kind": "paragraph", "rank": 1, "score": None, "seconds": 0.1,
                **({"error": "generation: anthropic returned HTTP 503"} if failed else {})}

    monkeypatch.setattr(evaluate.evaluation, "evaluate_row", evaluate_row)
    monkeypatch.setattr(evaluate.evaluation, "summarize", lambda results, k: {"n_errors": sum("error" in r for r in results)})
    monkeypatch.setattr(evaluate.evaluation, "_markdown", lambda *_: "")
    monkeypatch.setattr(evaluate.evaluation, "_track_row", lambda *_: None)
    rows = [{"id": "q1"}, {"id": "q2"}]
    summary = evaluate.run_eval(rows, tmp_path / "r", search=lambda *_: [], output_dir=tmp_path, retry_wait=0)
    assert calls == ["q1", "q1", "q2"] and summary["n_errors"] == 0
