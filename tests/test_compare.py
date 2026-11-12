import json

import pytest

from hgrag import compare


def _run(tmp_path, name, rows, k=10):
    d = tmp_path / name
    d.mkdir()
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"k_sections": k}), encoding="utf-8")
    return d


def _row(i, paper, f1, score=4, rank=1, **extra):
    return {"id": f"q{i}", "paper": paper, "type": "extractive", "answer_f1": f1, "score": score,
            "section_rank": rank, "context_recall": 0.8, **extra}


def test_clustered_se_grows_when_a_paper_moves_all_its_questions_together():
    diffs = [0.1, 0.1, -0.1, -0.1]
    independent = compare.clustered_se(diffs, ["a", "b", "c", "d"])
    grouped = compare.clustered_se(diffs, ["a", "a", "b", "b"])
    assert grouped > independent
    assert compare.clustered_se(diffs, ["a"] * 4) is None  # one paper: no estimate
    assert compare.clustered_se([], []) is None


def test_compare_pairs_the_questions_and_says_when_a_gain_is_noise(tmp_path):
    a = _run(tmp_path, "a", [_row(i, f"p{i % 4}", 0.5) for i in range(8)])
    # 4 questions gain 0.2, 4 lose 0.2 (two whole papers each): mean 0, nothing beyond the noise
    b = _run(tmp_path, "b", [_row(i, f"p{i % 4}", 0.7 if i % 4 < 2 else 0.3) for i in range(8)])
    c = compare.compare(a, b)
    f1 = next(m for m in c["metrics"] if m["metric"].startswith("answer F1"))
    assert (f1["a"], f1["b"], f1["diff"]) == (0.5, 0.5, 0.0)
    assert (f1["better"], f1["worse"]) == (4, 4) and not f1["significant"]
    assert c["n_common"] == 8


def test_compare_flags_a_clear_gain_and_skips_unanswerable_for_search_metrics(tmp_path):
    rows_a = [_row(i, f"p{i}", 0.2, rank=None) for i in range(6)] + [_row(6, "p6", 0.0, rank=None, unanswerable=True)]
    rows_b = [_row(i, f"p{i}", 0.6 + 0.01 * i, rank=2) for i in range(6)] + [_row(6, "p6", 1.0, rank=None, unanswerable=True)]
    c = compare.compare(_run(tmp_path, "a", rows_a), _run(tmp_path, "b", rows_b))
    by = {m["metric"]: m for m in c["metrics"]}
    assert by["answer F1 (all questions)"]["n"] == 7 and by["answer F1 (all questions)"]["significant"]
    assert by["section hit@10"]["n"] == 6 and (by["section hit@10"]["a"], by["section hit@10"]["b"]) == (0.0, 1.0)
    assert by["section hit@1"]["b"] == 0.0
    assert by["MRR"]["b"] == 0.5
    assert "extractive" in c["by_type"]


def test_compare_only_uses_questions_both_runs_answered(tmp_path):
    a = _run(tmp_path, "a", [_row(1, "p1", 0.5), _row(2, "p2", 0.5), {**_row(3, "p3", 0.5), "error": "boom"}])
    b = _run(tmp_path, "b", [_row(2, "p2", 0.9), _row(3, "p3", 0.9), _row(4, "p4", 0.9)])
    c = compare.compare(a, b)
    assert (c["n_common"], c["n_only_a"], c["n_only_b"]) == (1, 1, 2)
    with pytest.raises(compare.CompareError):
        compare.compare(a, _run(tmp_path, "c", [_row(9, "p9", 0.1)]))


def test_write_comparison_leaves_a_readable_file_in_the_second_run(tmp_path):
    a = _run(tmp_path, "a", [_row(i, f"p{i}", 0.4) for i in range(4)])
    b = _run(tmp_path, "b", [_row(i, f"p{i}", 0.5) for i in range(4)])
    c, out = compare.write_comparison(a, b)
    text = out.read_text(encoding="utf-8")
    assert out == b / "comparison.md" and "B - A" in text and "answer F1" in text
    assert "within the noise" in compare.format_comparison(c) or "beyond the noise" in compare.format_comparison(c)
