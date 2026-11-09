"""QASPER (Dasigi et al., 2021, CC BY 4.0) as a source of the ragbench runner.

Information-seeking questions on NLP papers, written by readers who had seen the title and abstract only, answered by
other NLP practitioners who marked the paragraphs (or tables, figures) that support the answer. The papers are on
arXiv under their QASPER id, so the runner parses the real PDFs; the evidence comes from the dataset's own text
(S2ORC), compared with our sections by word overlap like the Open RAG Bench gold sections.

The source exposes the same files as the Open RAG Bench one (``queries.json``, ``qrels.json``, ``answers.json``,
``pdf_urls.json``, ``corpus/<paper>.json``): the "gold section" of a question is its evidence, the paragraphs of all
annotators joined. Unanswerable questions (every annotator said so) have no evidence: they are judged on abstention
only. ``answer_f1`` is the official token F1 (SQuAD normalisation, best annotator); it favours short answers, so it
is low for the long answers of the generator and only comparable between our runs.

Splits: ``dev`` (281 papers, default: the one to tune the pipeline on) and ``test`` (416 papers, kept for the final
measure, as the dataset README asks).
"""

from __future__ import annotations

import io
import json
import re
import string
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import requests

ARCHIVES = {"dev": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz", "qasper-dev-v0.3.json"),
            "test": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
                     "qasper-test-v0.3.json")}
FLOAT = "FLOAT SELECTED"
NO_ANSWER = "The documents do not say: no answer can be given."


class QasperError(RuntimeError):
    pass


def references(qa: dict[str, Any]) -> list[dict[str, Any]]:
    """The answer of each annotator, typed as in the official evaluator: extractive, abstractive, boolean or none."""
    out = []
    for annotation in qa.get("answers") or []:
        a = annotation["answer"]
        if a.get("unanswerable"):
            out.append({"answer": "Unanswerable", "type": "none", "evidence": []})
        elif a.get("extractive_spans"):
            out.append({"answer": ", ".join(a["extractive_spans"]), "type": "extractive", "evidence": a.get("evidence") or []})
        elif a.get("free_form_answer"):
            out.append({"answer": a["free_form_answer"], "type": "abstractive", "evidence": a.get("evidence") or []})
        elif a.get("yes_no") is not None:
            out.append({"answer": "Yes" if a["yes_no"] else "No", "type": "boolean", "evidence": a.get("evidence") or []})
    return out


def _evidence(refs: list[dict[str, Any]]) -> list[str]:
    """The evidence paragraphs of all annotators, each once, in order; a table or figure is kept as its caption."""
    seen: dict[str, None] = {}
    for ref in refs:
        for text in ref["evidence"]:
            text = re.sub(rf"^{FLOAT}:\s*", "", text.strip())
            if text:
                seen.setdefault(text)
    return list(seen)


def expected_answer(refs: list[dict[str, Any]]) -> str:
    """The reference given to the judge: the answer of the annotators (all of them when they differ)."""
    answers = list(dict.fromkeys(r["answer"].strip() for r in refs if r["type"] != "none"))
    if not answers:
        return NO_ANSWER
    if len(answers) == 1:
        return answers[0]
    return "Any of these annotator answers is correct: " + " | ".join(answers)


def build_files(papers: dict[str, Any]) -> dict[str, Any]:
    """The source files of the runner, from the QASPER JSON (``{paper id: {title, full_text, qas...}}``)."""
    queries: dict[str, Any] = {}
    qrels: dict[str, Any] = {}
    answers: dict[str, str] = {}
    corpus: dict[str, Any] = {}
    for pid, paper in papers.items():
        sections = []
        for qa in paper.get("qas") or []:
            question = (qa.get("question") or "").strip()
            refs = references(qa)
            if not question or not refs:
                continue
            answerable = [r for r in refs if r["type"] != "none"]
            evidence = _evidence(answerable)
            qid = qa["question_id"]
            queries[qid] = {"query": question, "type": answerable[0]["type"] if answerable else "none",
                            "source": "figure-table" if any(FLOAT in t for r in answerable for t in r["evidence"]) else "text",
                            "unanswerable": not answerable, "references": [r["answer"] for r in refs]}
            qrels[qid] = {"doc_id": pid, "section_id": len(sections)}
            sections.append({"text": "\n\n".join(evidence)})
            answers[qid] = expected_answer(refs)
        corpus[pid] = {"id": pid, "title": paper.get("title"), "sections": sections}
    return {"queries.json": queries, "qrels.json": qrels, "answers.json": answers,
            "pdf_urls.json": {pid: f"https://arxiv.org/pdf/{pid}" for pid in papers},
            **{f"corpus/{pid}.json": doc for pid, doc in corpus.items()}}


class QasperSource:
    """The QASPER split as the files of a ragbench source, downloaded once (the archive) and cached as JSON."""

    negatives_have_questions = True  # every QASPER paper has questions: negatives are papers outside the sample
    name = "QASPER"

    def __init__(self, cache: Path, split: str = "dev"):
        if split not in ARCHIVES:
            raise QasperError(f"unknown QASPER split {split!r} (choose from {sorted(ARCHIVES)})")
        self.cache = Path(cache)
        self.split = split
        self._files: dict[str, Any] | None = None

    def _papers(self) -> dict[str, Any]:
        url, member = ARCHIVES[self.split]
        path = self.cache / member
        if not path.exists():
            resp = requests.get(url, timeout=300)
            if resp.status_code >= 400:
                raise QasperError(f"cannot fetch {url}: HTTP {resp.status_code}")
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
                f = tar.extractfile(member)
                if f is None:
                    raise QasperError(f"{member} is not in {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f.read())
        return json.loads(path.read_text(encoding="utf-8"))

    def json(self, name: str) -> Any:
        if self._files is None:
            self._files = build_files(self._papers())
        if name not in self._files:
            raise QasperError(f"{name} is not part of the QASPER {self.split} split")
        return self._files[name]


# ----------------------------------------------------------------------- official answer metric
def _normalize(text: str) -> str:
    """SQuAD normalisation, as in the QASPER evaluator: lower case, no punctuation, no articles, single spaces."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _normalize(prediction).split(), _normalize(reference).split()
    same = sum((Counter(pred) & Counter(ref)).values())
    if not same:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, refs: list[str]) -> float:
    """Best token F1 over the annotators, as the official Answer F1."""
    return max((token_f1(prediction, r) for r in refs), default=0.0)
