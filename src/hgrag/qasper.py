"""QASPER (Dasigi et al., 2021, CC BY 4.0) as a source of the ragbench runner.

Information-seeking questions on NLP papers, written by readers who had seen the title and abstract only, answered by
other NLP practitioners who marked the paragraphs (or tables, figures) that support the answer. The papers are on
arXiv under their QASPER id, so the runner parses the real PDFs; the evidence comes from the dataset's own text
(S2ORC), compared with our sections by word overlap like the Open RAG Bench gold sections.

A QASPER question is asked about one paper ("Do they release code?"): the official setting gives the paper with
the question. In a corpus, such a question names no paper, so the title of the paper is put before it
(``In the paper "<title>": <question>``); the question as written is kept in ``question_original``.

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


def scoped_question(title: str | None, question: str) -> str:
    """The question aimed at its paper, as the official setting does by giving the paper with the question."""
    title = " ".join((title or "").split())
    return f'In the paper "{title}": {question}' if title else question


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
            queries[qid] = {"query": scoped_question(paper.get("title"), question), "question_original": question,
                            "type": answerable[0]["type"] if answerable else "none",
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


# ------------------------------------------------------------------------------ short answer
SHORT_SYSTEM = """You turn the answer of a question-answering system into the short answer format of a reading \
comprehension benchmark. You receive the QUESTION, the SOURCES the system read and its full ANSWER.

Reply with the short answer only, no other text:
- a yes/no question: Yes or No, the conclusion the ANSWER leans to, even when it is hedged;
- otherwise the shortest span that answers it, copied word for word from the SOURCES when it is there (a name, a \
number, a short phrase; several items separated by commas);
- Unanswerable only when the ANSWER gives nothing that answers the question. An ANSWER that names some items, gives \
a partial answer or adds that something is not stated explicitly still answers: reply with what it gives."""
SHORT_CONTEXT_CHARS = 12000
SHORT_BUDGETS = (400, 1600)  # max_tokens of the call, then of the retry when the reply comes back empty


def short_answer(cfg: Any, question: str, answer: str, sections: list[dict[str, Any]]) -> str:
    """The short answer in the QASPER format (span, Yes/No or Unanswerable), from the full answer and its sources:
    the official answer F1 compares words, so a correct answer of several sentences scores low. Thinking is off for
    Haiku as for the judges (it counts in max_tokens: half the short answers came back empty at 200 tokens), and an
    empty reply is asked once more with a larger budget."""
    import dataclasses

    from .generation import complete, judge_thinking

    cfg = dataclasses.replace(cfg, thinking=judge_thinking(cfg.model) or cfg.thinking)
    sources = "\n\n".join(f"[{s.get('section_path')}]\n{s.get('text') or ''}" for s in sections)[:SHORT_CONTEXT_CHARS]
    user = f"QUESTION:\n{question}\n\nSOURCES:\n{sources}\n\nANSWER:\n{answer}"
    text = ""
    for budget in SHORT_BUDGETS:
        reply, _usage = complete(cfg, SHORT_SYSTEM, user, max_tokens=budget)
        text = " ".join(reply.split()).strip().strip('"')
        if text:
            break
    return text


# ------------------------------------------------------------------- answer in the QASPER format
# The generator answers directly in the format of the annotators, as the published systems do (no second call to
# shorten a long answer). The four kinds follow the annotation guidelines of the dataset: yes/no, extractive span,
# free-form, unanswerable. The rules answer the error families of the 2026-11-10 report: B (an answer turned into
# "Unanswerable"), C (lists given in part) and D (related information given for a question the paper does not answer).
QASPER_SYSTEM = """You answer a question about one scientific paper, using ONLY the numbered sources: sections of \
that paper, each rebuilt with all its paragraphs, lists, tables and figures. Your answer is compared word by word \
with the answers of human annotators, so follow these rules exactly.

Read every source before answering: the answer is often spread over several sections, a table or a figure.

Reply with exactly one of these forms, and nothing else (no citation, no explanation, no introduction):
1. Yes or No: only for a closed question that yes or no answers (Do, Does, Is, Are, Can, Was, Were, Has, Have...). \
Give the side the paper supports, even when it only implies it. A question that offers a choice ("X or Y?") is not \
a yes/no question: answer with the option the paper supports.
2. A span copied word for word from the sources: the preferred form whenever the paper states the answer (a name, \
a number with its unit, a short phrase). When the question asks for several things (which metrics, which datasets, \
what approaches, what baselines, what limitations), give every item the paper mentions, from all the sources, \
separated by commas, not only the first one.
3. One short sentence in your own words: only when no span of the paper answers the question, because the answer \
must be combined from several places or inferred.
4. Unanswerable: when the paper does not contain the answer. Related information (other methods, other settings, \
what the paper does instead, what it does not show) is not an answer: reply Unanswerable.

When the question asks for results, a performance, a gain or a size (how much, how big, how many, how well, what \
results), give the numbers with what they measure, read in the tables and figures; never a vague phrase such as \
"competitive results", "a large margin" or "significant improvement". When a value is printed only on a figure, \
read it on the figure image.

Stop after the answer: no detail the question does not ask for, no comparison, no qualification."""


def clean_answer(text: str) -> str:
    """The answer as the official evaluator should read it: no citation marks, quotes or markdown; the canonical
    Yes, No and Unanswerable."""
    text = re.sub(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]", "", text or "")
    text = " ".join(text.replace("**", "").split()).strip().strip('"').strip()
    head = text.lower().rstrip(".!")
    for canonical in ("Yes", "No", "Unanswerable"):
        if head == canonical.lower() or (canonical == "Unanswerable" and head.startswith("unanswerable")):
            return canonical
    return text
