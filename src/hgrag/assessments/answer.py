"""Step 7 - the generated answer: citations that point to a given source, numbers that the sources contain, a
refusal spotted; an LLM judge of its relevance to the question and its completeness (no reference answer)."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .. import config
from . import Assessment

_CITATION = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")
_NUMBER = re.compile(r"(?<![\w.^_])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.^_])[-−]?\d+(?:\.\d+)?")
_APPROX_WORDS = r"(?:~|≈|about|approximately|approx\.|around|roughly|nearly|near|close to|almost|some)"
# an approximate value, or the second bound of an approximate range ("roughly 111% to 184%")
_APPROX = re.compile(rf"{_APPROX_WORDS}\s*(?:[-−]?\d[\d.,]*\s*%?\s*(?:to|-|–|and)\s*)?$", re.I)
_REFUSAL = re.compile(
    r"\b(?:cannot|can't|can not|unable to)\s+(?:answer|determine|find|tell|say)"
    r"|\b(?:the\s+)?(?:sources?|documents?|context)\s+(?:provided\s+)?(?:do(?:es)?\s+not|don't|doesn't)\s+"
    r"(?:contain|provide|mention|include|address|discuss|give|state|cover|say|describe|explain|specify)"
    r"|\bno (?:source|document)s?\s+(?:gives?|states?|mentions?|contains?|provides?)", re.I)
_PARTIAL = re.compile(r"\b(?:but|however|although|though)\b", re.I)


def cited_sources(answer: str) -> list[int]:
    """Source numbers cited as [2], [1, 3] or [2-4], in order of appearance."""
    out: list[int] = []
    for group in _CITATION.findall(answer or ""):
        for part in group.split(","):
            span = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", part)
            if span and 0 <= int(span[2]) - int(span[1]) < 50:
                out += range(int(span[1]), int(span[2]) + 1)
            elif span:
                out += [int(span[1]), int(span[2])]
            else:
                out.append(int(part))
    return out


def first_sentence(answer: str) -> str:
    text = re.sub(r"^\W+", "", (answer or "").replace("**", "")).strip()
    end = re.search(r"(?<=[.!?])\s|\n", text)
    return text[:end.start()] if end else text


def is_refusal(answer: str) -> bool:
    """The first sentence, where the prompt asks for the direct answer, declines without announcing a partial
    answer ("The sources do not define X, but they show..." is a partial answer). A refusal may still cite the
    nearest source: on two questions about something else than the corpus, both did."""
    opening = first_sentence(answer)
    return bool(_REFUSAL.search(opening)) and not _PARTIAL.search(opening)


def citations(answer: str, n_sources: int) -> Assessment:
    """Every [n] points to one of the ``n_sources`` given sources, and an answer that does not decline cites at least
    one: a citation to a missing source is invented, an answer without any cannot be checked by the reader."""
    cited = cited_sources(answer)
    invalid = sorted({c for c in cited if not 1 <= c <= n_sources})
    if is_refusal(answer) and not cited:
        return Assessment("answer.citations", 0, passed=None, rationale="refusal: no citation expected")
    passed = bool(cited) and not invalid
    why = (f"{len(cited)} citation(s) of {len(set(cited))} source(s) out of {n_sources}"
           + (f"; not a given source: {invalid} (invented, or reference numbers of the paper copied from the text)"
              if invalid else "") + ("" if cited else "; no citation at all"))
    return Assessment("answer.citations", len(cited), passed=passed, rationale=why,
                      metadata={"cited": sorted(set(cited)), "invalid": invalid, "n_sources": n_sources})


def _canonical(token: str) -> str | None:
    try:
        return format(Decimal(token.replace(",", "").replace("−", "-")).normalize(), "f")
    except InvalidOperation:
        return None


def numbers(text: str, *, skip_approximate: bool = False) -> list[str]:
    """Numbers of a text, canonical (1,200.50 -> 1200.5), without citations and, with ``skip_approximate``, without
    those marked as approximate (the prompt asks for that mark on values read from a chart)."""
    text = _CITATION.sub(" ", (text or "").replace("**", "").replace("__", ""))
    out = []
    for m in _NUMBER.finditer(text):
        if skip_approximate and _APPROX.search(text[max(0, m.start() - 40):m.start()]):
            continue
        value = _canonical(m.group(0))
        if value is not None:
            out.append(value)
    return out


def unsourced_numbers(answer: str, context: str, *, max_unsourced: int | None = None) -> Assessment:
    """Numbers of the answer (2 digits or more, or with decimals) that appear nowhere in the context sent to the
    model: invented, converted (0.53 -> 53 %) or read from an image without being marked approximate."""
    max_unsourced = config.ASSESS_MAX_UNSOURCED_NUMBERS if max_unsourced is None else max_unsourced
    known = set(numbers(context)) | {v.lstrip("-") for v in numbers(context)}
    found = [v for v in numbers(answer, skip_approximate=True) if len(v.lstrip("-").replace(".", "")) >= 2 or "." in v]
    missing = sorted({v for v in found if v not in known and v.lstrip("-") not in known}, key=found.index)
    return Assessment("answer.unsourced_numbers", len(missing), passed=len(missing) <= max_unsourced,
                      metadata={"numbers": missing, "checked": len(set(found))},
                      rationale=f"{len(missing)} of {len(set(found))} number(s) of the answer not in the sources"
                                + (f": {', '.join(missing[:10])}" if missing else "") + f" (threshold {max_unsourced})")


def refusal(answer: str) -> Assessment:
    """The answer declines. Right when the corpus does not hold the answer (see ``retrieval.top_score``), a loss
    when the sources did hold it: an alert either way."""
    refused = is_refusal(answer)
    return Assessment("answer.refusal", refused, passed=not refused,
                      rationale=f"the answer declines: '{first_sentence(answer)[:120]}'" if refused
                      else "the answer does not decline")


# ------------------------------------------------------------------------------------------- LLM judge
ANSWER_JUDGE_SYSTEM = """You check an answer written by an assistant to a user's question. You do not know the \
right answer and you do not see the documents the assistant used: judge only how the answer responds to the question.

relevance 1-5: 5 = answers exactly what is asked; 4 = answers it with some digression; 3 = partly about something \
else; 2 = mostly about something else; 1 = does not address the question.
completeness 1-5: list the parts of the question (each thing asked, each item to compare, the yes/no when asked). \
5 = every part answered; 4 = every part, one thinly; 3 = one part unanswered; 2 = several parts unanswered; \
1 = none. A part the answer says the documents do not cover counts as unanswered.

Reply with ONLY a JSON object: {"parts": ["..."], "unanswered": ["..."], "relevance": <1-5>, \
"completeness": <1-5>, "rationale": "<one sentence>"}"""


def judge_answer(question: str, answer: str, *, model: str | None = None, api_key: str | None = None) -> list[Assessment]:
    from .. import generation
    from .description import JudgeReplyError, parse_reply

    model = model or config.ASSESS_JUDGE_MODEL
    cfg = generation.LLMConfig(provider="anthropic", model=model, max_tokens=1000, send_images=False, api_key=api_key,
                               thinking=generation.judge_thinking(model))
    names = ("answer.relevance", "answer.completeness")
    if not cfg.api_key:
        return [Assessment(n, None, source="llm_judge", source_id=model, error="no API key for the judge (ANTHROPIC_API_KEY)")
                for n in names]
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer or '(empty)'}"
    last: Exception | None = None
    for attempt in range(2):
        try:
            reply, _usage = generation.complete(cfg, ANSWER_JUDGE_SYSTEM, user + ("\n\nReply with the JSON object only."
                                                                                  if attempt else ""))
            data = parse_reply(reply, ("relevance", "completeness"))
            break
        except (JudgeReplyError, generation.GenerationError) as exc:
            last = exc
    else:
        return [Assessment(n, None, source="llm_judge", source_id=model, error=str(last)) for n in names]
    why = str(data.get("rationale", ""))
    return [
        Assessment("answer.relevance", data["relevance"], source="llm_judge", source_id=model,
                   passed=data["relevance"] >= config.ASSESS_JUDGE_OK, rationale=why),
        Assessment("answer.completeness", data["completeness"], source="llm_judge", source_id=model,
                   passed=data["completeness"] >= config.ASSESS_JUDGE_OK, rationale=why,
                   metadata={"parts": data.get("parts") or [], "unanswered": data.get("unanswered") or []}),
    ]


def assess(answer: str, msgs: dict[str, Any]) -> list[Assessment]:
    """The deterministic checks; ``msgs`` is the prompt (``generation.build_messages``) the answer was written from."""
    context = (msgs.get("sources_text") or "") + "\n" + "\n".join(i.get("caption") or "" for i in msgs.get("images") or [])
    return [citations(answer, len(msgs.get("sources") or [])), unsourced_numbers(answer, context), refusal(answer)]
