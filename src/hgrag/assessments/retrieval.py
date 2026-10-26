"""Step 5 - search and reranking, per question: best reranker score, agreement of the dense and keyword searches,
and an LLM judge of the context (relevance of each source, sufficiency of the whole)."""

from __future__ import annotations

from typing import Any

from .. import config
from . import Assessment


def top_score(diagnostics: dict[str, Any], *, min_score: float | None = None) -> Assessment:
    """Best reranker score of the question. On the ragbench sample, 7 of 8 questions about something else than the
    corpus scored 0.000 and 2 of 233 questions about it scored under 0.13: a low score means the corpus probably
    does not hold the answer. Without reranking the dense similarity is reported, with no threshold."""
    min_score = config.ASSESS_MIN_TOP_SCORE if min_score is None else min_score
    score, kind = diagnostics.get("top_score"), diagnostics.get("top_score_kind")
    if score is None:
        return Assessment("retrieval.top_score", None, passed=False, rationale="no candidate found")
    if kind != "rerank":
        return Assessment("retrieval.top_score", round(score, 4), metadata={"kind": kind},
                          rationale=f"best dense similarity {score:.3f} (no reranker: no threshold)")
    return Assessment("retrieval.top_score", round(score, 4), passed=score >= min_score, metadata={"kind": kind},
                      rationale=f"best reranker score {score:.3f} (threshold {min_score})"
                                + ("" if score >= min_score else ": the corpus probably does not hold the answer"))


def search_agreement(diagnostics: dict[str, Any], *, min_agreement: float | None = None) -> Assessment:
    """Share of the candidates that the dense and the keyword searches both found. Near 0 when the question is
    about something else than the corpus (0 to 0.10 on 8 such questions); a weak signal inside it (median 0.43 when
    the reference section was found, 0.33 when it was not)."""
    min_agreement = config.ASSESS_MIN_AGREEMENT if min_agreement is None else min_agreement
    agreement = diagnostics.get("agreement")
    if agreement is None:
        return Assessment("retrieval.search_agreement", None, rationale="no keyword search (dense only)")
    meta = {k: diagnostics.get(k) for k in ("n_candidates", "n_both", "n_dense_only", "n_keyword_only")}
    return Assessment("retrieval.search_agreement", round(agreement, 3), passed=agreement >= min_agreement,
                      metadata=meta,
                      rationale=f"{meta['n_both']}/{meta['n_candidates']} candidates found by both searches "
                                f"(threshold {min_agreement})")


# ------------------------------------------------------------------------------------------- LLM judge
CONTEXT_JUDGE_SYSTEM = """You check the context that a search system gives to a language model before it answers a \
question. You receive the question and numbered sources: sections of documents, and their figures as images \
labelled with the source number. Do not answer the question, and do not use your own knowledge: judge only what the \
sources state or show.

relevance of each source, 1-5: 5 = holds part of the answer; 4 = directly about what the question asks; 3 = same \
topic, but nothing that answers; 2 = loosely related; 1 = unrelated.
sufficiency of all the sources together, 1-5: 5 = everything needed to answer fully; 4 = the main answer, a detail \
missing; 3 = part of the answer; 2 = only background; 1 = nothing that answers.

Reply with ONLY a JSON object: {"relevance": {"1": <1-5>, "2": <1-5>, ...}, "missing": "<what the sources lack to \
answer, or an empty string>", "sufficiency": <1-5>, "rationale": "<one sentence>"}"""


def judge_context(query: str, msgs: dict[str, Any], *, model: str | None = None, api_key: str | None = None,
                  send_images: bool = True) -> list[Assessment]:
    """Relevance of each source and sufficiency of the context, as the generator receives it (``msgs`` from
    ``generation.build_messages``: the sources kept by the budget and their figures)."""
    from .. import generation
    from .description import JudgeReplyError, parse_reply

    model = model or config.ASSESS_JUDGE_MODEL
    cfg = generation.LLMConfig(provider="anthropic", model=model, max_tokens=1500, send_images=send_images,
                               api_key=api_key, thinking=generation.judge_thinking(model))
    names = ("retrieval.context_relevance", "retrieval.context_sufficiency")
    if not cfg.api_key:
        return [Assessment(n, None, source="llm_judge", source_id=model, error="no API key for the judge (ANTHROPIC_API_KEY)")
                for n in names]
    if not msgs.get("sources"):
        return [Assessment(n, None if n.endswith("relevance") else 1, source="llm_judge", source_id=model,
                           passed=False, rationale="no source") for n in names]
    images = msgs["images"] if send_images else []
    content = generation._content_parts(cfg, msgs, images) if images else f"{msgs['sources_text']}\n\n{msgs['question']}"
    last: Exception | None = None
    for attempt in range(2):
        try:
            if attempt:
                content = (content + "\n\nReply with the JSON object only." if isinstance(content, str)
                           else [*content, {"type": "text", "text": "Reply with the JSON object only."}])
            reply, _usage = generation.complete(cfg, CONTEXT_JUDGE_SYSTEM, content)
            data = parse_reply(reply, ("sufficiency",))
            relevance = {int(k): int(v) for k, v in (data.get("relevance") or {}).items()}
            break
        except (JudgeReplyError, generation.GenerationError, ValueError, TypeError, AttributeError) as exc:
            last = exc
    else:
        return [Assessment(n, None, source="llm_judge", source_id=model, error=str(last)) for n in names]
    n = len(msgs["sources"])
    relevant = [i for i in range(1, n + 1) if relevance.get(i, 1) >= config.ASSESS_JUDGE_OK]
    sources = [{"n": i, "section": s.get("section_path"), "relevance": relevance.get(i)}
               for i, s in enumerate(msgs["sources"], start=1)]
    why = str(data.get("rationale", ""))
    return [
        Assessment("retrieval.context_relevance", round(len(relevant) / n, 3), source="llm_judge", source_id=model,
                   passed=bool(relevant), metadata={"sources": sources},
                   rationale=f"{len(relevant)}/{n} sources relevant (score >= {config.ASSESS_JUDGE_OK}); "
                             f"first source {relevance.get(1)}"),
        Assessment("retrieval.context_sufficiency", data["sufficiency"], source="llm_judge", source_id=model,
                   passed=data["sufficiency"] >= config.ASSESS_MIN_SUFFICIENCY, rationale=why,
                   metadata={"missing": str(data.get("missing") or "")}),
    ]


def assess(diagnostics: dict[str, Any]) -> list[Assessment]:
    """The deterministic checks (the judge needs the prompt: ``judge_context``)."""
    return [top_score(diagnostics), search_agreement(diagnostics)]
