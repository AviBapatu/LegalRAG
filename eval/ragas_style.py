"""RAGAs-style generation metrics (Milestone 7, AGENTS.md §6).

Implements three normalized [0, 1] scores in the style of RAGAS:

* **faithfulness**        — how much of the generated answer is supported by
                            the retrieved context.
* **answer relevance**    — how well the generated answer addresses the query.
* **context relevance**   — how relevant the retrieved context is to the query.

This is a from-scratch, RAGAs-STYLE implementation and is NOT the official
RAGAS package (it shares neither code nor scorer state with RAGAS).

Two clearly separated scoring modes
-----------------------------------
1. **Deterministic heuristics (default)** — pure lexical overlap scoring. No
   LLM, no API key, no network. Fully deterministic and unit-testable. These
   are deliberately simple proxies and are labelled as heuristics in the
   evaluation report.

2. **LLM judge (optional)** — an injectable :class:`LLMJudge`. When supplied,
   the judge's score is used directly for that metric instead of the
   heuristic. The judge is isolated behind a narrow ``score(system=, user=)``
   interface; this module never constructs or calls a live external API by
   itself. Callers wire a real model (e.g. Groq/GPT) to the judge.

Scores are always clamped strictly to [0, 1]: an out-of-range judge response
raises ``ValueError`` instead of being silently accepted.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from retrieval.retriever import RetrievalResult

from .text_util import content_token_set, overlap_ratio, split_sentences, strip_citations

#: An answer that declines to answer does not address the query.
_I_DONT_KNOW_RE = re.compile(
    r"\bi\s*don['’]?t\s+know\b|\bi\s+do\s+not\s+know\b|\bunknown\b", re.IGNORECASE
)

#: A claim counts as supported by the context when this fraction of its
#: meaningful tokens is present in the concatenated context text.
CLAIM_SUPPORT_THRESHOLD = 0.5

#: A context chunk counts as relevant to the query when at least one
#: meaning-bearing query token appears in it.
CHUNK_RELEVANCE_MIN_OVERLAP = 0.0


class LLMJudge(Protocol):
    """A narrow 0-1 score judge.

    Implementations must return a ``float`` in [0, 1] for the given prompt.
    The evaluation report records whether an LLM judge was used.
    """

    def score(self, *, system: str, user: str) -> float: ...


def _judge_score(judge: LLMJudge, system: str, user: str) -> float:
    """Run a judge and require its output to be a normalized 0-1 float."""
    value = float(judge.score(system=system, user=user))
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"LLM judge returned out-of-range score {value!r}; expected [0, 1]"
        )
    return value


# ---------------------------------------------------------------------------
# Context normalization: evidence objects -> chunk texts
# ---------------------------------------------------------------------------


def context_texts(evidence: Sequence[object]) -> list[str]:
    """Extract the body text of evidence items (RetrievalResult/dict chunks)."""
    texts: list[str] = []
    for item in evidence or []:
        if isinstance(item, RetrievalResult):
            chunk = item.chunk or {}
        elif isinstance(item, dict):
            chunk = item.get("chunk") if isinstance(item.get("chunk"), dict) else item
        else:
            chunk = {}
        text = chunk.get("text") or chunk.get("original_text") or ""
        if str(text).strip():
            texts.append(str(text).strip())
    return texts


def _normalize_context(context: Sequence[object]) -> list[str]:
    """Return a list of plain chunk-text strings.

    A sequence of strings passes through unchanged (the friendly API), while
    evidence objects are converted via :func:`context_texts`.
    """
    if not context:
        return []
    if all(isinstance(item, str) for item in context):
        return [str(item).strip() for item in context if str(item).strip()]
    return context_texts(context)


# ---------------------------------------------------------------------------
# Deterministic heuristics
# ---------------------------------------------------------------------------


def _faithfulness_deterministic(answer: str, context: list[str]) -> float:
    claims = split_sentences(strip_citations(answer))
    if not claims:
        return 0.0
    joined_context = " ".join(context)
    supported = sum(
        1
        for claim in claims
        if overlap_ratio(claim, joined_context) >= CLAIM_SUPPORT_THRESHOLD
    )
    return supported / len(claims)


def _answer_relevance_deterministic(answer: str, query: str) -> float:
    if not str(answer).strip():
        return 0.0
    if _I_DONT_KNOW_RE.search(str(answer)):
        return 0.0
    query_tokens = content_token_set(str(query))
    if not query_tokens:
        return 0.0
    answer_tokens = set(str(answer).lower().split())
    return len(query_tokens & answer_tokens) / len(query_tokens)


def _context_relevance_deterministic(query: str, context: list[str]) -> float:
    if not context:
        return 0.0
    query_tokens = content_token_set(str(query))
    if not query_tokens:
        return 0.0
    relevant = sum(
        1
        for chunk in context
        if overlap_ratio(str(query), chunk) > CHUNK_RELEVANCE_MIN_OVERLAP
    )
    return relevant / len(context)


# ---------------------------------------------------------------------------
# LLM-judge prompts (used only when a judge is injected)
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = (
    "You are a strict factuality judge for a legal RAG system. Given a "
    "generated ANSWER and the RETRIEVED CONTEXT, rate the faithfulness of the "
    "answer: the fraction of the answer's factual claims that are fully "
    "supported by the context, as a number in [0, 1]. Contesting a claim the "
    "context supports counts as higher faithfulness; asserting facts absent "
    "from the context lowers it. Output ONLY the number, nothing else."
)

ANSWER_RELEVANCE_SYSTEM = (
    "You are a relevance judge for a legal RAG system. Given a QUERY and a "
    "generated ANSWER, rate how well the answer addresses the question asked "
    "as a number in [0, 1]. An off-topic answer scores 0; a complete, direct "
    "answer scores 1. Output ONLY the number, nothing else."
)

CONTEXT_RELEVANCE_SYSTEM = (
    "You are a retrieval-quality judge for a legal RAG system. Given a QUERY "
    "and the RETRIEVED CONTEXT, rate the relevance of the context to the query "
    "as a number in [0, 1]: the fraction of the retrieved context that is "
    "directly relevant to answering the query. Output ONLY the number, "
    "nothing else."
)


def _faithfulness_judge_prompt(answer: str, context: list[str]) -> tuple[str, str]:
    user = "\n".join(
        [
            "RETRIEVED CONTEXT:",
            "\n\n".join(f"- {text}" for text in context) or "(empty)",
            "",
            "ANSWER:",
            str(answer),
            "",
            "Faithfulness score (0-1):",
        ]
    )
    return FAITHFULNESS_SYSTEM, user


def _answer_relevance_judge_prompt(answer: str, query: str) -> tuple[str, str]:
    user = "\n".join(
        [
            "QUERY:",
            str(query),
            "",
            "ANSWER:",
            str(answer),
            "",
            "Answer relevance score (0-1):",
        ]
    )
    return ANSWER_RELEVANCE_SYSTEM, user


def _context_relevance_judge_prompt(query: str, context: list[str]) -> tuple[str, str]:
    user = "\n".join(
        [
            "QUERY:",
            str(query),
            "",
            "RETRIEVED CONTEXT:",
            "\n\n".join(f"- {text}" for text in context) or "(empty)",
            "",
            "Context relevance score (0-1):",
        ]
    )
    return CONTEXT_RELEVANCE_SYSTEM, user


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


def faithfulness(
    answer: str,
    context: Sequence[object],
    judge: LLMJudge | None = None,
) -> float:
    """Faithfulness in [0, 1]: how much of the answer the context supports.

    Args:
        answer: The generated answer text.
        context: Chunk texts, or evidence objects (RetrievalResult/dicts).
        judge: Optional ``LLMJudge``. When omitted, the deterministic
            token-overlap heuristic is used.
    """
    texts = _normalize_context(context)
    if judge is None:
        return _faithfulness_deterministic(str(answer), texts)
    system, user = _faithfulness_judge_prompt(str(answer), texts)
    return _judge_score(judge, system, user)


def answer_relevance(
    answer: str,
    query: str,
    judge: LLMJudge | None = None,
) -> float:
    """Answer relevance in [0, 1]: how well the answer addresses the query.

    Args:
        answer: The generated answer text.
        query: The user's query.
        judge: Optional ``LLMJudge``; omitted -> deterministic heuristic.
    """
    if judge is None:
        return _answer_relevance_deterministic(str(answer), str(query))
    system, user = _answer_relevance_judge_prompt(str(answer), str(query))
    return _judge_score(judge, system, user)


def context_relevance(
    query: str,
    context: Sequence[object],
    judge: LLMJudge | None = None,
) -> float:
    """Context relevance in [0, 1]: how relevant the retrieved context is.

    Args:
        query: The user's query.
        context: Chunk texts, or evidence objects (RetrievalResult/dicts).
        judge: Optional ``LLMJudge``; omitted -> deterministic heuristic.
    """
    texts = _normalize_context(context)
    if judge is None:
        return _context_relevance_deterministic(str(query), texts)
    system, user = _context_relevance_judge_prompt(str(query), texts)
    return _judge_score(judge, system, user)


def evaluate_rag_results(
    answer: str,
    query: str,
    context: Sequence[object],
    judge: LLMJudge | None = None,
) -> dict:
    """Return all three scores as ``{"faithfulness", "answer_relevance",
    "context_relevance"}``. Each is a normalized [0, 1] float.
    """
    return {
        "faithfulness": faithfulness(answer, context, judge=judge),
        "answer_relevance": answer_relevance(answer, query, judge=judge),
        "context_relevance": context_relevance(query, context, judge=judge),
    }


__all__ = [
    "CLAIM_SUPPORT_THRESHOLD",
    "CHUNK_RELEVANCE_MIN_OVERLAP",
    "LLMJudge",
    "answer_relevance",
    "context_relevance",
    "context_texts",
    "evaluate_rag_results",
    "faithfulness",
]