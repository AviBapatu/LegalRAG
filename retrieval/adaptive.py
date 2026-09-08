"""Adaptive retrieval loop (Milestone 6, AGENTS.md §4).

Implements the DRAG-BILQA-style controller:

    user query
        -> initial HybridRetriever retrieval
        -> generation
        -> confidence/support check
        -> if support is below threshold:
             LLM query rewrite
             -> retrieve again
             -> generate again
        -> return final result

The controller REUSES the Milestone 3 ``HybridRetriever`` unchanged (all dense
retrieval, BM25, RRF fusion, MMR reranking and structural-boost logic lives
new behavior is the loop orchestration, the deterministic support decision, and
the query-rewrite step.

Key properties:

- The initial retrieval ALWAYS happens.
- The support decision is deterministic and independently unit-testable; it
  does NOT invent a numeric confidence score. It uses signals the existing
  generation interface already produces: premise status, whether the answer
  says "I don't know", and whether any chunks were cited.
- At most ``max_extra_rounds`` extra retrieval rounds may follow the initial
  one (default 2). The loop can never run forever.
- If query rewriting fails (or the rewrite client is absent), adaptive
  retrieval stops gracefully and returns the best available result rather than
  looping or crashing.
- Complete retrieval + generation + decision information is preserved for every
  round so later evaluation (Milestone 7) can inspect exactly what happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .retriever import HybridRetriever, RetrievalResult

if TYPE_CHECKING:
    from generation.generate import GroqGenerator, GenerationResult
    from generation.prompt import PremiseVerification

# Defaults (driven by config where present).
DEFAULT_ENABLED = True
DEFAULT_SUPPORT_THRESHOLD = 1.0
DEFAULT_MAX_EXTRA_ROUNDS = 2

#: Normalized marker the Milestone 5 answer model is instructed to produce when
#: it cannot answer from the evidence. Matched case-insensitively.
_I_DONT_KNOW_RE = re.compile(
    r"\bi\s*don['’]?t\s+know\b|\bi\s+do\s+not\s+know\b|\bunknown\b", re.IGNORECASE
)

# Possible stop reasons, kept as stable strings for the result object.
REASON_SUPPORTED = "supported"
REASON_MAX_ROUNDS = "max_extra_rounds_reached"
REASON_REWRITE_FAILED = "query_rewrite_failed"
REASON_NO_REWRITER = "no_query_rewriter"


# ---------------------------------------------------------------------------
# Deterministic support decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportDecision:
    """Outcome of deciding whether another retrieval round is needed.

    Attributes:
        needs_another_round: Whether retrieval should be attempted again.
        reason: A stable short string describing why (see the REASON_*
            constants). Also used by the query rewrite to know what went wrong.
    """

    needs_another_round: bool
    reason: str

    @property
    def sufficient(self) -> bool:
        return not self.needs_another_round


def _cite_count(result: GenerationResult) -> int:
    return len(result.cited_chunk_ids or [])


def decide_support(
    generation_result: GenerationResult,
    evidence: Sequence[object],
) -> SupportDecision:
    """Deterministically decide whether a further retrieval round is needed.

    Uses only information the existing generation interface already provides;
    it does NOT compute a made-up numeric confidence score. The decision rules
    (each independently testable):

    1. A premise that is unsupported or contradicted is never treated as high
       confidence, so it always requests another retrieval round.
    2. An answer saying "I don't know" is insufficient and requests another
       round (when one is available).
    3. An answer that cites no evidence at all is considered insufficient.
    4. An answer that cites at least one chunk AND has no unsupported premise
       and does not say "I don't know" is considered sufficient, so the loop
       stops.

    ``evidence`` is accepted for future/interface symmetry and so callers pass
    the retrieval results that grounded the answer; the decision itself does
    not depend on it (the signal lives in the generation result), which keeps
    this deterministic and trivially unit-testable.
    """
    text = (generation_result.answer or "").strip()
    verification = generation_result.verification

    if verification is not None and verification.is_unsupported:
        return SupportDecision(
            needs_another_round=True, reason="unsupported_premise"
        )

    if _I_DONT_KNOW_RE.search(text):
        return SupportDecision(needs_another_round=True, reason="i_dont_know")

    if _cite_count(generation_result) == 0:
        return SupportDecision(needs_another_round=True, reason="no_citations")

    return SupportDecision(needs_another_round=False, reason=REASON_SUPPORTED)


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------


class QueryRewriter:
    """Rewrite a query to improve retrieval, using the Milestone 5 client.

    The rewrite NEVER answers the legal question itself. It only produces a
    focused search query based on (a) the original user query, (b) why the
    previous result was insufficient, and (c) the previously retrieved
    evidence/result.

    ``client`` is the same Groq-compatible abstraction (real or injected fake
    SDK) used by generation, so this is mockable and never requires a real API
    key in tests.
    """

    #: Instructs the model to output only the focused retrieval query.
    REWRITE_SYSTEM = (
        "You are a query rewriter for a legal-document retrieval system. "
        "Your only task is to rewrite the user's question into one or more "
        "focused search queries that will help retrieve the missing legal "
        "evidence. Do NOT answer the legal question. Do NOT state legal "
        "conclusions. Output ONLY the rewritten search query text, with no "
        "prelude, labels, or explanation."
    )

    def __init__(self, client):
        self.client = client

    def rewrite(
        self,
        original_query: str,
        reason: str,
        evidence: Sequence[object],
    ) -> str:
        """Produce a focused search query for the next retrieval round.

        Raises RuntimeError if the underlying client fails; the adaptive
        controller catches this and stops gracefully.
        """
        evidence_text = _format_evidence_for_rewrite(evidence)
        user = "\n".join(
            [
                "ORIGINAL QUESTION:",
                str(original_query).strip(),
                "",
                "WHY THE PREVIOUS RESULT WAS INSUFFICIENT:",
                reason,
                "",
                "PREVIOUSLY RETRIEVED EVIDENCE:",
                evidence_text or "(none)",
                "",
                "TASK:",
                "Write a single focused search query to retrieve the missing "
                "legal evidence. Output only the rewritten query.",
            ]
        )
        result = self.client.complete(
            {"system": self.REWRITE_SYSTEM, "user": user}
        )
        rewritten = (result or "").strip()
        if not rewritten:
            raise RuntimeError("query rewriter returned an empty query")
        return rewritten


def _format_evidence_for_rewrite(evidence: Sequence[object]) -> str:
    """Render evidence items for the rewrite prompt (only what's useful for
    search, never the legal answer)."""
    chunks: list[str] = []
    for item in evidence or []:
        if isinstance(item, RetrievalResult):
            cid = item.chunk_id
            text = _chunk_body(item.chunk)
            src = item.source_name
            section = item.section or (item.chunk or {}).get("section")
        else:
            mapping = item if isinstance(item, dict) else {}
            chunk = mapping.get("chunk") or mapping
            cid = chunk.get("chunk_id") or mapping.get("chunk_id")
            text = _chunk_body(chunk)
            src = chunk.get("source_name") or mapping.get("source_name")
            section = chunk.get("section") or mapping.get("section")
        label = f"[{cid}]"
        if src:
            label += f" (source: {src}"
            if section:
                label += f", section: {section}"
            label += ")"
        chunks.append(f"{label}\n{text}")
    return "\n\n".join(chunks)


def _chunk_body(chunk: dict) -> str:
    text = (chunk or {}).get("text") or (chunk or {}).get("original_text") or ""
    return str(text).strip()


# ---------------------------------------------------------------------------
# Adaptive result structures
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveRound:
    """Everything that happened in a single retrieval+generation round."""

    query: str
    retrieval_results: list[RetrievalResult]
    generation: GenerationResult
    decision: SupportDecision
    round_index: int = 0
    # True for the very first (mandatory) retrieval; False for adaptive rewrites.
    is_initial: bool = False
    # The rewritten query that produced this round (None for the initial round).
    rewritten_from: str | None = None


@dataclass
class AdaptiveResult:
    """Structured controller output, preserving full per-round detail."""

    original_query: str
    final_answer: str
    final_verification: PremiseVerification
    final_retrieval_results: list[RetrievalResult]
    rounds_used: int
    adaptive_triggered: bool
    rewritten_queries: list[str]
    rounds: list[AdaptiveRound]
    stop_reason: str
    support_threshold: float
    max_extra_rounds: int


# ---------------------------------------------------------------------------
# Adaptive retriever controller
# ---------------------------------------------------------------------------


class AdaptiveRetriever:
    """Run initial retrieval + generation, then conditionally rewrite/retrieve.

    Args:
        retriever: A Milestone 3 ``HybridRetriever`` (reused untouched).
        generator: A Milestone 5 ``GroqGenerator`` (reused untouched).
        rewriter: Optional ``QueryRewriter``. If None and an extra round is
            needed, adaptive retrieval stops gracefully (REASON_NO_REWRITER).
        max_extra_rounds: Maximum number of EXTRA retrieval rounds after the
            initial one (default 2). Bounds the loop; can never be infinite.
        support_threshold: Kept for configuration compatibility. With the
            deterministic support decision this is a nominal value; see
            ``decide_support`` for the actual deterministic rules.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GroqGenerator,
        rewriter: QueryRewriter | None = None,
        max_extra_rounds: int = DEFAULT_MAX_EXTRA_ROUNDS,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    ):
        if max_extra_rounds < 0:
            raise ValueError("max_extra_rounds must be >= 0")
        self.retriever = retriever
        self.generator = generator
        self.rewriter = rewriter
        self.max_extra_rounds = int(max_extra_rounds)
        self.support_threshold = float(support_threshold)

    def run(self, query: str) -> AdaptiveResult:
        """Execute the adaptive loop for ``query`` and return the full result."""
        rewritten_queries: list[str] = []
        rounds: list[AdaptiveRound] = []

        # Round 0: the initial retrieval ALWAYS happens.
        current_query = query.strip()
        current_evidence = self._retrieve(current_query)
        current_generation = self._generate(current_query, current_evidence)
        decision = decide_support(current_generation, current_evidence)
        rounds.append(
            AdaptiveRound(
                query=current_query,
                retrieval_results=current_evidence,
                generation=current_generation,
                decision=decision,
                round_index=0,
                is_initial=True,
            )
        )

        extra_used = 0
        stop_reason = None

        while decision.needs_another_round:
            if extra_used >= self.max_extra_rounds:
                stop_reason = REASON_MAX_ROUNDS
                break

            if self.rewriter is None:
                stop_reason = REASON_NO_REWRITER
                break

            # Rewrite the query for the next retrieval round.
            try:
                rewritten = self.rewriter.rewrite(
                    query, decision.reason, current_evidence
                )
            except Exception:
                # Graceful stop: return the best result we already have.
                stop_reason = REASON_REWRITE_FAILED
                break

            rewritten_queries.append(rewritten)
            extra_used += 1
            current_query = rewritten
            current_evidence = self._retrieve(current_query)
            current_generation = self._generate(current_query, current_evidence)
            decision = decide_support(current_generation, current_evidence)
            rounds.append(
                AdaptiveRound(
                    query=current_query,
                    retrieval_results=current_evidence,
                    generation=current_generation,
                    decision=decision,
                    round_index=extra_used,
                    is_initial=False,
                    rewritten_from=rewritten,
                )
            )

        if stop_reason is None:
            # Loop exited because the last decision was sufficient.
            stop_reason = decision.reason

        last = rounds[-1]
        return AdaptiveResult(
            original_query=query,
            final_answer=last.generation.answer,
            final_verification=last.generation.verification,
            final_retrieval_results=last.retrieval_results,
            rounds_used=len(rounds),
            adaptive_triggered=extra_used > 0,
            rewritten_queries=rewritten_queries,
            rounds=rounds,
            stop_reason=stop_reason,
            support_threshold=self.support_threshold,
            max_extra_rounds=self.max_extra_rounds,
        )

    def _retrieve(self, query: str) -> list[RetrievalResult]:
        return self.retriever.retrieve(query)

    def _generate(self, query: str, evidence: Sequence[object]) -> GenerationResult:
        return self.generator.generate(query, list(evidence))

    @classmethod
    def from_config(
        cls,
        retriever: HybridRetriever,
        generator: GroqGenerator,
        rewriter: QueryRewriter | None = None,
        adaptive_config: dict | None = None,
    ) -> "AdaptiveRetriever":
        """Build a controller from the ``adaptive`` config block.

        Missing keys fall back to sensible defaults. ``max_extra_rounds`` and
        ``support_threshold`` are the behavioral settings here; ``enabled``
        (in config) tells the caller whether to use this controller at all,
        which is a call-site concern, not a loop concern.
        """
        cfg = adaptive_config or {}
        return cls(
            retriever=retriever,
            generator=generator,
            rewriter=rewriter,
            max_extra_rounds=int(cfg.get("max_extra_rounds", DEFAULT_MAX_EXTRA_ROUNDS)),
            support_threshold=float(
                cfg.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD)
            ),
        )


__all__ = [
    "AdaptiveRetriever",
    "AdaptiveResult",
    "AdaptiveRound",
    "QueryRewriter",
    "SupportDecision",
    "decide_support",
]
