"""Shared fixtures/tools for the Milestone 8 app tests.

Everything here is offline and deterministic: fake retrievers/generators with
no Groq client, no API key, no embedding-model download, and no server.
"""

from __future__ import annotations

from generation.generate import GenerationResult
from generation.prompt import PremiseVerification
from retrieval.adaptive import (
    AdaptiveResult,
    AdaptiveRound,
    QueryRewriter,
    SupportDecision,
)
from retrieval.retriever import RetrievalResult


def sample_retrieval_result(
    chunk_id: str = "chunk-1",
    source_name: str = "nda",
    text: str = (
        "Confidential Information shall not be disclosed to third parties "
        "without prior written consent."
    ),
    section: str = "2",
    parent: str = "1",
    heading: str = "2",
    hierarchy_path: list[str] | None = None,
    fused_score: float = 0.9,
    dense_score: float = 0.8,
    bm25_score: float = 0.7,
) -> RetrievalResult:
    hierarchy_path = hierarchy_path or ([parent, section] if parent else [section])
    chunk = {
        "chunk_id": chunk_id,
        "source_name": source_name,
        "text": text,
        "original_text": text,
        "section": section,
        "parent": parent,
        "heading": heading,
        "hierarchy_path": hierarchy_path,
    }
    return RetrievalResult(
        chunk_id=chunk_id,
        source_name=source_name,
        chunk=chunk,
        fused_score=fused_score,
        dense_score=dense_score,
        bm25_score=bm25_score,
        section=section,
        parent=parent,
        heading=heading,
        hierarchy_path=hierarchy_path,
    )


class FakeRetriever:
    """Deterministic retriever returning preset chunks."""

    def __init__(self, results: list[RetrievalResult] | None = None):
        self.results = results or [sample_retrieval_result()]
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.queries.append(query)
        return [r for r in self.results]


class FailingRetriever:
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        raise RuntimeError("retrieval exploded")


class FakeGenerator:
    """Deterministic generator: returns a grounded answer citing chunk-1."""

    def __init__(
        self,
        answer: str | None = None,
        verification: PremiseVerification | None = None,
        cited: list[str] | None = None,
    ):
        self.answer = answer or (
            "The NDA prohibits disclosure to third parties. [citation:chunk-1]"
        )
        self.verification = verification or PremiseVerification(status="supported")
        self.cited = cited or ["chunk-1"]
        self.queries: list[str] = []

    def generate(self, query: str, evidence=None) -> GenerationResult:
        self.queries.append(query)
        return GenerationResult(
            answer=self.answer,
            query=query,
            verification=self.verification,
            cited_chunk_ids=list(self.cited),
            prompt={},
            model="fake-model",
            temperature=0.2,
            max_tokens=100,
        )


class FailingGenerator:
    def generate(self, query: str, evidence=None):
        raise RuntimeError("generation exploded")


class FakeRewriter:
    def __init__(self, rewritten: str = "rewritten clearer query"):
        self.rewritten = rewritten
        self.calls: list[tuple[str, str]] = []

    def rewrite(self, original_query: str, reason: str, evidence=None) -> str:
        self.calls.append((original_query, reason))
        return self.rewritten


class FailingRewriter:
    def rewrite(self, original_query: str, reason: str, evidence=None) -> str:
        raise RuntimeError("rewrite exploded")


class FakeAdaptive:
    """Deterministic adaptive controller backed by fake collaborators."""

    def __init__(
        self,
        retriever=None,
        generator=None,
        rewriter: QueryRewriter | None = None,
        max_extra_rounds: int = 2,
        support_threshold: float = 1.0,
        rounds: int = 1,
        adaptive_triggered: bool = False,
        rewritten_queries: list[str] | None = None,
        stop_reason: str = "supported",
        final_evidence: list[RetrievalResult] | None = None,
    ):
        self.retriever = retriever or FakeRetriever()
        self.generator = generator or FakeGenerator()
        self.rewriter = rewriter
        self.max_extra_rounds = max_extra_rounds
        self.support_threshold = support_threshold
        self._rounds = rounds
        self._adaptive_triggered = adaptive_triggered
        self._rewritten_queries = rewritten_queries or []
        self._stop_reason = stop_reason
        self._final_evidence = final_evidence
        self.run_called_with: list[str] = []

    def run(self, query: str) -> AdaptiveResult:
        self.run_called_with.append(query)
        evidence = (
            self._final_evidence
            if self._final_evidence is not None
            else self.retriever.retrieve(query)
        )
        generation = self.generator.generate(query, evidence)
        verification = generation.verification
        rounds: list[AdaptiveRound] = []
        for i in range(self._rounds):
            decision = SupportDecision(
                needs_another_round=(i < self._rounds - 1),
                reason=("supported" if i == self._rounds - 1 else "no_citations"),
            )
            rounds.append(
                AdaptiveRound(
                    query=query if i == 0 else self._rewritten_queries[i - 1],
                    retrieval_results=evidence,
                    generation=generation,
                    decision=decision,
                    round_index=i,
                    is_initial=(i == 0),
                    rewritten_from=(None if i == 0 else self._rewritten_queries[i - 1]),
                )
            )
        last = rounds[-1]
        return AdaptiveResult(
            original_query=query,
            final_answer=generation.answer or "",
            final_verification=verification,
            final_retrieval_results=evidence,
            rounds_used=self._rounds,
            adaptive_triggered=self._adaptive_triggered,
            rewritten_queries=list(self._rewritten_queries),
            rounds=rounds,
            stop_reason=self._stop_reason,
            support_threshold=self.support_threshold,
            max_extra_rounds=self.max_extra_rounds,
        )


class FailingAdaptive:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc or RuntimeError("adaptive exploded")
        self.max_extra_rounds = 2
        self.support_threshold = 1.0

    def run(self, query: str):
        raise self.exc