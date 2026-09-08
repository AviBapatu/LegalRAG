"""Failure-point tagging for RAG responses (Milestone 7, AGENTS.md §6).

Classifies a failed RAG response into the survey's seven failure points
(Hindi et al., Table 10 / Fig. 7: missing content, missed top-K, not in
context, not extracted, incorrect specificity, incomplete, ...) PLUS the
eighth category introduced by this project: an UNSUPPORTED PREMISE caught by
the Milestone 5 premise-verification step (§5).

The survey's seven failure points map onto these stable tag names:

* ``retrieval_failure``            — retrieval missed the relevant content:
                                     nothing retrieved, or the expected source
                                     document is absent from the top-k
                                     (survey: "missed top-K"/"missing content").
* ``context_integration_failure``  — the right document WAS retrieved but the
                                     answer does not actually draw on it
                                     (survey: "not in context").
* ``generation_failure``           — the generator produced an empty/uninfor-
                                     mative answer despite retrieved material
                                     (survey: "not extracted"/"incomplete").
* ``hallucination``                — the answer asserts claims the retrieved
                                     evidence does not support or contradicts.
* ``citation_failure``             — citations are missing while substantive
                                     claims are made, or citations reference
                                     chunks that were never retrieved.
* ``efficiency_failure``           — the response required too many adaptive
                                     retrieval rounds (survey: efficiency).
* ``interpretability_failure``     — retrieved chunks carried no section/
                                     hierarchy metadata, so the answer cannot
                                     be traced to structural context.
* ``unsupported_premise``          — (the project's 8th) the query's factual/
                                     legal premise is unsupported or
                                     contradicted by the evidence, per the
                                     Milestone 5 premise-verification step.

Tags are STRUCTURED, not free-form strings: each tag is a dataclass carrying
the tag id, a human label, the reason it fired, and the concrete evidence
(signals) that triggered it. A response may carry multiple tags, and tagging
is deterministic and purely rule-based over a normalized RAG-result record, so
it is unit-testable without any external API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from generation.prompt import (
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
)

from . import claim_check

TAG_RETRIEVAL_FAILURE = "retrieval_failure"
TAG_CONTEXT_INTEGRATION_FAILURE = "context_integration_failure"
TAG_GENERATION_FAILURE = "generation_failure"
TAG_HALLUCINATION = "hallucination"
TAG_CITATION_FAILURE = "citation_failure"
TAG_EFFICIENCY_FAILURE = "efficiency_failure"
TAG_INTERPRETABILITY_FAILURE = "interpretability_failure"
TAG_UNSUPPORTED_PREMISE = "unsupported_premise"

ALL_FAILURE_POINTS = (
    TAG_RETRIEVAL_FAILURE,
    TAG_CONTEXT_INTEGRATION_FAILURE,
    TAG_GENERATION_FAILURE,
    TAG_HALLUCINATION,
    TAG_CITATION_FAILURE,
    TAG_EFFICIENCY_FAILURE,
    TAG_INTERPRETABILITY_FAILURE,
    TAG_UNSUPPORTED_PREMISE,
)

STOP_REASON_MAX_ROUNDS = "max_extra_rounds_reached"

_I_DONT_KNOW_RE = re.compile(
    r"\bi\s*don['’]?t\s+know\b|\bi\s+do\s+not\s+know\b|\bunknown\b", re.IGNORECASE
)

#: Stable human labels for the eight categories.
FAILURE_POINT_LABELS = {
    TAG_RETRIEVAL_FAILURE: "Retrieval failure (missed top-K / missing content)",
    TAG_CONTEXT_INTEGRATION_FAILURE: "Context integration failure (not in context)",
    TAG_GENERATION_FAILURE: "Generation failure (not extracted / incomplete)",
    TAG_HALLUCINATION: "Hallucination (ungrounded claims)",
    TAG_CITATION_FAILURE: "Citation failure",
    TAG_EFFICIENCY_FAILURE: "Efficiency failure (too many retrieval rounds)",
    TAG_INTERPRETABILITY_FAILURE: "Interpretability failure (no structural trace)",
    TAG_UNSUPPORTED_PREMISE: "Unsupported premise",
}


@dataclass(frozen=True)
class FailureTag:
    """A structured failure-point tag.

    Attributes:
        tag: One of the ``TAG_*`` constants (also in ``ALL_FAILURE_POINTS``).
        reason: Why the tag fired, in plain language.
        evidence: The concrete signals that triggered the tag (retrieved
            sources, cited ids, claim ids, round counts, ...).
    """

    tag: str
    reason: str
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        """JSON-serializable form: ``{"tag", "label", "reason", "evidence"}``."""
        return {
            "tag": self.tag,
            "label": FAILURE_POINT_LABELS[self.tag],
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


def failure_point_label(tag: str) -> str:
    """Return the human label for a failure point (its id if unknown)."""
    return FAILURE_POINT_LABELS.get(tag, tag)


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------


def _normalize_verification(verification: object) -> dict:
    """Turn a PremiseVerification (or dict) into ``{"status", ...}``."""
    if verification is None:
        return {}
    if isinstance(verification, Mapping):
        return dict(verification)
    status = getattr(verification, "status", None)
    return {
        "status": status,
        "premises": list(getattr(verification, "premises", None) or []),
        "explanation": getattr(verification, "explanation", None) or "",
    }


def normalize_result(
    query_id: str,
    query: str,
    expected_document: str | None,
    retrieved: Sequence[object],
    answer: str | None,
    cited_chunk_ids: Sequence[str] | None,
    verification: object = None,
    rounds_used: int | None = None,
    stop_reason: str | None = None,
    adaptive_triggered: bool | None = None,
) -> dict:
    """Build the normalized RAG-result record the tagger consumes.

    ``retrieved`` may be Milestone 3 ``RetrievalResult`` objects or plain
    result/chunk dicts; each is normalized to a serializable chunk dict that
    preserves structural metadata for the interpretability check.
    """
    chunks: list[dict] = []
    for item in retrieved or []:
        if hasattr(item, "chunk_id"):  # RetrievalResult
            chunk = getattr(item, "chunk", None) or {}
            entry = {
                "chunk_id": str(getattr(item, "chunk_id") or chunk.get("chunk_id") or ""),
                "source_name": str(
                    getattr(item, "source_name") or chunk.get("source_name") or ""
                ),
                "section": getattr(item, "section", None) or chunk.get("section"),
                "parent": getattr(item, "parent", None) or chunk.get("parent"),
                "heading": getattr(item, "heading", None) or chunk.get("heading"),
                "hierarchy_path": list(
                    getattr(item, "hierarchy_path", None) or chunk.get("hierarchy_path") or []
                ),
                "text": str(
                    chunk.get("text") or chunk.get("original_text") or ""
                ),
            }
        else:
            mapping = item if isinstance(item, Mapping) else {}
            chunk = mapping.get("chunk") if isinstance(mapping.get("chunk"), Mapping) else mapping
            entry = {
                "chunk_id": str(chunk.get("chunk_id") or mapping.get("chunk_id") or ""),
                "source_name": str(
                    chunk.get("source_name") or mapping.get("source_name") or ""
                ),
                "section": chunk.get("section") or mapping.get("section"),
                "parent": chunk.get("parent") or mapping.get("parent"),
                "heading": chunk.get("heading") or mapping.get("heading"),
                "hierarchy_path": list(chunk.get("hierarchy_path") or mapping.get("hierarchy_path") or []),
                "text": str(chunk.get("text") or chunk.get("original_text") or ""),
            }
        chunks.append(entry)

    return {
        "query_id": query_id,
        "query": str(query),
        "expected_document": expected_document,
        "retrieved_sources": [c["source_name"] for c in chunks],
        "retrieved_chunks": chunks,
        "answer": str(answer or ""),
        "cited_chunk_ids": list(cited_chunk_ids or []),
        "verification": _normalize_verification(verification),
        "rounds_used": rounds_used,
        "stop_reason": stop_reason,
        "adaptive_triggered": adaptive_triggered,
    }


def _evidence_signature(record: dict) -> list[str]:
    """Small edge-free textual handles used as tag ``evidence`` entries."""
    signatures: list[str] = []
    sources = record.get("retrieved_sources") or []
    if sources:
        signatures.append("retrieved_sources=" + ", ".join(sources))
    cited = record.get("cited_chunk_ids") or []
    if cited:
        signatures.append("cited_chunk_ids=" + ", ".join(cited))
    rounds = record.get("rounds_used")
    if rounds is not None:
        signatures.append(f"rounds_used={rounds}")
    if record.get("stop_reason"):
        signatures.append(f"stop_reason={record['stop_reason']}")
    return signatures


def _chunk_has_hierarchy(chunk: dict) -> bool:
    return bool(
        chunk.get("section")
        or chunk.get("parent")
        or chunk.get("heading")
        or chunk.get("hierarchy_path")
    )


def _is_unknown_answer(answer: str) -> bool:
    return bool(_I_DONT_KNOW_RE.search(answer))


# ---------------------------------------------------------------------------
# Tagging (deterministic rule set)
# ---------------------------------------------------------------------------


def tag_result(record: dict, precomputed_claims=None) -> list[FailureTag]:
    """Classify a normalized RAG-result record into failure-point tags.

    Args:
        record: Output of :func:`normalize_result` (or anything with the same
            keys).
        precomputed_claims: Optional ``claim_check.ClaimLevelReport`` to reuse
            for the hallucination check (avoids recomputing it when the caller
            already ran claim-level faithfulness).

    Returns:
        A list of :class:`FailureTag`. A perfect response yields ``[]``; a
        response can legitimately yield several tags.
    """
    tags: list[FailureTag] = []
    evidence = _evidence_signature(record)

    expected = record.get("expected_document")
    sources = list(record.get("retrieved_sources") or [])
    chunks = list(record.get("retrieved_chunks") or [])
    answer = str(record.get("answer") or "").strip()
    cited = list(record.get("cited_chunk_ids") or [])
    verification = record.get("verification") or {}

    retrieved_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
    source_of_cited = {
        c.get("chunk_id"): c.get("source_name")
        for c in chunks
        if c.get("chunk_id") and c.get("source_name")
    }

    # 1. Retrieval failure (missed relevant content / nothing retrieved).
    if not sources:
        tags.append(
            FailureTag(TAG_RETRIEVAL_FAILURE, "no chunks were retrieved", evidence)
        )
    elif expected and expected not in sources:
        tags.append(
            FailureTag(
                TAG_RETRIEVAL_FAILURE,
                f"expected document {expected!r} is absent from the retrieved top-k",
                evidence,
            )
        )

    # 7. Interpretability failure (no structural hierarchy on retrieved chunks).
    if chunks and all(not _chunk_has_hierarchy(c) for c in chunks):
        tags.append(
            FailureTag(
                TAG_INTERPRETABILITY_FAILURE,
                "retrieved chunks carry no section/hierarchy metadata, so the "
                "answer cannot be traced back to the document structure",
                ["no section/parent/heading/hierarchy_path on any chunk"],
            )
        )

    if not answer:
        tags.append(FailureTag(TAG_GENERATION_FAILURE, "the answer is empty", evidence))

    if answer:
        # 3. Generation failure: an "I don't know" that ignores retrieved
        # evidence (or an unknown answer with nothing retrieved).
        if _is_unknown_answer(answer):
            if expected and expected in sources:
                tags.append(
                    FailureTag(
                        TAG_GENERATION_FAILURE,
                        "the answer says it cannot answer despite the expected "
                        "document being present in the retrieved context",
                        evidence,
                    )
                )
            else:
                tags.append(
                    FailureTag(
                        TAG_GENERATION_FAILURE,
                        "the answer declines to answer and no relevant context "
                        "was retrieved",
                        evidence,
                    )
                )

        # 2. Context integration failure: relevant context retrieved but not
        # used in the answer (the answer's citations never touch the expected
        # document).
        cited_retrieved = [c for c in cited if c in retrieved_ids]
        cited_sources = {source_of_cited[cid] for cid in cited if cid in source_of_cited}
        if (
            expected is not None
            and expected in sources
            and cited_retrieved
            and cited_sources
            and expected not in cited_sources
        ):
            tags.append(
                FailureTag(
                    TAG_CONTEXT_INTEGRATION_FAILURE,
                    f"the expected document {expected!r} was retrieved but none "
                    f"of the cited chunks come from it (cited sources: "
                    f"{sorted(cited_sources)})",
                    evidence,
                )
            )

        # 5. Citation failure: substantive answers must cite retrieved chunks.
        if not _is_unknown_answer(answer):
            if not cited:
                tags.append(
                    FailureTag(
                        TAG_CITATION_FAILURE,
                        "the answer makes substantive claims but cites no chunks",
                        evidence,
                    )
                )
            else:
                missing = [cid for cid in cited if cid not in retrieved_ids]
                if missing:
                    tags.append(
                        FailureTag(
                            TAG_CITATION_FAILURE,
                            "citations reference chunks that were not retrieved: "
                            + ", ".join(missing),
                            evidence,
                        )
                    )

    # 4. Hallucination: claims the evidence does not support or contradicts.
    if answer and not _is_unknown_answer(answer):
        claims = precomputed_claims
        if claims is None:
            claims = claim_check.claim_level_faithfulness(answer, chunks)
        bad_claims = [
            c for c in claims.checks if c.status in (claim_check.STATUS_UNSUPPORTED, claim_check.STATUS_CONTRADICTED)
        ]
        if bad_claims:
            tags.append(
                FailureTag(
                    TAG_HALLUCINATION,
                    f"{len(bad_claims)} of {len(claims.checks)} claims are not "
                    f"grounded in the retrieved evidence",
                    [f"{c.claim_id} ({c.status}): {c.claim}" for c in bad_claims],
                )
            )

    # 6. Efficiency failure: the adaptive loop needed extra rounds.
    rounds = record.get("rounds_used")
    stop_reason = record.get("stop_reason")
    if (rounds or 0) > 1 or stop_reason == STOP_REASON_MAX_ROUNDS:
        tags.append(
            FailureTag(
                TAG_EFFICIENCY_FAILURE,
                f"adaptive retrieval used {rounds} round(s)"
                + (f", stopped on {stop_reason!r}" if stop_reason else ""),
                evidence,
            )
        )

    # 8. Unsupported premise (project's eighth failure point).
    status = verification.get("status")
    if status in (STATUS_UNSUPPORTED, STATUS_CONTRADICTED):
        premises = verification.get("premises") or []
        tags.append(
            FailureTag(
                TAG_UNSUPPORTED_PREMISE,
                f"the query was checked against the evidence and found to have "
                f"an {status} premise",
                tuple(premises) or ("premise verification status: " + status,),
            )
        )

    return tags


def tag_results(records: Sequence[dict], precomputed_claims: Mapping[str, object] | None = None) -> dict:
    """Tag every record and return ``{query_id: [FailureTag, ...]}``.

    ``precomputed_claims`` optionally maps query_id to a ClaimLevelReport to
    reuse for hallucination detection.
    """
    tagged: dict[str, list[FailureTag]] = {}
    for record in records:
        rid = str(record.get("query_id") or "")
        claims = (precomputed_claims or {}).get(rid)
        tagged[rid] = tag_result(record, precomputed_claims=claims)
    return tagged


def failure_point_summary(tagged: Mapping[str, Sequence[FailureTag]]) -> dict:
    """Summarize tags across queries.

    Returns ``{"counts": {tag: n}, "tagged_queries": n,
    "tagged_queries_fraction": float-of-total}`` where every failure point is
    present in counts (zeroed if never fired).
    """
    counts = {tag: 0 for tag in ALL_FAILURE_POINTS}
    for tags in tagged.values():
        for failure_tag in tags:
            counts[failure_tag.tag] += 1
    tagged_queries = sum(1 for tags in tagged.values() if tags)
    total = len(tagged)
    return {
        "counts": counts,
        "tagged_queries": tagged_queries,
        "tagged_queries_fraction": (tagged_queries / total) if total else 0.0,
    }


__all__ = [
    "ALL_FAILURE_POINTS",
    "FAILURE_POINT_LABELS",
    "STOP_REASON_MAX_ROUNDS",
    "TAG_CITATION_FAILURE",
    "TAG_CONTEXT_INTEGRATION_FAILURE",
    "TAG_EFFICIENCY_FAILURE",
    "TAG_GENERATION_FAILURE",
    "TAG_HALLUCINATION",
    "TAG_INTERPRETABILITY_FAILURE",
    "TAG_RETRIEVAL_FAILURE",
    "TAG_UNSUPPORTED_PREMISE",
    "FailureTag",
    "failure_point_label",
    "failure_point_summary",
    "normalize_result",
    "tag_result",
    "tag_results",
]