"""Serialize Milestone 1-7 dataclasses into JSON-friendly dicts.

The API layer must expose rich per-round and per-chunk detail to the frontend.
Rather than calling ``dataclasses.asdict`` on every type (which would pull in
extra fields like full chunk bodies and raw prompts we don't want to leak),
each serializer here selects exactly the fields the Milestone 8 UI/spec asks
for, reusing the existing data structures without reimplementing any
retrieval/generation logic.
"""

from __future__ import annotations

from typing import Any, Mapping

from generation.prompt import PremiseVerification
from retrieval.adaptive import AdaptiveResult, AdaptiveRound
from retrieval.retriever import RetrievalResult


def premise_verification_to_dict(
    verification: PremiseVerification | None,
) -> dict[str, Any]:
    if verification is None:
        return {
            "status": None,
            "premises": [],
            "explanation": "",
            "is_unsupported": False,
        }
    return {
        "status": getattr(verification, "status", None),
        "premises": list(getattr(verification, "premises", None) or []),
        "explanation": getattr(verification, "explanation", None) or "",
        "is_unsupported": bool(getattr(verification, "is_unsupported", False)),
    }


def retrieval_result_to_dict(result: RetrievalResult) -> dict[str, Any]:
    """Serialize one retrieved chunk with its hierarchy + scores."""
    chunk = result.chunk or {}
    text = str(chunk.get("text") or chunk.get("original_text") or "")
    return {
        "chunk_id": result.chunk_id,
        "source_name": result.source_name or chunk.get("source_name"),
        "text": text,
        "section": result.section or chunk.get("section"),
        "parent": result.parent or chunk.get("parent"),
        "heading": result.heading or chunk.get("heading"),
        "hierarchy_path": list(result.hierarchy_path or []),
        "fused_score": result.fused_score,
        "dense_score": result.dense_score,
        "bm25_score": result.bm25_score,
        "structurally_boosted": bool(result.structurally_boosted),
    }


def adaptive_round_to_dict(round_: AdaptiveRound) -> dict[str, Any]:
    return {
        "round_index": round_.round_index,
        "is_initial": bool(round_.is_initial),
        "rewritten_from": round_.rewritten_from,
        "query": round_.query,
        "retrieval_results": [
            retrieval_result_to_dict(r) for r in round_.retrieval_results
        ],
        "generation": {
            "answer": (round_.generation.answer if round_.generation else ""),
            "query": (round_.generation.query if round_.generation else ""),
            "verification": premise_verification_to_dict(
                round_.generation.verification if round_.generation else None
            ),
            "cited_chunk_ids": list(
                (round_.generation.cited_chunk_ids if round_.generation else None) or []
            ),
        },
        "decision": {
            "needs_another_round": bool(round_.decision.needs_another_round),
            "reason": round_.decision.reason,
        },
    }


def adaptive_result_to_dict(result: AdaptiveResult) -> dict[str, Any]:
    """Serialize the full adaptive-retrieval controller output."""
    verification = result.final_verification
    return {
        "original_query": result.original_query,
        "final_answer": result.final_answer,
        "premise_verification": premise_verification_to_dict(verification),
        "adaptive_triggered": bool(result.adaptive_triggered),
        "rounds_used": int(result.rounds_used),
        "rewritten_queries": list(result.rewritten_queries or []),
        "stop_reason": result.stop_reason,
        "support_threshold": result.support_threshold,
        "max_extra_rounds": int(result.max_extra_rounds),
        "final_retrieval_results": [
            retrieval_result_to_dict(r) for r in result.final_retrieval_results
        ],
        "cited_chunk_ids": list(
            dict.fromkeys(
                cid for r in result.final_retrieval_results for cid in [r.chunk_id]
            )
        ),
        "failure_tags": [],
        "claim_level": None,
        "rounds": [adaptive_round_to_dict(r) for r in result.rounds],
    }


def claim_level_to_dict(report: Any) -> dict[str, Any] | None:
    """Serialize a ClaimLevelReport (from eval.claim_check) or None."""
    if report is None:
        return None
    details = getattr(report, "checks", None) or []
    counts = getattr(report, "counts", None) or {}
    return {
        "score": float(getattr(report, "score", 0.0)),
        "num_claims": int(getattr(report, "num_claims", 0)),
        "counts": {str(k): int(v) for k, v in dict(counts).items()},
        "details": [
            {
                "claim_id": getattr(c, "claim_id", None),
                "claim": getattr(c, "claim", None),
                "status": getattr(c, "status", None),
                "evidence": getattr(c, "evidence", None),
                "reason": getattr(c, "reason", None),
            }
            for c in details
        ],
    }


def failure_tags_to_dict(tags: Any) -> list[dict[str, Any]]:
    """Serialize a list of FailureTag objects (or already-dict tags)."""
    out: list[dict[str, Any]] = []
    for tag in tags or []:
        if isinstance(tag, Mapping):
            out.append(dict(tag))
            continue
        as_dict = getattr(tag, "as_dict", None)
        if callable(as_dict):
            out.append(dict(as_dict()))
        else:
            out.append(
                {
                    "tag": getattr(tag, "tag", None),
                    "reason": getattr(tag, "reason", None),
                    "evidence": list(getattr(tag, "evidence", None) or []),
                }
            )
    return out
