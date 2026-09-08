"""Claim-level faithfulness evaluation (Milestone 7, AGENTS.md §6).

Instead of scoring a whole generated answer with one faithfulness number,
decompose the answer into individual factual claims, then check each claim
against the retrieved evidence independently. Every claim is assigned a
support status:

* **supported**    — the evidence is consistent with (can ground) the claim.
* **unsupported**  — the evidence is silent on the claim (cannot confirm it).
* **contradicted** — the evidence directly conflicts with a specific detail
                     (e.g. the claim says "two years", the evidence says 3).

The overall claim-level faithfulness score is the fraction of claims that are
supported: ``len(supported) / len(all claims)``, a number in [0, 1].

This is more diagnostic than a single answer-level score: it catches cases
where most of an answer is grounded but one claim is not, and it preserves
per-claim detail for the UI/report to display.

Deterministic by default
------------------------
Claim extraction (sentence splitting) and claim checking (token-overlap plus
numeric-conflict detection) are pure functions. Both steps accept injectable
callables so an LLM-backed implementation can be swapped in for production;
tests inject fakes. No API key or network access is required by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from retrieval.retriever import RetrievalResult

from .text_util import find_numbers, overlap_ratio, split_sentences, strip_citations

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_CONTRADICTED = "contradicted"
VALID_STATUSES = (STATUS_SUPPORTED, STATUS_UNSUPPORTED, STATUS_CONTRADICTED)

#: A claim counts as supported when this fraction of its meaningful tokens is
#: grounded in the concatenated evidence text.
SUPPORT_THRESHOLD = 0.6

#: Stable label for the claim-id prefix (``claim-1``, ``claim-2``, ...).
DEFAULT_CLAIM_PREFIX = "claim"


class ClaimExtractor(Protocol):
    """Extracts individual factual claims from an answer."""

    def extract(self, answer: str) -> list[str]: ...


class ClaimChecker(Protocol):
    """Checks one claim against a normalized evidence list.

    ``evidence`` is a list of ``{"chunk_id": str, "text": str}`` dicts (see
    :func:`normalize_evidence`). Returns a :class:`ClaimCheck`.
    """

    def check(self, claim: str, evidence: Sequence[dict]) -> "ClaimCheck": ...


@dataclass(frozen=True)
class ClaimCheck:
    """The support verdict for a single claim.

    Attributes:
        claim_id: Stable identifier (``claim-1``, ``claim-2``, ...).
        claim: The claim text as asserted in the answer.
        status: One of ``supported`` / ``unsupported`` / ``contradicted``.
        evidence: Chunk ids consulted while deciding the status.
        reason: Why this status was assigned.
    """

    claim_id: str
    claim: str
    status: str
    evidence: list[str] = field(default_factory=list)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"unknown claim status {self.status!r}; expected one of {VALID_STATUSES}"
            )


@dataclass(frozen=True)
class ClaimLevelReport:
    """The full claim-level faithfulness result for one answer.

    Attributes:
        score: Overall claim-level faithfulness in [0, 1] =
            supported / total claims (0.0 when there are no claims).
        checks: One :class:`ClaimCheck` per extracted claim.
        counts: ``{supported: n, unsupported: n, contradicted: n}``.
        num_claims: Total number of claims found in the answer.
    """

    score: float
    checks: list[ClaimCheck]
    counts: dict[str, int]
    num_claims: int

    @property
    def unsupported_fraction(self) -> float:
        """Fraction of claims that are NOT supported, in [0, 1]."""
        if self.num_claims == 0:
            return 0.0
        bad = self.counts[STATUS_UNSUPPORTED] + self.counts[STATUS_CONTRADICTED]
        return bad / self.num_claims


# ---------------------------------------------------------------------------
# Evidence normalization
# ---------------------------------------------------------------------------


def normalize_evidence(evidence: Sequence[object]) -> list[dict]:
    """Convert evidence items into ``{"chunk_id", "text"}`` dicts.

    Accepts Milestone 3 ``RetrievalResult`` objects or plain result/chunk
    dicts. Chunks without readable text are included with an empty string so
    the presence of an id is still visible to the checker.
    """
    normalized: list[dict] = []
    for item in evidence or []:
        if isinstance(item, RetrievalResult):
            chunk = item.chunk or {}
            chunk_id = str(item.chunk_id or chunk.get("chunk_id") or "")
        elif isinstance(item, dict):
            chunk = item.get("chunk") if isinstance(item.get("chunk"), dict) else item
            chunk_id = str(chunk.get("chunk_id") or item.get("chunk_id") or "")
        else:
            continue
        text = chunk.get("text") or chunk.get("original_text") or ""
        normalized.append({"chunk_id": chunk_id, "text": str(text)})
    return normalized


# ---------------------------------------------------------------------------
# Claim extraction (deterministic sentence split)
# ---------------------------------------------------------------------------


def extract_claims(answer: str, extractor: ClaimExtractor | None = None) -> list[str]:
    """Return the factual claims of ``answer`` as individual strings.

    The default extractor splits the answer on sentence boundaries and strips
    inline citation tokens. An injectable ``extractor`` overrides it (e.g. an
    LLM-backed claim decomposer).
    """
    if extractor is not None:
        return [str(claim).strip() for claim in extractor.extract(str(answer)) if str(claim).strip()]
    return split_sentences(strip_citations(answer))


# ---------------------------------------------------------------------------
# Deterministic claim checking
# ---------------------------------------------------------------------------


def check_claim(
    claim: str,
    evidence: Sequence[dict],
    checker: ClaimChecker | None = None,
) -> ClaimCheck:
    """Assign a support status (supported/unsupported/contradicted) to a claim.

    Args:
        claim: A single extracted claim.
        evidence: Normalized evidence (see :func:`normalize_evidence`), i.e.
            ``[{"chunk_id": ..., "text": ...}]``.
        checker: Optional injectable checker; when omitted the deterministic
            heuristic below is used.

    Deterministic heuristic rules (documented, unit-tested):
        1. coverage = fraction of the claim's meaning-bearing tokens found in
           the concatenated evidence text.
        2. ``coverage >= SUPPORT_THRESHOLD``            -> **supported**.
        3. the claim and the evidence both carry numeric values and the
           claim's numbers are entirely disjoint from the evidence's numbers
           (e.g. "two years" vs. evidence stating 3) -> **contradicted**.
        4. otherwise                                   -> **unsupported**.
    """
    if checker is not None:
        return checker.check(claim, evidence)

    claim = str(claim).strip()
    chunks = [dict(entry) for entry in evidence or []]
    joined = " ".join(entry.get("text") or "" for entry in chunks)

    coverage = overlap_ratio(claim, joined)
    claim_numbers = find_numbers(claim)
    evidence_numbers = find_numbers(joined)
    conflicting = (
        bool(claim_numbers)
        and bool(evidence_numbers)
        and claim_numbers.isdisjoint(evidence_numbers)
    )
    consulted = [
        entry.get("chunk_id") or ""
        for entry in chunks
        if overlap_ratio(claim, entry.get("text") or "") > 0
    ]

    if coverage >= SUPPORT_THRESHOLD:
        status = STATUS_SUPPORTED
        reason = (
            f"{coverage:.0%} of the claim's meaning-bearing tokens are "
            f"grounded in the retrieved evidence"
        )
    elif conflicting:
        status = STATUS_CONTRADICTED
        reason = (
            "the claim asserts numeric detail(s) "
            f"{sorted(claim_numbers)} that conflict with the numbers found "
            f"in the evidence ({sorted(evidence_numbers)})"
        )
    else:
        status = STATUS_UNSUPPORTED
        reason = (
            "the retrieved evidence does not contain the claim's content "
            f"(token coverage {coverage:.0%} < {SUPPORT_THRESHOLD:.0%})"
        )

    return ClaimCheck(
        claim_id="",
        claim=claim,
        status=status,
        evidence=consulted,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Overall claim-level faithfulness
# ---------------------------------------------------------------------------


def claim_level_faithfulness(
    answer: str,
    evidence: Sequence[object],
    extractor: ClaimExtractor | None = None,
    checker: ClaimChecker | None = None,
    claim_id_prefix: str = DEFAULT_CLAIM_PREFIX,
) -> ClaimLevelReport:
    """Score an answer at claim level against ``evidence``.

    Args:
        answer: The generated answer text.
        evidence: Evidence objects (RetrievalResult/dict chunks).
        extractor: Optional claim extractor override.
        checker: Optional per-claim checker override.
        claim_id_prefix: Prefix for stable claim ids (``claim-1``, ...).

    Returns:
        A :class:`ClaimLevelReport` with the overall score, per-claim detail,
        and status counts. Claim detail is preserved so downstream UI and the
        report can render each claim's verdict.
    """
    claims = extract_claims(answer, extractor=extractor)
    normalized = normalize_evidence(evidence or [])
    if not claims:
        return ClaimLevelReport(
            score=0.0,
            checks=[],
            counts={STATUS_SUPPORTED: 0, STATUS_UNSUPPORTED: 0, STATUS_CONTRADICTED: 0},
            num_claims=0,
        )

    checks: list[ClaimCheck] = []
    for index, claim in enumerate(claims):
        result = check_claim(claim, normalized, checker=checker)
        checks.append(
            ClaimCheck(
                claim_id=f"{claim_id_prefix}-{index + 1}",
                claim=result.claim,
                status=result.status,
                evidence=result.evidence,
                reason=result.reason,
            )
        )

    counts = {
        STATUS_SUPPORTED: sum(1 for c in checks if c.status == STATUS_SUPPORTED),
        STATUS_UNSUPPORTED: sum(1 for c in checks if c.status == STATUS_UNSUPPORTED),
        STATUS_CONTRADICTED: sum(1 for c in checks if c.status == STATUS_CONTRADICTED),
    }
    return ClaimLevelReport(
        score=counts[STATUS_SUPPORTED] / len(checks),
        checks=checks,
        counts=counts,
        num_claims=len(checks),
    )


__all__ = [
    "DEFAULT_CLAIM_PREFIX",
    "STATUS_CONTRADICTED",
    "STATUS_SUPPORTED",
    "STATUS_UNSUPPORTED",
    "SUPPORT_THRESHOLD",
    "VALID_STATUSES",
    "ClaimCheck",
    "ClaimChecker",
    "ClaimExtractor",
    "ClaimLevelReport",
    "check_claim",
    "claim_level_faithfulness",
    "extract_claims",
    "normalize_evidence",
]