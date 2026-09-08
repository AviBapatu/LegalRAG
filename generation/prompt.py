"""Structured prompt builder for legal-RAG generation (Milestone 5).

Builds the prompts that drive premise verification and main answer generation.
Each prompt is a plain dict with ``system`` and ``user`` keys so the injected
client interface stays uniform and unit tests never need a real API key.

The main-answer prompt clearly separates, in order:

    a. the user query,
    b. the retrieved evidence (each chunk with its ``chunk_id`` citation key),
    c. the instructions.

The system message carries the standing legal-RAG instructions: ground answers
only in the supplied evidence, cite inline with ``[citation:chunk_id]``, say
"I don't know" when the evidence is insufficient, never present unsupported
legal claims as fact, and honor the premise-verification result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from retrieval.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Premise-verification status values
# ---------------------------------------------------------------------------

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_CONTRADICTED = "contradicted"

VALID_STATUSES = (STATUS_SUPPORTED, STATUS_UNSUPPORTED, STATUS_CONTRADICTED)

#: Shown when no evidence was retrieved, so the model still sees an explicit
#: signal that grounding material is absent. (Used on its own, it also forces
#: a "I don't know" answer.)
NO_EVIDENCE_TEXT = "(no retrieved evidence available for this query)"


@dataclass
class PremiseVerification:
    """Outcome of checking the query's factual/legal premises against evidence.

    Attributes:
        status: One of ``supported``, ``unsupported``, ``contradicted``.
        premises: The individual premises the verifier identified (if any).
        explanation: Why the verifier made that call.
    """

    status: str = STATUS_SUPPORTED
    premises: list[str] = field(default_factory=list)
    explanation: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"unknown premise status {self.status!r}; "
                f"expected one of {VALID_STATUSES}"
            )

    @property
    def is_unsupported(self) -> bool:
        return self.status in (STATUS_UNSUPPORTED, STATUS_CONTRADICTED)


# ---------------------------------------------------------------------------
# Shared instruction blocks
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are a legal research and analysis assistant. Answer questions about legal documents with care, fidelity, and restraint.

Rules you must follow:

1. GROUNDING — Base every claim exclusively on the RETRIEVED EVIDENCE. Do not use outside legal knowledge to add facts, provisions, or obligations that are not present in the evidence.
2. CITATION — After each factual or legal claim, add an inline citation of the form [citation:chunk_id], using the chunk_id shown in the RETRIEVED EVIDENCE. Never invent a chunk_id. If a claim has no supporting chunk, do not state it as fact.
3. SUFFICIENCY — If the retrieved evidence is not sufficient to answer the query, answer "I don't know" and say what evidence would be needed. Never guess, speculate, or interpolate.
4. NO UNSUPPORTED CLAIMS — Do not present unsupported legal claims as facts. Mark anything that goes beyond the evidence as explicitly uncertain.
5. PREMISE VERIFICATION — This is a dedicated check that runs before the main answer. A PREMISE VERIFICATION section reports whether each factual or legal assumption embedded in the query's wording is supported, unsupported, or contradicted by the evidence. If any premise is UNSUPPORTED or CONTRADICTED, your answer MUST first identify that premise and explain that the evidence does not support (or contradicts) it — never silently accept the premise as fact."""

ANSWER_INSTRUCTIONS = """- Answer the QUERY using ONLY the RETRIEVED EVIDENCE.
- Cite every claim inline with its chunk_id using the citation token [citation:chunk_id], e.g. "... [citation:nda-1]".
- If the evidence is insufficient, answer "I don't know" and note what evidence is missing.
- Do not present unsupported legal claims as facts; if you are unsure, say so.
- Honor the PREMISE VERIFICATION section: surface any unsupported or contradicted premise instead of assuming it is true."""

VERIFIER_SYSTEM_INSTRUCTIONS = """You are a premise verifier for a legal-RAG system. Your only job is to check whether the factual or legal assumptions (premises) embedded in the user's query are supported by, unsupported by, or contradicted by the retrieved evidence.

Definitions:
- supported: the premise is consistent with facts present in the evidence.
- unsupported: the evidence is silent on the premise and cannot confirm it.
- contradicted: the evidence directly contradicts the premise.

Rules:
- If the query makes no factual or legal assumption beyond the question itself, output STATUS: supported and PREMISES: none.
- Be conservative: a premise the evidence cannot confirm is UNSUPPORTED, never supported.
- A premise the evidence contradicts is CONTRADICTED, which takes precedence over supported/unsupported."""

VERIFIER_OUTPUT_FORMAT = """Output EXACTLY this format:

STATUS: supported|unsupported|contradicted
PREMISES: ["one premise", "another premise"]   <- or `none` if there are none
EXPLANATION: one sentence explaining how you reached that STATUS."""


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def _chunk_text(chunk: Mapping) -> str:
    text = chunk.get("text")
    if not text:
        text = chunk.get("original_text", "")
    return str(text).strip()


def _normalize_result(result: object) -> dict:
    """Normalize a RetrievalResult or a plain result/chunk dict."""
    if isinstance(result, RetrievalResult):
        chunk = result.chunk or {}
        return {
            "chunk_id": str(result.chunk_id),
            "source_name": str(result.source_name or chunk.get("source_name") or ""),
            "section": result.section or chunk.get("section"),
            "text": _chunk_text(chunk),
        }
    if isinstance(result, Mapping):
        chunk = result.get("chunk") or result
        return {
            "chunk_id": str(chunk.get("chunk_id") or result.get("chunk_id") or ""),
            "source_name": str(
                chunk.get("source_name") or result.get("source_name") or ""
            ),
            "section": chunk.get("section") or result.get("section"),
            "text": _chunk_text(chunk),
        }
    raise TypeError(
        f"unsupported evidence item {type(result).__name__}; expected "
        "RetrievalResult or a result/chunk dict"
    )


def format_evidence(
    evidence: Sequence[object] | Iterable[object],
) -> str:
    """Render retrieved chunks with chunk_id citation keys and source labels.

    Each chunk becomes a numbered block whose header carries the chunk_id the
    model must cite back with, plus its source document and section:

        [1] (chunk_id: nda-1, source: nda, section: 2)
        Confidential Information shall not be disclosed to third parties.
    """
    blocks: list[str] = []
    for i, result in enumerate(evidence, start=1):
        info = _normalize_result(result)
        header = f"[{i}] (chunk_id: {info['chunk_id']}"
        if info["source_name"]:
            header += f", source: {info['source_name']}"
        if info["section"]:
            header += f", section: {info['section']}"
        header += ")"
        blocks.append(f"{header}\n{info['text']}")
    if not blocks:
        return NO_EVIDENCE_TEXT
    return "\n\n".join(blocks)


def _format_verification(verification: PremiseVerification) -> str:
    lines = [f"STATUS: {verification.status.upper()}"]
    if verification.premises:
        lines.append("PREMISES:")
        lines.extend(f"- {p}" for p in verification.premises)
    if verification.explanation:
        lines += ["EXPLANATION:", verification.explanation]
    lines += [
        "",
        "The premise check above was performed against the retrieved evidence.",
        "If it reports an UNSUPPORTED or CONTRADICTED premise: DO NOT accept that premise as fact.",
        "Identify the premise and explain that the evidence does not support (or",
        "contradicts) it, then answer using only the evidence you actually have.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main-answer prompt
# ---------------------------------------------------------------------------


def build_prompt(
    query: str,
    evidence: Sequence[object] | Iterable[object],
    verification: PremiseVerification | None = None,
    system_instructions: str | None = None,
) -> dict:
    """Build the main-answer prompt.

    Args:
        query: The user's question.
        evidence: Retrieved chunks as ``RetrievalResult`` objects or result
            dicts (regression-compatible with Milestone 3 retrieval output).
        verification: Optional premise-verification result. When supplied it
            is inserted as a PREMISE VERIFICATION section between the evidence
            and the instructions, so the answer model sees the check.
        system_instructions: Override the default system block.

    Returns:
        ``{"system": ..., "user": ...}`` with the user message clearly
        separating QUERY / RETRIEVED EVIDENCE / INSTRUCTIONS.
    """
    sections = [
        "QUERY:",
        str(query).strip(),
        "",
        "RETRIEVED EVIDENCE:",
        format_evidence(evidence),
    ]
    if verification is not None:
        sections += ["", "PREMISE VERIFICATION:", _format_verification(verification)]
    sections += ["", "INSTRUCTIONS:", ANSWER_INSTRUCTIONS]
    return {
        "system": system_instructions or SYSTEM_INSTRUCTIONS,
        "user": "\n".join(sections),
    }


# ---------------------------------------------------------------------------
# Premise-verification prompt
# ---------------------------------------------------------------------------


def build_verification_prompt(
    query: str,
    evidence: Sequence[object] | Iterable[object],
) -> dict:
    """Build the dedicated premise-verification prompt.

    This is a distinct step run before main-answer generation. Its output is
    parsed back into a :class:`PremiseVerification` by ``parse_verification``.
    """
    user = "\n".join(
        [
            "QUERY:",
            str(query).strip(),
            "",
            "RETRIEVED EVIDENCE:",
            format_evidence(evidence),
            "",
            "PREMISE CHECK:",
            "Identify any factual or legal assumption the QUERY relies on and",
            "determine whether that assumption is supported, unsupported, or",
            "contradicted by the RETRIEVED EVIDENCE above.",
            "",
            VERIFIER_OUTPUT_FORMAT,
        ]
    )
    return {"system": VERIFIER_SYSTEM_INSTRUCTIONS, "user": user}


# ---------------------------------------------------------------------------
# Verification output parsing (deterministic, no extra LLM calls)
# ---------------------------------------------------------------------------

_STATUS_LINE_RE = re.compile(r"^STATUS:\s*(.+)$", re.MULTILINE)
_PREMISES_LINE_RE = re.compile(r"^PREMISES:\s*(.*)$", re.MULTILINE)
_EXPLANATION_RE = re.compile(r"^EXPLANATION:\s*(.*)$", re.MULTILINE)


def _normalize_status(value: str) -> str | None:
    v = value.strip().lower().rstrip(".")
    if "contradict" in v:
        return STATUS_CONTRADICTED
    if "unsupported" in v or "not supported" in v:
        return STATUS_UNSUPPORTED
    if "supported" in v or "confirm" in v or "consistent" in v:
        return STATUS_SUPPORTED
    return None


def parse_verification(raw: str) -> PremiseVerification:
    """Parse a verifier response into a :class:`PremiseVerification`.

    The parser is deliberately conservative: if the status cannot be read from
    the response, it defaults to ``unsupported`` so an unreliable response is
    surfaced rather than silently accepted. Premises are read from a JSON list
    (or a ``|``/comma-separated fallback); ``none`` means no premises.
    """
    raw = (raw or "").strip()
    status = STATUS_UNSUPPORTED
    found_status = False
    for line in raw.splitlines():
        if re.match(r"^\s*STATUS:", line, re.IGNORECASE):
            value = _STATUS_LINE_RE.match(line).group(1)
            norm = _normalize_status(value)
            if norm is not None:
                status = norm
            found_status = True
            break
    if not found_status:
        lowered = raw.lower()
        if "contradict" in lowered:
            status = STATUS_CONTRADICTED
        elif (
            "supported" in lowered
            and "unsupported" not in lowered
            and "not supported" not in lowered
        ):
            status = STATUS_SUPPORTED

    return PremiseVerification(
        status=status,
        premises=_parse_premises(raw),
        explanation=_parse_explanation(raw),
    )


def _parse_premises(raw: str) -> list[str]:
    m = _PREMISES_LINE_RE.search(raw)
    if not m:
        return []
    value = m.group(1).strip()
    if not value or value.lower() in ("none", "[]", '"none"', "n/a", "na"):
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
        value = value.strip("[]")
    parts = [p for sep in ("|", ",") for p in value.split(sep)]
    if not parts:
        parts = [value]
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def _parse_explanation(raw: str) -> str:
    m = _EXPLANATION_RE.search(raw)
    if not m:
        return ""
    return m.group(1).strip()