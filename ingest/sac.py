"""Summary-Augmented Chunking (SAC).

SAC generates a short "document fingerprint" summary (~150 chars) per document
and prepends it to every chunk derived from that document before embedding.
This directly targets Document-Level Retrieval Mismatch (DRM): the retriever
tends to pull chunks from the wrong document entirely because legal boilerplate
(NDAs, contracts) is structurally near-identical across documents (see
AGENTS.md §1, failure mode 1).

The summary can come from an LLM (when `use_llm=True`) or from a deterministic,
dependency-free heuristic summarizer (the default). The heuristic version exists
so the pipeline runs offline and reproducibly for testing and for the DRM
before/after comparison in Milestone 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .chunker import Chunk

# Delimiter placed between the summary and the chunk text so the model can
# still tell the two apart and so we can strip SAC for analysis.
SAC_SEPARATOR = "\n\n[Document summary]\n"


@dataclass
class SummaryResult:
    """A generated document summary.

    Attributes:
        summary: The short summary text (<= max_chars).
        strategy: How it was produced, "heuristic" or "llm".
    """

    summary: str
    strategy: str = "heuristic"


# Simple signal words for parties / doc type used by the heuristic summarizer.
_PARTY_HINTS = (
    "party", "parties", "company", "corporation", "inc.", "ltd", "llc",
    "enterprise", "discloser", "recipient", "licensor", "licensee",
    "buyer", "seller", "employer", "employee",
)
_DOC_TYPE_HINTS = (
    "agreement", "contract", "non-disclosure", "confidentiality", "nda",
    "licen", "employment", "purchase", "lease", "privacy policy",
)
# Generic words commonly wrapped in quotes in legal docs that are NOT parties.
_GENERIC_QUOTED = {
    "agreement", "the agreement", "this agreement", "contract", "the contract",
    "confidential information", "effective date", "party", "the parties",
    "the party", "document", "the document", "shall", "termination",
    "the termination", "secret", "the employee", "the employer",
}


def summarize_document(
    text: str,
    max_chars: int = 150,
    use_llm: bool = False,
    llm_func: Callable[[str], str] | None = None,
) -> SummaryResult:
    """Generate a short summary ("document fingerprint") for one document.

    Args:
        text: Full document text.
        max_chars: Maximum length for the produced summary (approximate; the
            heuristic is truncated, an LLM summary is not).
        use_llm: If True, use `llm_func` to summarize. Raises ValueError if
            `use_llm` is True but `llm_func` is None.
        llm_func: Callable accepting the document text and returning a summary.

    Returns:
        A SummaryResult.
    """
    if use_llm:
        if llm_func is None:
            raise ValueError("llm_func must be provided when use_llm=True")
        summary = llm_func(text)
        return SummaryResult(summary=summary, strategy="llm")

    summary = _heuristic_summary(text, max_chars)
    return SummaryResult(summary=summary, strategy="heuristic")


def _heuristic_summary(text: str, max_chars: int) -> str:
    """Deterministic, dependency-free document fingerprint.

    Scans the document for structural hints and builds a compact summary like:
        "Mutual NDA between 'Acme Inc.' and 'Global Corp.' — Confidentiality"
    Falls back to the first non-empty line if no hints are found.
    """
    text = text.strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = " ".join(lines)

    # Document type from the (approximate) first line + title-case-ish words.
    doc_type: str | None = None
    lower_lines = " ".join(line.lower() for line in lines[:5])
    for hint in _DOC_TYPE_HINTS:
        if hint in lower_lines:
            doc_type = _title(hint)
            break

    # Doc type may also be a title-cased heading line.
    if doc_type is None:
        for line in lines[:5]:
            words = line.split()
            if 1 <= len(words) <= 4 and line.istitle():
                doc_type = line
                break

    # Parties: quoted names, or capitalized phrases near party hints.
    parties = _extract_parties(text)

    parts: list[str] = []
    if doc_type:
        parts.append(doc_type)
    if parties:
        parts.append("between " + " and ".join(parties))
    # Core subject matter: drop stopwords for brevity.
    subject = _subject_matter(joined)
    if subject:
        parts.append(f"— {subject}")

    summary = " ".join(parts).strip()
    if not summary:
        summary = (lines[0] if lines else "")[:max_chars]

    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def _extract_parties(text: str) -> list[str]:
    """Find up to two likely party names (quoted or near party hints)."""
    import re

    parties: list[str] = []
    # Quoted names, e.g. "Acme Inc." — strong signal (but skip generic words).
    quoted = re.findall(r'"([^"]{2,60})"', text)
    for q in quoted:
        name = q.strip()
        if not name or name.lower() in _GENERIC_QUOTED:
            continue
        parties.append(name)
        if len(parties) >= 2:
            break
    if parties:
        return parties[:2]

    # Fallback: capitalized phrases on lines containing a party keyword.
    for line in text.splitlines():
        low = line.lower()
        if any(h in low for h in _PARTY_HINTS):
            words = line.split()
            cap_parts = []
            for w in words:
                stripped = w.strip(".,:;()\"'")
                if stripped and stripped[0].isupper() and len(stripped) > 1:
                    cap_parts.append(stripped)
            if cap_parts:
                parties.append(" ".join(cap_parts))
            if len(parties) >= 2:
                break
    return parties[:2]


def _subject_matter(text: str) -> str:
    """Strip filler words from the very start to get a short subject phrase."""
    stop = {
        "this", "the", "and", "for", "with", "between", "that", "of", "to",
    }
    words = text.split()
    kept: list[str] = []
    for w in words:
        low = w.lower().strip(".,:;()\"'")
        if low in stop:
            continue
        kept.append(w)
        if len(kept) >= 4:
            break
    return " ".join(kept)


def _title(word: str) -> str:
    return word.replace("-", " ").title()


def apply_sac_to_document(
    chunks: list[Chunk],
    summary: str,
    separator: str = SAC_SEPARATOR,
) -> list[Chunk]:
    """Prepend the document summary to every chunk of one document.

    Mutates each chunk in place: the embedded ``text`` becomes
    ``summary + separator + original_text`` and ``sac_applied`` is set to True
    so the effect of SAC can be measured (see eval harness, Milestone 4/6).
    The pre-SAC text is preserved in ``original_text``.
    """
    for chunk in chunks:
        if chunk.sac_applied:
            raise ValueError(f"chunk {chunk.chunk_id} already has SAC applied")
        chunk.sac_applied = True
        chunk.text = summary + separator + chunk.text
    return chunks
