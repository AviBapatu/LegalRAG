"""LegalRAG generation package: prompt building, premise verification, Groq generation.

Milestone 5 (AGENTS.md §5). The package exposes a structured prompt builder, a
premise-verification step, and a Groq-backed generator behind an isolated
client interface so unit tests never need a real API key or network access.
"""

from .prompt import (
    NO_EVIDENCE_TEXT,
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    PremiseVerification,
    build_prompt,
    build_verification_prompt,
    format_evidence,
    parse_verification,
)
from .generate import (
    API_KEY_ENV,
    GroqClient,
    GroqGenerator,
    GenerationConfig,
    GenerationResult,
    PremiseVerifier,
    extract_cited_chunk_ids,
)

__all__ = [
    "NO_EVIDENCE_TEXT",
    "STATUS_CONTRADICTED",
    "STATUS_SUPPORTED",
    "STATUS_UNSUPPORTED",
    "PremiseVerification",
    "build_prompt",
    "build_verification_prompt",
    "format_evidence",
    "parse_verification",
    "API_KEY_ENV",
    "GroqClient",
    "GroqGenerator",
    "GenerationConfig",
    "GenerationResult",
    "PremiseVerifier",
    "extract_cited_chunk_ids",
]