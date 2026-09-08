"""Groq answer generation and premise verification (Milestone 5).

The Groq SDK is kept behind a thin wrapper (``GroqClient``) exposing a
single ``complete(prompt: dict) -> str`` interface. Tests inject a fake SDK
object, so no API key or network access is required unless a real call is made.
The API key is read from the ``GROQ_API_KEY`` environment variable at the
moment a real client is built; it is never hard-coded and never written to
config.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from .prompt import (
    PremiseVerification,
    build_prompt,
    build_verification_prompt,
    parse_verification,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
API_KEY_ENV = "GROQ_API_KEY"

#: Inline citation token the answer model is instructed to produce.
CITATION_RE = re.compile(r"\[citation:([^\]]+)\]")


def _import_groq():
    """Lazily import the Groq SDK, isolated for test monkeypatching."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "the groq package is required for real Groq calls but is "
            "not installed; inject a fake SDK for testing"
        ) from exc
    return Groq


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings, driven by the ``generation`` config block.

    ``provider``, ``model``, ``temperature``, and ``max_tokens`` are all
    configurable; no secrets live here.
    """

    provider: str = "groq"
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_config_dict(cls, config: dict) -> "GenerationConfig":
        gen = config.get("generation") or {}
        return cls(
            provider=str(gen.get("provider", "groq")),
            model=str(gen.get("model", DEFAULT_MODEL)),
            temperature=float(gen.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(gen.get("max_tokens", DEFAULT_MAX_TOKENS)),
        )


class GroqClient:
    """Minimal, isolated wrapper around the Groq chat completions API.

    Args:
        config: A :class:`GenerationConfig`.
        api_key: Explicit key. Defaults to the ``GROQ_API_KEY``
            environment variable. Only needed when a real (non-injected) SDK
            is used.
        sdk: An object duck-typing the Groq SDK (``chat.completions.create``).
            Injected for tests; when None, the real SDK is built lazily.
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        api_key: str | None = None,
        sdk: object | None = None,
    ):
        self.config = config if config is not None else GenerationConfig()
        self.api_key = (
            api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        )
        self._sdk = sdk

    @property
    def sdk(self) -> object:
        """The underlying SDK client, built lazily from the API key."""
        if self._sdk is None:
            if not self.api_key:
                raise RuntimeError(
                    f"{API_KEY_ENV} is not set. Set it before making a real "
                    "Groq call, or inject a fake SDK for testing."
                )
            Groq = _import_groq()
            self._sdk = Groq(api_key=self.api_key)
        return self._sdk

    def complete(self, prompt: dict) -> str:
        """Send a prompt dict (``system``/``user``) and return the text."""
        messages = []
        if "system" in prompt and prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt.get("user", "")})
        
        response = self.sdk.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=messages,
        )
        
        text = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = (choice.message.content or "").strip()
        
        if not text:
            raise RuntimeError("Groq returned an empty response")
        return text


class PremiseVerifier:
    """Runs the dedicated premise-verification step before main generation.

    Uses the same client interface as generation, so it is testable with the
    same injected fake SDK.
    """

    def __init__(self, client: GroqClient):
        self.client = client

    def verify(
        self,
        query: str,
        evidence: Sequence[object] | None = None,
    ) -> PremiseVerification:
        prompt = build_verification_prompt(query, evidence or [])
        raw = self.client.complete(prompt)
        return parse_verification(raw)


@dataclass
class GenerationResult:
    """Structured generation output for downstream evaluation.

    Carries the answer plus everything the Milestone 6/7 eval harness needs:
    the premise-verification result, the chunk_ids actually cited, the exact
    prompt used, and the model parameters.
    """

    answer: str
    query: str
    verification: PremiseVerification
    cited_chunk_ids: list[str]
    prompt: dict
    model: str
    temperature: float
    max_tokens: int


class GroqGenerator:
    """Clean generation interface: query + retrieved results -> answer.

    Args:
        client: A Groq-compatible client (real or injected fake).
        verifier: Optional premise verifier; defaults to one sharing ``client``.
    """

    def __init__(
        self,
        client: GroqClient,
        verifier: PremiseVerifier | None = None,
    ):
        self.client = client
        self.verifier = verifier if verifier is not None else PremiseVerifier(client)

    def verify_premises(
        self,
        query: str,
        evidence: Sequence[object] | None = None,
    ) -> PremiseVerification:
        return self.verifier.verify(query, evidence)

    def generate(
        self,
        query: str,
        evidence: Sequence[object] | None = None,
        verification: PremiseVerification | None = None,
    ) -> GenerationResult:
        """Generate an answer grounded in ``evidence``.

        Args:
            query: User query.
            evidence: Retrieved chunks (Milestone 3 ``RetrievalResult`` objects
                or plain result dicts).
            verification: Optional precomputed premise check. When None, the
                premise-verification step runs first and its result is fed
                into the main-answer prompt (never silently dropped).
        """
        if verification is None:
            verification = self.verify_premises(query, evidence or [])
        prompt = build_prompt(query, evidence or [], verification=verification)
        answer = self.client.complete(prompt)
        return GenerationResult(
            answer=answer,
            query=query,
            verification=verification,
            cited_chunk_ids=extract_cited_chunk_ids(answer),
            prompt=prompt,
            model=self.client.config.model,
            temperature=self.client.config.temperature,
            max_tokens=self.client.config.max_tokens,
        )


def extract_cited_chunk_ids(answer: str) -> list[str]:
    """Return the chunk_ids cited in an answer, in first-appearance order."""
    return list(dict.fromkeys(CITATION_RE.findall(answer or "")))


__all__ = [
    "API_KEY_ENV",
    "CITATION_RE",
    "GroqClient",
    "GroqGenerator",
    "GenerationConfig",
    "GenerationResult",
    "PremiseVerifier",
    "extract_cited_chunk_ids",
]