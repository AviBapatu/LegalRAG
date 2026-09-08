"""Tests for Milestone 5 generation: mocked Groq client, premise verification,
config handling, API-key isolation, and regression with retrieval results.

No test here touches the network or needs GROQ_API_KEY.
"""

import os
import re
from pathlib import Path

import pytest

from generation.generate import (
    API_KEY_ENV,
    GroqClient,
    GroqGenerator,
    GenerationConfig,
    GenerationResult,
    PremiseVerifier,
    extract_cited_chunk_ids,
)
from generation.prompt import (
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    PremiseVerification,
)
from retrieval.retriever import RetrievalResult
from tests.conftest import sample_chunks

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

VERIFIER_SYSTEM_SIGNATURE = "premise verifier"


class FakeGroqResponse:
    def __init__(self, text):
        block = type("ContentBlock", (), {"content": text})()
        msg = type("Message", (), {"content": text})()
        choice = type("Choice", (), {"message": msg})()
        self.choices = [choice]


class _Messages:
    def __init__(self, sdk):
        self._sdk = sdk

    def create(self, **kwargs):
        return self._sdk._handle_create(**kwargs)


class FakeGroqSDK:
    """Duck-types the Groq SDK; records requests, never touches network."""

    def __init__(self, verifier_text=None, answer_text=None, responder=None):
        self.chat = type("Chat", (), {"completions": _Messages(self)})()
        self.verifier_text = verifier_text
        self.answer_text = answer_text
        self.responder = responder
        self.requests = []

    def _handle_create(self, **kwargs):
        self.requests.append(kwargs)
        messages = kwargs.get("messages") or []
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if self.responder is not None:
            text = self.responder(system, user, kwargs)
        elif VERIFIER_SYSTEM_SIGNATURE in system.lower():
            text = self.verifier_text or (
                "STATUS: supported\nPREMISES: none\nEXPLANATION: no premise."
            )
        else:
            text = self.answer_text or "Answer without citations."
        return FakeGroqResponse(text)


def make_evidence():
    chunks = sample_chunks()
    return [
        RetrievalResult.from_result_dict({"chunk": c, "fused_score": 1.0})
        for c in chunks
    ]


def make_config():
    return GenerationConfig(model="llama-test", temperature=0.5, max_tokens=256)


def make_client(sdk, config=None):
    return GroqClient(config=config or make_config(), api_key="test-key", sdk=sdk)


# ---------------------------------------------------------------------------
# Mocked generation
# ---------------------------------------------------------------------------


def test_mocked_generation_returns_answer_and_citations():
    sdk = FakeGroqSDK(
        answer_text="The NDA bars disclosure. [citation:nda-1] [citation:nda-0]"
    )
    client = make_client(sdk)
    generator = GroqGenerator(client)
    result = generator.generate("Can I disclose?", make_evidence())

    assert isinstance(result, GenerationResult)
    assert "The NDA bars disclosure." in result.answer
    assert result.cited_chunk_ids == ["nda-1", "nda-0"]
    assert result.verification.status == STATUS_SUPPORTED


def test_verification_runs_before_main_answer():
    sdk = FakeGroqSDK(
        verifier_text="STATUS: supported\nPREMISES: none\nEXPLANATION: no premise.",
        answer_text="Answer. [citation:nda-1]",
    )
    generator = GroqGenerator(make_client(sdk))
    generator.generate("Q?", make_evidence())

    assert len(sdk.requests) == 2
    first_system = next((m["content"] for m in sdk.requests[0]["messages"] if m.get("role") == "system"), "")
    assert VERIFIER_SYSTEM_SIGNATURE in first_system.lower()
    second_user = next((m["content"] for m in sdk.requests[1]["messages"] if m.get("role") == "user"), "")
    assert "PREMISE VERIFICATION:" in second_user
    assert "STATUS: SUPPORTED" in second_user


def test_supported_premise_flow():
    verifier_text = "STATUS: supported\nPREMISES: none\nEXPLANATION: ok."
    sdk = FakeGroqSDK(
        verifier_text=verifier_text,
        answer_text="Yes. [citation:nda-1]",
    )
    generator = GroqGenerator(make_client(sdk))
    result = generator.generate("Q?", make_evidence())
    assert result.verification.status == STATUS_SUPPORTED
    assert result.cited_chunk_ids == ["nda-1"]


def test_unsupported_premise_handling():
    verifier_text = (
        'STATUS: unsupported\nPREMISES: ["the NDA has no term limit"]\n'
        "EXPLANATION: no chunk states any term limit."
    )
    sdk = FakeGroqSDK(
        verifier_text=verifier_text,
        answer_text=(
            "Your premise that the NDA has no term limit is not supported by "
            "the evidence. [citation:nda-1]"
        ),
    )
    generator = GroqGenerator(make_client(sdk))
    result = generator.generate(
        "Since the NDA has no term limit, can I disclose after 2 years?",
        make_evidence(),
    )

    assert result.verification.status == STATUS_UNSUPPORTED
    assert result.verification.premises == ["the NDA has no term limit"]
    assert "status: unsupported" in result.prompt["user"].lower()
    assert "not supported by the evidence" in result.answer


def test_contradicted_premise_handling():
    verifier_text = (
        'STATUS: contradicted\nPREMISES: ["the NDA has no term limit"]\n'
        "EXPLANATION: nda-1 states a two-year term."
    )
    sdk = FakeGroqSDK(
        verifier_text=verifier_text,
        answer_text=(
            "That premise is contradicted by the evidence: the NDA has a "
            "two-year term. [citation:nda-1]"
        ),
    )
    generator = GroqGenerator(make_client(sdk))
    result = generator.generate(
        "Since the NDA has no term limit, can I disclose after 2 years?",
        make_evidence(),
    )

    assert result.verification.status == STATUS_CONTRADICTED
    assert "status: contradicted" in result.prompt["user"].lower()
    assert "contradicted by the evidence" in result.answer


def test_provided_verification_skips_verifier_call():
    verification = PremiseVerification(
        status=STATUS_UNSUPPORTED,
        premises=["premise"],
        explanation="silent.",
    )
    sdk = FakeGroqSDK(answer_text="Answer. [citation:nda-0]")
    generator = GroqGenerator(make_client(sdk))
    result = generator.generate("Q?", make_evidence(), verification=verification)

    assert len(sdk.requests) == 1  # only the answer call, no verifier call
    assert result.verification is verification
    assert "STATUS: UNSUPPORTED" in result.prompt["user"]


def test_premise_verifier_uses_client_interface():
    sdk = FakeGroqSDK(
        verifier_text="STATUS: contradicted\nPREMISES: none\nEXPLANATION: differs."
    )
    base_client = make_client(sdk)
    verifier = PremiseVerifier(base_client)
    result = verifier.verify("Q?", make_evidence())
    assert result.status == STATUS_CONTRADICTED
    assert len(sdk.requests) == 1
    system = next((m["content"] for m in sdk.requests[0]["messages"] if m.get("role") == "system"), "")
    assert VERIFIER_SYSTEM_SIGNATURE in system.lower()


def test_generation_result_structured_for_eval():
    sdk = FakeGroqSDK(
        verifier_text="STATUS: supported\nPREMISES: none\nEXPLANATION: ok.",
        answer_text="Salary is stated. [citation:emp-0]",
    )
    config = make_config()
    generator = GroqGenerator(make_client(sdk, config=config))
    result = generator.generate("What is the salary?", make_evidence())

    assert result.model == config.model
    assert result.temperature == config.temperature
    assert result.max_tokens == config.max_tokens
    assert result.query == "What is the salary?"
    assert set(result.prompt) == {"system", "user"}
    assert result.cited_chunk_ids == ["emp-0"]


def test_generate_accepts_plain_evidence_dicts():
    chunks = sample_chunks()
    evidence = [{"chunk": c, "source_name": c["source_name"]} for c in chunks]
    sdk = FakeGroqSDK(
        verifier_text="STATUS: supported\nPREMISES: none\nEXPLANATION: ok.",
        answer_text="Yes. [citation:nda-1]",
    )
    result = GroqGenerator(make_client(sdk)).generate("Q?", evidence)
    assert "RETRIEVED EVIDENCE:" in result.prompt["user"]
    assert "chunk_id: nda-1" in result.prompt["user"]


def test_empty_answer_raises():
    sdk = FakeGroqSDK(
        verifier_text="STATUS: supported\nPREMISES: none\nEXPLANATION: ok.",
        answer_text="   \n",
    )
    with pytest.raises(RuntimeError, match="empty response"):
        GroqGenerator(make_client(sdk)).generate("Q?", make_evidence())


# ---------------------------------------------------------------------------
# API-key handling (no secrets, no real calls)
# ---------------------------------------------------------------------------


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-groq-env-key")
    client = GroqClient()  # no sdk, no explicit key
    assert client.api_key == "sk-groq-env-key"


def test_explicit_api_key_overrides_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-groq-env-key")
    client = GroqClient(api_key="sk-groq-explicit")
    assert client.api_key == "sk-groq-explicit"


def test_missing_api_key_raises_only_when_calling(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    client = GroqClient(sdk=None)
    assert client.api_key is None
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        client.complete({"system": "s", "user": "u"})


def test_injected_sdk_works_without_api_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    sdk = FakeGroqSDK(
        verifier_text="STATUS: supported\nPREMISES: none\nEXPLANATION: ok.",
        answer_text="Sure. [citation:nda-1]",
    )
    client = GroqClient(config=make_config(), sdk=sdk)  # no api_key
    assert client.api_key is None
    result = GroqGenerator(client).generate("Q?", make_evidence())
    assert result.cited_chunk_ids == ["nda-1"]


def test_error_does_not_leak_key(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "groq" or name.startswith("groq."):
            raise ImportError("groq is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    client = GroqClient(api_key="sk-groq-topsecret", sdk=None)
    with pytest.raises(RuntimeError, match="groq package is required") as exc_info:
        client.complete({"system": "s", "user": "u"})
    assert "sk-groq-topsecret" not in str(exc_info.value)


def test_config_yaml_contains_no_secrets():
    import yaml

    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)

    def collect(d, prefix=""):
        items = {}
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(collect(v, full))
            else:
                items[full] = v
        return items

    forbidden = {"api_key", "apikey", "secret", "password", "access_token"}
    for key in collect(cfg):
        assert key.lower() not in forbidden, f"secret-like key found: {key}"
    for value in collect(cfg).values():
        assert not str(value).startswith("sk-groq"), "stray API key value in config"


# ---------------------------------------------------------------------------
# Configurable generation settings
# ---------------------------------------------------------------------------


def test_request_uses_configurable_model_temperature_max_tokens():
    config = GenerationConfig(
        model="llama-x-test-1", temperature=0.05, max_tokens=77
    )
    sdk = FakeGroqSDK(answer_text="Answer. [citation:nda-1]")
    client = make_client(sdk, config=config)
    GroqGenerator(client).generate("Q?", make_evidence())

    request = sdk.requests[-1]
    assert request["model"] == "llama-x-test-1"
    assert request["max_tokens"] == 77


def test_generation_config_from_config_dict():
    cfg = GenerationConfig.from_config_dict(
        {
            "generation": {
                "provider": "groq",
                "model": "llama-z",
                "temperature": 0.7,
                "max_tokens": 512,
            }
        }
    )
    assert cfg.provider == "groq"
    assert cfg.model == "llama-z"
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 512


def test_generation_config_defaults_when_block_missing():
    cfg = GenerationConfig.from_config_dict({})
    assert cfg.provider == "groq"
    assert isinstance(cfg.model, str)
    assert cfg.temperature > 0
    assert cfg.max_tokens > 0


def test_real_config_yaml_generation_block_is_sane():
    import yaml

    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    gen = GenerationConfig.from_config_dict(cfg)
    assert "generation" in cfg
    assert gen.provider in ("groq",)
    assert "llama" in gen.model
    assert 0.0 <= gen.temperature <= 1.0
    assert gen.max_tokens >= 1


def test_premise_verification_validates_status():
    with pytest.raises(ValueError):
        PremiseVerification(status="bogus")


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def test_extract_cited_chunk_ids_order_and_uniqueness():
    answer = "a [citation:nda-1] b [citation:nda-0] c [citation:nda-1]"
    assert extract_cited_chunk_ids(answer) == ["nda-1", "nda-0"]


def test_extract_cited_chunk_ids_none():
    assert extract_cited_chunk_ids("no citations") == []


def test_citation_regex_matches_answer_format():
    from generation.generate import CITATION_RE

    assert re.search(CITATION_RE, "[citation:nda-1]")
    assert not re.search(CITATION_RE, "citation nda-1")