"""Tests for the Milestone 5 structured prompt builder."""

import pytest

from generation.prompt import (
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
from retrieval.retriever import RetrievalResult
from tests.conftest import sample_chunks


def make_evidence():
    chunks = sample_chunks()
    return [
        RetrievalResult.from_result_dict({"chunk": c, "fused_score": 1.0})
        for c in chunks
    ]


def test_build_prompt_separates_query_evidence_instructions():
    prompt = build_prompt("Can I disclose?", make_evidence())
    user = prompt["user"]
    query_at = user.index("QUERY:")
    evidence_at = user.index("RETRIEVED EVIDENCE:")
    instructions_at = user.index("INSTRUCTIONS:")
    assert query_at < evidence_at < instructions_at
    assert "Can I disclose?" in user


def test_evidence_blocks_carry_chunk_id_and_source():
    text = format_evidence(make_evidence())
    assert "chunk_id: nda-1" in text
    assert "chunk_id: emp-0" in text
    assert "source: nda" in text
    assert "source: emp" in text
    assert "Confidential Information shall not be disclosed" in text


def test_instructions_require_inline_citations():
    prompt = build_prompt("Q", make_evidence())
    for block in (prompt["system"], prompt["user"]):
        assert "[citation:chunk_id]" in block


def test_instructions_require_i_dont_know():
    prompt = build_prompt("Q", make_evidence())
    for block in (prompt["system"], prompt["user"]):
        assert "don't know" in block or "I don't know" in block


def test_instructions_forbid_unsupported_legal_claims():
    prompt = build_prompt("Q", make_evidence())
    assert "unsupported legal claims" in prompt["system"].lower()


def test_system_contains_dedicated_premise_verification_instruction():
    prompt = build_prompt("Q", make_evidence())
    assert "PREMISE VERIFICATION" in prompt["system"]


def test_verification_section_included_when_supplied():
    verification = PremiseVerification(
        status=STATUS_UNSUPPORTED,
        premises=["the NDA has no term limit"],
        explanation="No chunk states a term limit.",
    )
    prompt = build_prompt("Q", make_evidence(), verification=verification)
    user = prompt["user"]
    assert "PREMISE VERIFICATION:" in user
    assert "STATUS: UNSUPPORTED" in user
    assert "the NDA has no term limit" in user


def test_unsupported_premise_surfaced_in_prompt():
    verification = PremiseVerification(
        status=STATUS_UNSUPPORTED,
        premises=["disclosure is allowed after 2 years"],
        explanation="The evidence is silent on any term limit.",
    )
    prompt = build_prompt("Q", make_evidence(), verification=verification)
    assert "STATUS: UNSUPPORTED" in prompt["user"]
    assert "disclosure is allowed after 2 years" in prompt["user"]
    assert "DO NOT accept that premise as fact" in prompt["user"]


def test_contradicted_premise_surfaced_in_prompt():
    verification = PremiseVerification(
        status=STATUS_CONTRADICTED,
        premises=["the NDA has no term limit"],
        explanation="nda-1 states a two-year term.",
    )
    prompt = build_prompt("Q", make_evidence(), verification=verification)
    assert "STATUS: CONTRADICTED" in prompt["user"]
    assert "the NDA has no term limit" in prompt["user"]


def test_empty_evidence_marks_missing_material():
    prompt = build_prompt("Q", [])
    assert NO_EVIDENCE_TEXT in prompt["user"]


def test_regression_with_milestone3_retrieval_result():
    prompt = build_prompt("salary?", make_evidence())
    assert "Employee shall receive an annual base salary" in prompt["user"]


def test_accepts_plain_evidence_dicts():
    chunk = dict(sample_chunks()[0])
    prompt = build_prompt("Q", [{"chunk": chunk, "source_name": chunk["source_name"]}])
    assert "chunk_id: nda-0" in prompt["user"]


def test_verification_prompt_asks_for_status_format():
    prompt = build_verification_prompt("Q", make_evidence())
    assert "premise verifier" in prompt["system"].lower()
    assert "RETRIEVED EVIDENCE:" in prompt["user"]
    assert "STATUS: supported|unsupported|contradicted" in prompt["user"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STATUS: supported\nPREMISES: none\nEXPLANATION: ok.", STATUS_SUPPORTED),
        (
            "STATUS: unsupported\nPREMISES: [\"no term limit\"]\nEXPLANATION: silent.",
            STATUS_UNSUPPORTED,
        ),
        (
            "STATUS: contradicted\nPREMISES: [\"2-year term\"]\nEXPLANATION: chunk says otherwise.",
            STATUS_CONTRADICTED,
        ),
        (
            "STATUS: CONTRADICTED\nPREMISES: none\nEXPLANATION: chunk contradicts.",
            STATUS_CONTRADICTED,
        ),
    ],
)
def test_parse_verification_statuses(raw, expected):
    result = parse_verification(raw)
    assert result.status == expected


def test_parse_verification_premises_and_explanation():
    result = parse_verification(
        'STATUS: unsupported\nPREMISES: ["a", "b"]\nEXPLANATION: evidence is silent.'
    )
    assert result.premises == ["a", "b"]
    assert result.explanation == "evidence is silent."


def test_parse_verification_no_premises():
    result = parse_verification("STATUS: supported\nPREMISES: none\nEXPLANATION: x")
    assert result.premises == []


def test_parse_verification_defaults_to_unsupported_when_unreadable():
    result = parse_verification("this is not a verifier response at all")
    assert result.status == STATUS_UNSUPPORTED