"""Tests for eval/claim_check.py: claim extraction, deterministic per-claim
support checking (supported/unsupported/contradicted), and claim-level
faithfulness scoring."""

import pytest

from eval.claim_check import (
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    SUPPORT_THRESHOLD,
    ClaimCheck,
    ClaimChecker,
    ClaimExtractor,
    ClaimLevelReport,
    check_claim,
    claim_level_faithfulness,
    extract_claims,
    normalize_evidence,
)


def _evidence(chunks_text, chunk_id="c1"):
    return [{"chunk_id": chunk_id, "text": chunks_text}]


class TestExtractClaims:
    def test_splits_sentences(self):
        claims = extract_claims(
            "The term is two years. The parties must keep it confidential."
        )
        assert len(claims) == 2
        assert claims[0].startswith("The term is two years")
        assert claims[1].startswith("The parties")

    def test_keeps_initials_together(self):
        claims = extract_claims("The N.D.A. term is two years. Parties comply.")
        assert len(claims) <= 2
        assert "N.D.A" in claims[0]

    def test_empty_answer(self):
        assert extract_claims("") == []

    def test_injectable_extractor(self):
        class Extractor:
            def extract(self, answer):
                return ["claim one", "claim two"]

        assert extract_claims("anything", extractor=Extractor()) == ["claim one", "claim two"]

    def test_strips_citation_markers(self):
        claims = extract_claims(
            "The term is two years. [citation:nda-1] The parties comply."
        )
        assert claims == ["The term is two years.", "The parties comply."]

    def test_citation_only_answer_has_no_claims(self):
        assert extract_claims("[citation:nda-1]") == []


class TestCheckClaim:
    def test_supported(self):
        result = check_claim(
            "The term is two years.",
            _evidence("The term is two years."),
        )
        assert result.status == STATUS_SUPPORTED
        assert "grounded" in result.reason
        assert result.evidence == ["c1"]

    def test_unsupported_silent_evidence(self):
        result = check_claim(
            "The parties elected a board of directors.",
            _evidence("The term is two years."),
        )
        assert result.status == STATUS_UNSUPPORTED
        assert "does not contain" in result.reason
        assert result.evidence == []

    def test_numeric_contradiction(self):
        # Token coverage below threshold AND the number in the claim (5) is
        # disjoint from the number in the evidence (2).
        result = check_claim(
            "The consulting arrangement lasts five years in Asia only.",
            _evidence("The contract lasts two years in a consulting engagement."),
        )
        assert result.status == STATUS_CONTRADICTED
        assert "conflict" in result.reason

    def test_contradiction_requires_low_coverage(self):
        # Same number, high coverage -> supported, not contradicted.
        result = check_claim(
            "The term is two years.",
            _evidence("The term is two years for both parties."),
        )
        assert result.status == STATUS_SUPPORTED

    def test_empty_evidence_is_unsupported(self):
        result = check_claim("Some claim.", [])
        assert result.status == STATUS_UNSUPPORTED

    def test_injectable_checker(self):
        class Checker:
            def check(self, claim, evidence):
                return ClaimCheck(claim_id="", claim=claim, status=STATUS_UNSUPPORTED, reason="fake")

        result = check_claim("X", _evidence("Y"), checker=Checker())
        assert result.status == STATUS_UNSUPPORTED
        assert result.reason == "fake"


class TestNormalizeEvidence:
    def test_dict_evidence(self):
        out = normalize_evidence([{"chunk_id": "c1", "text": "hello"}])
        assert out == [{"chunk_id": "c1", "text": "hello"}]

    def test_inner_chunk_dict(self):
        out = normalize_evidence([{"chunk_id": "wrap", "chunk": {"chunk_id": "inner", "text": "body"}}])
        assert out == [{"chunk_id": "inner", "text": "body"}]

    def test_non_mapping_skipped(self):
        assert normalize_evidence([42, "text"]) == []


class TestClaimLevelFaithfulness:
    def test_fraction_of_supported(self):
        answer = (
            "The consulting arrangement lasts five years in Asia only. "
            "The term is two years."
        )
        evidence = [
            {
                "chunk_id": "c1",
                "text": "The contract lasts two years in a consulting engagement.",
            }
        ]
        report = claim_level_faithfulness(answer, evidence)
        assert report.score == pytest.approx(0.5)
        assert report.num_claims == 2
        assert report.counts == {STATUS_SUPPORTED: 1, STATUS_UNSUPPORTED: 0, STATUS_CONTRADICTED: 1}
        assert report.unsupported_fraction == pytest.approx(0.5)

    def test_all_supported(self):
        answer = "The term is two years for both parties."
        report = claim_level_faithfulness(answer, _evidence("The term is two years for both parties."))
        assert report.score == 1.0
        assert report.counts[STATUS_UNSUPPORTED] == 0
        assert report.counts[STATUS_CONTRADICTED] == 0

    def test_empty_answer(self):
        report = claim_level_faithfulness("", _evidence("anything"))
        assert report.score == 0.0
        assert report.num_claims == 0
        assert report.unsupported_fraction == 0.0

    def test_claim_ids_are_stable(self):
        answer = "The term is two years. The parties comply."
        report = claim_level_faithfulness(answer, _evidence("The term is two years. The parties comply."))
        assert [c.claim_id for c in report.checks] == ["claim-1", "claim-2"]

    def test_custom_prefix(self):
        answer = "The term is two years."
        report = claim_level_faithfulness(answer, _evidence("The term is two years."), claim_id_prefix="c")
        assert report.checks[0].claim_id == "c-1"