"""Tests for eval/failure_point_tagger.py: the survey's seven failure points
plus the project's eighth (unsupported premise), as structured, deterministic
tags over a normalized RAG-result record."""

from eval.failure_point_tagger import (
    ALL_FAILURE_POINTS,
    STOP_REASON_MAX_ROUNDS,
    TAG_CITATION_FAILURE,
    TAG_CONTEXT_INTEGRATION_FAILURE,
    TAG_EFFICIENCY_FAILURE,
    TAG_GENERATION_FAILURE,
    TAG_HALLUCINATION,
    TAG_INTERPRETABILITY_FAILURE,
    TAG_RETRIEVAL_FAILURE,
    TAG_UNSUPPORTED_PREMISE,
    FailureTag,
    failure_point_label,
    failure_point_summary,
    normalize_result,
    tag_result,
    tag_results,
)


def _chunk(cid, source, text, section="1", parent=None):
    return {
        "chunk_id": cid,
        "source_name": source,
        "text": f"SUMMARY for {source}.\n\n{text}",
        "original_text": text,
        "section": section,
        "parent": parent,
        "heading": section,
        "hierarchy_path": [section] if parent is None else [parent, section],
    }


def _run(query_id="q1", *, answer="", cited=None, retrieved=None, expected="nda",
         verification=None, rounds=1, stop_reason=None):
    return normalize_result(
        query_id=query_id,
        query="what obligations bind the parties?",
        expected_document=expected,
        retrieved=retrieved or [],
        answer=answer,
        cited_chunk_ids=cited,
        verification=verification,
        rounds_used=rounds,
        stop_reason=stop_reason,
    )


def _grounded_answer(cid, body):
    return f"{body} [citation:{cid}]"


class TestNormalizeResult:
    def test_normalizes_plain_dicts(self):
        chunk = _chunk("c1", "nda", "Confidential information is protected.")
        record = _run(
            answer="The information is protected. [citation:c1]",
            cited=["c1"],
            retrieved=[chunk],
        )
        assert record["retrieved_sources"] == ["nda"]
        assert record["retrieved_chunks"][0]["chunk_id"] == "c1"
        assert record["verification"] == {}
        assert record["rounds_used"] == 1

    def test_missing_fields_default(self):
        record = _run()
        assert record["retrieved_sources"] == []
        assert record["answer"] == ""
        assert record["cited_chunk_ids"] == []


class TestTags:
    BODY = "Confidential Information shall not be disclosed to third parties."

    def _grounded(self, source="nda", extra=""):
        chunks = [_chunk("c1", source, self.BODY)]
        return _run(
            answer=_grounded_answer("c1", self.BODY + extra),
            cited=["c1"],
            retrieved=chunks,
            expected=source,
        )

    def test_perfect_response_has_no_tags(self):
        assert tag_result(self._grounded()) == []

    def test_retrieval_failure_nothing_retrieved(self):
        tags = tag_result(_run())
        assert any(t.tag == TAG_RETRIEVAL_FAILURE for t in tags)
        assert any(t.tag == TAG_GENERATION_FAILURE for t in tags)  # empty answer

    def test_retrieval_failure_wrong_document(self):
        chunks = [_chunk("c1", "emp", "The employee handbook.")]
        tags = tag_result(_run(retrieved=chunks, answer="The handbook. [citation:c1]", cited=["c1"], expected="nda"))
        assert any(t.tag == TAG_RETRIEVAL_FAILURE for t in tags)

    def test_context_integration_failure(self):
        chunks = [
            _chunk("c-nda", "nda", self.BODY),
            _chunk("c-emp", "emp", "Employee receives a salary."),
        ]
        answer = "The obligations are stated in the cited source. [citation:c-emp]"
        tags = tag_result(_run(retrieved=chunks, answer=answer, cited=["c-emp"], expected="nda"))
        assert any(t.tag == TAG_CONTEXT_INTEGRATION_FAILURE for t in tags)

    def test_generation_failure_idk_with_evidence(self):
        chunks = [_chunk("c1", "nda", self.BODY)]
        tags = tag_result(_run(retrieved=chunks, answer="I don't know.", expected="nda"))
        assert any(t.tag == TAG_GENERATION_FAILURE for t in tags)

    def test_hallucination_with_citation_failure(self):
        chunks = [_chunk("c1", "nda", self.BODY)]
        answer = (
            "The parties are legally required to disclose all trade secrets "
            "to competitors. [citation:never-retrieved-chunk-999]"
        )
        tags = tag_result(_run(retrieved=chunks, answer=answer, cited=["never-retrieved-chunk-999"], expected="nda"))
        tag_ids = {t.tag for t in tags}
        assert TAG_HALLUCINATION in tag_ids
        assert TAG_CITATION_FAILURE in tag_ids
        # The published document WAS retrieved but the cited chunk never was.
        assert TAG_RETRIEVAL_FAILURE not in tag_ids

    def test_citation_failure_no_citations(self):
        chunks = [_chunk("c1", "nda", self.BODY)]
        tags = tag_result(_run(retrieved=chunks, answer=self.BODY, expected="nda"))
        assert any(t.tag == TAG_CITATION_FAILURE for t in tags)

    def test_efficiency_failure_extra_rounds(self):
        record = self._grounded()
        record["rounds_used"] = 3
        tags = tag_result(record)
        assert [t.tag for t in tags] == [TAG_EFFICIENCY_FAILURE]

    def test_efficiency_failure_max_rounds_stop(self):
        record = self._grounded()
        record["rounds_used"] = 2
        record["stop_reason"] = STOP_REASON_MAX_ROUNDS
        tags = tag_result(record)
        assert any(t.tag == TAG_EFFICIENCY_FAILURE for t in tags)

    def test_interpretability_failure_no_hierarchy(self):
        chunk = _chunk("c1", "nda", self.BODY)
        for key in ("section", "parent", "heading", "hierarchy_path"):
            chunk[key] = None
        tags = tag_result(_run(retrieved=[chunk], answer=_grounded_answer("c1", self.BODY), cited=["c1"], expected="nda"))
        assert any(t.tag == TAG_INTERPRETABILITY_FAILURE for t in tags)

    def test_unsupported_premise(self):
        chunks = [_chunk("c1", "nda", self.BODY)]
        record = _run(
            retrieved=chunks,
            answer=_grounded_answer("c1", self.BODY),
            cited=["c1"],
            expected="nda",
            verification={
                "status": "unsupported",
                "premises": ["the agreement has no term limit"],
                "explanation": "no chunk mentions a term limit.",
            },
        )
        tags = tag_result(record)
        assert [t.tag for t in tags] == [TAG_UNSUPPORTED_PREMISE]

    def test_multiple_tags(self):
        chunks = [_chunk("c1", "nda", self.BODY)]
        record = _run(
            retrieved=chunks,
            answer="I don't know.",
            expected="nda",
            rounds=3,
            stop_reason=STOP_REASON_MAX_ROUNDS,
        )
        ids = {t.tag for t in tag_result(record)}
        assert {TAG_GENERATION_FAILURE, TAG_EFFICIENCY_FAILURE} <= ids


class TestFailureTagShape:
    def test_as_dict(self):
        tag = FailureTag(TAG_HALLUCINATION, "reason here", ("claim-1",))
        as_dict = tag.as_dict()
        assert as_dict["tag"] == TAG_HALLUCINATION
        assert as_dict["label"] == failure_point_label(TAG_HALLUCINATION)
        assert as_dict["reason"] == "reason here"
        assert as_dict["evidence"] == ["claim-1"]

    def test_failure_point_label_defaults_to_id(self):
        assert failure_point_label("not-a-real-tag") == "not-a-real-tag"

    def test_all_failure_points_have_labels(self):
        from eval.failure_point_tagger import FAILURE_POINT_LABELS

        assert set(ALL_FAILURE_POINTS) == set(FAILURE_POINT_LABELS)
        # Every known tag resolves to its label, distinct from the tag id.
        for tag in ALL_FAILURE_POINTS:
            assert failure_point_label(tag) == FAILURE_POINT_LABELS[tag]
            assert failure_point_label(tag) != tag


class TestTagResultsAndSummary:
    def test_tag_results_keyed_by_query_id(self):
        records = [self._clean("a"), self._clean("b")]
        tagged = tag_results(records)
        assert set(tagged) == {"a", "b"}

    @staticmethod
    def _clean(qid):
        body = "Confidential Information shall not be disclosed to third parties."
        chunk = _chunk("c1", "nda", body)
        return normalize_result(
            query_id=qid,
            query="?",
            expected_document="nda",
            retrieved=[chunk],
            answer=f"{body} [citation:c1]",
            cited_chunk_ids=["c1"],
        )

    def test_failure_point_summary_zeroes_unused(self):
        tagged = {"a": []}
        summary = failure_point_summary(tagged)
        assert summary["tagged_queries"] == 0
        assert summary["tagged_queries_fraction"] == 0.0
        assert set(summary["counts"]) == set(ALL_FAILURE_POINTS)
        assert all(v == 0 for v in summary["counts"].values())

    def test_failure_point_summary_counts(self):
        tagged = {"a": [FailureTag(TAG_HALLUCINATION, "r")], "b": []}
        summary = failure_point_summary(tagged)
        assert summary["tagged_queries"] == 1
        assert summary["tagged_queries_fraction"] == 0.5
        assert summary["counts"][TAG_HALLUCINATION] == 1