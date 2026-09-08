"""Tests for eval/run_eval.py: the unified Milestone 7 harness
(config resolution, RAG-result normalization, full-end-to-end evaluation,
deterministic reports, serialization, and the offline smoke rag_fn).

Everything here runs in tmp dirs with scripted retrieval so pytest never
builds real indexes, downloads models, or touches the in-repo data/eval corpus.
"""

import json

import pytest

from eval.run_eval import (
    DEFAULT_FULL_EVAL_DIR,
    FULL_EVAL_VERSION,
    build_smoke_rag_fn,
    normalize_rag_result,
    render_markdown_report,
    resolve_eval_config,
    resolve_k_values,
    resolve_top_k,
    run_full_evaluation,
    save_report,
)
from generation.prompt import PremiseVerification
from retrieval.retriever import RetrievalResult

BODY = "Confidential Information shall not be disclosed to third parties."


def _chunk(cid, source, body=BODY, section="1", parent=None):
    return {
        "chunk_id": cid,
        "source_name": source,
        "text": f"SUMMARY for {source}.\n\n{body}",
        "original_text": body,
        "section": section,
        "parent": parent,
        "heading": section,
        "hierarchy_path": [section] if parent is None else [parent, section],
    }


def _queries(n=2):
    return [
        {"query_id": f"q{i}", "query": f"query number {i}?", "expected_document": "nda"}
        for i in range(n)
    ]


def _grounded_rag_fn():
    chunk = _chunk("c1", "nda")

    def rag_fn(query, qrecord):
        return {
            "retrieved": [chunk],
            "answer": f"{BODY} [citation:c1]",
            "cited_chunk_ids": ["c1"],
            "verification": PremiseVerification(status="supported"),
            "rounds_used": 1,
        }

    return rag_fn


class TestConfigResolution:
    def test_resolve_eval_config(self):
        assert resolve_eval_config({"evaluation": {"k_values": [3]}}) == {"k_values": [3]}
        assert resolve_eval_config({}) == {}

    def test_resolve_k_values_priority(self):
        assert resolve_k_values({"k_values": [3, 5]}) == [3, 5]
        assert resolve_k_values({}, k_values=[7]) == [7]
        assert resolve_k_values({}) == [1, 3, 5, 10]

    def test_resolve_k_values_validation(self):
        with pytest.raises(ValueError):
            resolve_k_values({}, k_values=[])
        with pytest.raises(ValueError):
            resolve_k_values({}, k_values=[0, 3])
        with pytest.raises(ValueError):
            resolve_k_values({}, k_values=[5, 3])

    def test_resolve_top_k(self):
        assert resolve_top_k([1, 3, 5, 10]) == 10
        assert resolve_top_k([1, 3, 5], top_k=4) == 4
        with pytest.raises(ValueError):
            resolve_top_k([1], top_k=0)


class TestNormalizeResult:
    def test_normalizes_rag_function_output(self):
        chunk = _chunk("c1", "nda", "Confidential Information protected.")
        record = normalize_rag_result(
            {"query_id": "q1", "query": "q?", "expected_document": "nda"},
            {
                "retrieved": [chunk],
                "answer": "Confidential Information protected. [citation:c1]",
                "cited_chunk_ids": ["c1"],
            },
        )
        assert record["query_id"] == "q1"
        assert record["retrieved_sources"] == ["nda"]
        assert record["cited_chunk_ids"] == ["c1"]
        assert record["rounds_used"] is None  # absent from output -> None


class TestRunFullEvaluation:
    def test_clean_system_scores_perfectly(self):
        queries = _queries()
        result = run_full_evaluation(
            queries,
            _grounded_rag_fn(),
            eval_config={"evaluation": {}},
            timestamp="2024-01-01T00:00:00+00:00",
        )
        agg = result["aggregate"]
        assert agg["num_queries"] == 2
        assert agg["precision_at_k"][1] == 1.0
        assert agg["recall_at_k"][1] == 1.0
        assert agg["mrr"] == 1.0
        assert agg["drm_fraction"] == 0.0
        assert agg["faithfulness"] == 1.0
        assert agg["claim_level_faithfulness"] == 1.0
        assert agg["failure_tags"]["tagged_queries"] == 0
        assert len(result["per_query"]) == 2

    def test_deterministic_given_timestamp(self):
        result_a = run_full_evaluation(_queries(), _grounded_rag_fn(), timestamp="2024-01-01T00:00:00+00:00")
        result_b = run_full_evaluation(_queries(), _grounded_rag_fn(), timestamp="2024-01-01T00:00:00+00:00")
        assert result_a == result_b

    def test_flags_turn_modules_off(self):
        result = run_full_evaluation(
            _queries(),
            _grounded_rag_fn(),
            enable_ragas_style=False,
            enable_claim_check=False,
            enable_failure_tags=False,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        per_query = result["per_query"][0]
        assert "ragas" not in per_query
        assert "claims" not in per_query
        assert "failure_tags" not in per_query
        assert "faithfulness" not in result["aggregate"]
        assert "claim_level_faithfulness" not in result["aggregate"]
        assert "failure_tags" not in result["aggregate"]

    def test_flags_come_from_config(self):
        result = run_full_evaluation(
            _queries(),
            _grounded_rag_fn(),
            eval_config={"evaluation": {"enable_ragas_style": False}},
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert "ragas" not in result["per_query"][0]

    def test_judge_is_used_and_recorded(self):
        class Judge:
            def score(self, *, system, user):
                return 0.5

        result = run_full_evaluation(
            _queries(), _grounded_rag_fn(), judge=Judge(),
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert result["aggregate"]["faithfulness"] == 0.5
        assert result["envelope"]["scoring"]["judge"] == "Judge"

    def test_corpus_chunks_enrich_retrieved_text(self):
        textless = {
            "chunk_id": "c1",
            "source_name": "nda",
            "section": "1",
            "parent": None,
            "heading": "1",
            "hierarchy_path": ["1"],
        }
        corpus_chunks = [
            dict(textless, text=f"SUMMARY for nda.\n\n{BODY}", original_text=BODY)
        ]

        def rag_fn(query, qrecord):
            return {
                "retrieved": [dict(textless)],
                "answer": f"{BODY} [citation:c1]",
                "cited_chunk_ids": ["c1"],
                "verification": PremiseVerification(status="supported"),
                "rounds_used": 1,
            }

        result = run_full_evaluation(
            _queries(),
            rag_fn,
            corpus_chunks=corpus_chunks,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        # Without enrichment the evidence is text-empty and everything fails;
        # with it, the grounded answer scores perfectly.
        assert result["aggregate"]["faithfulness"] == 1.0
        assert result["aggregate"]["claim_level_faithfulness"] == 1.0
        assert result["aggregate"]["failure_tags"]["tagged_queries"] == 0

    def test_drm_fraction_reflects_wrong_document_retrieval(self):
        def rag_fn(query, qrecord):
            return {
                "retrieved": [_chunk("c-wrong", "emp")],
                "answer": "Something grounded. [citation:c-wrong]",
                "cited_chunk_ids": ["c-wrong"],
                "verification": PremiseVerification(status="supported"),
                "rounds_used": 1,
            }

        result = run_full_evaluation(_queries(), rag_fn, timestamp="2024-01-01T00:00:00+00:00")
        assert result["aggregate"]["drm_fraction"] == 1.0
        assert result["aggregate"]["top1_accuracy"] == 0.0
        assert all(q["drm_fraction"] == 1.0 for q in result["per_query"])
        assert result["aggregate"]["failure_tags"]["counts"]["retrieval_failure"] == 2


class TestSmokeRagFn:
    def _retrieval_results(self):
        return [
            RetrievalResult.from_result_dict({"chunk": _chunk("c-nda", "nda"), "source_name": "nda"}),
            RetrievalResult.from_result_dict({
                "chunk": _chunk("c-emp", "emp", "Employee receives a salary."),
                "source_name": "emp",
            }),
        ]

    class _StubRetriever:
        def __init__(self, results):
            self.results = results

        def retrieve(self, query, top_k=None):
            return self.results

    def _smoke(self, perturb):
        queries = _queries(n=9)
        fn = build_smoke_rag_fn(
            self._StubRetriever(self._retrieval_results()),
            queries,
            retrieve_top_k=3,
            perturb=perturb,
        )
        return queries, fn

    def test_clean_queries_are_grounded(self):
        queries, fn = self._smoke(perturb=False)
        for q in queries:
            out = fn(q["query"], q)
            assert out["cited_chunk_ids"] == ["c-nda"]
            assert "Confidential Information" in out["answer"]
            assert out["rounds_used"] == 1

    def test_every_perturbation_branch_fires(self):
        queries, fn = self._smoke(perturb=True)
        outputs = {q["query_id"]: fn(q["query"], q) for q in queries}
        n = len(queries)
        assert outputs[f"q{n - 1}"]["cited_chunk_ids"] == ["never-retrieved-chunk-999"]
        assert outputs[f"q{n - 2}"]["answer"] == "I don't know."
        assert outputs[f"q{n - 3}"]["verification"].status == "unsupported"
        assert outputs[f"q{n - 4}"]["cited_chunk_ids"] == ["c-emp"]
        efficiency = outputs[f"q{n - 5}"]
        assert efficiency["rounds_used"] == 3
        assert efficiency["stop_reason"] == "max_extra_rounds_reached"
        stripped = outputs[f"q{n - 6}"]["retrieved"]
        assert all(r.section is None and not r.hierarchy_path for r in stripped)
        # The remaining queries stay clean and grounded.
        assert outputs["q0"]["cited_chunk_ids"] == ["c-nda"]


class TestReportRender:
    def _result(self):
        # Mixed: one clean query + one hallucinating query.
        clean = _queries(2)
        chunk = _chunk("c1", "nda")

        def rag_fn(query, qrecord):
            if qrecord["query_id"] == "q0":
                return {
                    "retrieved": [chunk],
                    "answer": f"{BODY} [citation:c1]",
                    "cited_chunk_ids": ["c1"],
                    "verification": PremiseVerification(status="supported"),
                    "rounds_used": 1,
                }
            return {
                "retrieved": [chunk],
                "answer": (
                    "The parties are legally required to disclose trade secrets "
                    "to competitors. [citation:bogus]"
                ),
                "cited_chunk_ids": ["bogus"],
                "verification": PremiseVerification(status="supported"),
                "rounds_used": 1,
            }

        return run_full_evaluation(
            clean, rag_fn, timestamp="2024-01-01T00:00:00+00:00",
        )

    def test_markdown_has_all_sections(self):
        md = render_markdown_report(self._result())
        assert "Full Evaluation Report" in md
        assert "Aggregate metrics" in md
        assert "Failure-point tags" in md
        assert "Per-query results" in md
        assert "Reproduction metadata" in md
        assert "Citation failure" in md

    def test_markdown_deterministic(self):
        result = self._result()
        assert render_markdown_report(result) == render_markdown_report(result)

    def test_save_report_writes_distinct_names(self, tmp_path):
        result = self._result()
        json_path, md_path = save_report(result, tmp_path / "full_eval")
        # Must NOT collide with the Milestone 4 DRM file names.
        assert json_path.name == "results.json"
        assert md_path.name == "report.md"
        on_disk = json.loads(json_path.read_text())
        assert on_disk["envelope"]["version"] == FULL_EVAL_VERSION
        assert on_disk["aggregate"]["num_queries"] == 2
        assert on_disk["aggregate"]["failure_tags"]["counts"]["hallucination"] == 1
        assert on_disk["aggregate"]["failure_tags"]["counts"]["citation_failure"] == 1

    def test_save_report_deterministic_bytes(self, tmp_path):
        result = self._result()
        p1, _ = save_report(result, tmp_path / "a")
        p2, _ = save_report(result, tmp_path / "b")
        assert p1.read_bytes() == p2.read_bytes()