"""Tests for eval/metrics.py: from-scratch precision@k, recall@k, MRR, MAP and
the document/chunk-level run-record helpers (Milestone 7, AGENTS.md §6)."""

import pytest

from eval.metrics import (
    average_precision,
    chunk_level_pair,
    document_level_pair,
    evaluate_queries,
    evaluate_retrieval,
    mean_average_precision,
    mrr,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestCoreMetrics:
    def test_precision_at_k(self):
        # Relevant = {A}; ranked list has A at rank 2.
        assert precision_at_k(["B", "A", "C"], ["A"], k=1) == 0.0
        assert precision_at_k(["B", "A", "C"], ["A"], k=2) == 0.5
        assert precision_at_k(["B", "A", "C"], ["A"], k=3) == pytest.approx(1 / 3)

    def test_precision_at_k_fewer_than_k(self):
        # Only 2 results exist but k=10: denominator shrinks to 2.
        assert precision_at_k(["A", "B"], ["A"], k=10) == 0.5

    def test_precision_at_k_empty_ranking(self):
        assert precision_at_k([], ["A"], k=5) == 0.0

    def test_precision_at_k_rejects_nonpositive_k(self):
        with pytest.raises(ValueError):
            precision_at_k(["A"], ["A"], k=0)
        with pytest.raises(ValueError):
            recall_at_k(["A"], ["A"], k=-1)

    def test_recall_at_k_counts_distinct_relevant(self):
        # Duplicate rows cannot push recall above 1.0.
        assert recall_at_k(["A", "A", "A"], ["A"], k=3) == 1.0
        assert recall_at_k(["B", "A"], ["A", "B"], k=2) == 1.0
        assert recall_at_k(["B", "C"], ["A"], k=2) == 0.0

    def test_recall_at_k_empty_relevant(self):
        assert recall_at_k(["A", "B"], [], k=3) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["B", "A", "C"], ["A"]) == 0.5
        assert reciprocal_rank(["A"], ["A"]) == 1.0
        assert reciprocal_rank(["B", "C"], ["A"]) == 0.0

    def test_average_precision(self):
        # AP sums precision@(first rank of each relevant item), divided by the
        # total number of relevant items. B@1 -> 1/1, A@2 -> 2/2; the duplicate
        # B@3 adds nothing.
        ap = average_precision(["B", "A", "B", "C"], ["A", "B"])
        assert ap == pytest.approx(1.0)

    def test_average_precision_unretrieved_relevant_lowers(self):
        # A is relevant but never retrieved, so AP = (1/1)/2 = 0.5.
        assert average_precision(["B", "C"], ["A", "B"]) == pytest.approx(0.5)

    def test_average_precision_no_hits_and_empty(self):
        assert average_precision(["B", "C"], ["A"]) == 0.0
        assert average_precision(["A"], []) == 0.0

    def test_mrr_and_map(self):
        ranked_and_relevant = [
            (["A", "B", "C"], ["A"]),
            (["B", "A", "C"], ["A"]),
            (["C", "B"], ["A"]),
            (["B", "A", "C"], ["B"]),
        ]
        assert mrr(ranked_and_relevant) == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4)
        expected_aps = [
            average_precision(r, rel) for r, rel in ranked_and_relevant
        ]
        assert mean_average_precision(ranked_and_relevant) == pytest.approx(
            sum(expected_aps) / 4
        )

    def test_mrr_empty_input(self):
        assert mrr([]) == 0.0
        assert mean_average_precision([]) == 0.0


def _run(**overrides):
    run = {
        "query_id": "q1",
        "query": "question",
        "expected_document": "nda",
        "retrieved_sources": ["nda", "nda", "emp"],
        "retrieved_chunks": [
            {"chunk_id": "c1", "source_name": "nda"},
            {"chunk_id": "c2", "source_name": "nda"},
            {"chunk_id": "c3", "source_name": "emp"},
        ],
    }
    run.update(overrides)
    return run


class TestRecordHelpers:
    def test_document_level_pair(self):
        ranked, relevant = document_level_pair(_run())
        assert ranked == ["nda", "nda", "emp"]
        assert relevant == {"nda"}

    def test_chunk_level_pair(self):
        run = _run()
        corpus = [
            {"chunk_id": "c1", "source_name": "nda"},
            {"chunk_id": "c2", "source_name": "nda"},
            {"chunk_id": "c9", "source_name": "nda"},  # exists but not retrieved
        ]
        ranked, relevant = chunk_level_pair(run, corpus)
        assert ranked == ["c1", "c2", "c3"]
        assert relevant == {"c1", "c2", "c9"}


class TestEvaluate:
    def test_evaluate_retrieval_document_level(self):
        result = evaluate_retrieval(_run(), k_values=[1, 2, 3])
        assert result["precision_at_k"] == pytest.approx({1: 1.0, 2: 1.0, 3: 2 / 3})
        assert result["recall_at_k"] == pytest.approx({1: 1.0, 2: 1.0, 3: 1.0})
        assert result["reciprocal_rank"] == 1.0
        assert result["average_precision"] == 1.0

    def test_evaluate_retrieval_miss(self):
        run = _run(retrieved_sources=["emp", "emp"], expected_document="nda")
        result = evaluate_retrieval(run, k_values=[1, 2])
        assert result["precision_at_k"][1] == 0.0
        assert result["recall_at_k"][2] == 0.0
        assert result["reciprocal_rank"] == 0.0

    def test_evaluate_retrieval_unknown_level(self):
        with pytest.raises(ValueError):
            evaluate_retrieval(_run(), level="bogus")

    def test_evaluate_queries_aggregate(self):
        runs = [
            _run(query_id="q1", retrieved_sources=["nda", "nda", "emp"], expected_document="nda"),
            _run(query_id="q2", retrieved_sources=["emp", "nda", "emp"], expected_document="nda"),
            _run(query_id="q3", retrieved_sources=["emp", "emp", "emp"], expected_document="nda"),
        ]
        out = evaluate_queries(runs, k_values=[1, 3])
        agg = out["aggregate"]
        assert agg["precision_at_k"][1] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
        assert agg["precision_at_k"][3] == pytest.approx((2 / 3 + 1 / 3 + 0.0) / 3)
        assert agg["recall_at_k"][3] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
        assert agg["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert len(out["per_query"]) == 3

    def test_evaluate_queries_empty(self):
        out = evaluate_queries([], k_values=[1, 3])
        assert out["aggregate"]["mrr"] == 0.0
        assert out["aggregate"]["map"] == 0.0
        assert out["aggregate"]["precision_at_k"] == {1: 0.0, 3: 0.0}