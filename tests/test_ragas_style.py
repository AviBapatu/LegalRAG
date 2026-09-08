"""Tests for eval/ragas_style.py: RAGAs-STYLE faithfulness / answer relevance /
context relevance, deterministic heuristics, judge injection, and clamping."""

import pytest

from eval import ragas_style
from eval.ragas_style import (
    context_relevance,
    context_texts,
    evaluate_rag_results,
    faithfulness,
    answer_relevance,
)


class TestFaithfulness:
    def test_fully_supported_answer(self):
        context = [
            "The confidentiality obligations of the agreement bind both parties "
            "for two years."
        ]
        answer = "The confidentiality obligations bind both parties for two years."
        assert faithfulness(answer, context) == 1.0

    def test_partially_supported(self):
        context = ["The term of the agreement is two years."]
        answer = (
            "The term of the agreement is two years. "
            "The parties must disclose all trade secrets to the public."
        )
        score = faithfulness(answer, context)
        assert 0.0 < score < 1.0

    def test_unsupported_answer(self):
        context = ["The agreement has no term limit."]
        answer = "The parties must publicize all confidential information."
        assert faithfulness(answer, context) == 0.0

    def test_empty_answer_and_context(self):
        assert faithfulness("", ["context"]) == 0.0
        assert faithfulness("Some answer.", []) == 0.0

    def test_judge_overrides_heuristic(self):
        class Judge:
            def score(self, *, system, user):
                assert system and user
                return 0.75

        assert faithfulness("The term is five years.", ["two years"], judge=Judge()) == 0.75

    def test_judge_out_of_range_raises(self):
        class Judge:
            def score(self, *, system, user):
                return 1.5

        with pytest.raises(ValueError):
            faithfulness("x", ["y"], judge=Judge())


class TestAnswerRelevance:
    def test_direct_answer(self):
        query = "what is the term of the nda?"
        answer = "The term of the nda is two years."
        assert answer_relevance(answer, query) > 0.0

    def test_idontknow_scores_zero(self):
        query = "what is the term of the nda?"
        assert answer_relevance("I don't know.", query) == 0.0
        assert answer_relevance("I do not know.", query) == 0.0

    def test_empty_answer_and_empty_query(self):
        assert answer_relevance("", "a question") == 0.0
        assert answer_relevance("An answer.", "") == 0.0

    def test_judge_override(self):
        class Judge:
            def score(self, *, system, user):
                return 0.9

        assert answer_relevance("anything", "query", judge=Judge()) == 0.9


class TestContextRelevance:
    def test_all_chunks_relevant(self):
        query = "what obligations bind the parties?"
        context = [
            "The parties shall keep confidential information private.",
            "Obligations include non-disclosure of trade secrets.",
        ]
        assert context_relevance(query, context) == 1.0

    def test_no_overlap(self):
        query = "what color is the car?"
        context = ["The term of this agreement is two years."]
        assert context_relevance(query, context) == 0.0

    def test_empty_context(self):
        assert context_relevance("anything", []) == 0.0

    def test_mixed(self):
        query = "term of the nda"
        context = ["The nda term is two years.", "Parties negotiate pricing."]
        # Second chunk shares no content tokens with the query.
        assert context_relevance(query, context) == 0.5


class TestContextTexts:
    def test_dict_chunks_and_texts(self):
        chunks = [{"text": "alpha"}, {"original_text": "beta"}]
        assert context_texts(chunks) == ["alpha", "beta"]

    def test_passthrough_accepts_texts(self):
        assert context_relevance("term of nda", ["the nda term is two years"]) == 1.0


class TestEvaluateRagResults:
    def test_returns_all_three(self):
        result = evaluate_rag_results(
            "The term of the nda is two years.",
            "what is the term?",
            ["The term of the nda is two years."],
        )
        assert set(result) == {"faithfulness", "answer_relevance", "context_relevance"}
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_prompt_builders_exist(self):
        assert ragas_style.FAITHFULNESS_SYSTEM
        assert ragas_style.ANSWER_RELEVANCE_SYSTEM
        assert ragas_style.CONTEXT_RELEVANCE_SYSTEM