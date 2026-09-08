"""Tests for ingest.sac: summarization and SAC application."""

import pytest

from ingest.chunker import SentenceChunker
from ingest.sac import (
    SAC_SEPARATOR,
    apply_sac_to_document,
    summarize_document,
)

NDA_TEXT = (
    'This Mutual Non-Disclosure Agreement (the "Agreement") is between '
    '"Acme Corporation" and "Global Enterprises Ltd." covering confidential '
    "information for a potential business relationship. "
    "Section 1 defines Confidential Information."
)


class TestSummarizeDocument:
    def test_returns_summary_result(self):
        res = summarize_document(NDA_TEXT)
        assert res.strategy == "heuristic"
        assert isinstance(res.summary, str)

    def test_uses_llm_when_requested(self):
        calls = []
        def fake_llm(text):
            calls.append(text)
            return "LLM summary"
        res = summarize_document(NDA_TEXT, use_llm=True, llm_func=fake_llm)
        assert res.strategy == "llm"
        assert res.summary == "LLM summary"
        assert calls == [NDA_TEXT]

    def test_use_llm_without_func_raises(self):
        with pytest.raises(ValueError):
            summarize_document(NDA_TEXT, use_llm=True)

    def test_empty_text(self):
        res = summarize_document("")
        assert res.summary == ""

    def test_respects_max_chars(self):
        res = summarize_document(NDA_TEXT, max_chars=30)
        assert len(res.summary) <= 33

    def test_detects_parties(self):
        res = summarize_document(NDA_TEXT, max_chars=300)
        assert "Acme" in res.summary or "Global" in res.summary

    def test_respects_llm_summary_length_unbounded(self):
        def fake(text):
            return "x" * 500
        res = summarize_document(NDA_TEXT, use_llm=True, llm_func=fake, max_chars=30)
        assert len(res.summary) == 500


class TestApplySAC:
    def test_prepends_summary_to_each_chunk(self):
        chunks = SentenceChunker(min_chars=0).chunk("d", "One sentence. Two.")
        summary = "Doc summary"
        out = apply_sac_to_document(chunks, summary)
        assert out is chunks
        for ch in chunks:
            assert ch.sac_applied is True
            assert ch.text.startswith(summary + SAC_SEPARATOR)
            assert ch.original_text
            assert summary in ch.text

    def test_preserves_original_text(self):
        chunks = SentenceChunker(min_chars=0).chunk("d", "Hello. World.")
        orig = [c.original_text for c in chunks]
        apply_sac_to_document(chunks, "SUM")
        assert [c.original_text for c in chunks] == orig
        assert all(c.original_text != c.text for c in chunks)

    def test_empty_chunks(self):
        assert apply_sac_to_document([], "SUM") == []

    def test_double_application_raises(self):
        chunks = SentenceChunker(min_chars=0).chunk("d", "Hi.")
        apply_sac_to_document(chunks, "one")
        with pytest.raises(ValueError):
            apply_sac_to_document(chunks, "two")
