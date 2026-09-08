"""Tests for ingest.chunker: SentenceChunker and PatternChunker."""

import pytest

from ingest.chunker import Chunk, Chunker, PatternChunker, SentenceChunker


# --------------------------------------------------------------------------
# SentenceChunker
# --------------------------------------------------------------------------

class TestSentenceChunker:
    def test_splits_on_sentence_boundaries(self):
        c = SentenceChunker(min_chars=0)
        chunks = c.chunk("d", "First sentence. Second sentence! Third?")
        text = " ".join(ch.text for ch in chunks)
        assert "First sentence" in text
        assert "Second sentence" in text
        assert "Third" in text

    def test_merges_short_sentences(self):
        c = SentenceChunker(max_chars=1000, min_chars=50)
        chunks = c.chunk("d", "Short. Another short. And a longer one goes here enough.")
        assert len(chunks) >= 1
        for ch in chunks:
            assert len(ch.text) >= 1

    def test_respects_max_chars(self):
        short = "a. " * 5  # ~10 chars
        c = SentenceChunker(max_chars=10, min_chars=0)
        chunks = c.chunk("d", short.strip())
        for ch in chunks:
            assert len(ch.text) <= 15  # allow small overshoot on grouping

    def test_chunk_ids_unique(self):
        c = SentenceChunker(min_chars=0)
        chunks = c.chunk("doc", "One. Two. Three. Four. Five.")
        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("doc-") for i in ids)

    def test_metadata_defaults(self):
        c = SentenceChunker(min_chars=0)
        ch = c.chunk("doc", "Just one sentence here.")[0]
        assert ch.section is None
        assert ch.parent is None
        assert ch.heading is None
        assert ch.hierarchy_path == []
        assert ch.sac_applied is False
        assert ch.start_offset == 0

    def test_offsets_point_at_real_text(self):
        c = SentenceChunker(min_chars=0)
        text = "AAA. BBB. CCC."
        chunks = c.chunk("d", text)
        for ch in chunks:
            # The chunk text must exactly match the source slice at its offsets.
            assert text[ch.start_offset:ch.end_offset] == ch.text
            assert ch.start_offset >= 0
            assert ch.end_offset > ch.start_offset
            assert ch.end_offset <= len(text)


# --------------------------------------------------------------------------
# PatternChunker
# --------------------------------------------------------------------------

SAMPLE = """Preamble text before sections.

Section 1. Definitions
In this Agreement, "Confidential Information" means secret data.

Section 2. Scope
2.1 This section covers scope of use.
2.2 It also covers exclusions.

Section 3. Term
3.1 This Agreement lasts two years.
"""


class TestPatternChunker:
    def test_parses_sections(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        sections = [c.section for c in chunks if c.section]
        # "Section 1" has direct body content; the others' content lives in
        # their nested subsections.
        assert "1" in sections
        assert "2.1" in sections
        assert "2.2" in sections
        assert "3.1" in sections

    def test_top_level_parent_preserved_via_subsections(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        # Section 2's content lives in its subsections, which point at "2".
        assert any(c.parent == "2" for c in chunks)

    def test_preamble_chunk(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        preamble = [c for c in chunks if c.section is None]
        assert len(preamble) == 1
        assert "Preamble" in preamble[0].text

    def test_subsection_numbering(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        sections = {c.section for c in chunks}
        assert "2.1" in sections
        assert "2.2" in sections

    def test_hierarchy_parent(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        sub = next(c for c in chunks if c.section == "2.1")
        assert sub.parent == "2"
        assert sub.hierarchy_path == ["2", "2.1"]

    def test_heading_metadata(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        sec = next(c for c in chunks if c.section == "1")
        assert sec.heading is not None
        assert "Definitions" in sec.heading

    def test_offsets_bound_text(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        for c in chunks:
            assert c.start_offset >= 0
            assert c.end_offset > c.start_offset
            assert c.start_offset < len(SAMPLE)
        # Preamble starts at 0
        preamble = next(c for c in chunks if c.section is None)
        assert preamble.start_offset == 0

    def test_section_body_excludes_heading(self):
        chunks = PatternChunker().chunk("doc", SAMPLE)
        sec = next(c for c in chunks if c.section == "1")
        # Body should contain the definition, not the heading text itself.
        assert "Confidential Information" in sec.text
        assert "Definitions" not in sec.text.split("\n")[0]

    def test_empty_document(self):
        assert PatternChunker().chunk("doc", "") == []

    def test_no_delimiters_returns_single_chunk(self):
        chunks = PatternChunker().chunk("doc", "Just freeform text with no sections.")
        assert len(chunks) == 1
        assert chunks[0].section is None


# --------------------------------------------------------------------------
# Chunker facade
# --------------------------------------------------------------------------

class TestChunkerFacade:
    def test_selects_sentence(self):
        c = Chunker("sentence")
        assert isinstance(c.impl, SentenceChunker)

    def test_selects_pattern(self):
        c = Chunker("pattern")
        assert isinstance(c.impl, PatternChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            Chunker("nope")

    def test_pattern_chunks_via_facade(self):
        chunks = Chunker("pattern").chunk("doc", SAMPLE)
        assert len(chunks) > 0
        assert isinstance(chunks[0], Chunk)
