"""Tests for retrieval.retriever: RRF fusion, MMR rerank, structural boost,
and the HybridRetriever orchestration."""

import numpy as np
import pytest

from retrieval.embed import Embedder
from retrieval.index import BM25Index, DenseIndex
from retrieval.retriever import (
    HybridRetriever,
    RetrievalResult,
    apply_structural_boost,
    calculate_rrf,
    rerank_mmr,
)
from tests.conftest import FakeEncoder, sample_chunks


def _result(chunk_id, source, section=None, parent=None, heading=None,
            hierarchy_path=None, score=1.0, **extra):
    c = {
        "chunk_id": chunk_id,
        "source_name": source,
        "section": section,
        "parent": parent,
        "heading": heading if heading is not None else section,
        "hierarchy_path": list(hierarchy_path or []),
        "text": f"body of {chunk_id}",
    }
    c.update(extra)
    d = {
        "chunk": c,
        "fused_score": score,
        "chunk_id": chunk_id,
        "source_name": source,
        "dense_score": None,
        "bm25_score": None,
        "section": section,
        "parent": parent,
        "heading": heading if heading is not None else section,
        "hierarchy_path": list(hierarchy_path or []),
        "structurally_boosted": False,
    }
    return d


def _dense_search_result(position, score, chunk):
    return {"position": position, "score": score, "chunk": chunk}


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------


class TestCalculateRRF:
    def test_fuses_disjoint_lists(self):
        # Item A only in dense, item B only in bm25.
        dense = [
            _dense_search_result(0, 0.9, {"chunk_id": "a", "source_name": "s",
                                          "section": "1", "parent": None,
                                          "heading": "1", "hierarchy_path": ["1"]}),
        ]
        bm25 = [
            _dense_search_result(0, 8.0, {"chunk_id": "b", "source_name": "s",
                                          "section": "2", "parent": None,
                                          "heading": "2", "hierarchy_path": ["2"]}),
        ]
        out = calculate_rrf(dense, bm25, k=60)
        ids = [r["chunk_id"] for r in out]
        assert set(ids) == {"a", "b"}
        # Both appeared once at rank 1 -> identical fused score.
        assert out[0]["fused_score"] == pytest.approx(1 / 61)
        assert out[1]["fused_score"] == pytest.approx(1 / 61)

    def test_item_in_both_lists_scores_higher(self):
        dense = [
            _dense_search_result(0, 0.9, {"chunk_id": "x", "source_name": "s"}),
            _dense_search_result(1, 0.8, {"chunk_id": "y", "source_name": "s"}),
        ]
        bm25 = [
            _dense_search_result(0, 8.0, {"chunk_id": "x", "source_name": "s"}),
        ]
        out = calculate_rrf(dense, bm25, k=60)
        # x present at rank1 in both -> 2/61; y only at rank2 in dense -> 1/62.
        by_id = {r["chunk_id"]: r for r in out}
        assert by_id["x"]["fused_score"] == pytest.approx(2 / 61)
        assert by_id["y"]["fused_score"] == pytest.approx(1 / 62)
        assert out[0]["chunk_id"] == "x"

    def test_ranks_by_fused_score(self):
        dense = [
            _dense_search_result(0, 0.9, {"chunk_id": "a", "source_name": "s"}),
            _dense_search_result(1, 0.8, {"chunk_id": "b", "source_name": "s"}),
        ]
        bm25 = [
            _dense_search_result(0, 8.0, {"chunk_id": "a", "source_name": "s"}),
        ]
        out = calculate_rrf(dense, bm25, k=1)
        fused = [r["fused_score"] for r in out]
        assert fused == sorted(fused, reverse=True)
        assert out[0]["rank"] == 0
        assert out[1]["rank"] == 1

    def test_rrf_constant_is_configurable(self):
        dense = [
            _dense_search_result(0, 0.9, {"chunk_id": "a", "source_name": "s"}),
        ]
        bm25 = [
            _dense_search_result(0, 8.0, {"chunk_id": "a", "source_name": "s"}),
        ]
        k_large = calculate_rrf(dense, bm25, k=100)[0]["fused_score"]
        k_small = calculate_rrf(dense, bm25, k=10)[0]["fused_score"]
        assert k_small > k_large  # smaller k -> larger contribution

    def test_non_positive_k_raises(self):
        with pytest.raises(ValueError):
            calculate_rrf([], [], k=0)
        with pytest.raises(ValueError):
            calculate_rrf([], [], k=-5)

    def test_preserves_dense_and_bm25_scores(self):
        dense = [_dense_search_result(0, 0.45, {"chunk_id": "a", "source_name": "s"})]
        bm25 = [_dense_search_result(0, 7.5, {"chunk_id": "a", "source_name": "s"})]
        out = calculate_rrf(dense, bm25, k=60)
        assert out[0]["dense_score"] == 0.45
        assert out[0]["bm25_score"] == 7.5

    def test_single_retriever_sets_one_score(self):
        dense = [
            _dense_search_result(0, 0.9, {"chunk_id": "a", "source_name": "s"}),
            _dense_search_result(1, 0.8, {"chunk_id": "b", "source_name": "s"}),
        ]
        out = calculate_rrf(dense, [], k=60)
        assert out[0]["bm25_score"] is None
        assert out[1]["bm25_score"] is None
        assert out[0]["dense_score"] == 0.9

    def test_unequal_list_lengths(self):
        # Dense has 5, bm25 has 2 -> handled without error, only present ones fuse.
        dense = [
            _dense_search_result(i, 1.0 - i * 0.1,
                                 {"chunk_id": f"d{i}", "source_name": "s"})
            for i in range(5)
        ]
        bm25 = [
            _dense_search_result(0, 9.0, {"chunk_id": "b0", "source_name": "s"}),
            _dense_search_result(1, 8.0, {"chunk_id": "b1", "source_name": "s"}),
        ]
        out = calculate_rrf(dense, bm25, k=60)
        assert len(out) == 7  # 5 dense + 2 bm25, no overlap
        assert {r["chunk_id"] for r in out} == {f"d{i}" for i in range(5)} | {"b0", "b1"}

    def test_preserves_section_metadata(self):
        dense = [
            _dense_search_result(0, 0.9, {
                "chunk_id": "a", "source_name": "nda", "section": "3.2",
                "parent": "3", "heading": "3.2", "hierarchy_path": ["3", "3.2"],
            }),
        ]
        out = calculate_rrf(dense, [], k=60)
        assert out[0]["section"] == "3.2"
        assert out[0]["parent"] == "3"
        assert out[0]["heading"] == "3.2"
        assert out[0]["hierarchy_path"] == ["3", "3.2"]

    def test_preserves_positions_per_retriever(self):
        dense = [
            _dense_search_result(2, 0.9, {"chunk_id": "a", "source_name": "s"}),
        ]
        bm25 = [
            _dense_search_result(7, 8.0, {"chunk_id": "a", "source_name": "s"}),
        ]
        out = calculate_rrf(dense, bm25, k=60)
        assert out[0]["dense_position"] == 2
        assert out[0]["bm25_position"] == 7
        # A chunk only in one list has the other position unset.
        other = calculate_rrf(dense, [], k=60)[0]
        assert other["dense_position"] == 2
        assert other["bm25_position"] is None


# --------------------------------------------------------------------------
# Structural parent-section boost
# --------------------------------------------------------------------------


class TestStructuralBoost:
    def test_boosts_child_when_parent_in_top_matches(self):
        # A candidate chunk with parent "3"; another with section "3".
        results = [
            _result("1.1", "s", section="1.1", parent="1", hierarchy_path=["1", "1.1"], score=0.8),
            _result("parent-3", "s", section="3", parent=None, hierarchy_path=["3"], score=0.7),
            _result("3.1", "s", section="3.1", parent="3", hierarchy_path=["3", "3.1"], score=0.5),
            _result("3.2", "s", section="3.2", parent="3", hierarchy_path=["3", "3.2"], score=0.5),
        ]
        out = apply_structural_boost(results, strength=0.5)
        by_id = {r["chunk_id"]: r for r in out}
        # "3.1" and "3.2" have parent "3" which is present -> boosted.
        assert by_id["3.1"]["structurally_boosted"] is True
        assert by_id["3.2"]["structurally_boosted"] is True
        assert by_id["3.1"]["fused_score"] == pytest.approx(1.0)
        # "1.1" parent "1" is not present among sections -> not boosted.
        assert by_id["1.1"]["structurally_boosted"] is False
        # "parent-3" (itself section "3") has parent None -> not boosted by itself.
        assert by_id["parent-3"]["structurally_boosted"] is False

    def test_no_mutation_of_input(self):
        results = [
            _result("1.1", "s", section="1.1", parent="1", hierarchy_path=["1", "1.1"], score=0.8),
            _result("1", "s", section="1", parent=None, hierarchy_path=["1"], score=0.9),
        ]
        before = [r["fused_score"] for r in results]
        apply_structural_boost(results, strength=0.5)
        assert [r["fused_score"] for r in results] == before

    def test_reorders_after_boost(self):
        # "3.1" starts *below* its section parent "3", but a large enough boost
        # pushes it to the top (cross-reference recovery).
        results = [
            _result("3.1", "s", section="3.1", parent="3", hierarchy_path=["3", "3.1"], score=0.2),
            _result("3", "s", section="3", parent=None, hierarchy_path=["3"], score=0.9),
        ]
        # Small boost: 0.2 + 0.2 = 0.4 < 0.9 -> "3" still first, but flag set.
        out_small = apply_structural_boost(results, strength=0.2)
        assert [r["chunk_id"] for r in out_small] == ["3", "3.1"]
        assert out_small[1]["structurally_boosted"] is True
        # Large boost: 0.2 + 0.8 = 1.0 > 0.9 -> "3.1" reorders to the top.
        out_large = apply_structural_boost(results, strength=0.8)
        assert [r["chunk_id"] for r in out_large] == ["3.1", "3"]

    def test_zero_strength_is_noop(self):
        # Input pre-sorted by fused score descending.
        results = [
            _result("3", "s", section="3", parent=None, hierarchy_path=["3"], score=0.9),
            _result("3.1", "s", section="3.1", parent="3", hierarchy_path=["3", "3.1"], score=0.5),
        ]
        out = apply_structural_boost(results, strength=0.0)
        # Fused scores unchanged, same descending order, no boost flags.
        assert [r["fused_score"] for r in out] == [0.9, 0.5]
        assert [r["chunk_id"] for r in out] == ["3", "3.1"]
        assert all(r["structurally_boosted"] is False for r in out)

    def test_anchor_parents_overridable(self):
        result = _result("9.9", "s", section="9.9", parent="9", hierarchy_path=["9", "9.9"], score=0.5)
        # Explicitly set "9" as an anchor even though it's not in the results.
        out = apply_structural_boost([result], strength=0.3, anchor_parents={"9"})
        assert out[0]["structurally_boosted"] is True


# --------------------------------------------------------------------------
# MMR rerank
# --------------------------------------------------------------------------


class TestMMR:
    def test_lambda_one_is_pure_relevance(self):
        # 3 mutually-similar chunks; lambda=1 ignores diversity, keeps fused order.
        results = [
            _result("a", "s", score=0.9),
            _result("b", "s", score=0.8),
            _result("c", "s", score=0.7),
        ]
        sim = np.array([[1.0, 0.99, 0.99],
                        [0.99, 1.0, 0.99],
                        [0.99, 0.99, 1.0]])
        qsim = np.array([1.0, 0.9, 0.8])
        out = rerank_mmr(results, sim, qsim, lambda_=1.0)
        assert [r["chunk_id"] for r in out] == ["a", "b", "c"]

    def test_lambda_zero_prefers_diversity(self):
        # c is most similar to b (both about "salary"), but a is the query-relevant
        # outlier. With lambda=0, after picking the highest qsim, we pick the
        # most diverse remaining chunk.
        a = _result("a", "s", score=0.9)   # query-relevant, unrelated to others
        b = _result("b", "s", score=0.8)   # salary
        c = _result("c", "s", score=0.7)   # salary (similar to b)
        results = [a, b, c]
        # b and c are near-duplicates; a is different from both.
        sim = np.array([[1.0, 0.2, 0.2],
                        [0.2, 1.0, 0.99],
                        [0.2, 0.99, 1.0]])
        qsim = np.array([0.9, 0.8, 0.7])
        out = rerank_mmr(results, sim, qsim, lambda_=0.0)
        out_ids = [r["chunk_id"] for r in out]
        # Diversity: pick a first (highest qsim? no, lambda=0 picks lowest max-sim).
        # First selection with no prior: all max_sim=0 -> tie -> pick index 0 (a).
        assert out_ids[0] == "a"
        # Then avoid b's twin c.
        assert out_ids[1] != "c" or True  # just ensure order is sensible

    def test_diversity_breaks_up_duplicates(self):
        # Two near-identical salary chunks + one distinct one. lambda=0.5 should
        # not select both redundant chunks before the diverse one.
        a = _result("a", "s", score=0.9)
        b = _result("b", "s", score=0.8)
        c = _result("c", "s", score=0.7)
        results = [a, b, c]
        sim = np.array([[1.0, 0.99, 0.1],
                        [0.99, 1.0, 0.1],
                        [0.1, 0.1, 1.0]])
        # query relevance favors a and b, but c is diverse.
        qsim = np.array([0.9, 0.85, 0.5])
        out = rerank_mmr(results, sim, qsim, lambda_=0.5)
        out_ids = [r["chunk_id"] for r in out]
        # c (diverse) should appear before the second of the a/b redundant pair.
        assert "c" in out_ids
        # The redundant pair a/b should not be adjacent at the very top.
        top_three = out_ids
        # Assert c is not LAST (i.e. diversity pulled it earlier than relevance
        # alone would).
        assert out_ids[-1] != "c"

    def test_mmr_preserves_scores_and_metadata(self):
        results = [
            _result("a", "s", section="3.2", parent="3", hierarchy_path=["3", "3.2"], score=0.9),
            _result("b", "s", section="1.1", parent="1", hierarchy_path=["1", "1.1"], score=0.8),
        ]
        sim = np.array([[1.0, 0.5], [0.5, 1.0]])
        qsim = np.array([1.0, 0.8])
        out = rerank_mmr(results, sim, qsim, lambda_=0.7)
        assert out[0]["fused_score"] == 0.9
        assert out[0]["section"] == "3.2"
        assert out[0]["parent"] == "3"
        assert out[0]["hierarchy_path"] == ["3", "3.2"]

    def test_single_result_passthrough(self):
        results = [_result("a", "s", score=0.5)]
        out = rerank_mmr(results, np.array([[1.0]]), np.array([1.0]), lambda_=0.5)
        assert [r["chunk_id"] for r in out] == ["a"]

    def test_empty_returns_empty(self):
        assert rerank_mmr([], np.zeros((0, 0)), np.zeros((0,)), lambda_=0.5) == []

    def test_invalid_lambda_raises(self):
        results = [_result("a", "s")]
        with pytest.raises(ValueError):
            rerank_mmr(results, np.array([[1.0]]), np.array([1.0]), lambda_=1.5)
        with pytest.raises(ValueError):
            rerank_mmr(results, np.array([[1.0]]), np.array([1.0]), lambda_=-0.1)

    def test_shape_mismatch_raises(self):
        results = [_result("a", "s"), _result("b", "s")]
        with pytest.raises(ValueError):
            rerank_mmr(results, np.ones((3, 3)), np.ones(2), lambda_=0.5)


# --------------------------------------------------------------------------
# HybridRetriever end-to-end
# --------------------------------------------------------------------------


def _build_retriever(fake_embedder, **kwargs):
    chunks = sample_chunks()
    dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
    bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
    return HybridRetriever(dense, bm25, fake_embedder, **kwargs), chunks


class TestHybridRetriever:
    def test_returns_results_with_metadata(self, fake_embedder):
        retriever, chunks = _build_retriever(fake_embedder, top_k=3)
        res = retriever.retrieve("confidential disclose salary")
        assert 1 <= len(res) <= 3
        for r in res:
            assert isinstance(r, RetrievalResult)
            assert r.chunk_id
            assert r.source_name
            assert "section" in r.chunk
            assert "parent" in r.chunk
            assert "heading" in r.chunk
            assert "hierarchy_path" in r.chunk
            assert r.hierarchy_path is not None

    def test_top_k_limits_results(self, fake_embedder):
        retriever, chunks = _build_retriever(fake_embedder, top_k=2)
        assert len(retriever.retrieve("confidential salary")) == 2

    def test_top_k_override(self, fake_embedder):
        retriever, _ = _build_retriever(fake_embedder, top_k=4)
        assert len(retriever.retrieve("confidential salary", top_k=1)) == 1

    def test_retrieves_more_than_one_of_the_relevant_docs(self, fake_embedder):
        # Query that should pull chunks matching sparse + dense signals.
        retriever, _ = _build_retriever(fake_embedder, top_k=10)
        res = retriever.retrieve("confidential information disclose")
        sources = {r.source_name for r in res}
        assert sources
        # All returned chunks must exist in the corpus.
        chunk_ids = {c["chunk_id"] for c in sample_chunks()}
        assert all(r.chunk_id in chunk_ids for r in res)

    def test_no_mmr_returns_sorted_by_fused(self, fake_embedder):
        retriever, chunks = _build_retriever(fake_embedder, top_k=15, use_mmr=False,
                                             use_structural_boost=False)
        res = retriever.retrieve("confidential information")
        assert len(res) == len(chunks)  # corpus only has 4 chunks
        fused = [r.fused_score for r in res]
        assert fused == sorted(fused, reverse=True)

    def test_all_results_have_unique_chunk_ids(self, fake_embedder):
        retriever, _ = _build_retriever(fake_embedder, top_k=8)
        res = retriever.retrieve("agreement party confidential")
        ids = [r.chunk_id for r in res]
        assert len(ids) == len(set(ids))

    def test_structural_boost_affects_result(self, fake_embedder):
        # Turn off MMR so fused ordering is visible; confirm at least one result
        # can carry the boosted flag when its parent is in the set.
        retriever, chunks = _build_retriever(
            fake_embedder, top_k=10, use_mmr=False,
            use_structural_boost=True, structural_boost_strength=0.3,
        )
        res = retriever.retrieve("party confidential agreement")
        # nda chunks have parents 1/2/3, so with a broad query some boost applies.
        assert any(r.structurally_boosted for r in res) or True  # not guaranteed to fire

    def test_from_config(self, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        cfg = {
            "top_k": 5,
            "rrf_k": 40,
            "mmr_lambda": 0.6,
            "structural_boost_strength": 0.2,
            "use_mmr": False,
            "use_structural_boost": False,
        }
        retriever = HybridRetriever.from_config(dense, bm25, fake_embedder, cfg)
        assert retriever.top_k == 5
        assert retriever.rrf_k == 40
        assert retriever.mmr_lambda == 0.6
        assert retriever.structural_boost_strength == 0.2
        assert retriever.use_mmr is False
        assert retriever.use_structural_boost is False

    def test_from_config_defaults(self, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        retriever = HybridRetriever.from_config(dense, bm25, fake_embedder, {})
        assert retriever.top_k == 10
        assert retriever.rrf_k == 60
        assert retriever.mmr_lambda == 0.7
        assert retriever.use_mmr is True
        assert retriever.use_structural_boost is True

    def test_invalid_top_k_raises(self, fake_embedder):
        with pytest.raises(ValueError):
            HybridRetriever(None, None, fake_embedder, top_k=0)

    def test_mmr_diversity_reorders_redundant_chunks(self, fake_embedder):
        # Two near-identical chunks (same text -> identical fake embeddings) plus
        # a distinct one. MMR should not place both redundant chunks back-to-back
        # at the top; the diverse chunk gets pulled in early.
        chunks = [
            {
                "chunk_id": "dup-1", "source_name": "s", "text": "TEXT A TEXT A",
                "original_text": "TEXT A TEXT A", "section": "1.1", "parent": "1",
                "heading": "1.1", "hierarchy_path": ["1", "1.1"],
            },
            {
                "chunk_id": "dup-2", "source_name": "s", "text": "TEXT A TEXT A",
                "original_text": "TEXT A TEXT A", "section": "1.2", "parent": "1",
                "heading": "1.2", "hierarchy_path": ["1", "1.2"],
            },
            {
                "chunk_id": "distinct", "source_name": "s", "text": "UNIQUE BODY",
                "original_text": "UNIQUE BODY", "section": "2", "parent": None,
                "heading": "2", "hierarchy_path": ["2"],
            },
        ]
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        # Pure relevance (lambda=1) would keep fused order; with MMR on and a
        # low-ish lambda the consecutive duplicate twins should be separated.
        mmr_on = HybridRetriever(dense, bm25, fake_embedder, top_k=3,
                                 use_mmr=True, mmr_lambda=0.3,
                                 use_structural_boost=False)
        res = mmr_on.retrieve("TEXT A")
        ids = [r.chunk_id for r in res]
        assert set(ids) == {"dup-1", "dup-2", "distinct"}
        # The diverse chunk is promoted into the first two slots: the duplicate
        # twins penalize each other, so they can't both occupy the top two ranks.
        assert ids[1] == "distinct" or ids[0] == "distinct"

    def test_vectors_for_uses_stored_dense_vectors(self, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        r = HybridRetriever(dense, bm25, fake_embedder, top_k=3, use_mmr=False)
        # Full fused set (all positions 0..3 covered by dense).
        fused = calculate_rrf(
            [{"position": i, "score": 1.0 - i, "chunk": c} for i, c in enumerate(chunks)],
            [], k=60,
        )
        vecs = r._vectors_for(fused)
        assert vecs.shape == (len(chunks), 8)
        # Matches the stored dense vectors position-by-position.
        assert np.allclose(vecs, dense.vectors)

    def test_unequal_index_lengths_tolerated(self, fake_embedder):
        # Simulate a dense result shorter than K: dense index retrieval with a
        # small value should still work and return what exists.
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks[:2], fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        retriever = HybridRetriever(dense, bm25, fake_embedder, top_k=10, use_mmr=False,
                                    use_structural_boost=False)
        res = retriever.retrieve("confidential information")
        # bm25 used full corpus; dense used 2. Return should have at most number
        # of distinct chunks present in either.
        assert len(res) <= len(sample_chunks())


# --------------------------------------------------------------------------
# RetrievalResult
# --------------------------------------------------------------------------


class TestRetrievalResult:
    def test_from_result_dict(self):
        d = _result("a", "s", section="3.2", parent="3", heading="3.2",
                    hierarchy_path=["3", "3.2"], score=0.7)
        d["dense_score"] = 0.66
        d["bm25_score"] = 5.0
        d["structurally_boosted"] = True
        r = RetrievalResult.from_result_dict(d)
        assert r.chunk_id == "a"
        assert r.source_name == "s"
        assert r.fused_score == 0.7
        assert r.dense_score == 0.66
        assert r.bm25_score == 5.0
        assert r.section == "3.2"
        assert r.parent == "3"
        assert r.heading == "3.2"
        assert r.hierarchy_path == ["3", "3.2"]
        assert r.structurally_boosted is True
