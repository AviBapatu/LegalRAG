"""Tests for retrieval.index: FAISS dense index, BM25 sparse index,
namespacing, metadata mapping, persistence, and load/reuse behavior."""

import json

import numpy as np
import pytest

from retrieval.embed import Embedder
from retrieval.index import (
    BM25Index,
    DenseIndex,
    _load_metadata,
    index_namespace,
    index_path_for,
    load_or_build_bm25,
    load_or_build_dense,
    read_chunks,
    slugify_model_name,
)
from tests.conftest import FakeEncoder, sample_chunks

MODEL_A = "BAAI/bge-large-en-v1.5"
MODEL_B = "some-other-model"


def _write_chunks_jsonl(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c))
            fh.write("\n")


# --------------------------------------------------------------------------
# Namespacing
# --------------------------------------------------------------------------


class TestNamespacing:
    def test_slugify(self):
        assert slugify_model_name("BAAI/bge-large-en-v1.5") == "BAAI-bge-large-en-v1.5"

    def test_namespace_contains_model_and_sac(self):
        assert index_namespace(MODEL_A, True) == "BAAI-bge-large-en-v1.5-sac-on"
        assert index_namespace(MODEL_A, False) == "BAAI-bge-large-en-v1.5-sac-off"

    def test_namespace_differs_across_models(self):
        assert index_namespace(MODEL_A, True) != index_namespace(MODEL_B, True)

    def test_namespace_differs_across_sac(self):
        assert index_namespace(MODEL_A, True) != index_namespace(MODEL_A, False)

    def test_index_path_embeds_namespace(self, tmp_path):
        p = index_path_for(tmp_path, MODEL_A, True)
        assert p == tmp_path / "BAAI-bge-large-en-v1.5-sac-on"


# --------------------------------------------------------------------------
# Dense (FAISS) index
# --------------------------------------------------------------------------


class TestDenseIndex:
    def test_build_shape_and_metadata(self, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        assert dense.dimension == 8
        assert dense.faiss_index.ntotal == len(chunks)
        assert dense.metadata["embedding_model"] == "fake-model"
        assert dense.metadata["use_sac"] is True
        assert len(dense.metadata["chunks"]) == len(chunks)

    def test_build_uses_sac_text_field(self, fake_embedder):
        chunks = sample_chunks()
        sac_on = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        sac_off = DenseIndex.build(chunks, fake_embedder, use_sac=False)
        # SAC on embeds text (summary-augmented); off uses original_text, so
        # the two configs produce different vectors.
        assert not np.allclose(sac_on.vectors, sac_off.vectors)

    def test_position_to_chunk_mapping(self, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        for i, chunk in enumerate(chunks):
            record = dense.position_to_chunk(i)
            assert record["chunk_id"] == chunk["chunk_id"]
            assert record["source_name"] == chunk["source_name"]

    def test_search_maps_vectors_back_to_chunks(self, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        # Query with an embedding of one of the chunk texts: that chunk should
        # come back at the top of the ranking (deterministic embedding).
        query = fake_embedder.embed([chunks[1]["text"]])[0]
        results = dense.search(query, top_k=2)
        assert len(results) == 2
        for r in results:
            assert "position" in r and "score" in r and "chunk" in r
        top = results[0]
        assert top["chunk"]["chunk_id"] == chunks[1]["chunk_id"]
        assert top["position"] == 1

    def test_search_wrong_dimension_raises(self, fake_embedder):
        dense = DenseIndex.build(sample_chunks(), fake_embedder, use_sac=True)
        with pytest.raises(ValueError):
            dense.search(np.zeros(3, dtype=np.float32))

    def test_build_empty_chunks_raises(self, fake_embedder):
        with pytest.raises(ValueError):
            DenseIndex.build([], fake_embedder, use_sac=True)

    def test_persist_and_reload(self, tmp_path, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        dense.save(tmp_path)

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "vectors.npy").exists()
        assert (tmp_path / "metadata.json").exists()

        loaded = DenseIndex.load(tmp_path, "fake-model", use_sac=True)
        assert loaded.dimension == 8
        assert loaded.faiss_index.ntotal == len(chunks)
        assert [c["chunk_id"] for c in loaded.chunks] == [c["chunk_id"] for c in chunks]
        assert np.allclose(loaded.vectors, dense.vectors)

    def test_reloaded_search_identical(self, tmp_path, fake_embedder):
        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        dense.save(tmp_path)
        loaded = DenseIndex.load(tmp_path, "fake-model", use_sac=True)
        q = dense.vectors[0]
        assert loaded.search(q, top_k=1)[0]["chunk"]["chunk_id"] == chunks[0]["chunk_id"]


# --------------------------------------------------------------------------
# BM25 sparse index
# --------------------------------------------------------------------------


class TestBM25Index:
    def test_build_maps_positions_to_chunks(self, fake_embedder):
        chunks = sample_chunks()
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        assert len(bm25.bm25.get_scores(["confidential"])) == len(chunks)
        for i, chunk in enumerate(chunks):
            record = bm25.position_to_chunk(i)
            assert record["chunk_id"] == chunk["chunk_id"]
            assert record["source_name"] == chunk["source_name"]

    def test_search_finds_term_bearing_chunk(self, fake_embedder):
        chunks = sample_chunks()
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        results = bm25.search("confidential disclose", top_k=1)
        assert results
        assert results[0]["chunk"]["chunk_id"] == "nda-1"

    def test_search_ranked_by_term_matches(self, fake_embedder):
        chunks = sample_chunks()
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=False)
        results = bm25.search("salary", top_k=3)
        assert results[0]["chunk"]["chunk_id"] == "emp-0"

    def test_persist_and_reload(self, tmp_path, fake_embedder):
        chunks = sample_chunks()
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        bm25.save(tmp_path)

        assert (tmp_path / "corpus.json").exists()
        assert (tmp_path / "metadata.json").exists()

        loaded = BM25Index.load(tmp_path, "fake-model", use_sac=True)
        assert [c["chunk_id"] for c in loaded.chunks] == [c["chunk_id"] for c in chunks]

        original = bm25.search("confidential", top_k=2)
        reloaded = loaded.search("confidential", top_k=2)
        assert [r["chunk"]["chunk_id"] for r in original] == [
            r["chunk"]["chunk_id"] for r in reloaded
        ]

    def test_build_empty_chunks_raises(self, fake_embedder):
        with pytest.raises(ValueError):
            BM25Index.build([], fake_embedder, use_sac=True)


# --------------------------------------------------------------------------
# load_or_build + no silent overwrite
# --------------------------------------------------------------------------


class TestLoadOrBuild:
    def _write_chunks(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        _write_chunks_jsonl(sample_chunks(), path)
        return path

    def test_builds_and_reuses(self, tmp_path, fake_embedder):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        emb = Embedder("fake-model", encoder=FakeEncoder(dim=8))

        first = load_or_build_dense(chunks_path, index_root, emb, use_sac=True)
        assert isinstance(first, DenseIndex)

        # Replace the chunks file with different content: a subsequent
        # load_or_build must LOAD the existing index, not rebuild it.
        _write_chunks_jsonl([sample_chunks()[0]], chunks_path)
        second = load_or_build_dense(chunks_path, index_root, emb, use_sac=True)
        assert second.faiss_index.ntotal == len(sample_chunks())
        assert second.metadata["embedding_model"] == "fake-model"

    def test_dense_and_bm25_coexist_in_same_namespace(self, tmp_path):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        emb = Embedder("fake-model", encoder=FakeEncoder(dim=8))

        dense = load_or_build_dense(chunks_path, index_root, emb, use_sac=True)
        bm25 = load_or_build_bm25(chunks_path, index_root, emb, use_sac=True)

        ns = index_root / index_namespace("fake-model", True)
        assert (ns / "dense").exists()
        assert (ns / "bm25").exists()
        # Reloading the bm25 index must no longer see the dense metadata.
        reloaded = load_or_build_dense(chunks_path, index_root, emb, use_sac=True)
        reloaded_bm25 = load_or_build_bm25(chunks_path, index_root, emb, use_sac=True)
        assert reloaded.faiss_index.ntotal == dense.faiss_index.ntotal
        assert len(reloaded_bm25.chunks) == len(bm25.chunks)

    def test_bm25_builds_and_reuses(self, tmp_path, fake_embedder):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        emb = Embedder("fake-model", encoder=FakeEncoder(dim=8))

        first = load_or_build_bm25(chunks_path, index_root, emb, use_sac=True)
        assert isinstance(first, BM25Index)
        second = load_or_build_bm25(chunks_path, index_root, emb, use_sac=True)
        assert second is not None
        assert len(second.chunks) == len(sample_chunks())

    def test_exist_ok_false_refuses_overwrite(self, tmp_path, fake_embedder):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        emb = Embedder("fake-model", encoder=FakeEncoder(dim=8))
        load_or_build_dense(chunks_path, index_root, emb, use_sac=True)
        with pytest.raises(FileExistsError):
            load_or_build_dense(chunks_path, index_root, emb, use_sac=True, exist_ok=False)

    def test_sac_namespaced_directories_coexist(self, tmp_path):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        emb = Embedder("fake-model", encoder=FakeEncoder(dim=8))

        on = load_or_build_dense(chunks_path, index_root, emb, use_sac=True)
        off = load_or_build_dense(chunks_path, index_root, emb, use_sac=False)
        assert (index_root / index_namespace("fake-model", True)).exists()
        assert (index_root / index_namespace("fake-model", False)).exists()
        assert on.metadata["use_sac"] is True
        assert off.metadata["use_sac"] is False

    def test_model_namespaced_directories_coexist(self, tmp_path):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        emb_a = Embedder(MODEL_A, encoder=FakeEncoder(dim=8))
        emb_b = Embedder(MODEL_B, encoder=FakeEncoder(dim=8))

        load_or_build_dense(chunks_path, index_root, emb_a, use_sac=True)
        load_or_build_dense(chunks_path, index_root, emb_b, use_sac=True)
        assert (index_root / index_namespace(MODEL_A, True)).exists()
        assert (index_root / index_namespace(MODEL_B, True)).exists()


class TestMismatchProtection:
    def _write_chunks(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        _write_chunks_jsonl(sample_chunks(), path)
        return path

    def test_load_dense_with_wrong_model_raises(self, tmp_path, fake_embedder):
        DenseIndex.build(sample_chunks(), fake_embedder, use_sac=True).save(tmp_path)
        with pytest.raises(ValueError, match="embedding model"):
            DenseIndex.load(tmp_path, "other-model", use_sac=True)

    def test_load_dense_with_wrong_sac_raises(self, tmp_path, fake_embedder):
        DenseIndex.build(sample_chunks(), fake_embedder, use_sac=True).save(tmp_path)
        with pytest.raises(ValueError, match="use_sac"):
            DenseIndex.load(tmp_path, "fake-model", use_sac=False)

    def test_load_bm25_with_wrong_model_raises(self, tmp_path, fake_embedder):
        BM25Index.build(sample_chunks(), fake_embedder, use_sac=True).save(tmp_path)
        with pytest.raises(ValueError, match="embedding model"):
            BM25Index.load(tmp_path, "other-model", use_sac=True)

    def test_load_missing_metadata_raises(self, tmp_path, fake_embedder):
        # A directory containing only index files, no metadata, is untrustworthy.
        import faiss

        index = faiss.IndexFlatIP(8)
        index.add(np.zeros((2, 8), dtype=np.float32))
        faiss.write_index(index, str(tmp_path / "index.faiss"))
        with pytest.raises(ValueError, match="metadata"):
            DenseIndex.load(tmp_path, "fake-model", use_sac=True)

    def test_load_or_build_raises_on_config_collision(self, tmp_path):
        chunks_path = self._write_chunks(tmp_path)
        index_root = tmp_path / "index"
        # "a/b" and "a-b" slugify identically, so they collide on the same
        # index directory. The stored metadata must stop the second build
        # from silently reusing/overwriting an index it was not built with.
        emb_a = Embedder("a/b", encoder=FakeEncoder(dim=8))
        emb_collision = Embedder("a-b", encoder=FakeEncoder(dim=8))
        load_or_build_dense(chunks_path, index_root, emb_a, use_sac=True)
        with pytest.raises(ValueError, match="embedding model"):
            load_or_build_dense(chunks_path, index_root, emb_collision, use_sac=True)


# --------------------------------------------------------------------------
# read_chunks / metadata loading
# --------------------------------------------------------------------------


class TestReaders:
    def test_read_chunks_loads_jsonl(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        _write_chunks_jsonl(sample_chunks(), path)
        chunks = read_chunks(path)
        assert [c["chunk_id"] for c in chunks] == [c["chunk_id"] for c in sample_chunks()]

    def test_read_chunks_skips_blank_lines(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        _write_chunks_jsonl(sample_chunks(), path)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n\n")
        assert len(read_chunks(path)) == len(sample_chunks())

    def test_load_metadata_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_metadata(tmp_path, "dense")