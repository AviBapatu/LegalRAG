"""Tests for eval.drm_eval: DRM metric calculation, the SAC off/on comparison
pipeline, deterministic outputs, and result serialization.

All tests run offline against the FakeEncoder (tests.conftest), exercising the
real ingestion/index/retrieval code path on small tmp corpora.
"""

import json

import pytest

from eval.drm_eval import (
    compute_drm_metrics,
    load_queries,
    render_markdown_report,
    retrieve_all_queries,
    run_drm_comparison,
    save_results,
)
from ingest.chunker import PatternChunker
from ingest.loader import load_documents
from ingest.pipeline import run_ingestion
from ingest.sac import apply_sac_to_document, summarize_document
from retrieval.index import BM25Index, DenseIndex
from retrieval.retriever import HybridRetriever


def _write_nda_docs(raw_dir):
    """Write two NDA-style docs that share boilerplate but differ in subject
    and parties, so SAC (document fingerprint) can disambiguate them and they
    are the kind of corpus where DRM occurs."""
    docs = {
        "nda_alpha.txt": (
            'Mutual Non-Disclosure Agreement between "Alpha Corp." and '
            '"Beta Labs" to support a potential widget sourcing partnership.\n'
            "\n"
            "1. Confidential Information\n"
            '1.1 "Confidential Information" includes trade secrets and '
            "engineering information.\n"
            "2. Obligations\n"
            "2.1 Each party will hold the other party's Confidential "
            "Information in confidence using at least reasonable care.\n"
            "3. Term\n"
            "3.1 This Agreement continues for two years unless earlier "
            "terminated.\n"
        ),
        "nda_gamma.txt": (
            'Mutual Non-Disclosure Agreement between "Gamma Holdings" and '
            '"Delta Systems" to support a potential data analytics '
            "collaboration.\n"
            "\n"
            "1. Confidential Information\n"
            '1.1 "Confidential Information" includes trade secrets and '
            "engineering information.\n"
            "2. Obligations\n"
            "2.1 Each party will hold the other party's Confidential "
            "Information in confidence using at least reasonable care.\n"
            "3. Term\n"
            "3.1 This Agreement continues for two years unless earlier "
            "terminated.\n"
        ),
    }
    for name, text in docs.items():
        (raw_dir / name).write_text(text)


def _config(**overrides):
    cfg = {
        "ingest": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "chunks_file": "chunks.jsonl",
        },
        "chunking": {
            "strategy": "pattern",
            "use_sac": True,
            "sac_use_llm": False,
            "sac_summary_max_chars": 150,
            "pattern": {},
            "sentence": {},
        },
        "retrieval": {
            "top_k": 3,
            "rrf_k": 60,
            "mmr_lambda": 0.7,
            "structural_boost_strength": 0.1,
            "use_mmr": False,
            "use_structural_boost": True,
        },
    }
    cfg.update(overrides)
    return cfg


def _queries():
    return [
        {
            "query_id": "q1",
            "query": "what obligations apply to the widget sourcing partnership?",
            "expected_document": "nda_alpha",
        },
        {
            "query_id": "q2",
            "query": "how long does confidentiality last in the data analytics collaboration?",
            "expected_document": "nda_gamma",
        },
    ]


# --------------------------------------------------------------------------
# DRM metric calculation
# --------------------------------------------------------------------------


class TestComputeDrmMetrics:
    def test_hand_computed_values(self):
        runs = [
            {"query_id": "a", "expected_document": "A", "retrieved_sources": ["A", "B", "A"]},
            {"query_id": "b", "expected_document": "B", "retrieved_sources": ["A", "B", "B"]},
            {"query_id": "c", "expected_document": "C", "retrieved_sources": ["A", "B", "A"]},
        ]
        m = compute_drm_metrics(runs, k=3)
        assert m["num_queries"] == 3
        assert m["top_k"] == 3
        assert m["top1_accuracy"] == pytest.approx(1 / 3)
        # q1 (A in top3) and q2 (B in top3) hit; q3 misses.
        assert m["top3_accuracy"] == pytest.approx(2 / 3)
        # Document-level accuracy@k is identical to top3 accuracy here.
        assert m["doc_accuracy_top_k"] == pytest.approx(2 / 3)
        assert m["doc_accuracy"] == pytest.approx({1: 1 / 3, 3: 2 / 3, 5: 2 / 3})
        # Mismatches: q1 -> B (1), q2 -> A (1), q3 -> A, B, A (3) = 5 of 9.
        assert m["retrieval_mismatch_count"] == 5
        assert m["drm_fraction"] == pytest.approx(5 / 9)
        assert m["total_chunks_checked"] == 9

    def test_perfect_retrieval_has_zero_mismatch(self):
        runs = [
            {"query_id": "a", "expected_document": "A", "retrieved_sources": ["A", "A", "A"]},
            {"query_id": "b", "expected_document": "B", "retrieved_sources": ["B", "B", "B"]},
        ]
        m = compute_drm_metrics(runs, k=3)
        assert m["top1_accuracy"] == 1.0
        assert m["top3_accuracy"] == 1.0
        assert m["retrieval_mismatch_count"] == 0
        assert m["drm_fraction"] == 0.0

    def test_shorter_retrieved_lists_are_drm_scored_at_their_length(self):
        # Only 2 chunks returned but k=5: the 2 returned are scored, total=2.
        runs = [
            {"query_id": "a", "expected_document": "A", "retrieved_sources": ["B", "A"]},
        ]
        m = compute_drm_metrics(runs, k=5)
        assert m["total_chunks_checked"] == 2
        assert m["retrieval_mismatch_count"] == 1
        assert m["drm_fraction"] == pytest.approx(0.5)

    def test_doc_accuracy_varied_over_k(self):
        runs = [
            {"query_id": "a", "expected_document": "A", "retrieved_sources": ["B", "B", "B", "A", "B"]},
        ]
        m = compute_drm_metrics(runs, k=5)
        # At k=1..3 the expected doc is absent; at k=4 and k=5 it is present.
        assert m["doc_accuracy"][1] == 0.0
        assert m["doc_accuracy"][3] == 0.0
        assert m["doc_accuracy"][5] == 1.0

    def test_zero_k_raises(self):
        runs = [{"query_id": "a", "expected_document": "A", "retrieved_sources": ["A"]}]
        with pytest.raises(ValueError):
            compute_drm_metrics(runs, k=0)

    def test_empty_runs_raises(self):
        with pytest.raises(ValueError):
            compute_drm_metrics([], k=3)


class TestLoadQueries:
    def test_loads_queries(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(
            json.dumps(
                {
                    "queries": [
                        {"query_id": "a", "query": "q?", "expected_document": "doc"}
                    ]
                }
            )
        )
        queries = load_queries(path)
        assert queries == [
            {"query_id": "a", "query": "q?", "expected_document": "doc"}
        ]

    def test_missing_expected_document_raises(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(json.dumps({"queries": [{"query_id": "a", "query": "q?"}]}))
        with pytest.raises(ValueError, match="expected_document"):
            load_queries(path)

    def test_empty_expected_document_raises(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(
            json.dumps(
                {
                    "queries": [
                        {"query_id": "a", "query": "q?", "expected_document": ""}
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="empty expected_document"):
            load_queries(path)


# --------------------------------------------------------------------------
# SAC off/on comparison pipeline (real ingestion + index + retriever)
# --------------------------------------------------------------------------


class TestRunDrmComparison:
    def _setup(self, tmp_path, fake_embedder):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_nda_docs(raw)
        cfg = _config()
        return cfg, raw, tmp_path / "processed", tmp_path / "index", _queries()

    def test_returns_both_sac_sections_with_metrics(self, tmp_path, fake_embedder):
        cfg, raw, processed, index, queries = self._setup(tmp_path, fake_embedder)
        result = run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )

        assert set(result) == {"envelope", "sac_off", "sac_on", "comparison"}
        for key in ("sac_off", "sac_on"):
            section = result[key]
            assert section["use_sac"] == (key == "sac_on")
            m = section["metrics"]
            assert m["num_queries"] == 2
            assert m["top_k"] == 3
            for field in (
                "top1_accuracy",
                "top3_accuracy",
                "doc_accuracy_top_k",
                "retrieval_mismatch_count",
                "drm_fraction",
                "total_chunks_checked",
            ):
                assert field in m
            assert len(section["per_query"]) == 2
            for run in section["per_query"]:
                assert run["query_id"] and run["expected_document"]
                assert run["retrieved_sources"]
                assert len(run["retrieved_sources"]) == 3

        # Comparison table covers the required metrics with deltas.
        labels = {row["label"] for row in result["comparison"]}
        assert "Top-1 accuracy" in labels
        assert "Top-3 accuracy" in labels
        assert "Document accuracy@3" in labels
        assert "Retrieval mismatch count (top-k)" in labels
        assert "DRM fraction (wrong-document chunks in top-k)" in labels
        assert all("sac_off" in row and "sac_on" in row and "delta" in row
                   for row in result["comparison"])

    def test_sac_off_indexes_are_namespaced_apart(self, tmp_path, fake_embedder):
        cfg, raw, processed, index, queries = self._setup(tmp_path, fake_embedder)
        run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        # Both SAC namespaces exist under the eval index root, and the chunk
        # files used for ingestion are distinct.
        assert (index / "fake-model-sac-off" / "dense").exists()
        assert (index / "fake-model-sac-off" / "bm25").exists()
        assert (index / "fake-model-sac-on" / "dense").exists()
        assert (index / "fake-model-sac-on" / "bm25").exists()
        assert (processed / "chunks_sac_off.jsonl").exists()
        assert (processed / "chunks_sac_on.jsonl").exists()

    def test_reuses_existing_indexes(self, tmp_path, fake_embedder):
        # Running twice must not rebuild: load_or_build loads the namespace.
        cfg, raw, processed, index, queries = self._setup(tmp_path, fake_embedder)
        first = run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        second = run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert first["sac_off"]["metrics"] == second["sac_off"]["metrics"]
        assert first["sac_on"]["metrics"] == second["sac_on"]["metrics"]

    def test_deterministic_outputs(self, tmp_path, fake_embedder):
        cfg, raw, processed, index, queries = self._setup(tmp_path, fake_embedder)
        a = run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        b = run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert a == b  # identical dicts including envelope + per-query detail


# --------------------------------------------------------------------------
# Result serialization
# --------------------------------------------------------------------------


class TestSerialization:
    def _result(self, tmp_path, fake_embedder):
        cfg, raw, processed, index, queries = self._setup(tmp_path, fake_embedder)
        return run_drm_comparison(
            cfg,
            fake_embedder,
            queries,
            raw_dir=raw,
            processed_dir=processed,
            index_root=index,
            timestamp="2024-01-01T00:00:00+00:00",
        )

    def _setup(self, tmp_path, fake_embedder):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_nda_docs(raw)
        return _config(), raw, tmp_path / "processed", tmp_path / "index", _queries()

    @staticmethod
    def _stringify_keys(obj):
        """Deep-copy ``obj`` converting every dict key to str.

        JSON coerces int dict keys (``doc_accuracy: {1: ...}``) to strings on
        load, so compare the on-disk object against a stringified copy of the
        in-memory result.
        """
        if isinstance(obj, dict):
            return {str(k): TestSerialization._stringify_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [TestSerialization._stringify_keys(v) for v in obj]
        return obj

    def test_saves_json_and_markdown(self, tmp_path, fake_embedder):
        result = self._result(tmp_path, fake_embedder)
        out = tmp_path / "results"
        json_path, md_path = save_results(result, out)

        assert json_path.exists() and json_path.name == "drm_results.json"
        assert md_path.exists() and md_path.name == "drm_report.md"

        on_disk = json.loads(json_path.read_text())
        # The JSON round-trips to the same object (modulo int->str dict keys).
        assert on_disk == self._stringify_keys(result)
        # The Markdown mentions the comparison and both SAC sections.
        text = md_path.read_text()
        assert "DRM Evaluation" in text
        assert "SAC off" in text and "SAC on" in text
        assert "Comparison table" in text

    def test_json_is_stable_and_sorted(self, tmp_path, fake_embedder):
        result = self._result(tmp_path, fake_embedder)
        out = tmp_path / "results"
        path, _ = save_results(result, out)
        # sort_keys=True means keys are written in sorted order and two saves
        # of the same result are byte-identical.
        obj = json.loads(path.read_text())
        assert list(obj.keys()) == sorted(obj.keys())
        again = tmp_path / "results2"
        path2, _ = save_results(result, again)
        assert path.read_bytes() == path2.read_bytes()

    def test_markdown_is_deterministic(self, tmp_path, fake_embedder):
        result = self._result(tmp_path, fake_embedder)
        md_a = render_markdown_report(result)
        md_b = render_markdown_report(result)
        assert md_a == md_b
        assert "generated_at" in result["envelope"]


# --------------------------------------------------------------------------
# The retrieval used by the eval is the real HybridRetriever (no dup logic)
# --------------------------------------------------------------------------


class TestRetrievalRoundTrip:
    def test_sac_changes_which_chunks_embed(self, tmp_path, fake_embedder):
        # Sanity: SAC on/off genuinely embed different text, so the eval is
        # comparing two genuinely different indexes (this is what makes the
        # DRM before/after attributable to SAC).
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_nda_docs(raw)
        docs = load_documents(raw)
        chunker = PatternChunker()
        chunks_plain = []
        chunks_sac = []
        for doc in docs:
            plain = chunker.chunk(doc.source_name, doc.text)
            sac = chunker.chunk(doc.source_name, doc.text)
            apply_sac_to_document(sac, summarize_document(doc.text).summary)
            chunks_plain.extend(plain)
            chunks_sac.extend(sac)
        assert all(not c.sac_applied for c in chunks_plain)
        assert all(c.sac_applied for c in chunks_sac)

    def test_retrieve_uses_hybrid_retriever(self, tmp_path, fake_embedder):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_nda_docs(raw)
        cfg = _config()
        run_ingestion(
            cfg,
            raw_dir=raw,
            processed_dir=tmp_path / "processed",
            chunks_file="chunks.jsonl",
        )
        chunks_path = tmp_path / "processed" / "chunks.jsonl"
        dense = DenseIndex.build(
            [json.loads(l) for l in chunks_path.read_text().splitlines()],
            fake_embedder,
            use_sac=True,
        )
        bm25 = BM25Index.build(
            [json.loads(l) for l in chunks_path.read_text().splitlines()],
            fake_embedder,
            use_sac=True,
        )
        retriever = HybridRetriever(dense, bm25, fake_embedder, top_k=3, use_mmr=False)
        runs = retrieve_all_queries(retriever, _queries(), top_k=3)
        assert len(runs) == 2
        assert all(len(run["retrieved_sources"]) == 3 for run in runs)