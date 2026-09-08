"""Tests for eval/ablation.py: embedding-model ablation over the same query
set, explicit (never silent) embedder resolution, deterministic reports, and
serialization.

All retrieval runs offline with fake embedders; no model is ever downloaded
and pytest never touches the in-repo data/eval corpus (tmp dirs instead).
"""

import json

import pytest

from eval.ablation import (
    configured_ablation_models,
    render_markdown_report,
    resolve_embedders,
    run_ablation,
    save_results,
)
from retrieval.embed import Embedder


def _write_nda_docs(raw_dir):
    docs = {
        "nda_alpha.txt": (
            'Mutual Non-Disclosure Agreement between "Alpha Corp." and '
            '"Beta Labs" to support a potential widget sourcing partnership.\n'
            "\n1. Confidential Information\n"
            '1.1 "Confidential Information" includes trade secrets and '
            "engineering information.\n"
            "2. Obligations\n"
            "2.1 Each party will hold the other party's Confidential "
            "Information in confidence.\n"
        ),
        "nda_gamma.txt": (
            'Mutual Non-Disclosure Agreement between "Gamma Holdings" and '
            '"Delta Systems" to support a data analytics collaboration.\n'
            "\n1. Confidential Information\n"
            '1.1 "Confidential Information" includes trade secrets.\n'
            "2. Obligations\n"
            "2.1 Each party will hold the other party's Confidential "
            "Information in confidence.\n"
        ),
    }
    for name, text in docs.items():
        (raw_dir / name).write_text(text)


def _config():
    return {
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
        "evaluation": {
            "k_values": [1, 3, 5],
            "embedding_ablation_models": ["fake-a", "fake-b"],
        },
    }


def _queries():
    return [
        {"query_id": "q1", "query": "what obligations apply to the widget partnership?", "expected_document": "nda_alpha"},
        {"query_id": "q2", "query": "how does confidentiality work in the analytics collaboration?", "expected_document": "nda_gamma"},
    ]


def _embedders(fake_embedder, fake_embedder_other_model):
    return {"fake-a": fake_embedder, "fake-b": fake_embedder_other_model}


class TestConfiguredModels:
    def test_returns_models_in_order_deduped(self):
        cfg = _config()
        assert configured_ablation_models(cfg) == ["fake-a", "fake-b"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="embedding_ablation_models"):
            configured_ablation_models({"evaluation": {"embedding_ablation_models": []}})

    def test_missing_block_raises(self):
        with pytest.raises(ValueError):
            configured_ablation_models({})

    def test_invalid_entries_raise(self):
        with pytest.raises(ValueError):
            configured_ablation_models({"evaluation": {"embedding_ablation_models": ["ok", ""]}})


class TestResolveEmbedders:
    def test_injected_mapping_is_authoritative(self, fake_embedder, fake_embedder_other_model):
        resolved = resolve_embedders(_config(), {"fake-a": fake_embedder, "fake-b": fake_embedder_other_model})
        assert set(resolved) == {"fake-a", "fake-b"}
        assert resolved["fake-a"] is fake_embedder

    def test_missing_configured_model_raises_not_substituted(self, fake_embedder):
        # fake-a is configured but absent from the mapping -> hard error, never
        # a silent substitution with a different model.
        with pytest.raises(RuntimeError, match="no injected embedder"):
            resolve_embedders(_config(), {"fake-b": fake_embedder})

    def test_no_injection_builds_real_lazy_embedders(self):
        resolved = resolve_embedders(_config())
        assert all(isinstance(e, Embedder) for e in resolved.values())
        assert [e.model_name for e in resolved.values()] == ["fake-a", "fake-b"]


class TestRunAblation:
    def _setup(self, tmp_path, fake_embedder, fake_embedder_other_model, timestamp="2024-01-01T00:00:00+00:00"):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_nda_docs(raw)
        return (
            _config(),
            raw,
            tmp_path / "processed",
            tmp_path / "index",
            _queries(),
            _embedders(fake_embedder, fake_embedder_other_model),
            timestamp,
        )

    def test_runs_every_model(self, tmp_path, fake_embedder, fake_embedder_other_model):
        cfg, raw, processed, index, queries, embedders, ts = self._setup(tmp_path, fake_embedder, fake_embedder_other_model)
        result = run_ablation(
            cfg, queries, embedders=embedders,
            raw_dir=raw, processed_dir=processed, index_root=index, timestamp=ts,
        )
        assert result["envelope"]["models"] == ["fake-a", "fake-b"]
        assert set(result["models"]) == {"fake-a", "fake-b"}
        for section in result["models"].values():
            m = section["metrics"]
            assert section["use_sac"] is True
            assert set(m["precision_at_k"]) == {1, 3, 5}
            assert set(m["recall_at_k"]) == {1, 3, 5}
            assert 0.0 <= m["mrr"] <= 1.0
            assert 0.0 <= m["drm_fraction"] <= 1.0
            assert len(section["per_query"]) == 2

    def test_envelope_reproduction_metadata(self, tmp_path, fake_embedder, fake_embedder_other_model):
        cfg, raw, processed, index, queries, embedders, ts = self._setup(tmp_path, fake_embedder, fake_embedder_other_model)
        result = run_ablation(cfg, queries, embedders=embedders, raw_dir=raw, processed_dir=processed, index_root=index, timestamp=ts)
        env = result["envelope"]
        assert env["version"] == 1
        assert env["retrieval_depth"] == 3
        assert env["k_values"] == [1, 3, 5]
        assert env["num_queries"] == 2
        assert env["embedders"] == "injected"
        assert len(env["corpus"]) == 2

    def test_deterministic_outputs(self, tmp_path, fake_embedder, fake_embedder_other_model):
        cfg, raw, processed, index, queries, embedders, ts = self._setup(tmp_path, fake_embedder, fake_embedder_other_model)
        a = run_ablation(cfg, queries, embedders=embedders, raw_dir=raw, processed_dir=processed, index_root=index, timestamp=ts)
        b = run_ablation(cfg, queries, embedders=embedders, raw_dir=raw, processed_dir=processed, index_root=index, timestamp=ts)
        assert a == b

    def test_empty_k_values_raises(self, tmp_path, fake_embedder, fake_embedder_other_model):
        cfg, raw, processed, index, queries, embedders, ts = self._setup(tmp_path, fake_embedder, fake_embedder_other_model)
        with pytest.raises(ValueError, match="k_values"):
            run_ablation(cfg, queries, embedders=embedders, k_values=[], raw_dir=raw, processed_dir=processed, index_root=index)


class TestSerialization:
    @staticmethod
    def _stringify_keys(obj):
        if isinstance(obj, dict):
            return {str(k): TestSerialization._stringify_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [TestSerialization._stringify_keys(v) for v in obj]
        return obj

    def _result(self, tmp_path, fake_embedder, fake_embedder_other_model):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_nda_docs(raw)
        return run_ablation(
            _config(), _queries(),
            embedders=_embedders(fake_embedder, fake_embedder_other_model),
            raw_dir=raw, processed_dir=tmp_path / "processed",
            index_root=tmp_path / "index",
            timestamp="2024-01-01T00:00:00+00:00",
        )

    def test_saves_json_and_markdown(self, tmp_path, fake_embedder, fake_embedder_other_model):
        result = self._result(tmp_path, fake_embedder, fake_embedder_other_model)
        out = tmp_path / "out"
        json_path, md_path = save_results(result, out)
        assert json_path.name == "ablation_results.json"
        assert md_path.name == "ablation_report.md"
        assert json.loads(json_path.read_text()) == self._stringify_keys(result)
        text = md_path.read_text()
        assert "Embedding-Model Ablation" in text
        assert "fake-a" in text and "fake-b" in text

    def test_json_sorted_and_stable(self, tmp_path, fake_embedder, fake_embedder_other_model):
        result = self._result(tmp_path, fake_embedder, fake_embedder_other_model)
        path, _ = save_results(result, tmp_path / "a")
        path2, _ = save_results(result, tmp_path / "b")
        assert list(json.loads(path.read_text()).keys()) == sorted(
            json.loads(path.read_text()).keys()
        )
        assert path.read_bytes() == path2.read_bytes()

    def test_markdown_deterministic(self, tmp_path, fake_embedder, fake_embedder_other_model):
        result = self._result(tmp_path, fake_embedder, fake_embedder_other_model)
        assert render_markdown_report(result) == render_markdown_report(result)