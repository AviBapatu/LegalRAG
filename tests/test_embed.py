"""Tests for retrieval.embed: configurable embedding and chunk text selection."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from retrieval.embed import Embedder, chunk_to_embedding_text
from tests.conftest import FakeEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def _load_config():
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


class TestConfigModelHandling:
    def test_config_has_retrieval_block(self):
        cfg = _load_config()
        assert "retrieval" in cfg
        assert "embedding_model" in cfg["retrieval"]

    def test_default_embedding_model_is_bge_large(self):
        cfg = _load_config()
        assert cfg["retrieval"]["embedding_model"] == DEFAULT_MODEL

    def test_index_dir_configured(self):
        cfg = _load_config()
        assert cfg["retrieval"]["index_dir"] == "data/index"

    def test_model_is_config_driven_not_hardcoded_in_retrieval(self):
        # The Embedder has no built-in default model (that default lives only
        # in config.yaml), so the model name is always supplied explicitly.
        with pytest.raises(TypeError):
            Embedder()


class TestEmbedder:
    def test_rejects_empty_model_name(self):
        with pytest.raises(ValueError):
            Embedder("")

    def test_records_model_name(self, fake_embedder):
        assert fake_embedder.model_name == "fake-model"

    def test_embed_shape_type_and_normalization(self, fake_embedder):
        vectors = fake_embedder.embed(["a", "bb", "ccc"])
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, 8)
        assert vectors.dtype == np.float32
        # Rows are L2-normalized (cosine-equivalent inner product).
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_embed_single_string(self, fake_embedder):
        vectors = fake_embedder.embed("just one string")
        assert vectors.shape == (1, 8)

    def test_embed_deterministic(self, fake_embedder):
        a = fake_embedder.embed(["same text"])
        b = fake_embedder.embed(["same text"])
        assert np.array_equal(a, b)

    def test_embed_empty_list_returns_empty(self, fake_embedder):
        vectors = fake_embedder.embed([])
        assert vectors.shape[0] == 0

    def test_embedder_without_encoder_is_not_loaded(self):
        # Constructed with a name only -> nothing downloaded until embed().
        e = Embedder("fake-model")
        assert not e.is_loaded

    def test_dimension_depends_on_encoder(self):
        small = Embedder("m-a", encoder=FakeEncoder(dim=4))
        big = Embedder("m-b", encoder=FakeEncoder(dim=16))
        assert small.embed(["x"]).shape[1] == 4
        assert big.embed(["x"]).shape[1] == 16


class TestChunkToEmbeddingText:
    def _chunk(self):
        return {
            "chunk_id": "nda-0",
            "text": "SUMMARY text plus body",
            "original_text": "body without summary",
            "sac_applied": True,
        }

    def test_uses_augmented_text_when_sac(self):
        assert chunk_to_embedding_text(self._chunk(), use_sac=True) == "SUMMARY text plus body"

    def test_uses_original_text_when_no_sac(self):
        assert chunk_to_embedding_text(self._chunk(), use_sac=False) == "body without summary"

    def test_falls_back_to_text_when_no_original(self):
        chunk = {"chunk_id": "x", "text": "plain text"}
        assert chunk_to_embedding_text(chunk, use_sac=False) == "plain text"