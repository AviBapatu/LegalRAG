"""Dense embedding wrapper around SentenceTransformers.

The embedding model used to index the chunks is a key design decision: recent
legal RAG benchmarks show that embedding-model choice sets the ceiling on
downstream retrieval correctness more than the generator LLM does (AGENTS.md
§2, failure mode 2). To support the embedding-model ablation in Milestone 6,
the model name must always come from config and never be hard-coded here, and
the wrapper must stay small so alternative encoders can be dropped in.

The `Embedder` accepts either a model name (loaded lazily via
SentenceTransformer) or an injectable encoder object exposing ``encode()``. The
latter is what lets unit tests run without a huge model download: a caller can
construct an `Embedder` with a tiny/fake encoder while still recording the
canonical model name for index namespacing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Embedder:
    """Wrap an embedding model and produce dense vectors for text.

    Args:
        model_name: Identifier of the embedding model (e.g.
            ``BAAI/bge-large-en-v1.5``). Used for index namespacing and for
            lazily constructing a SentenceTransformer when no `encoder` is
            injected.
        encoder: Optional pre-built encoder object with an ``encode(texts,
            convert_to_numpy=..., normalize_embeddings=...)``-compatible API.
            When provided, it is used instead of loading a SentenceTransformer;
            this is primarily for tests / swappable models.
        device: Optional device string (e.g. "cpu", "cuda") passed to the
            SentenceTransformer loader.
    """

    def __init__(self, model_name: str, encoder=None, device: str | None = None):
        if not model_name:
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self._encoder = encoder
        self._device = device
        self._loaded: bool = encoder is not None

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(self.model_name, device=self._device)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying encoder has been materialized."""
        return self._loaded

    def embed(self, texts: Sequence[str] | str) -> np.ndarray:
        """Embed a sequence of texts into a float32 matrix.

        Args:
            texts: A string, or an iterable of strings to embed.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), dim)`` with dtype float32,
            L2-normalized rows.
        """
        if not self._loaded:
            self._load()
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        # Defensive L2 normalization: guarantees FAISS inner-product == cosine
        # similarity regardless of what the injected encoder returns.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def embed_texts(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    """Convenience wrapper: embed ``texts`` with an ``Embedder``."""
    return embedder.embed(texts)


def chunk_to_embedding_text(chunk: dict, use_sac: bool) -> str:
    """Return the text that should be embedded for a chunk dict.

    Each chunk line in ``data/processed/chunks.jsonl`` carries both ``text``
    and ``original_text`` (see ingest.chunker.Chunk.to_dict). When
    Summary-Augmented Chunking (SAC) is on, ``text`` has the document
    fingerprint prepended and is the correct thing to embed; ``original_text``
    is the pre-SAC body. This helper centralizes that selection so the index
    builds can be unambiguously namespaced by the SAC on/off state.

    Args:
        chunk: A chunk dict from chunks.jsonl.
        use_sac: Whether the index is being built with SAC applied.
            If True, embed ``text`` (summary-augmented); if False, embed
            ``original_text`` (identical to ``text`` when SAC never ran).
    """
    if use_sac:
        return str(chunk.get("text", ""))
    return str(chunk.get("original_text", chunk.get("text", "")))
