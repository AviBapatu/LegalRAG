"""Dense (FAISS) and sparse (BM25) vector indexes over processed chunks.

Milestone 2 (§2 of AGENTS.md): builds a FAISS dense index and a BM25 sparse
index from ``data/processed/chunks.jsonl`` and persists both under
``data/index/``. Index directories are namespaced by embedding model AND SAC
on/off state, so several configurations can coexist for the Milestone 4 DRM
comparison and the Milestone 6 embedding ablation.

Both indexes store a ``metadata.json`` describing exactly which
(embedding model, SAC state) built it, plus a per-position list of chunk
records (chunk_id, source_name, ...) so vector positions always map back to
their source chunk/document. When a path already exists, loading verifies the
stored metadata instead of silently reusing or overwriting an index built with
different parameters.

On disk, each (model, SAC) configuration owns an index directory (its
namespace); the dense and sparse indexes for that configuration live side by
side in ``<namespace>/dense`` and ``<namespace>/bm25`` so a dense and a sparse
index for the same run never collide.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embed import Embedder, chunk_to_embedding_text

# Sentinel used in metadata to mark sparse (BM25) indexes, which have no real
# embedding dimension and are only namespaced consistently with the model.
_BM25_DIMENSION = "bm25"

METADATA_VERSION = 1

INDEX_FILE = "index.faiss"
VECTORS_FILE = "vectors.npy"
METADATA_FILE = "metadata.json"
CORPUS_FILE = "corpus.json"  # tokenized documents for BM25

# Subdirectories holding each index kind inside a (model, SAC) namespace.
DENSE_SUBDIR = "dense"
BM25_SUBDIR = "bm25"


def slugify_model_name(model_name: str) -> str:
    """Turn a model identifier into a safe directory token.

    ``BAAI/bge-large-en-v1.5`` -> ``BAAI-bge-large-en-v1.5``.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", model_name)


def index_namespace(model_name: str, use_sac: bool) -> str:
    """Deterministic namespacing token for an index directory.

    Combined embedding-model + SAC flag: the two dimensions that change the
    embedded text of a chunk. Same config -> same token -> same index can be
    reused; different config -> different token -> indexes coexist.
    """
    sac = "sac-on" if use_sac else "sac-off"
    return f"{slugify_model_name(model_name)}-{sac}"


def index_path_for(index_root: str | Path, model_name: str, use_sac: bool) -> Path:
    """Resolve the on-disk directory for one (model, SAC) configuration.

    Returns the *namespace* directory; the dense and BM25 indexes each live in
    a subdirectory of it (``<namespace>/dense`` and ``<namespace>/bm25``).
    """
    return Path(index_root) / index_namespace(model_name, use_sac)


# --------------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------------


def _chunk_record(chunk: dict) -> dict:
    """Pick the small subset of chunk fields that matters for mapping/reporting."""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source_name": chunk.get("source_name"),
        "section": chunk.get("section"),
        "parent": chunk.get("parent"),
        "heading": chunk.get("heading"),
        # Preserve the structural hierarchy so the retriever (§3) can apply its
        # parent-section boost without re-loading the chunk body.
        "hierarchy_path": list(chunk.get("hierarchy_path") or []),
        "sac_applied": bool(chunk.get("sac_applied")),
    }


def _make_metadata(index_type: str, model_name: str, use_sac: bool, chunks: Sequence[dict], dimension: Any) -> dict:
    return {
        "version": METADATA_VERSION,
        "index_type": index_type,
        "embedding_model": model_name,
        "use_sac": bool(use_sac),
        "dimension": dimension,
        "chunks": [_chunk_record(c) for c in chunks],
    }


def _check_expected(metadata: dict, index_type: str, model_name: str, use_sac: bool) -> None:
    """Raise ValueError when stored metadata does not match what was requested.

    The embedding model is compared by its *original* name (not the slug used
    for directory naming) so that two model names whose slugs collide are still
    detected as a mismatch rather than silently reused.
    """
    stored_model = metadata.get("embedding_model")
    if metadata.get("index_type") != index_type:
        raise ValueError(
            f"index at this path is type {metadata.get('index_type')!r}, "
            f"not {index_type!r}"
        )
    if stored_model != model_name:
        raise ValueError(
            f"index was built with embedding model {stored_model!r}, "
            f"not {model_name!r}; refusing to reuse/overwrite "
            "(choose a different namespace or remove the existing index)"
        )
    if bool(metadata.get("use_sac")) != bool(use_sac):
        raise ValueError(
            f"index was built with use_sac={bool(metadata.get('use_sac'))}, "
            f"requested use_sac={bool(use_sac)}; refusing to reuse/overwrite"
        )


def read_chunks(chunks_path: str | Path) -> list[dict]:
    """Load the JSONL chunks file into a list of dicts (one per line)."""
    chunks_path = Path(chunks_path)
    chunks: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


# --------------------------------------------------------------------------
# Dense (FAISS) index
# --------------------------------------------------------------------------


class DenseIndex:
    """A FAISS inner-product index over chunk embeddings.

    Vector position ``i`` corresponds to ``metadata["chunks"][i]``, so results
    can always be mapped back to chunk IDs / source documents. Uses
    ``IndexFlatIP`` because chunk embeddings are L2-normalized (see
    ``Embedder.embed``), making inner product equal to cosine similarity.
    """

    def __init__(self, faiss_index, metadata: dict, vectors: np.ndarray | None = None):
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.vectors = vectors
        self.chunks: list[dict] = metadata.get("chunks", [])
        self.dimension = self.faiss_index.d

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, chunks: Sequence[dict], embedder: Embedder, use_sac: bool) -> "DenseIndex":
        """Embed chunks with ``embedder`` and build an in-memory FAISS index."""
        if not chunks:
            raise ValueError("cannot build a dense index from zero chunks")
        import faiss

        texts = [chunk_to_embedding_text(c, use_sac) for c in chunks]
        vectors = embedder.embed(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"embedder returned {vectors.ndim}-D vectors; expected 2-D"
            )
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"embedder returned {vectors.shape[0]} vectors for "
                f"{len(chunks)} chunks"
            )
        dimension = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        metadata = _make_metadata("dense", embedder.model_name, use_sac, chunks, dimension)
        return cls(index, metadata, vectors=vectors)

    # -- persistence ------------------------------------------------------

    def save(self, index_dir: str | Path) -> Path:
        """Write the FAISS index, raw vectors, and metadata into ``index_dir``."""
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_dir / INDEX_FILE))
        np.save(index_dir / VECTORS_FILE, np.asarray(self.vectors, dtype=np.float32))
        with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh)
        return index_dir

    @classmethod
    def load(cls, index_dir: str | Path, model_name: str, use_sac: bool) -> "DenseIndex":
        """Load a persisted dense index, verifying it matches (model, SAC)."""
        import faiss

        index_dir = Path(index_dir)
        metadata = _load_metadata(index_dir, "dense")
        _check_expected(metadata, "dense", model_name, use_sac)
        index = faiss.read_index(str(index_dir / INDEX_FILE))
        vectors_path = index_dir / VECTORS_FILE
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        return cls(index, metadata, vectors=vectors)

    # -- querying / mapping ------------------------------------------------

    def position_to_chunk(self, position: int) -> dict | None:
        """Return the chunk record at a vector position (None if out of range)."""
        if 0 <= position < len(self.chunks):
            return self.chunks[position]
        return None

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """Return up to ``top_k`` ``{position, score, chunk}`` results.

        ``query_vector`` must be a normalized, L2-normalized vector of
        ``self.dimension`` (embedding of the query is the retriever's job).
        """
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(
                f"query vector has dimension {q.shape[1]}, expected {self.dimension}"
            )
        scores, positions = self.faiss_index.search(q, top_k)
        results: list[dict] = []
        for score, pos in zip(scores[0], positions[0]):
            pos = int(pos)
            if pos < 0:  # FAISS returns -1 when fewer than top_k results exist
                continue
            results.append(
                {
                    "position": pos,
                    "score": float(score),
                    "chunk": self.position_to_chunk(pos),
                }
            )
        return results


# --------------------------------------------------------------------------
# Sparse (BM25) index
# --------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """A BM25Okapi index over the same chunks, namespaced like the dense index.

    The corpus is the same embedding text used for dense embedding
    (``chunk_to_embedding_text`` under the same SAC flag), so the sparse and
    dense indexes are directly comparable. Document position ``i`` maps to
    ``metadata["chunks"][i]`` exactly like the FAISS index.
    """

    def __init__(self, bm25, tokenized: list[list[str]], metadata: dict):
        self.bm25 = bm25
        self.tokenized = tokenized
        self.metadata = metadata
        self.chunks: list[dict] = metadata.get("chunks", [])

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, chunks: Sequence[dict], embedder: Embedder, use_sac: bool) -> "BM25Index":
        """Tokenize chunks and build a BM25Okapi model."""
        if not chunks:
            raise ValueError("cannot build a BM25 index from zero chunks")
        from rank_bm25 import BM25Okapi

        texts = [chunk_to_embedding_text(c, use_sac) for c in chunks]
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        metadata = _make_metadata("bm25", embedder.model_name, use_sac, chunks, _BM25_DIMENSION)
        return cls(bm25, tokenized, metadata)

    # -- persistence ------------------------------------------------------

    def save(self, index_dir: str | Path) -> Path:
        """Write the tokenized corpus, chunk metadata, and settings to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / CORPUS_FILE).open("w", encoding="utf-8") as fh:
            json.dump(self.tokenized, fh)
        with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh)
        return index_dir

    @classmethod
    def load(cls, index_dir: str | Path, model_name: str, use_sac: bool) -> "BM25Index":
        """Load a persisted BM25 index, verifying it matches (model, SAC).

        rank_bm25 models are plain Python objects, but we rebuild them from
        the stored tokenized corpus on load instead of pickling, which keeps
        the on-disk format version-agnostic and inspectable.
        """
        from rank_bm25 import BM25Okapi

        index_dir = Path(index_dir)
        metadata = _load_metadata(index_dir, "bm25")
        _check_expected(metadata, "bm25", model_name, use_sac)
        with (index_dir / CORPUS_FILE).open("r", encoding="utf-8") as fh:
            tokenized = json.load(fh)
        bm25 = BM25Okapi(tokenized)
        return cls(bm25, tokenized, metadata)

    # -- querying / mapping ------------------------------------------------

    def position_to_chunk(self, position: int) -> dict | None:
        """Return the chunk record at a document position (None if out of range)."""
        if 0 <= position < len(self.chunks):
            return self.chunks[position]
        return None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return up to ``top_k`` ``{position, score, chunk}`` results."""
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1]
        results: list[dict] = []
        for pos in order[:top_k]:
            score = float(scores[pos])
            results.append(
                {
                    "position": int(pos),
                    "score": score,
                    "chunk": self.position_to_chunk(int(pos)),
                }
            )
        return results


# --------------------------------------------------------------------------
# Metadata loader + load-or-build orchestration
# --------------------------------------------------------------------------


def _load_metadata(index_dir: Path, index_type: str) -> dict:
    metadata_path = index_dir / METADATA_FILE
    if not metadata_path.exists():
        if any(index_dir.iterdir()):
            # Something is there but no metadata -> unsafe to reuse blindly.
            raise ValueError(
                f"index at {index_dir} has no {METADATA_FILE}; refusing to "
                "guess its embedding model / SAC configuration"
            )
        raise FileNotFoundError(f"no index found at {index_dir}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if metadata.get("index_type") != index_type:
        raise ValueError(
            f"index at {index_dir} is type {metadata.get('index_type')!r}, "
            f"not {index_type!r}"
        )
    return metadata


def _load_or_build_path(index_dir: Path, exist_ok: bool) -> Path:
    """Return the index dir; raise instead of overwriting a mismatched one.

    If ``exist_ok`` is True, a directory that already contains a metadata file
    is fine (it will be validated against (model, SAC) by the caller). If
    ``exist_ok`` is False and the directory already exists with metadata, we
    refuse to silently overwrite it.
    """
    if not exist_ok and (index_dir / METADATA_FILE).exists():
        raise FileExistsError(
            f"index directory {index_dir} already exists; refusing to "
            "overwrite. Load it instead, or delete it explicitly if you "
            "really want to rebuild."
        )
    return index_dir


def load_or_build_dense(
    chunks_path: str | Path,
    index_root: str | Path,
    embedder: Embedder,
    use_sac: bool,
    exist_ok: bool = True,
) -> DenseIndex:
    """Load an existing dense index for (model, SAC) or build it from chunks.

    Args:
        chunks_path: Path to ``chunks.jsonl``.
        index_root: Root directory for persisted indexes.
        embedder: Embedder; its ``model_name`` drives namespacing.
        use_sac: Whether the index was built with SAC applied to chunks.
        exist_ok: If False and the namespaced directory already exists,
            raise FileExistsError instead of silently rebuilding.

    Returns:
        A ``DenseIndex``. When a matching index already exists it is loaded
        (fast path); otherwise it is built and saved.
    """
    chunks_path = Path(chunks_path)
    index_dir = index_path_for(index_root, embedder.model_name, use_sac) / DENSE_SUBDIR
    index_dir = _load_or_build_path(index_dir, exist_ok)

    if (index_dir / METADATA_FILE).exists():
        return DenseIndex.load(index_dir, embedder.model_name, use_sac)

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")
    chunks = read_chunks(chunks_path)
    dense = DenseIndex.build(chunks, embedder, use_sac)
    dense.save(index_dir)
    return dense


def load_or_build_bm25(
    chunks_path: str | Path,
    index_root: str | Path,
    embedder: Embedder,
    use_sac: bool,
    exist_ok: bool = True,
) -> BM25Index:
    """Load an existing BM25 index for (model, SAC) or build it from chunks.

    Mirrors ``load_or_build_dense``; the same ``embedding_model`` is used for
    namespacing so the sparse and dense indexes for the same run sit side by
    side.
    """
    chunks_path = Path(chunks_path)
    index_dir = index_path_for(index_root, embedder.model_name, use_sac) / BM25_SUBDIR
    index_dir = _load_or_build_path(index_dir, exist_ok)

    if (index_dir / METADATA_FILE).exists():
        return BM25Index.load(index_dir, embedder.model_name, use_sac)

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")
    chunks = read_chunks(chunks_path)
    bm25 = BM25Index.build(chunks, embedder, use_sac)
    bm25.save(index_dir)
    return bm25