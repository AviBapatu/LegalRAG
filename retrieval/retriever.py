"""Hybrid retrieval: dense + BM25 fusion, MMR rerank, and structural boost.

Milestone 3 (§3 of AGENTS.md): given a query, run dense (FAISS) and sparse
(BM25) retrieval in parallel, merge their rankings with Reciprocal Rank Fusion
(RRF), then apply a Maximal Marginal Relevance (MMR) diversity rerank and an
optional structural parent-section boost. The hierarchy metadata produced by
Milestone 1 (``section``/``parent``/``heading``/``hierarchy_path``) is used
(explicitly NOT flattened away) to boost cross-referenced provisions, fixing the
"flattened legal structure" failure mode from AGENTS.md.

This module reuses the Milestone 2 ``DenseIndex``, ``BM25Index`` and
``Embedder`` classes directly via their ``.search()`` APIs; it does not rebuild
or re-index anything.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .embed import Embedder
from .index import BM25Index, DenseIndex

# Default smoothing offset for Reciprocal Rank Fusion.
DEFAULT_RRF_K = 60
# Default MMR lambda: 1.0 is pure relevance, 0.0 is pure diversity.
DEFAULT_MMR_LAMBDA = 0.7
# Default weight applied when a chunk's parent section is already a top match.
DEFAULT_STRUCTURAL_BOOST = 0.1


@dataclass
class RetrievalResult:
    """A single retrieved chunk with per-retriever and fused scores.

    Preserves the chunk and its structural metadata so callers never have to
    re-look-up the source document, section hierarchy, or other fields.
    """

    chunk_id: str
    source_name: str
    chunk: dict
    fused_score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    # Structural metadata from Milestone 1, kept intact.
    section: str | None = None
    parent: str | None = None
    heading: str | None = None
    hierarchy_path: list[str] = field(default_factory=list)
    # Whether this chunk received the structural parent-section boost.
    structurally_boosted: bool = False

    @classmethod
    def from_result_dict(cls, result: dict) -> "RetrievalResult":
        """Build a RetrievalResult from a fused/reranked result dict."""
        chunk = result.get("chunk") or {}
        return cls(
            chunk_id=str(chunk.get("chunk_id")),
            source_name=str(result.get("source_name") or chunk.get("source_name")),
            chunk=chunk,
            fused_score=float(result.get("fused_score", 0.0)),
            dense_score=result.get("dense_score"),
            bm25_score=result.get("bm25_score"),
            section=chunk.get("section"),
            parent=chunk.get("parent"),
            heading=chunk.get("heading"),
            hierarchy_path=list(chunk.get("hierarchy_path") or []),
            structurally_boosted=bool(result.get("structurally_boosted", False)),
        )


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def calculate_rrf(
    dense_results: Sequence[dict],
    bm25_results: Sequence[dict],
    k: float = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse dense and BM25 ranked results using Reciprocal Rank Fusion.

    RRF score for an item is ``sum(1 / (k + rank_i))`` over the rank positions
    ``rank_i`` (1-based) at which the item appeared in each ranked list.
    Items present in only one list get a single term. ``k`` is the smoothing
    constant (standard choice 60); it is configurable here.

    Each item in ``dense_results`` / ``bm25_results`` is an index ``search()``
    result: ``{"position", "score", "chunk": {...}}``. Items are fused by their
    ``chunk["chunk_id"]`` so that chunk/source/section metadata and the
    individual dense/BM25 scores are preserved on the fused result.

    Returns:
        A list of result dicts of the form::

            {
                "chunk_id": ...,
                "source_name": ...,
                "chunk": {...},            # original index chunk record + scores
                "fused_score": ...,
                "rank": 0,                 # 0-based fused rank
                "dense_score": ... | None,
                "bm25_score": ... | None,
                "section": ...,
                "parent": ...,
                "heading": ...,
                "hierarchy_path": [...],
            }

        sorted descending by fused score.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive")

    fused: dict[Any, dict] = {}

    def add(results: Sequence[dict], score_key: str, position_key: str) -> None:
        for position, item in enumerate(results):
            chunk = item.get("chunk") or {}
            cid = chunk.get("chunk_id")
            if cid is None:
                raise ValueError("index search result missing chunk_id")
            rank = position + 1  # 1-based rank
            entry = fused.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "chunk": dict(chunk),
                    "fused_score": 0.0,
                    "dense_score": None,
                    "bm25_score": None,
                    "dense_position": None,
                    "bm25_position": None,
                    "section": chunk.get("section"),
                    "parent": chunk.get("parent"),
                    "heading": chunk.get("heading"),
                    "hierarchy_path": list(chunk.get("hierarchy_path") or []),
                },
            )
            entry["fused_score"] += 1.0 / (k + rank)
            entry[score_key] = float(item.get("score"))
            # The item's own position in the underlying index (not its rank in
            # this result list) lets us map back to stored vectors.
            entry[position_key] = item.get("position")

    add(dense_results, "dense_score", "dense_position")
    add(bm25_results, "bm25_score", "bm25_position")

    ranked = sorted(fused.values(), key=lambda e: e["fused_score"], reverse=True)
    for i, entry in enumerate(ranked):
        entry["rank"] = i
        entry["source_name"] = (entry["chunk"] or {}).get("source_name")
    return ranked


# --------------------------------------------------------------------------
# Structural parent-section boost
# --------------------------------------------------------------------------


def _anchor_sections(results: Sequence[dict]) -> set[str]:
    """Return the set of section ids present among the result chunks' sections.

    A slightly-lower-ranked sibling subsection is boosted when its *parent*
    section already appears among these top matches (as some chunk's section).
    Only section values are anchors -- a chunk's own parent value on its own
    never satisfies the condition (we need an actual ancestor chunk present).
    """
    sections: set[str] = set()
    for r in results:
        chunk = r.get("chunk") or {}
        section = chunk.get("section")
        if section:
            sections.add(str(section))
    return sections


def apply_structural_boost(
    fused_results: Sequence[dict],
    strength: float = DEFAULT_STRUCTURAL_BOOST,
    anchor_parents: set[str] | None = None,
) -> list[dict]:
    """Boost candidates whose parent section is already among the top matches.

    Given the fused ranking, for each candidate chunk that has a ``parent``
    section (from Milestone 1's hierarchy metadata), if that parent appears as
    some other candidate's section in the current result set, add ``strength``
    to the candidate's fused score. This helps recover cross-referenced
    provisions that pure similarity would miss, without flattening the
    hierarchy (the raw metadata is preserved on the chunk).

    Args:
        fused_results: Output of ``calculate_rrf`` (or anything with the same
            ``fused_score`` / ``chunk`` shape).
        strength: Amount added to the fused score of a boosted candidate.
        anchor_parents: Optional explicit set of "anchoring" section ids to
            test against. When None, the set is derived from the current
            results (the section ids present among them).

    Returns:
        A NEW list (input is not mutated) with ``fused_score`` possibly
        increased and ``structurally_boosted`` flags set, re-sorted descending.
    """
    if strength <= 0:
        # No-op path; return a copy unchanged.
        out = [dict(r) for r in fused_results]
        for r in out:
            r["structurally_boosted"] = False
        return _resort(out)

    if anchor_parents is None:
        anchor_parents = _anchor_sections(fused_results)

    out: list[dict] = []
    for r in fused_results:
        r = dict(r)
        chunk = r.get("chunk") or {}
        parent = chunk.get("parent")
        boosted = bool(parent) and str(parent) in anchor_parents
        if boosted:
            r["fused_score"] = float(r["fused_score"]) + float(strength)
        r["structurally_boosted"] = bool(boosted)
        out.append(r)
    return _resort(out)


def _resort(results: Sequence[dict]) -> list[dict]:
    """Sort a list of result dicts descending by fused_score, re-ranking."""
    ranked = sorted(results, key=lambda e: float(e["fused_score"]), reverse=True)
    for i, entry in enumerate(ranked):
        entry["rank"] = i
    return ranked


# --------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) reranking
# --------------------------------------------------------------------------


def rerank_mmr(
    fused_results: Sequence[dict],
    similarities: np.ndarray,
    query_similarities: Sequence[float],
    lambda_: float = DEFAULT_MMR_LAMBDA,
) -> list[dict]:
    """Rerank fused results by Maximal Marginal Relevance.

    MMR greedily builds an ordered list. At each step it selects the candidate
    chunk maximizing::

        lambda * rel(i) - (1 - lambda) * max_sim(i, selected)

    where ``rel(i)`` is the (normalized) query relevance of chunk ``i`` and
    ``max_sim`` is its maximum similarity to already-selected chunks. A strong
    value of ``lambda`` (near 1) favors relevance; a weak value (near 0)
    favors diversity, avoiding redundant collinear chunks.

    Args:
        fused_results: Fused/hybrid result dicts (from ``calculate_rrf`` and/or
            ``apply_structural_boost``), ordered by ``fused_score``.
        similarities: Square ``NxN`` matrix of pairwise cosine similarities
            among the corresponding candidate chunks, with diagonal ~1.
        query_similarities: Per-candidate query-to-chunk cosine similarities
            (``N`` values), one per result.
        lambda_: MMR trade-off in [0, 1].

    Returns:
        A NEW list ordered by MMR criterion (the MMR score itself is not the
        fused score; the fused score is preserved on each result dict).
    """
    n = len(fused_results)
    if n == 0:
        return []
    if lambda_ < 0 or lambda_ > 1:
        raise ValueError("MMR lambda must be in [0, 1]")

    similarities = np.asarray(similarities, dtype=np.float64)
    query_similarities = np.asarray(query_similarities, dtype=np.float64)

    if similarities.shape != (n, n):
        raise ValueError(
            f"similarities must be {n}x{n}, got {similarities.shape}"
        )
    if query_similarities.shape != (n,):
        raise ValueError(
            f"query_similarities must have length {n}, got {query_similarities.shape}"
        )

    # Normalize relevance to [0,1] so lambda is meaningful regardless of the
    # scale of the fused scores.
    rel = np.asarray([float(r["fused_score"]) for r in fused_results])
    rmin, rmax = float(rel.min()), float(rel.max())
    if rmax > rmin:
        rel_norm = (rel - rmin) / (rmax - rmin)
    else:
        rel_norm = np.ones(n)

    remaining = set(range(n))
    selected: list[int] = []

    while remaining:
        best_i = -1
        best_mmr = float("-inf")
        for i in remaining:
            if selected:
                max_sim = float(similarities[i, selected].max())
            else:
                max_sim = 0.0
            mmr = lambda_ * float(rel_norm[i]) - (1 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_i = i
        selected.append(best_i)
        remaining.discard(best_i)

    out = [dict(fused_results[i]) for i in selected]
    for i, entry in enumerate(out):
        entry["rank"] = i
    return out


# --------------------------------------------------------------------------
# HybridRetriever
# --------------------------------------------------------------------------


class HybridRetriever:
    """Combine dense + BM25 retrieval, fuse with RRF, rerank with MMR and a
    structural parent-section boost.

    Uses the existing Milestone 2 ``DenseIndex`` and ``BM25Index`` objects
    directly; nothing is re-instantiated or re-indexed here.

    Args:
        dense_index: A ``DenseIndex``.
        bm25_index: A ``BM25Index``.
        embedder: An ``Embedder`` used to embed the query; its vectors must
            be compatible with the dense index's dimension.
        top_k: Number of results returned by ``retrieve``.
        rrf_k: RRF smoothing constant.
        mmr_lambda: MMR trade-off.
        structural_boost_strength: Weight of the structural parent-section
            boost; <=0 disables the boost.
        use_mmr: Whether to apply MMR diversity reranking.
        use_structural_boost: Whether to apply the structural boost.
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        bm25_index: BM25Index,
        embedder: Embedder,
        top_k: int = 10,
        rrf_k: float = DEFAULT_RRF_K,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        structural_boost_strength: float = DEFAULT_STRUCTURAL_BOOST,
        use_mmr: bool = True,
        use_structural_boost: bool = True,
    ):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.structural_boost_strength = structural_boost_strength
        self.use_mmr = use_mmr
        self.use_structural_boost = use_structural_boost

    @classmethod
    def from_config(
        cls,
        dense_index: DenseIndex,
        bm25_index: BM25Index,
        embedder: Embedder,
        retrieval_config: dict,
    ) -> "HybridRetriever":
        """Build a retriever from a config dict (the ``retrieval`` block)."""
        return cls(
            dense_index=dense_index,
            bm25_index=bm25_index,
            embedder=embedder,
            top_k=int(retrieval_config.get("top_k", 10)),
            rrf_k=float(retrieval_config.get("rrf_k", DEFAULT_RRF_K)),
            mmr_lambda=float(retrieval_config.get("mmr_lambda", DEFAULT_MMR_LAMBDA)),
            structural_boost_strength=float(
                retrieval_config.get("structural_boost_strength", DEFAULT_STRUCTURAL_BOOST)
            ),
            use_mmr=bool(retrieval_config.get("use_mmr", True)),
            use_structural_boost=bool(
                retrieval_config.get("use_structural_boost", True)
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Run full hybrid retrieval for a query.

        Args:
            query: Natural-language query text.
            top_k: Override the number of returned results (defaults to
                ``self.top_k``).
            dense_top_k: Override the number of chunks fetched from the dense
                index. If None, an intermediate ``fetch_k`` is fetched.
            bm25_top_k: Override the number fetched from BM25. If None,
                ``fetch_k`` is fetched.

        The dense and BM25 queries are independent, so they run in parallel
        (two threads, one index each). We fetch an intermediate ``fetch_k``
        from each index so the structural boost has enough candidates to
        examine; the final result list still returns only ``top_k``.

        Returns:
            A list of ``RetrievalResult``, ranked by fused (post-boost) score;
            when MMR is enabled the order reflects the MMR diversity rerank.
        """
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")

        # The intermediate fetch size: enough candidates to make fusion +
        # structural boost meaningful while keeping retrieval bounded. We fetch
        # a multiple of k (never below k) capped so we don't over-fetch.
        fetch_k = max(k, int(k * 2))

        dk = fetch_k if dense_top_k is None else dense_top_k
        bk = fetch_k if bm25_top_k is None else bm25_top_k

        # Embed the query once.
        query_vector = self.embedder.embed(query)[0]

        # Dense + BM25 retrieval in parallel: the two index searches are
        # independent, so they run concurrently on separate indexes.
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self.dense_index.search, query_vector, dk)
            bm25_future = pool.submit(self.bm25_index.search, query, bk)
            dense_results = dense_future.result()
            bm25_results = bm25_future.result()

        # 1. Reciprocal Rank Fusion.
        fused = calculate_rrf(dense_results, bm25_results, k=self.rrf_k)

        # 2. Structural parent-section boost.
        if self.use_structural_boost and self.structural_boost_strength > 0:
            fused = apply_structural_boost(
                fused, strength=self.structural_boost_strength
            )

        # 3. MMR diversity rerank.
        reranked = fused
        if self.use_mmr and len(fused) > 1:
            reranked = self._apply_mmr(fused, query_vector)

        # Build ordered result objects from the (possibly reranked) fused list.
        chosen = reranked[:k]
        return [RetrievalResult.from_result_dict(r) for r in chosen]

    def _apply_mmr(
        self, fused_results: Sequence[dict], query_vector: np.ndarray
    ) -> list[dict]:
        """Compute pairwise chunk similarities and run MMR over `fused_results`."""
        vectors = self._vectors_for(fused_results)
        query_sim = [float(np.dot(query_vector, v)) for v in vectors]
        similarities = vectors @ vectors.T
        return rerank_mmr(
            fused_results,
            similarities,
            query_sim,
            lambda_=self.mmr_lambda,
        )

    def _vectors_for(self, fused_results: Sequence[dict]) -> np.ndarray:
        """Return a row-per-result embedding matrix of the fused chunks.

        Prefers the dense index's stored vectors (via each entry's
        ``dense_position``); falls back to re-embedding the chunk text for
        results that did not come from the dense index.
        """
        vectors: list[np.ndarray] = []
        stored = getattr(self.dense_index, "vectors", None)
        for r in fused_results:
            pos = r.get("dense_position")
            if (
                pos is not None
                and stored is not None
                and 0 <= int(pos) < stored.shape[0]
            ):
                vectors.append(np.asarray(stored[int(pos)], dtype=np.float32))
            else:
                text = str((r.get("chunk") or {}).get("text", ""))
                vectors.append(self.embedder.embed([text])[0])
        return np.stack(vectors)


# Keep a concise public API surface.
__all__ = [
    "RetrievalResult",
    "calculate_rrf",
    "apply_structural_boost",
    "rerank_mmr",
    "HybridRetriever",
]
