"""Standard retrieval metrics for the Milestone 7 evaluation harness.

Implemented entirely from scratch (no evaluation-library wrappers) so every
formula is inspectable and matches the definitions the survey points to
(Hindi et al., AGENTS.md §6 / Table 6 & 9).

Relevance definition
--------------------
Relevance is DOCUMENT-LEVEL by default and is driven by each query's
ground-truth ``expected_document`` (the same signal DRM uses in
``eval/drm_eval.py``): a retrieved item is relevant iff its source document
equals the expected document. So ``precision@k`` counts, among the top-k
retrieved chunks, how many come from the right source document, and
``recall@k`` reports whether the expected document was found in the top-k
(the single relevant "item" at the document level). This is exactly the signal
that matters for legal RAG retrieval quality and the one the embedding
ablation in ``eval/ablation.py`` measures.

A CHUNK-LEVEL view is also provided: a retrieved chunk is relevant iff its
``source_name`` equals the expected document. At this level the recall
denominator is the number of the expected document's chunks that are known in
the corpus, so recall@k measures how many of a document's expected chunks were
recovered in the top-k.

Edge cases handled
------------------
* no relevant results retrieved  -> precision/MRR/AP = 0.0, recall = 0.0
* fewer than ``k`` results       -> the denominator shrinks to the number
                                    actually retrieved (a 3-chunk corpus with
                                    k=10 is not penalized as if it were full)
* an empty relevant set          -> recall = 0.0 (there is nothing to recall)
* ``k < 1``                      -> ``ValueError``
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

RELEVANCE_DOCUMENT = "document"
RELEVANCE_CHUNK = "chunk"
_VALID_LEVELS = (RELEVANCE_DOCUMENT, RELEVANCE_CHUNK)

#: Default depths evaluated by the harness.
DEFAULT_K_VALUES = (1, 3, 5, 10)


def _require_positive_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


# ---------------------------------------------------------------------------
# Core metric functions (pure, operate on rank-ordered ids + relevant id set)
# ---------------------------------------------------------------------------


def precision_at_k(ranked_ids: Sequence[Any], relevant_ids: Iterable[Any], k: int) -> float:
    """Precision@k: fraction of the top-k retrieved items that are relevant.

    A retrieved item is relevant if it is in ``relevant_ids``. When fewer than
    ``k`` items were retrieved the denominator is the number of items that were
    actually retrieved, so a small corpus is scored at its true size.
    """
    _require_positive_k(k)
    relevant = set(relevant_ids)
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(ranked_ids: Sequence[Any], relevant_ids: Iterable[Any], k: int) -> float:
    """Recall@k: fraction of the relevant items recovered in the top-k.

    Counts DISTINCT relevant items (duplicate rows in a ranked list cannot
    inflate recall beyond 1.0). With document-level relevance this is 1.0 iff
    the expected document appears somewhere in the top-k, otherwise 0.0.
    Returns ``0.0`` when the relevant set is empty.
    """
    _require_positive_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = list(ranked_ids)[:k]
    recovered = len({item for item in top if item in relevant})
    return recovered / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[Any], relevant_ids: Iterable[Any]) -> float:
    """Reciprocal rank of the first relevant hit (0.0 when there is no hit).

    Returns ``1 / (position + 1)`` for the first rank that contains a relevant
    item, where position is 0-based; ``0.0`` if none of the retrieved items is
    relevant.
    """
    relevant = set(relevant_ids)
    for index, item in enumerate(ranked_ids):
        if item in relevant:
            return 1.0 / (index + 1)
    return 0.0


def average_precision(ranked_ids: Sequence[Any], relevant_ids: Iterable[Any]) -> float:
    """Average Precision: the mean of precision values at relevant ranks.

    Each relevant item counts ONCE wherever it first appears in the ranked
    list; duplicate rows cannot inflate the score. The sum of (precision@rank
    where a new relevant item is first seen) is divided by the TOTAL number of
    relevant items, so an item never retrieved or one that is relevant but
    absent lowers the score. No relevant items retrieved -> ``0.0``; an empty
    relevant set -> ``0.0``.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    seen: set[Any] = set()
    ap = 0.0
    for index, item in enumerate(ranked_ids):
        if item in relevant and item not in seen:
            seen.add(item)
            ap += len(seen) / (index + 1)
    return ap / len(relevant)


def mrr(ranked_and_relevant: Sequence[tuple[Sequence[Any], Iterable[Any]]]) -> float:
    """Mean Reciprocal Rank over a set of (ranked_ids, relevant_ids) queries."""
    items = list(ranked_and_relevant)
    if not items:
        return 0.0
    return sum(reciprocal_rank(ranked, relevant) for ranked, relevant in items) / len(items)


def mean_average_precision(
    ranked_and_relevant: Sequence[tuple[Sequence[Any], Iterable[Any]]]
) -> float:
    """Mean Average Precision (MAP) over a set of queries."""
    items = list(ranked_and_relevant)
    if not items:
        return 0.0
    return sum(average_precision(ranked, relevant) for ranked, relevant in items) / len(items)


# ---------------------------------------------------------------------------
# Run-record helpers (the run shape produced by eval.drm_eval.retrieve_all_queries)
# ---------------------------------------------------------------------------


def document_level_pair(run: dict) -> tuple[list[Any], set[Any]]:
    """Return ``(ranked sources, {expected_document})`` for a run record.

    ``run`` provides ``expected_document`` and ``retrieved_sources`` (the
    source documents of the retrieved chunks in rank order).
    """
    return (
        list(run.get("retrieved_sources") or []),
        {run.get("expected_document")},
    )


def chunk_level_pair(run: dict, corpus_chunks: Sequence[dict]) -> tuple[list[Any], set[Any]]:
    """Return ``(ranked chunk ids, {ids of the expected doc's corpus chunks})``.

    ``run`` provides ``expected_document`` and ``retrieved_chunks`` (each with
    a ``chunk_id``). ``corpus_chunks`` is the full known chunk list, so the
    recall denominator is the number of the expected document's chunks that
    actually exist, not just the ones retrieved.
    """
    ranked = [
        c.get("chunk_id")
        for c in (run.get("retrieved_chunks") or [])
        if c.get("chunk_id")
    ]
    expected = run.get("expected_document")
    relevant = {
        c.get("chunk_id")
        for c in (corpus_chunks or [])
        if c.get("source_name") == expected and c.get("chunk_id")
    }
    return (ranked, relevant)


def evaluate_retrieval(
    run: dict,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    corpus_chunks: Sequence[dict] | None = None,
    level: str = RELEVANCE_DOCUMENT,
) -> dict:
    """Per-query retrieval metrics for one run record.

    Returns ``{"precision_at_k": {k: ...}, "recall_at_k": {k: ...},
    "reciprocal_rank": ..., "average_precision": ...}`` under the chosen
    relevance level.
    """
    if level not in _VALID_LEVELS:
        raise ValueError(f"unknown relevance level {level!r}; expected one of {_VALID_LEVELS}")
    if level == RELEVANCE_DOCUMENT:
        ranked, relevant = document_level_pair(run)
    else:
        ranked, relevant = chunk_level_pair(run, corpus_chunks)
    return {
        "precision_at_k": {k: precision_at_k(ranked, relevant, k) for k in k_values},
        "recall_at_k": {k: recall_at_k(ranked, relevant, k) for k in k_values},
        "reciprocal_rank": reciprocal_rank(ranked, relevant),
        "average_precision": average_precision(ranked, relevant),
    }


def evaluate_queries(
    runs: Sequence[dict],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    corpus_chunks: Sequence[dict] | None = None,
    level: str = RELEVANCE_DOCUMENT,
) -> dict:
    """Retrieval metrics over a whole query set.

    Args:
        runs: Run records in the ``eval.drm_eval`` shape (``query_id``,
            ``expected_document``, ``retrieved_sources``, ``retrieved_chunks``).
        k_values: Depths at which precision/recall are reported.
        corpus_chunks: Needed only for chunk-level recall.
        level: ``document`` (default) or ``chunk``.

    Returns:
        ``{"per_query": [metrics per run], "aggregate": {precision_at_k,
        recall_at_k, mrr, map}}``. Both are deterministic given the runs.
    """
    per_query: list[dict] = []
    pairs: list[tuple[list[Any], set[Any]]] = []
    for run in runs:
        if level == RELEVANCE_DOCUMENT:
            ranked, relevant = document_level_pair(run)
        else:
            ranked, relevant = chunk_level_pair(run, corpus_chunks)
        per_query.append(
            {
                "precision_at_k": {k: precision_at_k(ranked, relevant, k) for k in k_values},
                "recall_at_k": {k: recall_at_k(ranked, relevant, k) for k in k_values},
                "reciprocal_rank": reciprocal_rank(ranked, relevant),
                "average_precision": average_precision(ranked, relevant),
            }
        )
        pairs.append((ranked, relevant))

    k_list = list(k_values)
    if not per_query:
        aggregate = {
            "mrr": 0.0,
            "map": 0.0,
            "precision_at_k": {k: 0.0 for k in k_list},
            "recall_at_k": {k: 0.0 for k in k_list},
        }
    else:
        aggregate = {
            "mrr": mrr(pairs),
            "map": mean_average_precision(pairs),
            "precision_at_k": {
                k: sum(p["precision_at_k"][k] for p in per_query) / len(per_query)
                for k in k_list
            },
            "recall_at_k": {
                k: sum(p["recall_at_k"][k] for p in per_query) / len(per_query)
                for k in k_list
            },
        }
    return {"per_query": per_query, "aggregate": aggregate}


__all__ = [
    "DEFAULT_K_VALUES",
    "RELEVANCE_CHUNK",
    "RELEVANCE_DOCUMENT",
    "average_precision",
    "chunk_level_pair",
    "document_level_pair",
    "evaluate_queries",
    "evaluate_retrieval",
    "mean_average_precision",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]