"""LegalRAG retrieval package: dense embeddings, FAISS index, and BM25 index."""

from .embed import Embedder, chunk_to_embedding_text
from .index import DenseIndex, BM25Index
from .retriever import (
    RetrievalResult,
    calculate_rrf,
    apply_structural_boost,
    rerank_mmr,
    HybridRetriever,
)
from .adaptive import (
    AdaptiveRetriever,
    AdaptiveResult,
    AdaptiveRound,
    QueryRewriter,
    SupportDecision,
    decide_support,
)

__all__ = [
    "Embedder",
    "chunk_to_embedding_text",
    "DenseIndex",
    "BM25Index",
    "RetrievalResult",
    "calculate_rrf",
    "apply_structural_boost",
    "rerank_mmr",
    "HybridRetriever",
    "AdaptiveRetriever",
    "AdaptiveResult",
    "AdaptiveRound",
    "QueryRewriter",
    "SupportDecision",
    "decide_support",
]
