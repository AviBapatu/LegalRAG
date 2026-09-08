"""LegalRAG ingestion package: loading, chunking, and Summary-Augmented Chunking."""

from .loader import Document, load_documents
from .chunker import Chunk, Chunker, SentenceChunker, PatternChunker
from .sac import summarize_document, apply_sac_to_document
from .pipeline import run_ingestion

__all__ = [
    "Document",
    "load_documents",
    "Chunk",
    "Chunker",
    "SentenceChunker",
    "PatternChunker",
    "summarize_document",
    "apply_sac",
    "run_ingestion",
]
