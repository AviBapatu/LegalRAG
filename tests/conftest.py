"""Shared fixtures for retrieval tests.

The FakeEncoder lets unit tests exercise the full embed/index pipeline without
downloading a real SentenceTransformer model: it produces small deterministic
dense vectors, so index building, persistence, namespacing, and metadata
mapping are all covered offline.
"""

import hashlib

import numpy as np
import pytest

from retrieval.embed import Embedder


class FakeEncoder:
    """Small deterministic encoder with a SentenceTransformer-like ``encode``."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False):
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                vec = vec / norm if norm else vec
            vectors.append(vec)
        return np.stack(vectors)


@pytest.fixture
def fake_encoder():
    return FakeEncoder(dim=8)


@pytest.fixture
def fake_embedder(fake_encoder):
    return Embedder("fake-model", encoder=fake_encoder)


@pytest.fixture
def fake_embedder_other_model():
    return Embedder("other-model", encoder=FakeEncoder(dim=8))


def sample_chunks():
    """Small chunk dicts in the shape produced by ingest.pipeline (Milestone 1).

    The chunks are SAC-applied: ``text`` carries a document-fingerprint prefix
    while ``original_text`` is the raw body, so index builds with SAC on vs. off
    genuinely embed different strings.
    """
    spec = [
        ("nda-0", "nda", "1", None,
         "Mutual NDA between Acme Corp. and Global Enterprises."),
        ("nda-1", "nda", "2", None,
         "Confidential Information shall not be disclosed to third parties."),
        ("emp-0", "emp", "2.1", "2",
         "Employee shall receive an annual base salary of one hundred twenty thousand dollars."),
        ("emp-1", "emp", "3.2", "3",
         "Either party may terminate upon thirty days written notice."),
    ]
    chunks = []
    for chunk_id, source, section, parent, body in spec:
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_name": source,
                "text": f"SUMMARY for {source}.\n\n{body}",
                "original_text": body,
                "section": section,
                "parent": parent,
                "heading": section,
                "hierarchy_path": [section] if parent is None else [parent, section],
                "start_offset": 0,
                "end_offset": len(body),
                "sac_applied": True,
            }
        )
    return chunks