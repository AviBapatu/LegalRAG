"""LegalRAG FastAPI application (Milestone 8, AGENTS.md §7).

The ``create_app`` factory wires the routes to a lazily-constructed service
layer. Importing this package NEVER requires a Groq API key, network
access, a downloaded BGE model, or a running server: heavy collaborators
(real embedder / indexes / Groq client) are only constructed when a real
(default) ``Application`` is built and asked for a component that needs them.
Tests inject fakes so nothing heavy is ever touched.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
