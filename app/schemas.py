"""Pydantic request/response models for the API.

These keep the route handlers declarative about what the API accepts and
returns while the heavy lifting stays in the service layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query body."""

    query: str = Field(..., description="The user's legal-document question.")


class QueryResponse(BaseModel):
    """POST /query response (serializable from the service-layer dict)."""

    original_query: str
    final_answer: str
    premise_verification: dict[str, Any] = Field(default_factory=dict)
    adaptive_triggered: bool = False
    rounds_used: int = 1
    rewritten_queries: list[str] = Field(default_factory=list)
    final_retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    failure_tags: list[dict[str, Any]] = Field(default_factory=list)
    claim_level: dict[str, Any] | None = None
    stop_reason: str | None = None
    support_threshold: float | None = None
    max_extra_rounds: int | None = None
    rounds: list[dict[str, Any]] = Field(default_factory=list)
