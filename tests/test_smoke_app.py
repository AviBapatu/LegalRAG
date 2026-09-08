"""Deterministic pytest coverage of the Milestone 8 smoke scenarios.

The two required scenarios are exercised exactly as the offline smoke script
(``python -m app.smoke``) does, but against a temp eval-results file so these
tests never depend on the repo's committed Milestone 7 results being present.
All components are injected fakes; no server, API key, or model download.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.service import Application
from app.smoke import AdaptiveScripted

from .app_fakes import FakeGenerator


def _write_eval_results(tmp_path) -> str:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "num_queries": 24,
                    "k_values": [1, 3, 5, 10],
                    "drm_depth": 10,
                    "precision_at_k": {"1": 0.58, "3": 0.40, "5": 0.36, "10": 0.30},
                    "recall_at_k": {"1": 0.58, "3": 0.79, "5": 0.91, "10": 1.0},
                    "mrr": 0.718,
                    "map": 0.718,
                    "drm_fraction": 0.695,
                    "faithfulness": 0.75,
                    "answer_relevance": 0.19,
                    "context_relevance": 0.887,
                    "claim_level_faithfulness": 0.75,
                    "failure_tags": {
                        "counts": {"hallucination": 5},
                        "tagged_queries": 6,
                        "tagged_queries_fraction": 0.25,
                    },
                },
                "envelope": {"version": 1},
                "per_query": [],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def smoke_client(tmp_path):
    from app.config import AppSettings

    settings = AppSettings(eval_results_path=_write_eval_results(tmp_path))
    application = Application(settings=settings, config=None)
    client = TestClient(create_app(settings=settings, app_=application))
    return client, application


class TestSmokeScenarios:
    QUERY = "May I disclose confidential information to third parties?"

    def test_example1_adaptive_rewrite_then_success(self, smoke_client):
        client, application = smoke_client
        application.adaptive = AdaptiveScripted("example1")
        resp = client.post("/query", json={"query": self.QUERY})
        assert resp.status_code == 200
        data = resp.json()
        assert data["adaptive_triggered"] is True
        assert data["rounds_used"] == 2
        assert len(data["rewritten_queries"]) == 1
        assert len(data["rounds"]) == 2
        assert data["rounds"][0]["decision"]["needs_another_round"] is True
        assert data["rounds"][1]["decision"]["reason"] == "supported"
        assert data["rounds"][0]["is_initial"] is True
        assert data["rounds"][1]["is_initial"] is False
        assert data["stop_reason"] == "supported"
        assert "cited_chunk_ids" in data
        assert data["premise_verification"]["status"] == "supported"

    def test_example2_sufficient_no_rewrite(self, smoke_client):
        client, application = smoke_client
        application.adaptive = AdaptiveScripted("example2")
        resp = client.post("/query", json={"query": self.QUERY})
        assert resp.status_code == 200
        data = resp.json()
        assert data["adaptive_triggered"] is False
        assert data["rounds_used"] == 1
        assert data["rewritten_queries"] == []
        assert len(data["rounds"]) == 1
        assert data["stop_reason"] == "supported"

    def test_frontend_served(self, smoke_client):
        client, _ = smoke_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert "LegalRAG" in resp.text
        assert "fetch('/query'" in resp.text
        assert "fetch('/eval'" in resp.text

    def test_eval_returns_milestone7_metrics(self, smoke_client):
        client, _ = smoke_client
        resp = client.get("/eval")
        assert resp.status_code == 200
        agg = resp.json()["aggregate"]
        assert agg["mrr"] == 0.718
        assert agg["drm_fraction"] == 0.695
        assert agg["faithfulness"] == 0.75
        assert agg["failure_tags"]["tagged_queries"] == 6
        assert agg["precision_at_1"] == 0.58