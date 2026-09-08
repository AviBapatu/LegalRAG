"""pytest suite for the Milestone 8 FastAPI application (AGENTS.md §7/§8).

Covers the API routes, the application/service wiring, and the frontend.

Deterministic, offline guarantees enforced throughout:
- no server is started (FastAPI TestClient is in-process)
- no Groq call, no GROQ_API_KEY required
- no embedding-model download (all fakes are injected)
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.config import AppSettings
from app.service import Application
from eval import claim_check, failure_point_tagger
from generation.prompt import (
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    PremiseVerification,
)
from retrieval.adaptive import AdaptiveResult
from retrieval.retriever import RetrievalResult

from .app_fakes import (
    FailingAdaptive,
    FailingGenerator,
    FakeAdaptive,
    FakeGenerator,
    FakeRetriever,
    FakeRewriter,
    sample_retrieval_result,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_application(
    *,
    adaptive=None,
    retriever=None,
    generator=None,
    config=None,
    eval_results=None,
    eval_results_path=None,
) -> Application:
    base_config = {
        "adaptive": {"enabled": True, "max_extra_rounds": 2, "support_threshold": 1.0},
        "evaluation": {
            "enable_ragas_style": True,
            "enable_claim_check": True,
            "enable_failure_tags": True,
        },
    }
    base_config.update(config or {})
    settings = AppSettings(eval_results_path=eval_results_path or "nonexistent.json")
    app = Application(settings=settings, config=base_config)
    if adaptive is not None:
        app.adaptive = adaptive
    if retriever is not None:
        app.retriever = retriever
    if generator is not None:
        app.generator = generator
    return app


def write_eval_results(tmp_path, aggregate=None, envelope=None, per_query=None):
    default_aggregate = {
        "num_queries": 24,
        "k_values": [1, 3, 5, 10],
        "drm_depth": 10,
        "precision_at_k": {"1": 0.58, "3": 0.40, "5": 0.36, "10": 0.30},
        "recall_at_k": {"1": 0.58, "3": 0.79, "5": 0.91, "10": 1.0},
        "mrr": 0.718,
        "map": 0.718,
        "drm_fraction": 0.695,
        "top1_accuracy": 0.583,
        "top3_accuracy": 0.791,
        "faithfulness": 0.75,
        "answer_relevance": 0.19,
        "context_relevance": 0.887,
        "claim_level_faithfulness": 0.75,
        "unsupported_claim_fraction": 0.23,
        "failure_tags": {
            "counts": {"retrieval_failure": 1, "hallucination": 5},
            "tagged_queries": 6,
            "tagged_queries_fraction": 0.25,
        },
    }
    default_aggregate.update(aggregate or {})
    default_envelope = {
        "version": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "num_queries": 24,
        "env": {},
        "scoring": {},
    }
    default_envelope.update(envelope or {})
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "aggregate": default_aggregate,
                "envelope": default_envelope,
                "per_query": per_query or [],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def adaptive_app(tmp_path):
    """An Application wired with all-fake adaptive components."""
    application = make_application(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        adaptive=FakeAdaptive(),
    )
    client = TestClient(create_app(app_=application))
    return client, application


@pytest.fixture
def eval_app(tmp_path):
    path = write_eval_results(tmp_path)
    client = TestClient(create_app(settings=AppSettings(eval_results_path=path)))
    return client


# ---------------------------------------------------------------------------
# API: GET /
# ---------------------------------------------------------------------------


class TestRootRoute:
    def test_root_serves_frontend(self):
        client = TestClient(create_app())
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "<!DOCTYPE html>" in resp.text

    def test_root_contains_expected_ui_elements(self):
        client = TestClient(create_app())
        resp = client.get("/")
        body = resp.text
        for expected in [
            "query-input",
            "submit-btn",
            "loading",
            "answer-text",
            "premise-status",
            "adaptive-badge",
            "rounds",
            "rewritten-list",
            "chunks-list",
            "claims-detail",
            "tags-list",
            "load-eval-btn",
            "eval-metrics",
        ]:
            assert expected in body, f"missing UI element {expected}"

    def test_frontend_references_query_and_eval(self):
        client = TestClient(create_app())
        resp = client.get("/")
        assert "'/query'" in resp.text or '"/query"' in resp.text
        assert "'/eval'" in resp.text or '"/eval"' in resp.text

    def test_uses_no_frontend_framework(self):
        client = TestClient(create_app())
        resp = client.get("/")
        for marker in ["react", "vue", "angular", "import/", "require("]:
            assert marker.lower() not in resp.text.lower()


# ---------------------------------------------------------------------------
# API: POST /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_valid_query(self, adaptive_app):
        client, _ = adaptive_app
        resp = client.post("/query", json={"query": "What is the term?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_query"] == "What is the term?"
        assert data["final_answer"]
        assert isinstance(data["premise_verification"], dict)
        assert "adaptive_triggered" in data
        assert data["rounds_used"] >= 1
        assert isinstance(data["retrieved"] if "retrieved" in data else data["final_retrieval_results"], list)
        assert data["cited_chunk_ids"]
        assert "rounds" in data
        assert "failure_tags" in data
        assert "claim_level" in data

    def test_empty_query_rejected(self, adaptive_app):
        client, _ = adaptive_app
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422
        assert resp.json()["detail"]

    def test_whitespace_query_rejected(self, adaptive_app):
        client, _ = adaptive_app
        resp = client.post("/query", json={"query": "   \n  "})
        assert resp.status_code == 422
        assert resp.json()["detail"]

    def test_missing_query_field(self, adaptive_app):
        client, _ = adaptive_app
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_response_required_fields_present(self, adaptive_app):
        client, _ = adaptive_app
        resp = client.post("/query", json={"query": "test"})
        data = resp.json()
        for field in [
            "original_query",
            "final_answer",
            "premise_verification",
            "adaptive_triggered",
            "rounds_used",
            "rewritten_queries",
            "final_retrieval_results",
            "cited_chunk_ids",
            "failure_tags",
            "claim_level",
            "stop_reason",
            "rounds",
        ]:
            assert field in data, f"missing field {field}"

    def test_retrieval_failure_returns_503(self):
        application = make_application(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            adaptive=FailingAdaptive(RuntimeError("retrieval boom")),
        )
        client = TestClient(create_app(app_=application))
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 503
        assert "retrieval" in resp.json()["detail"].lower() or "adaptive" in resp.json()["detail"].lower()

    def test_generation_failure_returns_503(self):
        # Non-adaptive path where generation fails.
        application = make_application(
            retriever=FakeRetriever(),
            generator=FailingGenerator(),
            config={"adaptive": {"enabled": False}},
        )
        client = TestClient(create_app(app_=application))
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 503
        assert "generation" in resp.json()["detail"].lower()

    def test_retrieval_failure_non_adaptive_returns_503(self):
        application = make_application(
            retriever=FailingRetrieverMock(),
            generator=FakeGenerator(),
            config={"adaptive": {"enabled": False}},
        )
        client = TestClient(create_app(app_=application))
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 503
        assert "retrieval" in resp.json()["detail"].lower()

    def test_no_secrets_in_error(self, adaptive_app):
        # Errors must never leak API keys.
        client, _ = adaptive_app
        resp = client.post("/query", json={})
        assert "api" not in resp.text.lower() or "sk-" not in resp.text.lower()
        assert "GROQ" not in resp.text.upper()


class FailingRetrieverMock:
    def retrieve(self, query, top_k=None):
        raise RuntimeError("retrieval exploded")


# ---------------------------------------------------------------------------
# API: GET /eval
# ---------------------------------------------------------------------------


class TestEvalEndpoint:
    def test_eval_returns_metrics(self, eval_app):
        client = eval_app
        resp = client.get("/eval")
        assert resp.status_code == 200
        data = resp.json()
        agg = data["aggregate"]
        for metric in [
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "precision_at_10",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "map",
            "drm_fraction",
            "faithfulness",
            "answer_relevance",
            "context_relevance",
            "claim_level_faithfulness",
            "failure_tags",
        ]:
            assert metric in agg, f"missing metric {metric}"

    def test_eval_missing_result_returns_404(self):
        application = make_application(eval_results_path="/nonexistent/results.json")
        client = TestClient(create_app(app_=application))
        resp = client.get("/eval")
        assert resp.status_code == 404
        assert "results" in resp.json()["detail"].lower()

    def test_eval_failure_tags_present(self, eval_app):
        client = eval_app
        resp = client.get("/eval")
        tags = resp.json()["aggregate"]["failure_tags"]
        assert "counts" in tags
        assert "tagged_queries" in tags
        assert "tagged_queries_fraction" in tags

    def test_eval_injected_in_memory(self, tmp_path):
        # An injected in-memory result dict bypasses the filesystem entirely.
        results = {
            "aggregate": {
                "num_queries": 2,
                "precision_at_k": {"1": 1.0},
                "recall_at_k": {"1": 1.0},
                "mrr": 1.0,
                "map": 1.0,
                "drm_fraction": 0.0,
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
                "context_relevance": 1.0,
                "claim_level_faithfulness": 1.0,
                "failure_tags": {"counts": {}, "tagged_queries": 0, "tagged_queries_fraction": 0.0},
            },
            "envelope": {},
            "per_query": [],
        }
        application = make_application()
        application.risk_taggers_claims = results
        client = TestClient(create_app(app_=application))
        resp = client.get("/eval")
        assert resp.status_code == 200
        assert resp.json()["aggregate"]["mrr"] == 1.0


# ---------------------------------------------------------------------------
# Application wiring
# ---------------------------------------------------------------------------


class TestApplicationWiring:
    def test_injected_fake_retriever_and_generator_used(self):
        retriever = FakeRetriever()
        generator = FakeGenerator()
        application = make_application(
            retriever=retriever, generator=generator, adaptive=None,
            config={"adaptive": {"enabled": False}},
        )
        result = application.answer_query("What is the term?")
        assert retriever.queries == ["What is the term?"]
        assert generator.queries == ["What is the term?"]
        assert result["final_answer"] == generator.answer

    def test_adaptive_integration_called(self):
        adaptive = FakeAdaptive()
        application = make_application(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            adaptive=adaptive,
        )
        result = application.answer_query("query")
        assert adaptive.run_called_with == ["query"]
        assert result["rounds_used"] == 1
        assert result["adaptive_triggered"] is False

    def test_non_adaptive_path_builds_flat_response(self):
        application = make_application(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            config={"adaptive": {"enabled": False}},
        )
        result = application.answer_query("q")
        assert result["adaptive_triggered"] is False
        assert result["rounds_used"] == 1
        assert "rounds" in result

    def test_retrieved_metadata_preserved(self):
        rr = sample_retrieval_result(
            chunk_id="nda-7",
            source_name="nda_acme",
            section="3.2",
            parent="3",
            heading="Confidentiality",
            hierarchy_path=["3", "3.2"],
            fused_score=0.55,
            dense_score=0.44,
            bm25_score=0.33,
        )
        adaptive = FakeAdaptive(final_evidence=[rr], rounds=1)
        application = make_application(adaptive=adaptive)
        result = application.answer_query("q")
        retrieved = result["final_retrieval_results"]
        assert len(retrieved) == 1
        entry = retrieved[0]
        assert entry["chunk_id"] == "nda-7"
        assert entry["source_name"] == "nda_acme"
        assert entry["section"] == "3.2"
        assert entry["parent"] == "3"
        assert entry["heading"] == "Confidentiality"
        assert entry["hierarchy_path"] == ["3", "3.2"]
        assert entry["fused_score"] == 0.55
        assert entry["dense_score"] == 0.44
        assert entry["bm25_score"] == 0.33

    def test_premise_verification_preserved(self):
        verification = PremiseVerification(
            status=STATUS_UNSUPPORTED,
            premises=["the NDA has no term limit"],
            explanation="no chunk mentions a term limit",
        )
        adaptive = FakeAdaptive(
            generator=FakeGenerator(verification=verification),
            rounds=1,
        )
        application = make_application(adaptive=adaptive)
        result = application.answer_query("q")
        pv = result["premise_verification"]
        assert pv["status"] == STATUS_UNSUPPORTED
        assert pv["premises"] == ["the NDA has no term limit"]
        assert pv["is_unsupported"] is True

    def test_adaptive_rounds_and_rewrites_preserved(self):
        adaptive = FakeAdaptive(
            rounds=2,
            adaptive_triggered=True,
            rewritten_queries=["rewritten clearer query"],
            stop_reason="max_extra_rounds_reached",
        )
        application = make_application(adaptive=adaptive)
        result = application.answer_query("q")
        assert result["adaptive_triggered"] is True
        assert result["rounds_used"] == 2
        assert result["rewritten_queries"] == ["rewritten clearer query"]
        assert result["stop_reason"] == "max_extra_rounds_reached"
        assert len(result["rounds"]) == 2
        assert result["rounds"][1]["is_initial"] is False
        assert result["rounds"][1]["rewritten_from"] == "rewritten clearer query"

    def test_claim_level_results_preserved(self):
        # An answer that fully grounds in the retrieved chunk -> supported claim.
        adaptive = FakeAdaptive(final_evidence=[sample_retrieval_result()])
        application = make_application(adaptive=adaptive)
        result = application.answer_query("q")
        claims = result["claim_level"]
        assert claims is not None
        assert "score" in claims
        assert "num_claims" in claims
        assert "details" in claims
        assert all("claim_id" in d and "status" in d for d in claims["details"])

    def test_failure_tags_preserved(self):
        # "I don't know" answer with no citations triggers generation/tag sets.
        adaptive = FakeAdaptive(
            generator=FakeGenerator(answer="I don't know.", cited=[]),
            rounds=1,
        )
        application = make_application(adaptive=adaptive)
        result = application.answer_query("q")
        assert isinstance(result["failure_tags"], list)
        # The tagger classifying "I don't know" should be stable: no retrieval
        # failure tag is emitted when chunks were retrieved.
        tags = [t["tag"] for t in result["failure_tags"]]
        assert "retrieval_failure" not in tags

    def test_unsupported_premise_tag_emitted(self):
        verification = PremiseVerification(
            status=STATUS_UNSUPPORTED,
            premises=["the agreement has no term"],
            explanation="no chunk mentions a term",
        )
        adaptive = FakeAdaptive(
            generator=FakeGenerator(
                answer="Your premise is unsupported. [citation:chunk-1]",
                verification=verification,
                cited=["chunk-1"],
            ),
            final_evidence=[sample_retrieval_result()],
            rounds=1,
        )
        application = make_application(adaptive=adaptive)
        result = application.answer_query("q")
        tags = [t["tag"] for t in result["failure_tags"]]
        assert failure_point_tagger.TAG_UNSUPPORTED_PREMISE in tags

    def test_config_not_found_error(self, tmp_path, monkeypatch):
        application = Application(
            settings=AppSettings(
                config_path=str(tmp_path / "config.yaml"),
                eval_results_path="x.json",
            ),
            config=None,
        )
        with pytest.raises(Exception) as excinfo:
            application.load_project_config()
        assert "config" in str(excinfo.value).lower()

    def test_importable_without_key_or_download(self):
        # create_app + Application import/construct without touching the real
        # model download code path, the Groq client, or a server.
        application = make_application(adaptive=FakeAdaptive())
        result = application.answer_query("test")
        assert result["final_answer"]


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


class TestFrontend:
    def test_root_is_html(self):
        client = TestClient(create_app())
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_query_eval_references(self):
        client = TestClient(create_app())
        resp = client.get("/")
        body = resp.text
        assert "fetch('/query'" in body
        assert "fetch('/eval'" in body

    def test_answers_rendered(self):
        client = TestClient(create_app())
        resp = client.get("/")
        for marker in ["answer-text", "chunks-list", "premise-status", "failure", "claims"]:
            assert marker in resp.text