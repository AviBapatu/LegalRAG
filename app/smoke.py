"""Local smoke test for the Milestone 8 API (no API key, no downloads, no server).

Runs the two required smoke scenarios against the real FastAPI app with
injected fake components, then demonstrates the three API surfaces:

    Example 1: query -> initial retrieval -> insufficient evidence
               -> adaptive rewrite -> second retrieval -> final answer
    Example 2: query -> sufficient evidence -> final answer (no rewrite)

Plus:
    - GET / returns the frontend
    - POST /query returns structured retrieval/generation information
    - GET /eval returns the existing Milestone 7 metrics

Run from the repo root:

    .venv/bin/python -m app.smoke

Everything is offline: the retriever/generator are fakes, the eval summary is
spoon-fed from the persisted eval/results/full_eval/results.json, and no
Groq key or embedding model is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from app.config import AppSettings
from app.service import Application
from generation.prompt import PremiseVerification
from retrieval.adaptive import (
    AdaptiveResult,
    AdaptiveRound,
    SupportDecision,
)
from retrieval.retriever import RetrievalResult

from tests.app_fakes import FakeGenerator, sample_retrieval_result


def _retrieval_result(chunk_id: str, source: str, text: str, section: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        source_name=source,
        chunk={
            "chunk_id": chunk_id,
            "source_name": source,
            "text": text,
            "original_text": text,
            "section": section,
            "parent": "1",
            "heading": section,
            "hierarchy_path": ["1", section],
        },
        fused_score=0.9,
        dense_score=0.8,
        bm25_score=0.7,
        section=section,
        parent="1",
        heading=section,
        hierarchy_path=["1", section],
    )


def _round(
    query: str,
    evidence,
    generation,
    decision,
    index: int,
    is_initial: bool,
    rewritten_from: str | None,
) -> AdaptiveRound:
    return AdaptiveRound(
        query=query,
        retrieval_results=evidence,
        generation=generation,
        decision=decision,
        round_index=index,
        is_initial=is_initial,
        rewritten_from=rewritten_from,
    )


class AdaptiveScripted:
    """Scripted two-round adaptive flow demonstrating both smoke examples.

    ``mode == "example1"``: the first round produces an "I don't know" answer
    (insufficient evidence), a rewrite happens, and the second round succeeds.
    ``mode == "example2"``: the first round is already supported; no rewrite.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.max_extra_rounds = 2
        self.support_threshold = 1.0

    def run(self, query: str) -> AdaptiveResult:
        first_evidence = [
            _retrieval_result(
                "chunk-a1",
                "nda_acme",
                "Non-disclosure obligations. [irrelevant boilerplate]",
                "1",
            )
        ]
        second_evidence = [
            _retrieval_result(
                "chunk-b2",
                "nda_acme",
                "Confidential Information shall not be disclosed to third parties.",
                "2",
            )
        ]

        if self.mode == "example1":
            # Round 0: insufficient (I don't know, no citation) -> rewrite.
            g0 = FakeGenerator(
                answer="I don't know.",
                cited=[],
                verification=PremiseVerification(status="supported"),
            ).generate(query, first_evidence)
            d0 = SupportDecision(needs_another_round=True, reason="i_dont_know")
            # Round 1: sufficient -> supported answer citing chunk-b2.
            g1 = FakeGenerator(
                answer="The NDA prohibits disclosure. [citation:chunk-b2]",
                cited=["chunk-b2"],
                verification=PremiseVerification(status="supported"),
            ).generate("rewritten: confidential information disclosure restrictions", second_evidence)
            d1 = SupportDecision(needs_another_round=False, reason="supported")
            return AdaptiveResult(
                original_query=query,
                final_answer=g1.answer,
                final_verification=g1.verification,
                final_retrieval_results=second_evidence,
                rounds_used=2,
                adaptive_triggered=True,
                rewritten_queries=["rewritten: confidential information disclosure restrictions"],
                rounds=[
                    _round(query, first_evidence, g0, d0, 0, True, None),
                    _round(
                        "rewritten: confidential information disclosure restrictions",
                        second_evidence,
                        g1,
                        d1,
                        1,
                        False,
                        "rewritten: confidential information disclosure restrictions",
                    ),
                ],
                stop_reason="supported",
                support_threshold=1.0,
                max_extra_rounds=2,
            )

        # example2: single sufficient round.
        g0 = FakeGenerator(
            answer="The NDA prohibits disclosure. [citation:chunk-b2]",
            cited=["chunk-b2"],
            verification=PremiseVerification(status="supported"),
        ).generate(query, second_evidence)
        d0 = SupportDecision(needs_another_round=False, reason="supported")
        return AdaptiveResult(
            original_query=query,
            final_answer=g0.answer,
            final_verification=g0.verification,
            final_retrieval_results=second_evidence,
            rounds_used=1,
            adaptive_triggered=False,
            rewritten_queries=[],
            rounds=[_round(query, second_evidence, g0, d0, 0, True, None)],
            stop_reason="supported",
            support_threshold=1.0,
            max_extra_rounds=2,
        )


def build_app(eval_results_path: str | None) -> TestClient:
    settings = AppSettings(
        eval_results_path=eval_results_path or "eval/results/full_eval/results.json",
        config_path="config.yaml",
    )
    application = Application(settings=settings, config=None)
    client = TestClient(create_app(settings=settings, app_=application))
    return client, application


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    eval_path = repo_root / "eval/results/full_eval/results.json"

    client, application = build_app(str(eval_path))

    for mode in ["example1", "example2"]:
        application.adaptive = AdaptiveScripted(mode)
        application.retriever = None
        application.generator = None
        query = "May I disclose confidential information to third parties?"
        resp = client.post("/query", json={"query": query})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        print(f"\n===== {mode} =====")
        print(f"original query      : {data['original_query']}")
        print(f"final answer        : {data['final_answer']}")
        print(f"adaptive_triggered  : {data['adaptive_triggered']}")
        print(f"rounds_used         : {data['rounds_used']}")
        print(f"rewritten_queries   : {data['rewritten_queries']}")
        print(f"stop_reason         : {data['stop_reason']}")
        print(f"premise status      : {data['premise_verification']['status']}")
        print(f"cited chunk ids     : {data['cited_chunk_ids']}")
        if mode == "example1":
            assert data["adaptive_triggered"] is True
            assert data["rounds_used"] == 2
            assert len(data["rewritten_queries"]) == 1
            assert len(data["rounds"]) == 2
        else:
            assert data["adaptive_triggered"] is False
            assert data["rounds_used"] == 1
            assert data["rewritten_queries"] == []

    # GET / -> frontend
    frontend = client.get("/")
    assert frontend.status_code == 200
    assert "LegalRAG" in frontend.text
    print("\n===== GET / (frontend) =====")
    print(f"status             : {frontend.status_code}")
    print(f"content-type       : {frontend.headers.get('content-type')}")
    print(f"served title       : yes (LegalRAG page)")

    # GET /eval -> Milestone 7 metrics
    ev = client.get("/eval")
    assert ev.status_code == 200, ev.text
    agg = ev.json()["aggregate"]
    print("\n===== GET /eval (Milestone 7 metrics) =====")
    print(f"precision@1        : {agg['precision_at_1']}")
    print(f"precision@5        : {agg['precision_at_5']}")
    print(f"recall@3           : {agg['recall_at_3']}")
    print(f"recall@10          : {agg['recall_at_10']}")
    print(f"mrr                : {agg['mrr']}")
    print(f"map                : {agg['map']}")
    print(f"drm_fraction       : {agg['drm_fraction']}")
    print(f"faithfulness       : {agg['faithfulness']}")
    print(f"answer_relevance   : {agg['answer_relevance']}")
    print(f"context_relevance  : {agg['context_relevance']}")
    print(f"claim faithfulness : {agg['claim_level_faithfulness']}")
    ft = agg["failure_tags"]
    print(f"failure tags       : {ft['tagged_queries']} tagged / {ft['counts']}")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))