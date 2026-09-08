"""Tests for Milestone 6 adaptive retrieval (retrieval/adaptive.py).

Covers the deterministic support decision, the query-rewrite client, the
controller loop (initial retrieval always runs, bounded extra rounds, graceful
rewrite failure, full per-round bookkeeping), config loading, and a real
``HybridRetriever`` + ``QueryRewriter`` integration path — all local and
deterministic, with mocked generation and a fake Groq SDK. No test here
needs an API key or network access.
"""

from pathlib import Path

import pytest

from generation.generate import (
    GroqClient,
    GenerationConfig,
    GenerationResult,
)
from generation.prompt import (
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    PremiseVerification,
)
from retrieval.adaptive import (
    REASON_MAX_ROUNDS,
    REASON_NO_REWRITER,
    REASON_REWRITE_FAILED,
    REASON_SUPPORTED,
    AdaptiveRetriever,
    AdaptiveResult,
    AdaptiveRound,
    QueryRewriter,
    SupportDecision,
    decide_support,
)
from retrieval.index import BM25Index, DenseIndex
from retrieval.retriever import HybridRetriever, RetrievalResult

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ev(chunk_id, source="nda", text="body of chunk", section=None):
    """A Milestone 3-shaped RetrievalResult for use as retrieved evidence."""
    return RetrievalResult.from_result_dict(
        {
            "chunk": {
                "chunk_id": chunk_id,
                "source_name": source,
                "text": text,
                "section": section,
                "parent": None,
                "heading": section,
                "hierarchy_path": [section] if section else [],
            },
            "fused_score": 1.0,
        }
    )


def gen(
    answer,
    verification=None,
    cited=None,
    query="q",
):
    """A GenerationResult with the same shape the Milestone 5 generator emits."""
    return GenerationResult(
        answer=answer,
        query=query,
        verification=verification if verification is not None else PremiseVerification(),
        cited_chunk_ids=list(cited or []),
        prompt={"system": "", "user": ""},
        model="test-model",
        temperature=0.0,
        max_tokens=1,
    )


class ScriptedRetriever:
    """Stands in for HybridRetriever: returns one scripted list per call."""

    def __init__(self, *result_lists):
        self.result_lists = [list(r) for r in result_lists]
        self.queries: list[str] = []

    def retrieve(self, query: str):
        self.queries.append(query)
        idx = min(len(self.queries) - 1, len(self.result_lists) - 1)
        if idx < 0:
            return []
        return list(self.result_lists[idx])


class StubGenerator:
    """Stands in for GroqGenerator: a scripted sequence of answers."""

    def __init__(self, script):
        self.script = list(script)  # each item: (answer, verification, cited)
        self.calls = []  # list of (query, evidence)

    def generate(self, query, evidence=None):
        self.calls.append((query, list(evidence or [])))
        answer, verification, cited = self.script.pop(0)
        return gen(answer, verification, cited, query=query)


class StubRewriter:
    """Stands in for QueryRewriter.rewrite with scripted outputs."""

    def __init__(self, rewrites=None):
        self.rewrites = list(rewrites or [])
        self.calls = []

    def rewrite(self, original_query, reason, evidence):
        self.calls.append((original_query, reason, list(evidence or [])))
        if not self.rewrites:
            raise RuntimeError("no more rewrites available")
        return self.rewrites.pop(0)


def make_controller(retriever, generator, rewriter=None, **kwargs):
    return AdaptiveRetriever(
        retriever=retriever,
        generator=generator,
        rewriter=rewriter,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Deterministic support decision
# ---------------------------------------------------------------------------


class TestDecideSupport:
    def test_supported_premise_with_citations_is_sufficient(self):
        d = decide_support(gen("Yes. [citation:nda-1]", cited=["nda-1"]), [ev("nda-1")])
        assert isinstance(d, SupportDecision)
        assert d.sufficient is True
        assert d.needs_another_round is False
        assert d.reason == REASON_SUPPORTED

    def test_supported_premise_without_citations_is_insufficient(self):
        d = decide_support(gen("Yes.", cited=[]), [ev("nda-1")])
        assert d.needs_another_round is True
        assert d.reason == "no_citations"

    def test_i_dont_know_triggers_another_round(self):
        d = decide_support(gen("I don't know.", cited=["nda-1"]), [ev("nda-1")])
        assert d.needs_another_round is True
        assert d.reason == "i_dont_know"

    def test_i_do_not_know_variant_triggers_another_round(self):
        d = decide_support(gen("I do not know."), [ev("nda-1")])
        assert d.needs_another_round is True
        assert d.reason == "i_dont_know"

    def test_unknown_answer_triggers_another_round(self):
        d = decide_support(gen("Unknown."), [ev("nda-1")])
        assert d.needs_another_round is True

    def test_unsupported_premise_not_high_confidence(self):
        verification = PremiseVerification(
            status=STATUS_UNSUPPORTED,
            premises=["the NDA has no term limit"],
            explanation="no chunk states a term.",
        )
        d = decide_support(gen("x [citation:nda-1]", verification, cited=["nda-1"]), [ev("nda-1")])
        assert d.needs_another_round is True
        assert d.reason == "unsupported_premise"

    def test_contradicted_premise_not_high_confidence(self):
        verification = PremiseVerification(
            status=STATUS_CONTRADICTED,
            premises=["the NDA has no term limit"],
            explanation="nda-1 states a term.",
        )
        d = decide_support(gen("x [citation:nda-1]", verification, cited=["nda-1"]), [ev("nda-1")])
        assert d.needs_another_round is True
        assert d.reason == "unsupported_premise"

    def test_empty_answer_is_insufficient(self):
        d = decide_support(gen(""), [ev("nda-1")])
        assert d.needs_another_round is True
        assert d.reason == "no_citations"


# ---------------------------------------------------------------------------
# Query rewrite interface
# ---------------------------------------------------------------------------


class FakeGroqResponse:
    def __init__(self, text):
        msg = type("Message", (), {"content": text})()
        choice = type("Choice", (), {"message": msg})()
        self.choices = [choice]

class _Messages:
    def __init__(self, sdk):
        self._sdk = sdk

    def create(self, **kwargs):
        return self._sdk._handle_create(**kwargs)

class FakeRewriteSDK:
    """Duck-types the Groq SDK for the rewrite client; never touches net."""

    def __init__(self, responder=None, default_text="targeted search query"):
        self.chat = type("Chat", (), {"completions": _Messages(self)})()
        self.responder = responder
        self.default_text = default_text
        self.requests = []

    def _handle_create(self, **kwargs):
        self.requests.append(kwargs)
        messages = kwargs.get("messages") or []
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if self.responder is not None:
            text = self.responder(system, user)
        else:
            text = self.default_text
        return FakeGroqResponse(text)


def make_rewriter(sdk):
    client = GroqClient(config=GenerationConfig(), sdk=sdk)
    return QueryRewriter(client)


class TestQueryRewriter:
    def test_rewrite_never_answers_the_legal_question(self):
        system = QueryRewriter.REWRITE_SYSTEM
        assert "Do NOT answer the legal question" in system
        assert "search quer" in system.lower()
        assert "legal conclusion" in system.lower() or "legal claims" in system.lower()

    def test_produces_focused_query_from_prompt(self):
        sdk = FakeRewriteSDK(default_text="confidential information obligations")
        rewriter = make_rewriter(sdk)
        out = rewriter.rewrite("Can I disclose?", "no_citations", [ev("nda-1")])
        assert out == "confidential information obligations"
        req = sdk.requests[-1]
        user_content = next((m["content"] for m in req["messages"] if m.get("role") == "user"), "")
        assert "ORIGINAL QUESTION:" in user_content
        assert "Can I disclose?" in user_content
        assert "no_citations" in user_content
        assert "nda-1" in user_content

    def test_failure_raises_empty_result(self):
        sdk = FakeRewriteSDK(default_text="   \n")
        rewriter = make_rewriter(sdk)
        with pytest.raises(RuntimeError, match="empty"):
            rewriter.rewrite("q", "i_dont_know", [ev("nda-1")])

    def test_no_api_key_required_when_sdk_injected(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        sdk = FakeRewriteSDK(default_text="salary clause")
        rewriter = make_rewriter(sdk)  # GroqClient built with no api_key
        assert rewriter.client.api_key is None
        assert rewriter.rewrite("salary?", "no_citations", []) == "salary clause"


# ---------------------------------------------------------------------------
# Adaptive controller: loop behavior
# ---------------------------------------------------------------------------


class TestAdaptiveController:
    def test_initial_retrieval_always_occurs(self):
        retriever = ScriptedRetriever([ev("nda-0"), ev("nda-1")])
        controller = make_controller(
            retriever,
            StubGenerator([("Yes. [citation:nda-1]", None, ["nda-1"])]),
        )
        result = controller.run("Can I disclose?")
        assert retriever.queries == ["Can I disclose?"]
        assert result.rounds_used == 1

    def test_sufficient_evidence_stops_after_first_round(self):
        retriever = ScriptedRetriever([ev("nda-1")])
        rewriter = StubRewriter(["rewritten"])
        controller = make_controller(
            retriever,
            StubGenerator([("Yes. [citation:nda-1]", None, ["nda-1"])]),
            rewriter,
        )
        result = controller.run("Can I disclose?")
        assert result.rounds_used == 1
        assert result.adaptive_triggered is False
        assert result.rewritten_queries == []
        assert rewriter.calls == []
        assert result.final_answer == "Yes. [citation:nda-1]"
        assert result.stop_reason == REASON_SUPPORTED

    def test_i_dont_know_triggers_another_retrieval(self):
        retriever = ScriptedRetriever(
            [ev("nda-0")],  # round 0 evidence
            [ev("nda-2", text="term clause")],  # round 1 evidence
        )
        rewriter = StubRewriter(["term obligations clause"])
        controller = make_controller(
            retriever,
            StubGenerator(
                [
                    ("I don't know.", None, []),
                    ("Term is two years. [citation:nda-2]", None, ["nda-2"]),
                ]
            ),
            rewriter,
        )
        result = controller.run("What is the term?")
        assert result.rounds_used == 2
        assert result.adaptive_triggered is True
        assert result.rewritten_queries == ["term obligations clause"]
        assert result.stop_reason == REASON_SUPPORTED

    def test_no_citations_triggers_another_retrieval(self):
        retriever = ScriptedRetriever([ev("nda-0")], [ev("nda-1")])
        rewriter = StubRewriter(["confidential obligations"])
        controller = make_controller(
            retriever,
            StubGenerator(
                [
                    ("Some answer without citations.", None, []),
                    ("Confidential info is protected. [citation:nda-1]", None, ["nda-1"]),
                ]
            ),
            rewriter,
        )
        result = controller.run("disclose?")
        assert result.rounds_used == 2
        assert result.adaptive_triggered is True

    def test_unsupported_premise_triggers_another_retrieval(self):
        unsupported = PremiseVerification(
            status=STATUS_UNSUPPORTED, premises=["no term limit"], explanation="silent"
        )
        retriever = ScriptedRetriever([ev("nda-0")], [ev("nda-1")])
        controller = make_controller(
            retriever,
            StubGenerator(
                [
                    ("Answer. [citation:nda-0]", unsupported, ["nda-0"]),
                    ("Evidence supports this. [citation:nda-1]", None, ["nda-1"]),
                ]
            ),
            StubRewriter(["term clause"]),
        )
        result = controller.run("Since there's no term limit?")
        assert result.rounds_used == 2
        # The round that surfaced the unsupported premise must NOT be the final,
        # confidence-accepting round.
        assert result.rounds[0].generation.verification.is_unsupported is True
        assert result.rounds[-1].generation.verification.is_unsupported is False

    def test_contradicted_premise_triggers_another_retrieval(self):
        contradicted = PremiseVerification(
            status=STATUS_CONTRADICTED,
            premises=["the NDA has no term"],
            explanation="nda-1 contradicts it",
        )
        retriever = ScriptedRetriever([ev("nda-0")], [ev("nda-1")])
        controller = make_controller(
            retriever,
            StubGenerator(
                [
                    ("Answer. [citation:nda-0]", contradicted, ["nda-0"]),
                    ("Corrected answer. [citation:nda-1]", None, ["nda-1"]),
                ]
            ),
            StubRewriter(["term clause"]),
        )
        result = controller.run("Since the NDA is perpetual?")
        assert result.rounds_used == 2
        assert result.rounds[0].generation.verification.status == STATUS_CONTRADICTED

    def test_supported_evidence_stops(self):
        retriever = ScriptedRetriever([ev("emp-0", "emp", text="salary 120k")])
        rewriter = StubRewriter(["salary amount"])
        controller = make_controller(
            retriever,
            StubGenerator([("Salary is 120k. [citation:emp-0]", None, ["emp-0"])]),
            rewriter,
        )
        result = controller.run("What is the salary?")
        assert result.rounds_used == 1
        assert result.adaptive_triggered is False
        assert rewriter.calls == []  # rewrite is only called when a round is needed

    def test_max_two_extra_rounds_enforced(self):
        retriever = ScriptedRetriever(  # fewer scripted lists than calls: last repeats
            [ev("nda-0")], [ev("nda-1")], [ev("nda-2")]
        )
        rewriter = StubRewriter(["rw1", "rw2"])
        always_insufficient = [
            ("I don't know.", None, []),
            ("I don't know.", None, []),
            ("I don't know.", None, []),
        ]
        controller = make_controller(
            retriever, StubGenerator(always_insufficient), rewriter,
            max_extra_rounds=2,
        )
        result = controller.run("q")
        assert result.rounds_used == 3  # initial + 2 extra
        assert result.adaptive_triggered is True
        assert result.rewritten_queries == ["rw1", "rw2"]
        assert result.stop_reason == REASON_MAX_ROUNDS
        assert rewriter.calls == [
            ("q", "i_dont_know", [ev("nda-0")]),
            ("q", "i_dont_know", [ev("nda-1")]),
        ]

    def test_zero_extra_rounds_is_bounded(self):
        retriever = ScriptedRetriever([ev("nda-0")])
        controller = make_controller(
            retriever,
            StubGenerator([("I don't know.", None, [])]),
            rewriter=StubRewriter(["should-not-be-used"]),
            max_extra_rounds=0,
        )
        result = controller.run("q")
        assert result.rounds_used == 1
        assert result.stop_reason == REASON_MAX_ROUNDS

    def test_stops_before_maximum_when_sufficient(self):
        retriever = ScriptedRetriever(
            [ev("nda-0")], [ev("nda-1")], [ev("nda-2")]
        )
        rewriter = StubRewriter(["rw1", "rw2"])
        controller = make_controller(
            retriever,
            StubGenerator(
                [
                    ("I don't know.", None, []),
                    ("Good. [citation:nda-1]", None, ["nda-1"]),
                ]
            ),
            rewriter,
            max_extra_rounds=2,
        )
        result = controller.run("q")
        assert result.rounds_used == 2  # stopped before the 2nd extra round
        assert result.stop_reason == REASON_SUPPORTED
        assert result.rewritten_queries == ["rw1"]
        assert len(rewriter.calls) == 1

    def test_rewrite_failure_returns_best_available_result(self):
        evidence = [ev("nda-0")]
        retriever = ScriptedRetriever(evidence)
        controller = make_controller(
            retriever,
            StubGenerator([("I don't know.", None, [])]),
            StubRewriter([]),  # rewrite() raises -> graceful stop
        )
        result = controller.run("q")
        assert result.rounds_used == 1
        assert result.stop_reason == REASON_REWRITE_FAILED
        assert result.final_answer == "I don't know."
        assert result.final_retrieval_results == evidence
        assert result.adaptive_triggered is False

    def test_no_rewriter_stops_gracefully(self):
        retriever = ScriptedRetriever([ev("nda-0")])
        controller = make_controller(
            retriever,
            StubGenerator([("I don't know.", None, [])]),
            rewriter=None,
        )
        result = controller.run("q")
        assert result.rounds_used == 1
        assert result.stop_reason == REASON_NO_REWRITER

    def test_no_infinite_loop(self):
        retriever = ScriptedRetriever([ev("nda-0")], [ev("nda-1")], [ev("nda-2")])
        controller = make_controller(
            retriever,
            StubGenerator(
                [
                    ("I don't know.", None, []),
                    ("I don't know.", None, []),
                    ("I don't know.", None, []),
                    ("I don't know.", None, []),  # would overflow if unbounded
                ]
            ),
            StubRewriter(["rw1", "rw2"]),
            max_extra_rounds=2,
        )
        result = controller.run("q")
        assert result.rounds_used == 3
        assert result.stop_reason == REASON_MAX_ROUNDS

    def test_rewritten_query_passed_to_retrieval_and_generation(self):
        retriever = ScriptedRetriever(
            [ev("nda-0")], [ev("nda-2", text="term clause")]
        )
        rewriter = StubRewriter(["term obligations clause"])
        generator = StubGenerator(
            [
                ("I don't know.", None, []),
                ("Two years. [citation:nda-2]", None, ["nda-2"]),
            ]
        )
        controller = make_controller(retriever, generator, rewriter)
        result = controller.run("What is the term?")
        assert retriever.queries == ["What is the term?", "term obligations clause"]
        # generation round 2 got the rewritten query and round-2 evidence
        assert generator.calls[1][0] == "term obligations clause"
        assert generator.calls[1][1][0].chunk_id == "nda-2"
        assert result.final_answer == "Two years. [citation:nda-2]"

    def test_per_round_results_preserved(self):
        retriever = ScriptedRetriever(
            [ev("nda-0")], [ev("nda-1")]
        )
        generator = StubGenerator(
            [
                ("I don't know. [citation:nda-0]", None, ["nda-0"]),
                ("Final. [citation:nda-1]", None, ["nda-1"]),
            ]
        )
        rewriter = StubRewriter(["rw-query"])
        controller = make_controller(retriever, generator, rewriter)
        result = controller.run("orig")
        assert len(result.rounds) == 2
        first, second = result.rounds
        assert isinstance(first, AdaptiveRound)
        assert first.is_initial is True
        assert first.round_index == 0
        assert first.query == "orig"
        assert first.rewritten_from is None
        assert first.decision.sufficient is False
        assert [r.chunk_id for r in first.retrieval_results] == ["nda-0"]
        assert first.generation.answer == "I don't know. [citation:nda-0]"
        assert second.is_initial is False
        assert second.round_index == 1
        assert second.query == "rw-query"
        assert second.rewritten_from == "rw-query"
        assert [r.chunk_id for r in second.retrieval_results] == ["nda-1"]
        # generator call genealogy: round1 used round0 evidence
        assert generator.calls[0][1][0].chunk_id == "nda-0"
        assert generator.calls[1][1][0].chunk_id == "nda-1"

    def test_final_data_preserved_on_result(self):
        retriever = ScriptedRetriever([ev("nda-1")])
        controller = make_controller(
            retriever,
            StubGenerator([("Yes. [citation:nda-1]", None, ["nda-1"])]),
        )
        result = controller.run("Can I disclose?")
        assert isinstance(result, AdaptiveResult)
        assert result.original_query == "Can I disclose?"
        assert result.final_answer == "Yes. [citation:nda-1]"
        assert result.final_verification.status == STATUS_SUPPORTED
        assert result.final_retrieval_results[0].chunk_id == "nda-1"
        assert result.rounds_used == 1
        assert result.adaptive_triggered is False
        assert result.stop_reason == REASON_SUPPORTED
        assert result.support_threshold == 1.0
        assert result.max_extra_rounds == 2

    def test_invalid_max_extra_rounds_raises(self):
        with pytest.raises(ValueError):
            make_controller(ScriptedRetriever([ev("nda-0")]),
                            StubGenerator([("fine", None, ["nda-0"])]),
                            max_extra_rounds=-1)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfig:
    def test_real_config_yaml_has_adaptive_block(self):
        import yaml

        with CONFIG_PATH.open() as fh:
            cfg = yaml.safe_load(fh)
        assert "adaptive" in cfg
        block = cfg["adaptive"]
        assert block["enabled"] in (True, False)
        assert block["max_extra_rounds"] >= 0
        assert block["support_threshold"] >= 0

    def test_from_config_dict(self):
        controller = AdaptiveRetriever.from_config(
            retriever=ScriptedRetriever([ev("nda-0")]),
            generator=StubGenerator([("fine", None, ["nda-0"])]),
            rewriter=StubRewriter([]),
            adaptive_config={
                "enabled": True,
                "support_threshold": 0.5,
                "max_extra_rounds": 3,
            },
        )
        assert controller.max_extra_rounds == 3
        assert controller.support_threshold == 0.5

    def test_from_config_defaults_when_block_missing(self):
        controller = AdaptiveRetriever.from_config(
            retriever=ScriptedRetriever([ev("nda-0")]),
            generator=StubGenerator([("fine", None, ["nda-0"])]),
        )
        assert controller.max_extra_rounds == 2
        assert controller.support_threshold == 1.0


# ---------------------------------------------------------------------------
# Integration with the real Milestone 3 retriever + Milestone 5 client
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_real_retriever_sufficient_first_round(self, fake_embedder):
        from tests.conftest import sample_chunks

        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        retriever = HybridRetriever(dense, bm25, fake_embedder, top_k=3)
        generator = StubGenerator(
            [
                (
                    "Confidential information may not be disclosed. [citation:nda-1]",
                    None,
                    ["nda-1"],
                )
            ]
        )
        rewriter = StubRewriter(["confidential info obligations"])
        controller = make_controller(retriever, generator, rewriter)
        result = controller.run("confidential information disclose salary")
        assert result.rounds_used == 1
        assert result.adaptive_triggered is False
        assert rewriter.calls == []
        assert result.stop_reason == REASON_SUPPORTED
        assert result.final_retrieval_results  # came from the real indexes

    def test_real_retriever_rewrites_and_retrieves_again(self, fake_embedder):
        from tests.conftest import sample_chunks

        chunks = sample_chunks()
        dense = DenseIndex.build(chunks, fake_embedder, use_sac=True)
        bm25 = BM25Index.build(chunks, fake_embedder, use_sac=True)
        retriever = HybridRetriever(dense, bm25, fake_embedder, top_k=3)
        generator = StubGenerator(
            [
                ("I don't know.", None, []),
                ("The NDA governs disclosures. [citation:nda-1]", None, ["nda-1"]),
            ]
        )
        rewriter = make_rewriter(FakeRewriteSDK(default_text="confidential obligations"))
        controller = make_controller(retriever, generator, rewriter)
        result = controller.run("disclose confidential information")
        assert result.rounds_used == 2
        assert result.adaptive_triggered is True
        assert len(result.rewritten_queries) == 1
        assert result.stop_reason == REASON_SUPPORTED
        assert rewriter.client.api_key is None  # no API key anywhere