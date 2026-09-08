"""Service layer (Milestone 8, AGENTS.md §7/§8).

The application layer wires together the Milestone 1-7 components:

    query -> adaptive retrieval -> generation -> evaluation metadata

All logic lives here, NOT in the route handlers. The ``Application`` class is a
dependency container: every collaborator (retriever, generator, controller,
eval-loader) is an injectable attribute, so tests can swap in fakes and the
package stays importable without an Anthropic key, network access, a BGE
download, or a running server.

Real (default) construction is lazy: building a real ``Application`` does not
touch the Anthropic SDK, the embedding model, or the index files. Only the
methods that actually need a component construct it, and that construction is
explicitly guarded so callers/tests can avoid it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# Light, import-safe imports only. Heavy components (GroqClient, Embedder,
# DenseIndex/BM25Index) are imported lazily so importing this module never
# requires an API key or a model download.
from generation.prompt import PremiseVerification

from .config import AppSettings
from .serialization import (
    adaptive_result_to_dict,
    claim_level_to_dict,
    failure_tags_to_dict,
    retrieval_result_to_dict,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base error raised by the application service layer."""


class InvalidQueryError(AppError):
    """Raised when the query is empty or otherwise invalid."""


class EvalResultsNotFoundError(AppError):
    """Raised when the Milestone 7 evaluation result file is unavailable."""


class ConfigNotFoundError(AppError):
    """Raised when the project config.yaml is required but missing."""


class RetrievalError(AppError):
    """Raised when retrieval fails for a query."""


class GenerationError(AppError):
    """Raised when answer generation fails."""


# ---------------------------------------------------------------------------
# Service container
# ---------------------------------------------------------------------------


@dataclass
class Application:
    """Dependency container wiring the RAG pipeline for the API.

    All collaborators default to ``None`` and are constructed lazily on first
    use via the ``_build_*`` methods. Tests inject fakes by assigning the
    attributes directly.
    """

    settings: AppSettings = field(default_factory=AppSettings)

    # Injectable collaborators (tests assign fakes here).
    config: Mapping | None = None
    retriever: Any = None
    generator: Any = None
    adaptive: Any = None
    query_rewriter: Any = None
    risk_taggers_claims: Any = None

    # Lazily-built real component cache.
    _real_retriever: Any = field(default=None, repr=False)
    _real_generator: Any = field(default=None, repr=False)
    _real_adaptive: Any = field(default=None, repr=False)
    _real_config: Mapping | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Config loading (Milestone 1-7 config.yaml)
    # ------------------------------------------------------------------

    def load_project_config(self) -> Mapping:
        if self.config is not None:
            return self.config
        if self._real_config is not None:
            return self._real_config
        path = Path(self.settings.config_path)
        if not path.exists():
            raise ConfigNotFoundError(
                f"project config not found at {self.settings.config_path}. "
                "Set 'config_path' or inject a config mapping."
            )
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ConfigNotFoundError("PyYAML is required to load config.yaml") from exc
        with path.open("r", encoding="utf-8") as fh:
            self._real_config = yaml.safe_load(fh) or {}
        return self._real_config

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_real_retriever(self):
        """Construct the HybridRetriever from the project corpus.

        This is the only method that touches the embedding model / index files.
        For the default offline demo config the eval corpus is used with the
        configured embedding model; callers can force an offline deterministic
        encoder by supplying a retriever injectable instead.
        """
        config = self.load_project_config()
        try:
            from eval.drm_eval import build_retriever, EVAL_RAW_DIR, EVAL_PROCESSED_DIR
            from eval.drm_eval import EVAL_INDEX_ROOT, CHUNK_FILE_SAC_ON
            from retrieval.embed import Embedder
        except Exception as exc:
            raise RetrievalError(
                f"could not import retrieval dependencies: {exc}"
            ) from exc

        model = (
            (config.get("retrieval") or {}).get(
                "embedding_model", "BAAI/bge-large-en-v1.5"
            )
        )
        repo_root = Path(__file__).resolve().parent.parent
        try:
            embedder = Embedder(model)
            retriever = build_retriever(
                config,
                embedder,
                use_sac=True,
                raw_dir=repo_root / EVAL_RAW_DIR,
                processed_dir=repo_root / EVAL_PROCESSED_DIR,
                index_root=repo_root / EVAL_INDEX_ROOT,
                chunk_file=CHUNK_FILE_SAC_ON,
            )
        except Exception as exc:
            raise RetrievalError(f"failed to build retriever: {exc}") from exc
        return retriever

    @property
    def effective_retriever(self) -> Any:
        if self.retriever is not None:
            return self.retriever
        if self._real_retriever is None:
            self._real_retriever = self._build_real_retriever()
        return self._real_retriever

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_real_generator(self):
        """Construct the GroqGenerator (real Groq client).

        Guards against missing API key so callers get a clear GenerationError
        rather than leaking SDK details.
        """
        config = self.load_project_config()
        try:
            from generation.generate import (
                GroqClient,
                GroqGenerator,
                GenerationConfig,
            )
        except Exception as exc:
            raise GenerationError(
                f"could not import generation dependencies: {exc}"
            ) from exc

        gen_config = GenerationConfig.from_config_dict(config)
        client = GroqClient(config=gen_config)
        try:
            # Verify the key is present before building, to fail early/clearly.
            client.sdk
        except RuntimeError as exc:
            raise GenerationError(str(exc)) from exc
        generator = GroqGenerator(client=client)
        return generator

    @property
    def effective_generator(self) -> Any:
        if self.generator is not None:
            return self.generator
        if self._real_generator is None:
            self._real_generator = self._build_real_generator()
        return self._real_generator

    # ------------------------------------------------------------------
    # Adaptive retrieval controller
    # ------------------------------------------------------------------

    def _build_real_adaptive(self):
        config = self.load_project_config()
        try:
            from generation.generate import GroqClient, GenerationConfig
            from retrieval.adaptive import AdaptiveRetriever, QueryRewriter
        except Exception as exc:
            raise RetrievalError(f"could not build adaptive controller: {exc}") from exc

        generator = self.effective_generator
        retriever = self.effective_retriever

        adaptive_cfg = (config or {}).get("adaptive") or {}
        enabled = bool(adaptive_cfg.get("enabled", True))

        rewriter = None
        if enabled:
            try:
                gen_config = GenerationConfig.from_config_dict(config)
                client = GroqClient(config=gen_config)
                rewriter = QueryRewriter(client=client)
            except Exception:
                rewriter = None

        max_extra = int(adaptive_cfg.get("max_extra_rounds", 2))
        threshold = float(adaptive_cfg.get("support_threshold", 1.0))

        controller = AdaptiveRetriever(
            retriever=retriever,
            generator=generator,
            rewriter=rewriter,
            max_extra_rounds=max_extra,
            support_threshold=threshold,
        )
        return controller

    @property
    def uses_adaptive(self) -> bool:
        if self.adaptive is not None:
            return True
        config = self.load_project_config()
        adaptive_cfg = (config or {}).get("adaptive") or {}
        return bool(adaptive_cfg.get("enabled", True))

    @property
    def effective_adaptive(self) -> Any:
        if self.adaptive is not None:
            return self.adaptive
        if self._real_adaptive is None:
            self._real_adaptive = self._build_real_adaptive()
        return self._real_adaptive

    # ------------------------------------------------------------------
    # Query -> adaptive retrieval -> generation -> evaluation metadata
    # ------------------------------------------------------------------

    def answer_query(self, query: str) -> dict[str, Any]:
        """Run the full RAG pipeline for a user query.

        Returns a serializable dict with the adaptive-retrieval result plus,
        where available, claim-level faithfulness and failure-tag metadata.

        Raises:
            InvalidQueryError: when the query is blank.
            RetrievalError: when retrieval fails.
            GenerationError: when generation fails.
        """
        if not query or not query.strip():
            raise InvalidQueryError("query must be a non-empty string")

        if self.uses_adaptive:
            try:
                result = self.effective_adaptive.run(query)
            except (RetrievalError, GenerationError):
                raise
            except Exception as exc:
                raise RetrievalError(f"adaptive retrieve/generate failed: {exc}") from exc
            payload = adaptive_result_to_dict(result)
        else:
            # Non-adaptive path: single retrieve + generate.
            retriever = self.effective_retriever
            generator = self.effective_generator
            try:
                evidence = retriever.retrieve(query)
            except Exception as exc:
                raise RetrievalError(f"retrieval failed: {exc}") from exc
            try:
                generation = generator.generate(query, evidence)
            except Exception as exc:
                raise GenerationError(f"generation failed: {exc}") from exc

            payload = {
                "original_query": query,
                "final_answer": generation.answer,
                "premise_verification": {
                    "status": (
                        generation.verification.status
                        if generation.verification
                        else None
                    ),
                    "premises": list(
                        (generation.verification.premises if generation.verification else None)
                        or []
                    ),
                    "explanation": (
                        generation.verification.explanation
                        if generation.verification
                        else ""
                    ),
                    "is_unsupported": bool(
                        generation.verification.is_unsupported
                        if generation.verification
                        else False
                    ),
                },
                "adaptive_triggered": False,
                "rounds_used": 1,
                "rewritten_queries": [],
                "stop_reason": "supported",
                "support_threshold": 1.0,
                "max_extra_rounds": 0,
                "final_retrieval_results": [
                    retrieval_result_to_dict(r) for r in evidence
                ],
                "cited_chunk_ids": list(generation.cited_chunk_ids or []),
                "failure_tags": [],
                "claim_level": None,
                "rounds": [],
            }

        # Attach claim-level faithfulness + failure tags when configured.
        self._attach_eval_metadata(payload)
        return payload

    def _attach_eval_metadata(self, payload: dict[str, Any]) -> None:
        """Compute failure tags and claim-level faithfulness for a query result.

        Uses the existing Milestone 7 modules (deterministic, no LLM key
        needed). Returns early if they are unavailable so the response stays
        valid rather than hard-failing on optional metadata.
        """
        try:
            from eval import claim_check, failure_point_tagger
        except Exception:
            return

        config = self.load_project_config()
        eval_cfg = (config or {}).get("evaluation") or {}
        enable_claims = bool(eval_cfg.get("enable_claim_check", True))
        enable_tags = bool(eval_cfg.get("enable_failure_tags", True))

        if not (enable_claims or enable_tags):
            return

        final_results = payload.get("final_retrieval_results") or []

        try:
            record = failure_point_tagger.normalize_result(
                query_id=payload.get("original_query", ""),
                query=payload.get("original_query", ""),
                expected_document=None,
                retrieved=final_results,
                answer=payload.get("final_answer", ""),
                cited_chunk_ids=payload.get("cited_chunk_ids") or [],
                verification=payload.get("premise_verification") or {},
                rounds_used=payload.get("rounds_used", 1),
                stop_reason=payload.get("stop_reason"),
                adaptive_triggered=payload.get("adaptive_triggered", False),
            )
            claims_report = None
            if enable_claims:
                chunks = record.get("retrieved_chunks") or []
                claims_report = claim_check.claim_level_faithfulness(
                    record["answer"], chunks
                )
                payload["claim_level"] = claim_level_to_dict(claims_report)

            if enable_tags:
                tags = failure_point_tagger.tag_result(
                    record, precomputed_claims=claims_report
                )
                payload["failure_tags"] = failure_tags_to_dict(tags)
        except Exception:
            # Evaluation metadata is best-effort; never fail the query for it.
            payload.setdefault("failure_tags", [])
            payload.setdefault("claim_level", None)

    # ------------------------------------------------------------------
    # Evaluation results
    # ------------------------------------------------------------------

    def load_eval_results(self) -> dict[str, Any]:
        """Load the Milestone 7 evaluation summary from disk.

        Reads the persisted ``results.json`` (written by ``eval.run_eval``).
        Does NOT re-run the harness.
        """
        results = self.risk_taggers_claims
        if isinstance(results, dict):
            return dict(results)

        path = Path(self.settings.eval_results_path)
        if not path.exists():
            raise EvalResultsNotFoundError(
                f"evaluation results not found at {self.settings.eval_results_path}. "
                "Run the Milestone 7 harness (python -m eval.run_eval) first."
            )
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise EvalResultsNotFoundError(
                f"could not read evaluation results at {self.settings.eval_results_path}: {exc}"
            ) from exc
        return data

    def eval_summary(self) -> dict[str, Any]:
        """Return the flattened /eval view of the Milestone 7 result file.

        Exposes the aggregate metrics (precision/recall@k, MRR, MAP, DRM,
        faithfulness, answer/context relevance, claim-level faithfulness, and
        the failure-tag summary) plus the envelope metadata.
        """
        data = self.load_eval_results()
        aggregate = data.get("aggregate") or {}
        envelope = data.get("envelope") or {}

        precision = aggregate.get("precision_at_k") or {}
        recall = aggregate.get("recall_at_k") or {}

        # Normalize @k keys to ints for friendly JSON output.
        def norm_k(mapping: Mapping) -> dict[str, float]:
            return {str(k): float(v) for k, v in dict(mapping).items()}

        return {
            "aggregate": {
                "precision_at_1": norm_k(precision).get("1"),
                "precision_at_3": norm_k(precision).get("3"),
                "precision_at_5": norm_k(precision).get("5"),
                "precision_at_10": norm_k(precision).get("10"),
                "recall_at_1": norm_k(recall).get("1"),
                "recall_at_3": norm_k(recall).get("3"),
                "recall_at_5": norm_k(recall).get("5"),
                "recall_at_10": norm_k(recall).get("10"),
                "mrr": aggregate.get("mrr"),
                "map": aggregate.get("map"),
                "drm_fraction": aggregate.get("drm_fraction"),
                "top1_accuracy": aggregate.get("top1_accuracy"),
                "top3_accuracy": aggregate.get("top3_accuracy"),
                "faithfulness": aggregate.get("faithfulness"),
                "answer_relevance": aggregate.get("answer_relevance"),
                "context_relevance": aggregate.get("context_relevance"),
                "claim_level_faithfulness": aggregate.get(
                    "claim_level_faithfulness"
                ),
                "unsupported_claim_fraction": aggregate.get(
                    "unsupported_claim_fraction"
                ),
                "failure_tags": aggregate.get("failure_tags"),
                "num_queries": aggregate.get("num_queries"),
                "k_values": aggregate.get("k_values"),
                "drm_depth": aggregate.get("drm_depth"),
            },
            "envelope": {
                "version": envelope.get("version"),
                "generated_at": envelope.get("generated_at"),
                "num_queries": envelope.get("num_queries"),
                "env": envelope.get("env"),
                "scoring": envelope.get("scoring"),
            },
            "per_query": data.get("per_query") or [],
        }
