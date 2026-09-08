"""LegalRAG evaluation package.

Milestone 4 (DRM) + Milestone 7 (full evaluation harness):

* ``drm_eval``         — SAC off/on Document Retrieval Mismatch comparison.
* ``metrics``          — precision@k / recall@k / MRR / MAP (from scratch).
* ``text_util``        — shared tokenization / sentence-splitting helpers.
* ``ragas_style``      — RAGAs-STYLE faithfulness/answer/context relevance.
* ``claim_check``      — claim-level faithfulness (supported/unsupported/
                         contradicted per claim).
* ``failure_point_tagger`` — the survey's 7 failure points + the project's
                         8th (unsupported premise), as structured tags.
* ``ablation``         — embedding-model ablation (precision/recall/MRR/DRM).
* ``run_eval``         — unified harness: metrics + DRM + RAGAs + claims +
                         failure tags, JSON + Markdown reports.

All modules run offline by default (deterministic heuristics and injectable
fake encoders/judges); no Groq key or model download is ever required by
pytest or the ``--smoke`` CLI entry points.

Submodules are exported LAZILY (PEP 562 ``__getattr__``) so that
``python -m eval.drm_eval`` / ``python -m eval.ablation`` /
``python -m eval.run_eval`` do not hit the "found in sys.modules" re-import
warning: importing ``eval`` never eagerly imports a submodule.
"""

from . import drm_eval  # noqa: F401  (drives lazy attribute resolution)

_ATTRIBUTES = {
    # name -> (submodule, attribute)
    "compute_drm_metrics": ("drm_eval", "compute_drm_metrics"),
    "load_queries": ("drm_eval", "load_queries"),
    "render_markdown_report": ("drm_eval", "render_markdown_report"),
    "retrieve_all_queries": ("drm_eval", "retrieve_all_queries"),
    "run_drm_comparison": ("drm_eval", "run_drm_comparison"),
    "save_results": ("drm_eval", "save_results"),
    "configured_ablation_models": ("ablation", "configured_ablation_models"),
    "render_ablation_report": ("ablation", "render_markdown_report"),
    "run_ablation": ("ablation", "run_ablation"),
    "claim_level_faithfulness": ("claim_check", "claim_level_faithfulness"),
    "ALL_FAILURE_POINTS": ("failure_point_tagger", "ALL_FAILURE_POINTS"),
    "FailureTag": ("failure_point_tagger", "FailureTag"),
    "tag_result": ("failure_point_tagger", "tag_result"),
    "tag_results": ("failure_point_tagger", "tag_results"),
    "average_precision": ("metrics", "average_precision"),
    "evaluate_queries": ("metrics", "evaluate_queries"),
    "evaluate_retrieval": ("metrics", "evaluate_retrieval"),
    "mean_average_precision": ("metrics", "mean_average_precision"),
    "mrr": ("metrics", "mrr"),
    "precision_at_k": ("metrics", "precision_at_k"),
    "recall_at_k": ("metrics", "recall_at_k"),
    "reciprocal_rank": ("metrics", "reciprocal_rank"),
    "answer_relevance": ("ragas_style", "answer_relevance"),
    "context_relevance": ("ragas_style", "context_relevance"),
    "evaluate_rag_results": ("ragas_style", "evaluate_rag_results"),
    "faithfulness": ("ragas_style", "faithfulness"),
    "FULL_EVAL_VERSION": ("run_eval", "FULL_EVAL_VERSION"),
    "run_full_evaluation": ("run_eval", "run_full_evaluation"),
    "save_report": ("run_eval", "save_report"),
}


def __getattr__(name: str):
    import importlib

    if name in _ATTRIBUTES:
        module_name, attr = _ATTRIBUTES[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    # Also allow ``from eval import metrics`` to resolve the submodule.
    submodule = importlib.util.find_spec(f".{name}", __name__)
    if submodule is not None:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALL_FAILURE_POINTS",
    "FULL_EVAL_VERSION",
    "FailureTag",
    "answer_relevance",
    "average_precision",
    "claim_level_faithfulness",
    "compute_drm_metrics",
    "configured_ablation_models",
    "context_relevance",
    "evaluate_queries",
    "evaluate_rag_results",
    "evaluate_retrieval",
    "faithfulness",
    "load_queries",
    "mean_average_precision",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "render_ablation_report",
    "render_markdown_report",
    "retrieve_all_queries",
    "run_ablation",
    "run_drm_comparison",
    "run_full_evaluation",
    "save_report",
    "save_results",
    "tag_result",
    "tag_results",
]