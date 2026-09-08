"""Unified Milestone 7 evaluation harness (AGENTS.md §6).

The single entry point that evaluates a collection of RAG results end to end
and produces, per query AND as aggregates:

* precision@k / recall@k for ``evaluation.k_values``      (eval/metrics.py)
* MRR and MAP                                              (eval/metrics.py)
* DRM (Docl-level Retrieval Mismatch)                      (eval/drm_eval.py)
* faithfulness / answer relevance / context relevance      (eval/ragas_style.py)
* claim-level faithfulness                                 (eval/claim_check.py)
* failure-point tags (survey's 7 + "unsupported premise")  (eval/failure_point_tagger.py)

The harness does NOT reimplement retrieval, generation, or any metric. It
knowledge only retrieves through an injected ``rag_fn`` and composes the
existing Milestone 1-4 pipeline modules and eval modules.

Offline / deterministic guarantees
----------------------------------
* ``run_full_evaluation`` is a pure function of its inputs + ``timestamp``;
  identical inputs + a fixed timestamp yield identical serialized output.
* No module requires a Groq key or network access.
* ``python -m eval.run_eval --smoke`` runs entirely offline: a deterministic
  fake encoder stands in for the embedding model, the Milestone 4 NDA corpus
  in ``data/eval`` is used, and answers are synthesized deterministically from
  the retrieved chunks (with a few scripted failures to exercise the tagger).
  Nothing is downloaded and no external API is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from generation.prompt import PremiseVerification

from retrieval.embed import Embedder
from retrieval.index import slugify_model_name

from . import claim_check, failure_point_tagger, ragas_style
from .drm_eval import (
    CHUNK_FILE_SAC_ON,
    EVAL_INDEX_ROOT,
    EVAL_PROCESSED_DIR,
    EVAL_QUERIES_PATH,
    EVAL_RAW_DIR,
    build_retriever,
    compute_drm_metrics,
    load_config,
    load_queries,
)
from .metrics import DEFAULT_K_VALUES, evaluate_queries, evaluate_retrieval
from retrieval.index import read_chunks

#: Milestone 7 harness version, recorded in every report for reproducibility.
FULL_EVAL_VERSION = 1

#: Where the full harness writes its reports. Distinct from the Milestone 4
#: DRM files (eval/results/drm_results.json, drm_report.md), which are never
#: touched by this module.
DEFAULT_FULL_EVAL_DIR = Path("eval/results/full_eval")
DEFAULT_ABLATION_DIR = Path("eval/results/ablation")

#: Human-readable summary of which (deterministic vs LLM-judge) produced the
#: generation scores; the smoke/demo run records "deterministic heuristics".
GENERATION_SCORING_NOTES = {
    "deterministic": (
        "faithfulness/answer-relevance/context-relevance used the "
        "deterministic lexical heuristics (no LLM judge injected)"
    ),
    "llm_judge": (
        "faithfulness/answer-relevance/context-relevance used an injected "
        "LLM judge; scores reflect that judge's output"
    ),
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def resolve_eval_config(config: Mapping) -> dict:
    """Return the ``evaluation`` block (empty dict when absent)."""
    return dict((config or {}).get("evaluation") or {})


def resolve_k_values(eval_config: Mapping, k_values: Sequence[int] | None = None) -> list[int]:
    """Resolve the k values from an explicit arg, the evaluation block, or the
    module default; validates them to positive ints in ascending order."""
    if k_values is None:
        k_values = eval_config.get("k_values") or DEFAULT_K_VALUES
    k_list = [int(k) for k in k_values]
    if not k_list:
        raise ValueError("k_values must be a non-empty sequence")
    if any(k < 1 for k in k_list):
        raise ValueError(f"all k_values must be >= 1, got {k_list}")
    if k_list != sorted(set(k_list)):
        raise ValueError(f"k_values must be distinct and ascending, got {k_list}")
    return k_list


def resolve_top_k(k_values: Sequence[int], top_k: int | None = None) -> int:
    """DRM depth: the explicit ``top_k`` or the largest evaluated depth."""
    depth = int(top_k if top_k is not None else max(k_values))
    if depth < 1:
        raise ValueError("top_k must be >= 1")
    return depth


# ---------------------------------------------------------------------------
# RAG-result normalization
# ---------------------------------------------------------------------------


def normalize_rag_result(query_record: Mapping, rag_result: Mapping) -> dict:
    """Wrap one query + its RAG-system output into the normalized record the
    metrics/failure-tagger consume (see validation_point_tagger.normalize_result)."""
    return failure_point_tagger.normalize_result(
        query_id=str(query_record.get("query_id") or ""),
        query=str(query_record.get("query") or ""),
        expected_document=query_record.get("expected_document"),
        retrieved=list((rag_result or {}).get("retrieved") or []),
        answer=str((rag_result or {}).get("answer") or ""),
        cited_chunk_ids=list((rag_result or {}).get("cited_chunk_ids") or []),
        verification=(rag_result or {}).get("verification"),
        rounds_used=(rag_result or {}).get("rounds_used"),
        stop_reason=(rag_result or {}).get("stop_reason"),
        adaptive_triggered=(rag_result or {}).get("adaptive_triggered"),
    )


def _claims_report_dict(report: claim_check.ClaimLevelReport) -> dict:
    """Serialize a ClaimLevelReport into a plain dict."""
    return {
        "score": report.score,
        "num_claims": report.num_claims,
        "counts": dict(report.counts),
        "details": [asdict(check) for check in report.checks],
    }


def _enrich_retrieved_text(records: Sequence[dict], corpus_chunks: Sequence[dict] | None) -> list[dict]:
    """Backfill body text onto retrieved chunk records from a known corpus.

    The persisted indexes (Milestone 2) map vector positions back to chunk
    *records* (chunk_id/source/hierarchy only — no body text), so a retriever
    built from an index returns chunks without ``text``. The evaluation corpus
    (chunks.jsonl) has the full bodies; when provided here, any retrieved
    chunk entry whose text is empty is enriched by chunk_id. This keeps the
    generation-facing scores (faithfulness / claims / ragas context) meaningful
    for index-backed retrievers without touching retrieval itself.
    """
    if not corpus_chunks:
        return list(records)
    by_id = {
        str(c.get("chunk_id")): c
        for c in corpus_chunks
        if c.get("chunk_id")
    }
    for record in records:
        for entry in record.get("retrieved_chunks") or []:
            if entry.get("text"):
                continue
            source = by_id.get(str(entry.get("chunk_id") or ""))
            if not source:
                continue
            entry["text"] = str(source.get("text") or source.get("original_text") or "")
            entry["original_text"] = str(source.get("original_text") or source.get("text") or "")
    return list(records)


def _mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------


def evaluate_query_record(
    record: dict,
    k_values: Sequence[int],
    drm_depth: int,
    judge: ragas_style.LLMJudge | None = None,
    enable_ragas_style: bool = True,
    enable_claim_check: bool = True,
    enable_failure_tags: bool = True,
) -> dict:
    """Evaluate one normalized record into a serializable per-query result."""
    query = record.get("query") or ""

    per_query: dict = {
        "query_id": record.get("query_id"),
        "query": query,
        "expected_document": record.get("expected_document"),
        "answer": record.get("answer"),
        "cited_chunk_ids": list(record.get("cited_chunk_ids") or []),
        "verification_status": (record.get("verification") or {}).get("status"),
        "rounds_used": record.get("rounds_used"),
        "stop_reason": record.get("stop_reason"),
        "adaptive_triggered": record.get("adaptive_triggered"),
        "retrieved_sources": list(record.get("retrieved_sources") or []),
        "retrieval": evaluate_retrieval(record, k_values=k_values),
        "drm_fraction": compute_drm_metrics([record], k=drm_depth)["drm_fraction"],
    }

    chunks = record.get("retrieved_chunks") or []
    if enable_ragas_style:
        per_query["ragas"] = ragas_style.evaluate_rag_results(
            record.get("answer") or "", query, chunks, judge=judge
        )

    claims_report = None
    if enable_claim_check:
        claims_report = claim_check.claim_level_faithfulness(record.get("answer") or "", chunks)
        per_query["claims"] = _claims_report_dict(claims_report)

    if enable_failure_tags:
        tags = failure_point_tagger.tag_result(record, precomputed_claims=claims_report)
        per_query["failure_tags"] = [tag.as_dict() for tag in tags]

    return per_query


# ---------------------------------------------------------------------------
# Full evaluation orchestrator
# ---------------------------------------------------------------------------


def run_full_evaluation(
    queries: Sequence[dict],
    rag_fn: Callable[[str, dict], Mapping],
    *,
    eval_config: Mapping | None = None,
    k_values: Sequence[int] | None = None,
    top_k: int | None = None,
    judge: ragas_style.LLMJudge | None = None,
    corpus_chunks: Sequence[dict] | None = None,
    timestamp: str | None = None,
    enable_ragas_style: bool | None = None,
    enable_claim_check: bool | None = None,
    enable_failure_tags: bool | None = None,
) -> dict:
    """Evaluate every query through ``rag_fn`` and return aggregate + per-query.

    Args:
        queries: Query records with ``query_id`` / ``query`` /
            ``expected_document`` (the harness's ground truth).
        rag_fn: Callable ``(query_text, query_record) -> dict`` returning the
            RAG system's output: ``answer``, ``cited_chunk_ids``, ``retrieved``
            (RetrievalResult/dict chunks), and optionally ``verification``
            (PremiseVerification or dict), ``rounds_used``, ``stop_reason``,
            ``adaptive_triggered``. This is the ONLY place retrieval/generation
            behavior is plugged in; the harness itself never retrieves.
        eval_config: The ``evaluation`` config block (or full config dict).
        k_values: Depths for precision@k / recall@k and the DRM depth default.
        top_k: DRM depth override (defaults to max(k_values)).
        judge: Optional RAGAs-style LLM judge for generation scores.
        corpus_chunks: Optional full chunk list for chunk-level recall.
        timestamp: Injectable timestamp for deterministic reports.
        enable_ragas_style / enable_claim_check / enable_failure_tags:
            Overrides for the ``evaluation`` config flags (default True).

    Returns:
        A deterministic dict ``{"envelope", "aggregate", "per_query"}``.
    """
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    eval_cfg = resolve_eval_config(eval_config or {})

    if enable_ragas_style is None:
        enable_ragas_style = bool(eval_cfg.get("enable_ragas_style", True))
    if enable_claim_check is None:
        enable_claim_check = bool(eval_cfg.get("enable_claim_check", True))
    if enable_failure_tags is None:
        enable_failure_tags = bool(eval_cfg.get("enable_failure_tags", True))

    k_list = resolve_k_values(eval_cfg, k_values)
    drm_depth = resolve_top_k(k_list, top_k)

    records: list[dict] = []
    per_query: list[dict] = []
    for query_record in queries:
        query_id = str(query_record.get("query_id") or "")
        query_text = str(query_record.get("query") or "")
        rag_result = rag_fn(query_text, dict(query_record))
        record = normalize_rag_result(query_record, rag_result)
        records.append(record)
    records = _enrich_retrieved_text(records, corpus_chunks)
    for record in records:
        per_query.append(
            evaluate_query_record(
                record,
                k_values=k_list,
                drm_depth=drm_depth,
                judge=judge,
                enable_ragas_style=enable_ragas_style,
                enable_claim_check=enable_claim_check,
                enable_failure_tags=enable_failure_tags,
            )
        )

    retrieval = evaluate_queries(records, k_values=k_list)
    drm = compute_drm_metrics(records, k=drm_depth)

    aggregate: dict = {
        "num_queries": len(records),
        "k_values": k_list,
        "drm_depth": drm_depth,
        "precision_at_k": retrieval["aggregate"]["precision_at_k"],
        "recall_at_k": retrieval["aggregate"]["recall_at_k"],
        "mrr": retrieval["aggregate"]["mrr"],
        "map": retrieval["aggregate"]["map"],
        "drm_fraction": drm["drm_fraction"],
        "drm_fraction_mean": _mean(q["drm_fraction"] for q in per_query),
        "top1_accuracy": drm["top1_accuracy"],
        "top3_accuracy": drm["top3_accuracy"],
    }
    if enable_ragas_style:
        aggregate["faithfulness"] = _mean(q["ragas"]["faithfulness"] for q in per_query)
        aggregate["answer_relevance"] = _mean(q["ragas"]["answer_relevance"] for q in per_query)
        aggregate["context_relevance"] = _mean(q["ragas"]["context_relevance"] for q in per_query)

    if enable_claim_check:
        aggregate["claim_level_faithfulness"] = _mean(q["claims"]["score"] for q in per_query)
        total_claims = sum(q["claims"]["num_claims"] for q in per_query)
        bad_claims = sum(
            q["claims"]["counts"][claim_check.STATUS_UNSUPPORTED]
            + q["claims"]["counts"][claim_check.STATUS_CONTRADICTED]
            for q in per_query
        )
        aggregate["unsupported_claim_fraction"] = (bad_claims / total_claims) if total_claims else 0.0

    tagged: Mapping[str, Sequence[failure_point_tagger.FailureTag]] = (
        failure_point_tagger.tag_results(records)
        if enable_failure_tags
        else {}
    )
    if enable_failure_tags:
        summary = failure_point_tagger.failure_point_summary(tagged)
        aggregate["failure_tags"] = {
            "counts": summary["counts"],
            "tagged_queries": summary["tagged_queries"],
            "tagged_queries_fraction": summary["tagged_queries_fraction"],
        }

    judge_note = GENERATION_SCORING_NOTES["llm_judge"] if judge is not None else GENERATION_SCORING_NOTES["deterministic"]
    return {
        "envelope": {
            "version": FULL_EVAL_VERSION,
            "generated_at": timestamp,
            "k_values": k_list,
            "drm_depth": drm_depth,
            "scoring": {
                "judge": None if judge is None else type(judge).__name__,
                "note": judge_note,
                "enable_ragas_style": enable_ragas_style,
                "enable_claim_check": enable_claim_check,
                "enable_failure_tags": enable_failure_tags,
            },
            "queries": [
                {"query_id": q["query_id"], "expected_document": q["expected_document"]}
                for q in queries
            ],
            "num_queries": len(queries),
            "env": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "config": _config_summary(eval_cfg),
                "level": "document",
                "corpus_chunks_provided": bool(corpus_chunks),
            },
            "notes": [
                "Milestone 7 full evaluation harness (eval/run_eval.py).",
                "Relevance is document-level: a retrieved chunk is relevant iff "
                "its source document equals the query's expected_document.",
                "The Milestone 4 DRM evaluation files "
                "(eval/results/drm_results.json, eval/results/drm_report.md) "
                "are intentionally not modified by this harness.",
            ],
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }


def _config_summary(eval_config: Mapping) -> dict:
    return {key: value for key, value in eval_config.items() if key != "embedding_ablation_models"}


# ---------------------------------------------------------------------------
# Report rendering + persistence
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.1%}" if isinstance(value, float) else str(value)


def render_markdown_report(result: dict) -> str:
    """Render the full-evaluation report as Markdown (deterministic)."""
    env = result["envelope"]
    agg = result["aggregate"]

    lines = [
        "# LegalRAG — Full Evaluation Report (Milestone 7)",
        "",
        (
            f"{agg['num_queries']} queries evaluated at depths "
            f"{env['k_values']} (DRM depth {env['drm_depth']}). Generated: "
            f"{env['generated_at']}."
        ),
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k in env["k_values"]:
        lines.append(f"| Precision@{k} | **{_fmt(agg['precision_at_k'][k])}** |")
        lines.append(f"| Recall@{k} | **{_fmt(agg['recall_at_k'][k])}** |")
    lines += [
        f"| MRR | **{_fmt(agg['mrr'])}** |",
        f"| MAP | **{_fmt(agg['map'])}** |",
        f"| DRM fraction | **{_fmt(agg['drm_fraction'])}** |",
        f"| Top-1 accuracy | **{_fmt(agg['top1_accuracy'])}** |",
        f"| Top-3 accuracy | **{_fmt(agg['top3_accuracy'])}** |",
    ]
    if "faithfulness" in agg:
        lines += [
            f"| Faithfulness | **{_fmt(agg['faithfulness'])}** |",
            f"| Answer relevance | **{_fmt(agg['answer_relevance'])}** |",
            f"| Context relevance | **{_fmt(agg['context_relevance'])}** |",
        ]
    if "claim_level_faithfulness" in agg:
        lines += [
            f"| Claim-level faithfulness | **{_fmt(agg['claim_level_faithfulness'])}** |",
            f"| Unsupported claim fraction | **{_fmt(agg['unsupported_claim_fraction'])}** |",
        ]
    if "failure_tags" in agg:
        lines += [
            f"| Queries with ≥1 failure tag | **{agg['failure_tags']['tagged_queries']}** "
            f"({_fmt(agg['failure_tags']['tagged_queries_fraction'])}) |",
        ]
    lines.append("")

    if "failure_tags" in agg:
        lines.append("## Failure-point tags (survey's 7 + unsupported premise)")
        lines.append("")
        lines.append("| # | Failure point | Count |")
        lines.append("|---|---|---|")
        for index, tag in enumerate(failure_point_tagger.ALL_FAILURE_POINTS, start=1):
            lines.append(
                f"| {index} | {failure_point_tagger.failure_point_label(tag)} | "
                f"{agg['failure_tags']['counts'][tag]} |"
            )
        lines.append("")

    lines.append("## Per-query results")
    lines.append("")
    lines.append(
        "| query_id | expected | P@1 | R@1 | RR | DRM | faithful | claims | tags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for q in result["per_query"]:
        ragas_cell = _fmt(q.get("ragas", {}).get("faithfulness", 0.0)) if "ragas" in q else "-"
        claims_cell = _fmt(q.get("claims", {}).get("score", 0.0)) if "claims" in q else "-"
        tag_names = ",".join(t["tag"] for t in q.get("failure_tags", [])) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(q["query_id"]),
                    str(q.get("expected_document") or "-"),
                    _fmt(q["retrieval"]["precision_at_k"][1]),
                    _fmt(q["retrieval"]["recall_at_k"][1]),
                    _fmt(q["retrieval"]["reciprocal_rank"]),
                    _fmt(q["drm_fraction"]),
                    ragas_cell,
                    claims_cell,
                    tag_names,
                ]
            )
            + " |"
        )
    lines.append("")

    if result["per_query"] and "claims" in result["per_query"][0]:
        lines.append("## Claim-level detail")
        lines.append("")
        for q in result["per_query"]:
            details = q.get("claims", {}).get("details", [])
            flagged = [
                d for d in details
                if d["status"] in (claim_check.STATUS_UNSUPPORTED, claim_check.STATUS_CONTRADICTED)
            ]
            if not flagged:
                continue
            lines.append(f"### {q['query_id']}")
            lines.append("")
            for detail in flagged:
                lines.append(
                    f"- `{detail['claim_id']}` **{detail['status']}**: "
                    f"*{detail['claim'][:140]}*"
                )
                lines.append(f"  - {detail['reason']}")
            lines.append("")

    lines.append("## Reproduction metadata")
    lines.append("")
    lines.append(f"- version: {env['version']}")
    lines.append(f"- k_values: {env['k_values']}, drm_depth: {env['drm_depth']}")
    lines.append(f"- scoring: {env['scoring']['note']}")
    lines.append(
        f"- config: `{json.dumps(env['env']['config'], sort_keys=True)}`"
    )
    lines.append(f"- python: {env['env']['python']} ({env['env']['platform']})")
    lines.append("- " + env["notes"][0])
    lines.append("- " + env["notes"][1])
    lines.append("- " + env["notes"][2])
    lines.append("")
    return "\n".join(lines)


def save_report(result: dict, results_dir: str | Path) -> tuple[Path, Path]:
    """Write the full-eval report as JSON + Markdown under ``results_dir``.

    Uses ``results.json`` / ``report.md`` (distinct names from the Milestone 4
    DRM files). The JSON is written with ``sort_keys=True`` so identical
    results serialize byte-identically.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    md_path = results_dir / "report.md"
    md_path.open("w", encoding="utf-8").write(render_markdown_report(result))
    return json_path, md_path


# ---------------------------------------------------------------------------
# Deterministic offline smoke path (no model download, no API key)
# ---------------------------------------------------------------------------


class _SmokeEncoder:
    """Deterministic SentenceTransformer-like encoder for the offline demo."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False):
        import numpy as np

        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.md5(str(text).encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                vec = vec / norm if norm else vec
            vectors.append(vec)
        return np.stack(vectors)


def _chunk_original_text(chunk: dict) -> str:
    return str(chunk.get("original_text") or chunk.get("text") or "").strip()


def _first_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return (sentences[0] if sentences else text).strip()


def build_smoke_rag_fn(
    retriever,
    queries: Sequence[dict],
    retrieve_top_k: int,
    perturb: bool = False,
    corpus_chunks: Sequence[dict] | None = None,
):
    """Deterministic RAG function used by the offline smoke run.

    Retrieval uses the REAL ``HybridRetriever`` passed in; answers are
    synthesized deterministically from the retrieved chunks. When ``perturb``
    is true, a handful of scripted queries are progressively perturbed so the
    smoke report demonstrates every failure-point category (this is a
    harness-demo construct -- the real CLI path uses ``perturb=False``).

    Retrieval results from an index-backed retriever carry no chunk body text,
    so ``corpus_chunks`` (the chunks.jsonl list) is used, when provided, to
    recover the body text for answer synthesis by ``chunk_id``.
    """
    text_by_id: dict[str, str] = {}
    for chunk in corpus_chunks or []:
        cid = chunk.get("chunk_id")
        if cid:
            text_by_id[str(cid)] = str(
                chunk.get("original_text") or chunk.get("text") or ""
            )
    index_of = {str(q.get("query_id")): i for i, q in enumerate(queries)}
    n = len(queries)

    def _body_for(cid: str, chunk: dict) -> str:
        text = text_by_id.get(str(cid))
        if text is None:
            text = _chunk_original_text(chunk)
        return _first_sentence(text)

    def rag_fn(query: str, qrecord: dict) -> dict:
        results = retriever.retrieve(query, top_k=retrieve_top_k)
        expected = qrecord.get("expected_document")
        idx = index_of.get(str(qrecord.get("query_id")), 0)
        verification = PremiseVerification(status="supported")
        base: dict = {
            "retrieved": results,
            "verification": verification,
            "rounds_used": 1,
        }
        if not results:
            return {**base, "answer": "I don't know.", "cited_chunk_ids": []}

        def preferred(chunk_results):
            for r in chunk_results:
                if r.source_name == expected:
                    return r
            return chunk_results[0]

        if perturb and idx == n - 1:
            # Hallucination + citation failure: cites a never-retrieved chunk
            # and asserts claims the evidence does not contain.
            return {
                **base,
                "answer": (
                    "The parties are legally required to disclose their trade "
                    "secrets to competitors within ninety days. "
                    "[citation:never-retrieved-chunk-999]"
                ),
                "cited_chunk_ids": ["never-retrieved-chunk-999"],
            }
        if perturb and idx == n - 2:
            # Generation failure: declines to answer despite retrieved evidence.
            return {**base, "answer": "I don't know.", "cited_chunk_ids": []}
        if perturb and idx == n - 3:
            # Unsupported premise (the project's 8th failure point).
            pick = preferred(results)
            return {
                **base,
                "verification": PremiseVerification(
                    status="unsupported",
                    premises=["the agreement has no term limit"],
                    explanation="no retrieved chunk mentions a term limit.",
                ),
                "answer": (
                    "Your premise that the agreement has no term limit is not "
                    f"supported by the evidence. [citation:{pick.chunk_id}]"
                ),
                "cited_chunk_ids": [pick.chunk_id],
            }
        if perturb and idx == n - 4:
            # Context-integration failure: cites a chunk from a DIFFERENT
            # document even though the expected document was retrieved.
            wrong = next((r for r in results if r.source_name != expected), None)
            if wrong is not None:
                return {
                    **base,
                    "answer": (
                        "The governing obligations are stated in the cited "
                        f"source. [citation:{wrong.chunk_id}]"
                    ),
                    "cited_chunk_ids": [wrong.chunk_id],
                }
        if perturb and idx == n - 5:
            # Efficiency failure: adaptive loop burned its extra rounds.
            pick = preferred(results)
            return {
                **base,
                "rounds_used": 3,
                "adaptive_triggered": True,
                "stop_reason": "max_extra_rounds_reached",
                "answer": (
                    "Confidential information is protected as stated in the "
                    f"cited provision. [citation:{pick.chunk_id}]"
                ),
                "cited_chunk_ids": [pick.chunk_id],
            }
        if perturb and idx == n - 6:
            # Interpretability failure: structural metadata stripped away.
            stripped = []
            for r in results:
                r.chunk = dict(r.chunk)
                r.chunk.pop("section", None)
                r.chunk.pop("parent", None)
                r.chunk.pop("heading", None)
                r.chunk.pop("hierarchy_path", None)
                r.section = None
                r.parent = None
                r.heading = None
                r.hierarchy_path = []
                stripped.append(r)
            pick = preferred(stripped)
            return {
                **base,
                "retrieved": stripped,
                "answer": (
                    "Confidential information is protected as stated in the "
                    f"cited provision. [citation:{pick.chunk_id}]"
                ),
                "cited_chunk_ids": [pick.chunk_id],
            }

        pick = preferred(results)
        body = _body_for(pick.chunk_id, pick.chunk)
        return {
            **base,
            "answer": f"{body} [citation:{pick.chunk_id}]",
            "cited_chunk_ids": [pick.chunk_id],
        }

    return rag_fn


def _corpus_manifest(raw_dir: str | Path) -> list[dict]:
    manifest = []
    for path in sorted(Path(raw_dir).iterdir()):
        if not path.is_file():
            continue
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# CLI (offline by default)
# ---------------------------------------------------------------------------


def _build_retriever_offline(config: dict, repo_root: Path, smoke: bool):
    """Build the real HybridRetriever over the Milestone 4 corpus.

    ``smoke=True`` uses the deterministic fake encoder (no download);
    ``smoke=False`` uses the configured real embedding model.
    """
    model = (config.get("retrieval") or {}).get("embedding_model", "BAAI/bge-large-en-v1.5")
    if smoke:
        embedder = Embedder(f"smoke-{slugify_model_name(model)}", encoder=_SmokeEncoder())
    else:
        embedder = Embedder(model)
    return build_retriever(
        config,
        embedder,
        use_sac=True,
        raw_dir=repo_root / EVAL_RAW_DIR,
        processed_dir=repo_root / EVAL_PROCESSED_DIR,
        index_root=repo_root / EVAL_INDEX_ROOT,
        chunk_file=CHUNK_FILE_SAC_ON,
    ), embedder.model_name


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full evaluation harness offline.

    Default mode is fully offline (fake deterministic encoder, synthesized
    answers, deterministic generation metrics). ``--real`` opts into using the
    configured embedding model (may download it on first use); generation
    metrics remain deterministic and no Groq call is ever made.
    ``--smoke-perturb`` enables the scripted failure injections that
    demonstrate every failure-point category in the demo report.
    """
    parser = argparse.ArgumentParser(description="LegalRAG Milestone 7 full evaluation harness")
    parser.add_argument("--real", action="store_true", help="use the real configured embedding model (may download it)")
    parser.add_argument("--smoke-perturb", action="store_true", help="inject scripted failures to demonstrate all 8 failure tags")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / "config.yaml")
    queries = load_queries(repo_root / EVAL_QUERIES_PATH)
    top_k = int((config.get("retrieval") or {}).get("top_k", 10))

    retriever, model_name = _build_retriever_offline(config, repo_root, smoke=not args.real)
    corpus_path = repo_root / EVAL_PROCESSED_DIR / CHUNK_FILE_SAC_ON
    corpus_chunks = read_chunks(corpus_path) if corpus_path.exists() else None
    rag_fn = build_smoke_rag_fn(
        retriever,
        queries,
        retrieve_top_k=top_k,
        perturb=args.smoke_perturb,
        corpus_chunks=corpus_chunks,
    )

    result = run_full_evaluation(
        queries,
        rag_fn,
        eval_config=config,
        top_k=top_k,
        corpus_chunks=corpus_chunks,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    # Attach reproduction metadata that depends on the run's own inputs.
    result["envelope"]["env"]["embedding_model"] = model_name
    result["envelope"]["env"]["embedder_source"] = "real-download" if args.real else "injected-fake-encoder"
    result["envelope"]["env"]["corpus"] = _corpus_manifest(repo_root / EVAL_RAW_DIR)
    result["envelope"]["env"]["corpus_dir"] = str(repo_root / EVAL_RAW_DIR)

    json_path, md_path = save_report(result, repo_root / DEFAULT_FULL_EVAL_DIR)

    agg = result["aggregate"]
    print("LegalRAG — Milestone 7 full evaluation (offline harness)")
    print("=" * 60)
    print(f"embedder source : {result['envelope']['env']['embedder_source']}")
    print(f"queries         : {agg['num_queries']}")
    print(f"mrr             : {agg['mrr']:.4f}")
    print(f"map             : {agg['map']:.4f}")
    print(f"drm_fraction    : {agg['drm_fraction']:.4f}")
    if "faithfulness" in agg:
        print(f"faithfulness    : {agg['faithfulness']:.4f}")
        print(f"answer relev.   : {agg['answer_relevance']:.4f}")
        print(f"context relev.  : {agg['context_relevance']:.4f}")
    if "claim_level_faithfulness" in agg:
        print(f"claim faithfulness: {agg['claim_level_faithfulness']:.4f}")
    print(f"\nJSON:     {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "DEFAULT_ABLATION_DIR",
    "DEFAULT_FULL_EVAL_DIR",
    "FULL_EVAL_VERSION",
    "build_smoke_rag_fn",
    "evaluate_query_record",
    "main",
    "normalize_rag_result",
    "render_markdown_report",
    "resolve_eval_config",
    "resolve_k_values",
    "resolve_top_k",
    "run_full_evaluation",
    "save_report",
]