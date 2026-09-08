"""Milestone 4 — Document Retrieval Mismatch (DRM) evaluation: SAC off vs. on.

Demonstrates that Summary-Augmented Chunking (SAC) reduces DRM — the fraction
of top-k retrieved chunks that come from the WRONG source document (AGENTS.md
failure mode 1 / Milestone 4). The eval corpus is a set of NDA/contract-style
documents that share near-identical boilerplate, with queries whose known
ground-truth document is one specific source.

This module *reuses* the existing pipeline end to end:
  * ingest.pipeline.run_ingestion          (Milestone 1, SAC toggleable)
  * retrieval.index.load_or_build_dense/bm25 (Milestone 2, namespaced indexes)
  * retrieval.retriever.HybridRetriever    (Milestone 3, fusion + rerank)
No retrieval logic is duplicated here; the only thing this module adds is the
DRM scoring, the SAC on/off comparison, and result serialization.

Run with the real embedding model:

    cd <repo root>
    python -m eval.drm_eval

This reads config.yaml, ingests ``data/eval/raw`` with SAC off and on, builds
namespaced indexes under ``data/eval/index``, retrieves every query from
``data/eval/queries.json``, and writes the comparison report under
``eval/results/`` as JSON + Markdown.

All directory paths and the embedder are injectable so pytest can exercise the
same code path offline with a fake encoder (see tests/test_drm_eval.py).
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from ingest.pipeline import run_ingestion
from retrieval.embed import Embedder
from retrieval.index import load_or_build_bm25, load_or_build_dense
from retrieval.retriever import HybridRetriever, RetrievalResult

DEFAULT_RESULTS_DIR = Path("eval/results")

# On-disk locations for the DRM evaluation corpus (all live in the repo).
EVAL_RAW_DIR = Path("data/eval/raw")
EVAL_PROCESSED_DIR = Path("data/eval/processed")
EVAL_INDEX_ROOT = Path("data/eval/index")
EVAL_QUERIES_PATH = Path("data/eval/queries.json")

# Chunk files are (model, SAC)-independent JSONL baskets; the chunk file used
# for ingestion differs only by SAC flag so the two configs never collide.
CHUNK_FILE_SAC_OFF = "chunks_sac_off.jsonl"
CHUNK_FILE_SAC_ON = "chunks_sac_on.jsonl"

# Evaluation set version, bumped when the corpus/queries change. Recorded in
# the report so a result is never mistaken for a different corpus.
EVAL_VERSION = 1


def load_config(config_path: str | Path) -> dict:
    """Load the full LegalRAG config.yaml into a dict."""
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_queries(queries_path: str | Path) -> list[dict]:
    """Load the eval query set.

    Expects ``{"queries": [{"query_id", "query", "expected_document"}, ...]}``.
    """
    with Path(queries_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    queries = payload.get("queries", [])
    for q in queries:
        if "query_id" not in q or "query" not in q or "expected_document" not in q:
            raise ValueError(
                "each query needs query_id, query, and expected_document"
            )
        if not q["expected_document"]:
            raise ValueError(f"query {q.get('query_id')!r} has an empty expected_document")
    return list(queries)


# --------------------------------------------------------------------------
# DRM scoring
# --------------------------------------------------------------------------


def compute_drm_metrics(query_runs: Sequence[dict], k: int) -> dict:
    """Compute the Milestone 4 metrics for one SAC configuration.

    Args:
        query_runs: One record per query, as returned by
            ``retrieve_all_queries``: ``{query_id, expected_document,
            retrieved_sources: [source, ...]}`` where ``retrieved_sources``
            lists the source documents of the top-k retrieved chunks in rank
            order.
        k: The retrieval depth over which DRM is measured (the number of
            retrieved chunks per query that are scored; ``len(retrieved)``
            may be smaller if the corpus is smaller).

    Returns:
        A dict with:
            * ``num_queries``
            * ``top_k``
            * ``top1_accuracy``  — fraction of queries whose top-1 chunk is from
              the ground-truth document.
            * ``top3_accuracy``  — fraction whose ground-truth document appears
              in the top-3 chunks.
            * ``doc_accuracy``   — ``{1, 3, 5, top_k} -> fraction of queries
              whose ground-truth document appears in the first k chunks.``
            * ``doc_accuracy_top_k`` — the document-level retrieval accuracy at
              the configured ``k``.
            * ``retrieval_mismatch_count`` — number of top-k retrieved chunks
              that come from a document other than the ground-truth one.
            * ``drm_fraction`` — ``retrieval_mismatch_count / total_chunks``:
              the fraction of top-k retrieved chunks from the WRONG document
              (the DRM metric per AGENTS.md).
            * ``total_chunks_checked``

        All values are deterministic given the query runs and ``k``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    runs = list(query_runs)
    num = len(runs)
    if num == 0:
        raise ValueError("cannot compute DRM metrics over zero queries")

    top1_hits = 0
    top3_hits = 0
    mismatches = 0
    total = 0
    doc_hits = {depth: 0 for depth in {1, 3, 5, k}}

    for run in runs:
        expected = run["expected_document"]
        sources = list(run.get("retrieved_sources", []))
        prefix = sources[:k]
        if prefix and prefix[0] == expected:
            top1_hits += 1
        if expected in sources[:3]:
            top3_hits += 1
        for depth in doc_hits:
            if expected in sources[:depth]:
                doc_hits[depth] += 1
        for source in prefix:
            total += 1
            if source != expected:
                mismatches += 1

    doc_hits = {depth: count / num for depth, count in doc_hits.items()}
    return {
        "num_queries": num,
        "top_k": k,
        "top1_accuracy": top1_hits / num,
        "top3_accuracy": top3_hits / num,
        "doc_accuracy": doc_hits,
        "doc_accuracy_top_k": doc_hits[k],
        "retrieval_mismatch_count": mismatches,
        "drm_fraction": mismatches / total if total else 0.0,
        "total_chunks_checked": total,
    }


def _comparison_table(sac_off: dict, sac_on: dict) -> list[dict]:
    """Build the before/after comparison rows for the two SAC configs.

    Each row is ``{metric, sac_off, sac_on, delta}`` where ``delta`` is
    ``sac_on - sac_off``. For accuracy metrics a positive delta is an
    improvement; for ``drm_fraction``/``retrieval_mismatch_count`` a negative
    delta is an improvement (this is called out in the report).
    """
    rows = [
        ("top1_accuracy", "Top-1 accuracy"),
        ("top3_accuracy", "Top-3 accuracy"),
    ]
    # Document-level accuracy at the depths we surface in the report.
    for depth in (1, 3, 5, sac_off["top_k"]):
        rows.append((f"doc_accuracy.{depth}", f"Document accuracy@{depth}"))
    rows += [
        ("retrieval_mismatch_count", "Retrieval mismatch count (top-k)"),
        ("drm_fraction", "DRM fraction (wrong-document chunks in top-k)"),
    ]

    table = []
    for key, label in rows:
        if "." in key and key.startswith("doc_accuracy"):
            off = sac_off["doc_accuracy"][int(key.split(".")[1])]
            on = sac_on["doc_accuracy"][int(key.split(".")[1])]
        else:
            off = sac_off[key]
            on = sac_on[key]
        table.append(
            {
                "metric": key,
                "label": label,
                "sac_off": off,
                "sac_on": on,
                "delta": on - off,
            }
        )
    return table


# --------------------------------------------------------------------------
# Retrieval orchestration (reuses the existing Milestone 1-3 pipeline)
# --------------------------------------------------------------------------


def build_retriever(
    config: dict,
    embedder: Embedder,
    use_sac: bool,
    raw_dir: str | Path,
    processed_dir: str | Path,
    index_root: str | Path,
    chunk_file: str,
) -> HybridRetriever:
    """Ingest + index + construct the hybrid retriever for one SAC state.

    Uses ``run_ingestion`` (Milestone 1), ``load_or_build_dense`` /
    ``load_or_build_bm25`` (Milestone 2), and ``HybridRetriever.from_config``
    (Milestone 3). The chunking ``use_sac`` flag is set on a shallow copy of
    the config so the caller's dict is never mutated.
    """
    cfg = copy.deepcopy(config)

    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)
    index_root = Path(index_root)

    cfg["chunking"]["use_sac"] = use_sac
    run_ingestion(
        cfg,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunks_file=chunk_file,
    )

    chunks_path = processed_dir / chunk_file
    dense = load_or_build_dense(chunks_path, index_root, embedder, use_sac)
    bm25 = load_or_build_bm25(chunks_path, index_root, embedder, use_sac)
    return HybridRetriever.from_config(dense, bm25, embedder, cfg["retrieval"])


def _to_retrieval_run(query: dict, results: Sequence[RetrievalResult]) -> dict:
    """Turn one query + its retrieved results into a compact run record."""
    expected = query["expected_document"]
    chunks = []
    for result in results:
        chunks.append(
            {
                "chunk_id": result.chunk_id,
                "source_name": result.source_name,
                "mismatch": result.source_name != expected,
            }
        )
    return {
        "query_id": query["query_id"],
        "query": query["query"],
        "expected_document": expected,
        "retrieved_sources": [c["source_name"] for c in chunks],
        "retrieved_chunks": chunks,
    }


def retrieve_all_queries(
    retriever: HybridRetriever,
    queries: Sequence[dict],
    top_k: int,
) -> list[dict]:
    """Run every eval query through ``retriever`` and record the top-k results.

    Args:
        retriever: A ``HybridRetriever`` (never reimplemented here).
        queries: Query records with ``query_id`` / ``query`` /
            ``expected_document``.
        top_k: Number of results kept per query (the retriever returns at
            most this many).

    Returns:
        A list of per-query run records (see ``_to_retrieval_run``).
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    runs = []
    for query in queries:
        results = retriever.retrieve(query["query"], top_k=top_k)
        runs.append(_to_retrieval_run(query, results[:top_k]))
    return runs


# --------------------------------------------------------------------------
# Experiment orchestration + serialization
# --------------------------------------------------------------------------


def _corpus_manifest(raw_dir: str | Path) -> list[dict]:
    """List each corpus file with its sha256 so results are reproducible."""
    manifest = []
    for path in sorted(Path(raw_dir).iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return manifest


def _env_metadata(config: dict) -> dict:
    """Reproduction metadata from the environment and the effective config."""
    retrieval = config.get("retrieval", {})
    chunking = config.get("chunking", {})
    return {
        "eval_version": EVAL_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": {
            "chunking_strategy": chunking.get("strategy"),
            "sac_summary_max_chars": chunking.get("sac_summary_max_chars"),
            "embedding_model": retrieval.get("embedding_model"),
            "retrieval_top_k": retrieval.get("top_k"),
            "rrf_k": retrieval.get("rrf_k"),
            "mmr_lambda": retrieval.get("mmr_lambda"),
            "structural_boost_strength": retrieval.get("structural_boost_strength"),
            "use_mmr": retrieval.get("use_mmr"),
            "use_structural_boost": retrieval.get("use_structural_boost"),
        },
    }


def run_drm_comparison(
    config: dict,
    embedder: Embedder,
    queries: Sequence[dict],
    raw_dir: str | Path = EVAL_RAW_DIR,
    processed_dir: str | Path = EVAL_PROCESSED_DIR,
    index_root: str | Path = EVAL_INDEX_ROOT,
    timestamp: str | None = None,
) -> dict:
    """Run the full SAC-off vs SAC-on DRM comparison.

    Pipeline for each SAC state: ingest (Milestone 1) -> index (Milestone 2)
    -> hybrid retrieval over every query (Milestone 3) -> DRM scoring. The two
    runs are identical except for ``chunking.use_sac``, and both embed with the
    same model, so the before/after difference is attributable to SAC.

    Args:
        config: Full application config dict.
        embedder: Embedder used for both the SAC-off and SAC-on indexes.
        queries: Eval query records (see ``load_queries``).
        raw_dir / processed_dir / index_root: Evaluation-specific directories
            (defaults to the in-repo ``data/eval/*`` locations).
        timestamp: Injectable timestamp used only in the report envelope.
            Defaults to the current UTC time. Tests pass a fixed value so the
            serialized report is byte-for-byte deterministic.

    Returns:
        A dict containing ``envelope`` (metadata), ``sac_off`` and ``sac_on``
        (each with ``metrics`` + ``per_query``), and ``comparison`` rows.
    """
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    top_k = int(config.get("retrieval", {}).get("top_k", 10))

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    index_root = Path(index_root)

    sections: dict[str, dict] = {}
    for use_sac, key, chunk_file in (
        (False, "sac_off", CHUNK_FILE_SAC_OFF),
        (True, "sac_on", CHUNK_FILE_SAC_ON),
    ):
        retriever = build_retriever(
            config,
            embedder,
            use_sac=use_sac,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            index_root=index_root,
            chunk_file=chunk_file,
        )
        runs = retrieve_all_queries(retriever, queries, top_k=top_k)
        sections[key] = {
            "use_sac": use_sac,
            "metrics": compute_drm_metrics(runs, k=top_k),
            "per_query": runs,
        }

    return {
        "envelope": {
            "generated_at": timestamp,
            "embedding_model": embedder.model_name,
            "num_queries": len(queries),
            "retrieval_depth": top_k,
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "index_root": str(index_root),
            "queries": [{"query_id": q["query_id"], "expected_document": q["expected_document"]} for q in queries],
            "corpus": _corpus_manifest(raw_dir),
            "env": _env_metadata(config),
        },
        "sac_off": sections["sac_off"],
        "sac_on": sections["sac_on"],
        "comparison": _comparison_table(
            sections["sac_off"]["metrics"], sections["sac_on"]["metrics"]
        ),
    }


# --------------------------------------------------------------------------
# Report rendering + persistence
# --------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Render a metric as a percentage when it is a fraction, else a plain int."""
    if isinstance(value, float):
        return f"{value:.1%}"
    return str(value)


def render_markdown_report(result: dict) -> str:
    """Render the DRM comparison report as Markdown (deterministic)."""
    env = result["envelope"]
    sac_off = result["sac_off"]
    sac_on = result["sac_on"]

    lines: list[str] = []
    lines.append("# DRM Evaluation — SAC off vs. SAC on")
    lines.append("")
    lines.append(
        f"Document Retrieval Mismatch (DRM) before/after report for "
        f"Summary-Augmented Chunking (SAC). Embedding model: "
        f"`{env['embedding_model']}`. Queries: {env['num_queries']} over "
        f"{len(env['corpus'])} documents. Retrieval depth: "
        f"{env['retrieval_depth']}. Generated: {env['generated_at']}."
    )
    lines.append("")

    lines.append("## Comparison table")
    lines.append("")
    lines.append("| Metric | SAC off | SAC on | Delta |")
    lines.append("|---|---|---|---|")
    for row in result["comparison"]:
        lines.append(
            f"| {row['label']} | {_fmt(row['sac_off'])} | "
            f"{_fmt(row['sac_on'])} | {_fmt(row['delta'])} |"
        )
    lines.append("")
    lines.append(
        "Delta is `SAC on - SAC off`. A positive delta is an improvement for "
        "the accuracy metrics; a negative delta is an improvement for mismatch "
        "count and DRM fraction."
    )
    lines.append("")

    for key, section in (("sac_off", sac_off), ("sac_on", sac_on)):
        m = section["metrics"]
        lines.append(f"## {key} (use_sac={section['use_sac']})")
        lines.append("")
        lines.append(f"- Top-1 accuracy: **{_fmt(m['top1_accuracy'])}**")
        lines.append(f"- Top-3 accuracy: **{_fmt(m['top3_accuracy'])}**")
        lines.append(
            f"- Document accuracy@{m['top_k']}: "
            f"**{_fmt(m['doc_accuracy_top_k'])}**"
        )
        lines.append(
            f"- Retrieval mismatch count: **{m['retrieval_mismatch_count']}** "
            f"of {m['total_chunks_checked']} chunks"
        )
        lines.append(f"- DRM fraction: **{_fmt(m['drm_fraction'])}**")
        lines.append("")
        lines.append("| query_id | expected | top-3 retrieved | top-3 hit |")
        lines.append("|---|---|---|---|")
        for run in section["per_query"]:
            top3 = run["retrieved_sources"][:3]
            hit = "yes" if run["expected_document"] in top3 else "no"
            lines.append(
                f"| {run['query_id']} | {run['expected_document']} | "
                f"{', '.join(top3) or '-'} | {hit} |"
            )
        lines.append("")

    lines.append("## Reproduction metadata")
    lines.append("")
    lines.append("- eval_version: " + str(env["env"]["eval_version"]))
    lines.append(f"- config: `{json.dumps(env['env']['config'], sort_keys=True)}`")
    lines.append(f"- python: {env['env']['python']} ({env['env']['platform']})")
    lines.append("")
    lines.append("Corpus files (sha256):")
    lines.append("")
    lines.append("| file | bytes | sha256 |")
    lines.append("|---|---|---|")
    for f in env["corpus"]:
        lines.append(f"| {f['file']} | {f['bytes']} | `{f['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def save_results(result: dict, results_dir: str | Path) -> tuple[Path, Path]:
    """Write the report as JSON and Markdown under ``results_dir``.

    Returns:
        ``(json_path, markdown_path)``. The JSON file is written with
        ``sort_keys=True`` so it is byte-stable for a given result dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "drm_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    md_path = results_dir / "drm_report.md"
    md_path.open("w", encoding="utf-8").write(render_markdown_report(result))
    return json_path, md_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    """Run the DRM evaluation with the real config + embedding model.

    Reads ``config.yaml`` from the repo root and the eval corpus/queries from
    ``data/eval/``, then writes the JSON + Markdown report to ``eval/results/``
    and prints the before/after table to stdout.
    """
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / "config.yaml")
    queries = load_queries(repo_root / EVAL_QUERIES_PATH)

    retrieval_cfg = config.get("retrieval", {})
    model = retrieval_cfg.get("embedding_model", "BAAI/bge-large-en-v1.5")
    embedder = Embedder(model)

    result = run_drm_comparison(
        config,
        embedder,
        queries,
        raw_dir=repo_root / EVAL_RAW_DIR,
        processed_dir=repo_root / EVAL_PROCESSED_DIR,
        index_root=repo_root / EVAL_INDEX_ROOT,
    )

    json_path, md_path = save_results(result, str(repo_root / DEFAULT_RESULTS_DIR))

    print("SAC OFF vs SAC ON — DRM before/after")
    print("=" * 46)
    print(f"{'metric':<42}{'off':>10}{'on':>10}")
    for row in result["comparison"]:
        print(f"{row['label']:<42}{_fmt(row['sac_off']):>10}{_fmt(row['sac_on']):>10}")

    print(f"\nJSON:      {json_path}")
    print(f"Markdown:  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())