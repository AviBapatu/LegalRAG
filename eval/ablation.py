"""Embedding-model ablation for retrieval quality (Milestone 7, AGENTS.md §6).

Recent legal-RAG benchmarks show the embedding model sets the ceiling on
downstream correctness more than the generator LLM does (AGENTS.md failure mode
2 / §2). This module runs the SAME evaluation query set through the swappable
embedding models listed in ``evaluation.embedding_ablation_models``
(config.yaml) and reports, per model:

* precision@k and recall@k  (eval/metrics.py, document-level relevance)
* DRM                        (eval.drm_eval.compute_drm_metrics, unchanged)

The ablation REUSES the existing pipeline end to end exactly like
``eval/drm_eval.py`` does:

* ``eval.drm_eval.build_retriever`` / ``retrieve_all_queries``  (Milestones 1-3)
* ``eval.drm_eval.compute_drm_metrics``                        (Milestone 4)
* ``eval.metrics.evaluate_queries``                            (Milestone 7)

No retrieval algorithm is reimplemented anywhere in this module.

Model availability is handled explicitly and never silently substituted:
* Callers may inject a ``model_name -> Embedder`` mapping (tests inject fake
  embedders here; pytest NEVER downloads BGE or any other model).
* A configured model missing from an injected mapping raises ``RuntimeError``
  instead of substituting another model.
* Without an injected mapping, real ``Embedder(model_name)`` objects are built
  lazily; any failure to load/embed surfaces as a clear exception from the
  normal index build, never as a silent fallback.
"""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from retrieval.embed import Embedder

from .drm_eval import (
    CHUNK_FILE_SAC_OFF,
    CHUNK_FILE_SAC_ON,
    EVAL_INDEX_ROOT,
    EVAL_PROCESSED_DIR,
    EVAL_RAW_DIR,
    build_retriever,
    compute_drm_metrics,
    retrieve_all_queries,
)
from .metrics import DEFAULT_K_VALUES, evaluate_queries

#: Evaluation-set version for ablation reports; bumped when the corpus or the
#: comparison semantics change. Independent of the Milestone 4 DRM version.
ABLATION_VERSION = 1


def configured_ablation_models(config: dict) -> list[str]:
    """Return the embedding models listed for the ablation.

    Raises ``ValueError`` when the list is empty or missing, since an ablation
    over zero models is meaningless.
    """
    models = list(
        (config.get("evaluation") or {}).get("embedding_ablation_models") or []
    )
    if not models:
        raise ValueError(
            "evaluation.embedding_ablation_models is empty; configure at least "
            "one embedding model for the ablation"
        )
    invalid = [m for m in models if not isinstance(m, str) or not m]
    if invalid:
        raise ValueError(f"invalid embedding model entries: {invalid}")
    models = list(dict.fromkeys(models))  # dedupe, preserve order
    return models


def resolve_embedders(
    config: dict,
    embedders: Mapping[str, Embedder] | None = None,
) -> dict[str, Embedder]:
    """Resolve an ``Embedder`` for every configured ablation model.

    Args:
        config: Full application config (``evaluation.embedding_ablation_models``
            drives the resolution).
        embedders: Optional pre-built mapping of model name -> Embedder. When
            provided it is authoritative: every configured model MUST be
            present and a missing entry raises ``RuntimeError`` (no silent
            substitution). When omitted, real lazy ``Embedder`` objects are
            built from the model names.

    Returns:
        ``{model_name: Embedder}`` for every distinct configured model.
    """
    models = configured_ablation_models(config)
    if embedders is not None:
        missing = [model for model in models if model not in embedders]
        if missing:
            raise RuntimeError(
                "the following configured embedding model(s) have no injected "
                f"embedder and cannot be quietly substituted: {missing}"
            )
        return {model: embedders[model] for model in models}
    return {model: Embedder(model) for model in models}


def _corpus_manifest(raw_dir: str | Path) -> list[dict]:
    """List corpus files with sha256 so an ablation report is reproducible."""
    manifest: list[dict] = []
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


def run_ablation(
    config: dict,
    queries: Sequence[dict],
    embedders: Mapping[str, Embedder] | None = None,
    raw_dir: str | Path = EVAL_RAW_DIR,
    processed_dir: str | Path = EVAL_PROCESSED_DIR,
    index_root: str | Path = EVAL_INDEX_ROOT,
    use_sac: bool = True,
    k_values: Sequence[int] | None = None,
    timestamp: str | None = None,
) -> dict:
    """Run the embedding-model ablation over the same query set.

    For each configured model: ingest + index (Milestone 1-2 pipeline via
    ``drm_eval.build_retriever``), hybrid retrieval over every query (Milestone
    3), then standard retrieval metrics (Milestone 7) and DRM (Milestone 4).
    All models share the same corpus, queries, SAC state, and retrieval depth,
    so the per-model differences are attributable to the embedding model.

    Args:
        config: Full application config dict.
        queries: Eval query records (see ``drm_eval.load_queries``).
        embedders: Optional model-name -> Embedder mapping. When provided it is
            authoritative (missing configured models raise). Tests inject fake
            embedders here; real runs omit it and use lazy real embedders.
        raw_dir / processed_dir / index_root: Evaluation-specific directories
            (defaults to the in-repo ``data/eval/*`` locations).
        use_sac: Whether retrieval indexes carry Summary-Augmented Chunking.
        k_values: Depths for precision@k / recall@k (defaults to
            ``evaluation.k_values`` or the module default).
        timestamp: Injectable timestamp for deterministic reports.

    Returns:
        ``{"envelope": {...}, "models": {model_name: {"use_sac", "metrics",
        "per_query"}}}``.
    """
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    models = configured_ablation_models(config)
    resolved = resolve_embedders(config, embedders)

    if k_values is not None:
        k_list = list(k_values)
    else:
        k_list = list((config.get("evaluation") or {}).get("k_values") or DEFAULT_K_VALUES)
    if not k_list:
        raise ValueError("k_values must be a non-empty sequence")

    top_k = int((config.get("retrieval") or {}).get("top_k", max(k_list)))
    if top_k < 1:
        raise ValueError("retrieval.top_k must be >= 1")

    chunk_file = CHUNK_FILE_SAC_ON if use_sac else CHUNK_FILE_SAC_OFF
    sections: dict[str, dict] = {}
    for model in models:
        embedder = resolved[model]
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
        standard = evaluate_queries(runs, k_values=k_list)
        drm = compute_drm_metrics(runs, k=top_k)
        sections[model] = {
            "use_sac": bool(use_sac),
            "metrics": {
                "precision_at_k": standard["aggregate"]["precision_at_k"],
                "recall_at_k": standard["aggregate"]["recall_at_k"],
                "mrr": standard["aggregate"]["mrr"],
                "map": standard["aggregate"]["map"],
                **{
                    key: drm[key]
                    for key in (
                        "drm_fraction",
                        "retrieval_mismatch_count",
                        "total_chunks_checked",
                        "top1_accuracy",
                        "top3_accuracy",
                        "doc_accuracy_top_k",
                    )
                },
            },
            "per_query": runs,
        }

    retrieval_cfg = config.get("retrieval") or {}
    chunking_cfg = config.get("chunking") or {}
    return {
        "envelope": {
            "version": ABLATION_VERSION,
            "generated_at": timestamp,
            "models": models,
            "num_queries": len(queries),
            "retrieval_depth": top_k,
            "k_values": list(k_list),
            "use_sac": bool(use_sac),
            "embedders": (
                "injected"
                if embedders is not None
                else [embedder.model_name for embedder in resolved.values()]
            ),
            "queries": [
                {"query_id": q["query_id"], "expected_document": q["expected_document"]}
                for q in queries
            ],
            "raw_dir": str(raw_dir),
            "corpus": _corpus_manifest(raw_dir),
            "env": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "config": {
                    "chunking_strategy": chunking_cfg.get("strategy"),
                    "use_sac": bool(chunking_cfg.get("use_sac")),
                    "embedding_model": retrieval_cfg.get("embedding_model"),
                    "retrieval_top_k": retrieval_cfg.get("top_k"),
                    "rrf_k": retrieval_cfg.get("rrf_k"),
                    "mmr_lambda": retrieval_cfg.get("mmr_lambda"),
                    "use_mmr": retrieval_cfg.get("use_mmr"),
                    "use_structural_boost": retrieval_cfg.get("use_structural_boost"),
                },
            },
        },
        "models": sections,
    }


# ---------------------------------------------------------------------------
# Report serialization  (writes to eval/results/ablation/, never touches the
# Milestone 4 DRM files under eval/results/)
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:.1%}" if isinstance(value, float) else str(value)


def render_markdown_report(result: dict) -> str:
    """Render the ablation comparison as Markdown (deterministic)."""
    env = result["envelope"]
    lines = [
        "# Embedding-Model Ablation",
        "",
        (
            f"Same query set ({env['num_queries']} queries) and corpus "
            f"({len(env['corpus'])} documents), retrieved with SAC "
            f"{'on' if env['use_sac'] else 'off'} at depth "
            f"{env['retrieval_depth']} across {len(env['models'])} embedding "
            f"model(s). Generated: {env['generated_at']}."
        ),
        "",
        "| Model | P@1 | P@5 | R@1 | R@5 | MRR | MAP | DRM |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, section in result["models"].items():
        m = section["metrics"]
        p1 = m["precision_at_k"]
        r1 = m["recall_at_k"]
        p1_val = p1.get(1, p1[min(p1)])
        p5_val = p1.get(5, p1[max(p1)])
        r1_val = r1.get(1, r1[min(r1)])
        r5_val = r1.get(5, r1[max(r1)])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{model}",
                    _fmt(p1_val),
                    _fmt(p5_val),
                    _fmt(r1_val),
                    _fmt(r5_val),
                    _fmt(m["mrr"]),
                    _fmt(m["map"]),
                    _fmt(m["drm_fraction"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Per-model details")
    lines.append("")
    for model, section in result["models"].items():
        m = section["metrics"]
        lines.append(f"### {model}")
        lines.append("")
        for k in sorted(m["precision_at_k"]):
            lines.append(
                f"- P@{k}: **{_fmt(m['precision_at_k'][k])}**, "
                f"R@{k}: **{_fmt(m['recall_at_k'][k])}**"
            )
        lines.append(
            f"- MRR: **{_fmt(m['mrr'])}**, MAP: **{_fmt(m['map'])}**, "
            f"DRM: **{_fmt(m['drm_fraction'])}** "
            f"({m['retrieval_mismatch_count']} of {m['total_chunks_checked']} "
            "chunks mismatch)"
        )
        lines.append("")

    lines.append("## Reproduction metadata")
    lines.append("")
    lines.append("- version: " + str(env["version"]))
    lines.append(f"- k_values: {env['k_values']}")
    lines.append(f"- use_sac: {env['use_sac']}")
    lines.append(f"- python: {env['env']['python']} ({env['env']['platform']})")
    lines.append(
        f"- config: `{__import__('json').dumps(env['env']['config'], sort_keys=True)}`"
    )
    lines.append("")
    lines.append("Corpus files (sha256):")
    lines.append("")
    lines.append("| file | bytes | sha256 |")
    lines.append("|---|---|---|")
    for file in env["corpus"]:
        lines.append(f"| {file['file']} | {file['bytes']} | `{file['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def save_results(result: dict, results_dir: str | Path) -> tuple[Path, Path]:
    """Write the ablation report as JSON + Markdown under ``results_dir``.

    Uses ``ablation_results.json`` / ``ablation_report.md`` (distinct from the
    Milestone 4 ``drm_results.json`` / ``drm_report.md``). The JSON is written
    with ``sort_keys=True`` so identical results serialize byte-identically.
    """
    import json

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "ablation_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    md_path = results_dir / "ablation_report.md"
    md_path.open("w", encoding="utf-8").write(render_markdown_report(result))
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the embedding ablation offline and write ablation_results.json.

    Default mode uses two deterministic fake embedders (no model download). Use
    ``--real`` to run the configured model(s) from ``config.yaml`` (this may
    download them on first use).
    """
    import argparse
    import copy
    import hashlib
    import sys

    from .drm_eval import EVAL_QUERIES_PATH, load_config, load_queries

    parser = argparse.ArgumentParser(description="LegalRAG embedding-model ablation")
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the configured real embedding model(s) from config.yaml (may download)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / "config.yaml")
    queries = load_queries(repo_root / EVAL_QUERIES_PATH)

    if args.real:
        result = run_ablation(
            config,
            queries,
            raw_dir=repo_root / EVAL_RAW_DIR,
            processed_dir=repo_root / EVAL_PROCESSED_DIR,
            index_root=repo_root / EVAL_INDEX_ROOT,
        )
        source = "real-download"
    else:
        import numpy as np

        class _FakeEncoder:
            def __init__(self, dim: int = 8):
                self.dim = dim

            def encode(self, texts, convert_to_numpy=False, normalize_embeddings=False):
                vectors = []
                for text in texts:
                    seed = int.from_bytes(
                        hashlib.md5(str(text).encode("utf-8")).digest()[:8], "big"
                    )
                    rng = np.random.default_rng(seed)
                    vec = rng.normal(size=self.dim).astype(np.float32)
                    if normalize_embeddings:
                        norm = np.linalg.norm(vec)
                        vec = vec / norm if norm else vec
                    vectors.append(vec)
                return np.stack(vectors)

        # Override the configured models with two fake, non-colliding names so
        # the offline run never touches or clobbers real model namespaces.
        cfg = copy.deepcopy(config)
        fake_models = ["smoke-embed-a", "smoke-embed-b"]
        cfg["evaluation"] = dict(config.get("evaluation") or {})
        cfg["evaluation"]["embedding_ablation_models"] = fake_models
        embedders = {
            name: Embedder(name, encoder=_FakeEncoder(dim=8)) for name in fake_models
        }
        result = run_ablation(
            cfg,
            queries,
            embedders=embedders,
            raw_dir=repo_root / EVAL_RAW_DIR,
            processed_dir=repo_root / EVAL_PROCESSED_DIR,
            index_root=repo_root / EVAL_INDEX_ROOT,
        )
        source = "injected-fake-embedders"

    result["envelope"]["env"]["embedder_source"] = source
    json_path, md_path = save_results(result, repo_root / "eval/results/ablation")

    print("LegalRAG — embedding-model ablation (offline)")
    print("=" * 60)
    print(f"embedder source : {source}")
    print(f"queries         : {result['envelope']['num_queries']}")
    print(f"{'model':<20}{'P@1':>8}{'MRR':>8}{'DRM':>8}")
    for model, section in result["models"].items():
        m = section["metrics"]
        print(
            f"{model:<20}{_fmt(m['precision_at_k'].get(1, list(m['precision_at_k'].values())[0])):>8}"
            f"{_fmt(m['mrr']):>8}{_fmt(m['drm_fraction']):>8}"
        )
    print(f"\nJSON:     {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "ABLATION_VERSION",
    "configured_ablation_models",
    "render_markdown_report",
    "resolve_embedders",
    "run_ablation",
    "save_results",
]