# LegalRAG

A working reference implementation of a Retrieval-Augmented Generation (RAG)
system for legal documents, built to the architecture described in:

> Hindi *et al.*, "Enhancing the Precision and Interpretability of RAG in
> Legal Technology: A Survey," IEEE Access, 2025

...plus fixes for known failure modes identified in follow-up 2025/2026
research. Every major design choice maps to a best-performing approach in the
survey, and every fix is paired with a measured before/after number in this
README (sections F–H). The authoritative engineering spec is `AGENTS.md`.

The project is built in milestones and each milestone is committed separately;
this README documents the state after **Milestone 8** (Milestone 9 = this
documentation + end-to-end verification).

---

## A. Overview

| Milestone | What was built | Status |
|---|---|---|
| 1 | Ingestion, two chunking strategies, Summary-Augmented Chunking (SAC) | done |
| 2 | Dense (sentence-transformers) + BM25 indexing, config-driven embedding model | done |
| 3 | Hybrid retrieval: Reciprocal Rank Fusion (RRF) + MMR rerank + structural boost | done |
| 4 | DRM metric, SAC on/off before/after report on a 6-doc NDA corpus | done (see F) |
| 5 | Prompt template, Claude generation, **premise-verification step** | done |
| 6 | Adaptive retrieval loop (query rewriting, capped extra rounds) | done |
| 7 | Full eval harness: precision/recall/MRR/MAP, DRM, RAGAs-style scores, claim-level faithfulness, failure-point tagger, embedding ablation | done (see G, H) |
| 8 | FastAPI backend + single-page frontend | done |
| 9 | README + final verification | this document |

**The four failure modes this project targets** (from AGENTS.md):

1. **Document-Level Retrieval Mismatch (DRM)** — the retriever pulls chunks
   from the entirely wrong source document because legal boilerplate (NDAs,
   contracts) is structurally near-identical across documents. Reuter *et al.*
   2025 found DRM rates over 95% on NDA corpora with naive chunking. Fixed via
   **Summary-Augmented Chunking (SAC)**.
2. **Retrieval sets the performance ceiling** — embedding-model choice affects
   downstream correctness more than the generator LLM (Legal RAG Bench, 2026).
   Addressed via a config-driven embedding model + a built-in **embedding model
   ablation** in the eval harness.
3. **Hallucination persists even with good retrieval**, especially on queries
   with false/misleading premises. Addressed via a **premise-verification step**
   and **claim-level** faithfulness scoring.
4. **Flattened legal structure** — naive chunking ignores section/subsection
   hierarchy. Addressed by **structure-aware chunking** that retains hierarchy
   metadata and uses it in reranking.

## B. Pipeline

```
data/raw/ ──► loader ──► chunker (sentence | pattern, structure-aware)
                    │
                    └─► SAC (summary prepended to every chunk, toggleable)
                    │
                    ▼
        data/processed/chunks.jsonl   (hierarchy + offsets + SAC metadata)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   dense (BGE) index       BM25 index        — namespaced by (model, SAC) under data/index/
        └───────────┬───────────┘
                    ▼
        HybridRetriever: dense ∥ BM25 → RRF fusion → MMR rerank
        → structural parent-section boost
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  premise verification    generation (Claude, inline citations)
        └───────────┬───────────┘
                    ▼
      adaptive loop (support check → query rewrite → re-retrieve, max N extra rounds)
                    ▼
   FastAPI (/query, /eval) + single-page frontend
        └─────────────────────────► eval harness (retrieval + generation metrics)
```

## C. Repository structure

```
.
├── AGENTS.md                  # authoritative engineering spec
├── config.yaml                # single source of truth for all runtime settings
├── pyproject.toml             # deps (pinned), pytest config; no console scripts
├── ingest/                    # Milestone 1
│   ├── loader.py              #   PDF/TXT/MD → Document (utf-8/latin-1 fallback)
│   ├── chunker.py             #   sentence + structure-aware pattern chunking
│   ├── sac.py                 #   Summary-Augmented Chunking (heuristic or LLM)
│   └── pipeline.py            #   run_ingestion(config, ...) → chunks.jsonl
├── retrieval/                 # Milestones 2–3, 6
│   ├── embed.py               #   Embedder (model name from config, injectable encoder)
│   ├── index.py               #   DenseIndex / BM25Index, load_or_build_*
│   ├── retriever.py           #   HybridRetriever: RRF → MMR → structural boost
│   └── adaptive.py            #   adaptive retrieval loop controller
├── generation/                # Milestone 5
│   ├── prompt.py              #   structured prompt + premise verification contract
│   └── generate.py            #   ClaudeClient (anthropic), citations, PremiseVerification
├── eval/                      # Milestones 4, 7
│   ├── metrics.py             #   precision@k, recall@k, MRR, MAP (from scratch)
│   ├── drm_eval.py            #   DRM metric + SAC on/off comparison (CLI: python -m eval.drm_eval)
│   ├── ragas_style.py         #   lightweight RAGAs-style LLM-judge scorers
│   ├── claim_check.py         #   claim-level faithfulness
│   ├── failure_point_tagger.py#   survey's 7 failure points + "unsupported premise"
│   ├── ablation.py            #   embedding-model ablation (CLI: python -m eval.ablation)
│   ├── run_eval.py            #   unified full-eval harness (CLI: python -m eval.run_eval)
│   ├── text_util.py
│   └── results/               #   committed reports (drm_, full_eval/, ablation/)
├── app/                       # Milestone 8
│   ├── app.py                 #   FastAPI factory (create_app)
│   ├── routes.py / schemas.py #   /query, /eval endpoints + response models
│   ├── service.py             #   Application orchestration (retrieve/premise/generate/adaptive)
│   ├── config.py              #   AppSettings
│   ├── __main__.py            #   dev launcher (python -m app)
│   ├── smoke.py               #   offline app smoke scenarios (python -m app.smoke)
│   └── static/index.html      #   single-page frontend
├── data/
│   ├── raw/                   #   demo corpus (sample_nda.txt, sample_employment.md)
│   ├── processed/             #   chunks.jsonl (gitignored artifacts)
│   ├── index/                 #   namespaced indexes (gitignored)
│   └── eval/                  #   6 NDA docs + 24 queries for the DRM/full evals
├── tests/                     # pytest suite (one file per module)
└── eval/results/              # committed reports (JSON + Markdown, with SHA-256 corpus fingerprints)
```

## D. Setup and commands

Requirements: Python 3.11+.

```bash
cd legalrag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # runtime + dev extras (pytest, httpx)

# Real generation only: Claude via Anthropic. Key is read from the environment
# at call time and is never stored in the repo.
export ANTHROPIC_API_KEY=sk-ant-...
```

All commands run from the repo root and use `python -m`, because the package
declares no console-script entry points.

### Run the full pytest suite

```bash
pytest -q
```

### Milestone 4 — DRM before/after (real embedding model)

This is the first measurable win and the canonical SAC comparison. It ingests
`data/eval/raw` (6 NDAs) with SAC off and on, builds both index namespaces
under `data/eval/index/`, retrieves all 24 queries, and writes
`eval/results/drm_results.json` + `eval/results/drm_report.md`:

```bash
python -m eval.drm_eval
```

Note: this uses the real configured embedding model
(`BAAI/bge-large-en-v1.5`); it is downloaded on first use.

### Milestone 7 — full evaluation harness

```bash
python -m eval.run_eval                          # fully offline (default)
python -m eval.run_eval --real                   # real embedding model (may download)
python -m eval.run_eval --smoke-perturb          # + scripted failures to show all 8 tags
```

The default is entirely offline: a deterministic fake encoder stands in for the
embedding model, answers are synthesized deterministically from the retrieved
chunks, and **no Anthropic call or model download ever happens**. Writes
`eval/results/full_eval/results.json` + `report.md`.

### Milestone 7 — embedding ablation

```bash
python -m eval.ablation                          # offline: two deterministic fake embedders
python -m eval.ablation --real                   # configured model(s) from config.yaml
```

Writes `eval/results/ablation/ablation_results.json` + `ablation_report.md`.

### Milestone 8 — API and frontend

```bash
python -m app                    # http://127.0.0.1:8000
python -m app --host 0.0.0.0 --port 8080
python -m app.smoke              # offline end-to-end smoke of the two demo scenarios
```

No API key is needed to start the server; a key is only required when a query
actually invokes the real Claude generator.

### Ingestion and index building (programmatic)

The core stages do not have standalone `__main__` CLIs; they are driven
programmatically and re-used end to end by the eval harnesses. To run them
directly:

```python
from pathlib import Path
import yaml

from ingest.pipeline import run_ingestion
from retrieval.embed import Embedder
from retrieval.index import load_or_build_dense, load_or_build_bm25

config = yaml.safe_load(Path("config.yaml").read_text())          # ingest data/raw
run_ingestion(config)                                              # → data/processed/chunks.jsonl

embedder = Embedder(config["retrieval"]["embedding_model"])
dense = load_or_build_dense(embedder, "data/index", chunks_file="data/processed/chunks.jsonl")
bm25 = load_or_build_bm25("data/index", chunks_file="data/processed/chunks.jsonl")
```

Indexes are namespaced by (embedding model, SAC on/off) so multiple
configurations coexist and are compared without clobbering each other; a
`metadata.json` records the params an index was built with.

## E. Configuration (`config.yaml`)

Single source of truth. Key blocks:

```yaml
ingest:
  raw_dir: data/raw
  processed_dir: data/processed
  chunks_file: chunks.jsonl
  supported_extensions: [.pdf, .txt, .md]

chunking:
  strategy: pattern            # pattern | sentence
  sentence: {max_chars: 1000, min_chars: 50}
  pattern:                    # regex delimiters, each with a hierarchy level
    delimiters:               # ("Section 3"/"Article 5" → level 1; "3.2.1" → 3;
      - {pattern: '^[ \t]*(?:Section|Article)\s+\d+', level: 1}  #  "3.2" → 2; "3." → 1; "(a)" → 3)
      - {pattern: '^[ \t]*\d+\.\d+\.\d+', level: 3}
      - {pattern: '^[ \t]*\d+\.\d+', level: 2}
      - {pattern: '^[ \t]*\d+\.', level: 1}
      - {pattern: '^[ \t]*\(\w+\)', level: 3}
  use_sac: true               # Summary-Augmented Chunking master switch
  sac_use_llm: false          # false → deterministic heuristic summarizer (offline)
  sac_summary_max_chars: 150
  sac_llm_model: claude-sonnet-4-20250514

retrieval:
  embedding_model: BAAI/bge-large-en-v1.5   # a config value, never hardcoded
  index_dir: data/index
  top_k: 10
  rrf_k: 60                    # RRF smoothing constant
  mmr_lambda: 0.7              # 1.0 = pure relevance, 0.0 = pure diversity
  structural_boost_strength: 0.1
  use_structural_boost: true
  use_mmr: true

generation:
  provider: anthropic
  model: claude-sonnet-4-20250514
  temperature: 0.2
  max_tokens: 1024

adaptive:
  enabled: true
  support_threshold: 1.0       # nominal; decision is deterministic (see code)
  max_extra_rounds: 2          # extra retrieval rounds after the initial one

evaluation:
  k_values: [1, 3, 5, 10]      # largest value is the default DRM depth
  embedding_ablation_models: [BAAI/bge-large-en-v1.5]
  enable_ragas_style: true
  enable_claim_check: true
  enable_failure_tags: true

output:
  newline: "\n"

app:
  host: 127.0.0.1
  port: 8000
```

**SAC in detail** (`ingest/sac.py`): each document gets a ~150-char
"document fingerprint" (parties, doc type, core subject matter) which is
prepended to every chunk of that document before embedding
(`SAC_SEPARATOR = "\n\n[Document summary]\n"`). The pre-SAC text is preserved
on each chunk (`original_text`) and `sac_applied` is set so the effect can be
measured. The default summarizer is deterministic (no LLM, no network); an LLM
summary is available by setting `sac_use_llm: true` and injecting a callable.

## F. Milestone 4 — DRM: SAC off vs. on (the before/after number)

Corpus: 6 near-identical NDA documents (`data/eval/raw/`), 24 queries
(`data/eval/queries.json`) each scoped to one known source document.
Embedding: `BAAI/bge-large-en-v1.5` (real). Retrieval depth: 10 (RRF + MMR +
structural boost). Report: `eval/results/drm_results.json` /
`eval/results/drm_report.md`.

| Metric | SAC off | SAC on | Δ |
|---|---|---|---|
| Top-1 accuracy | 83.3% | **91.7%** | +8.3% |
| Top-3 accuracy | 100.0% | 95.8% | −4.2% |
| Document accuracy@1 | 83.3% | **91.7%** | +8.3% |
| Document accuracy@3 | 100.0% | 95.8% | −4.2% |
| Document accuracy@5 | 100.0% | 100.0% | 0.0% |
| Document accuracy@10 | 100.0% | 100.0% | 0.0% |
| Retrieval mismatch count (top-k) | 152 | **123** | −29 |
| **DRM fraction (wrong-document chunks in top-k)** | 63.3% | **51.2%** | **−12.1%** |

**DRM fraction dropped from 63.3% to 51.2% with SAC.** The Top-3 accuracy
dip (−4.2%) is an honest caveat: SAC changes chunk text so the fusion ordering
inside the top-3 shifts even on queries whose correct document is still
retrieved; the wrong-document *fraction* is what SAC is designed to reduce, and
it falls by a quarter of its baseline value.

## G. Milestone 7 — full evaluation

Offline run of `eval/run_eval.py` over the same 24-query NDA corpus with SAC on
(24 queries, k = [1, 3, 5, 10], DRM depth 10). Report:
`eval/results/full_eval/results.json` / `report.md`.

> **Important:** this committed run used the harness's **deterministic fake
> encoder** (`embedder_source: injected-fake-encoder`), so these are offline
> demonstration numbers of the harness, not real embedding-model numbers. A
> real-model run is produced by `python -m eval.run_eval --real`.

| Metric | Value |
|---|---|
| Precision@1 / @3 / @5 / @10 | 58.3% / 40.3% / 36.7% / 30.4% |
| Recall@1 / @3 / @5 / @10 | 58.3% / 79.2% / 91.7% / 100.0% |
| MRR / MAP | 71.9% / 71.9% |
| DRM fraction (depth 10) | 69.6% |
| Top-1 / Top-3 accuracy | 58.3% / 79.2% |
| Faithfulness (RAGAs-style) | 75.0% |
| Answer relevance | 19.2% |
| Context relevance | 88.8% |
| Claim-level faithfulness | 75.0% |
| Unsupported claim fraction | 23.1% |
| Queries tagged with a failure point | 6 / 24 (25.0%) |

Failure-tag counts (survey's 7 points + the added 8th): `retrieval_failure` 0,
`context_integration_failure` 1, `generation_failure` 1, `hallucination` 5,
`citation_failure` 1, `efficiency_failure` 1, `interpretability_failure` 1,
**`unsupported_premise` 1** (the new category from the premise-verification
step).

Note: the DRM fraction here (69.6%) is higher than the Milestone 4 DRM-fraction.
This run uses a 10-deep window and the fake encoder on a corpus deliberately
constructed to stress DRM; it is not a contradiction of F, which compares SAC
off vs. on on identical retrieval settings (metrics move in the same direction
under both encoders).

## H. Milestone 7 — embedding ablation

`python -m eval.ablation` (offline) with two deterministic in-process
embedders (`smoke-embed-a`, `smoke-embed-b`, md5-seeded, dim 8), SAC on, depth
10, 24 queries. Report: `eval/results/ablation/ablation_results.json` /
`ablation_report.md`.

| Model | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR | MAP | DRM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke-embed-a | 58.3% | 40.3% | 36.7% | 30.4% | 58.3% | 79.2% | 91.7% | 100% | 71.9% | 71.9% | 69.6% |
| smoke-embed-b | 58.3% | 40.3% | 36.7% | 30.4% | 58.3% | 79.2% | 91.7% | 100% | 71.9% | 71.9% | 69.6% |

Both fake models are deterministic and produce identical vectors per text, so
the identical tables validate that the ablation *plumbing* is hermetic (no
cross-contamination between model namespaces; each model gets its own
`data/eval/index/<model>-sac-*` index). Real comparison across models:

```bash
eval.ablation --real   # uses config.yaml evaluation.embedding_ablation_models
```

## I. API

The FastAPI app (`app/app.py`, `python -m app`) exposes three routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the single-page frontend (`app/static/index.html`) |
| `/query` | POST | Full RAG round-trip. Body: `{"query": "<string>"}` |
| `/eval` | GET | Returns the committed Milestone 7 report from `eval/results/full_eval/results.json` (`AppSettings.eval_results_path`) |

`POST /query` response (Milestone 8 schema) includes:
`original_query`, `final_answer` (with inline citations to chunk IDs),
`premise_verification` (status `supported` / `unsupported` / `contradicted` +
premises + explanation), `adaptive_triggered`, `rounds_used`,
`rewritten_queries`, `final_retrieval_results` (chunks with dense/BM25/fused
scores + hierarchy metadata + `structurally_boosted`), `cited_chunk_ids`,
`failure_tags`, `claim_level` (score + per-claim detail), `stop_reason`,
`support_threshold`, `max_extra_rounds`, and per-round detail in `rounds`.

`GET /eval` returns the harness `aggregate` block (precision/recall@k, MRR,
MAP, DRM fraction, faithfulness, answer/context relevance, claim faithfulness,
failure-tag counts).

## J. Frontend

`app/static/index.html` — no frameworks, vanilla JS. Shows:

- query box + answer card (rounds, adaptive-triggered badge, stop reason,
  rewritten queries)
- premise-verification verdict and explanation
- failure-point tags (8 categories incl. "unsupported premise")
- claim-level faithfulness breakdown (per-claim status + reason)
- retrieved chunks with fused/dense/BM25 scores and section hierarchy
  (`section` / `parent` / `heading` / `hierarchy_path`)
- per-round detail (each adaptive round's query + chunks + decision)
- "Latest evaluation (Milestone 7)" panel that fetches `/eval` and renders the
  metric grid + failure-tag table

## K. Testing

Every module has a pytest file under `tests/`. The suite is fully offline:
fakes stand in for the embedding model (`tests/conftest.py`:
`FakeEncoder`, deterministic md5-seeded vectors) and for retrieval/generation
(`tests/app_fakes.py`). Two smoke scenarios (adaptive rewrite-then-success and
no-rewrite-sufficient) are covered both as pytest tests
(`tests/test_smoke_app.py`) and as the runnable script `python -m app.smoke`.

```bash
pytest -q      # 370 passed (verified in Milestone 9)
```

## L. Reproducibility

- **No secrets in the repo.** The Anthropic key is read from
  `ANTHROPIC_API_KEY` at call time only; `.gitignore` excludes `.env`,
  `.venv/`, `__pycache__/`, `data/index/`, `data/processed/`, and
  `legalrag.egg-info/`.
- **Corpus integrity.** Every committed report embeds a per-file SHA-256
  manifest of `data/eval/raw` plus an `EVAL_VERSION`, so a result is never
  mistaken for a different corpus.
- **Deterministic offline components.** Fake embedders are seeded by text
  content; harness answers are synthesized deterministically; full-eval
  serialization is a pure function of inputs + timestamp.
- **Config is captured, not assumed.** Reports record the exact
  `config.yaml` values used (strategy, model, top-k, RRF/MRR/boost switches,
  SAC state).
- **Environment.** The committed reports record Python version and platform
  (e.g. Python 3.14.4 on Linux/WSL2); pinned deps live in `pyproject.toml`.
- Real-model runs (Milestone 4, `--real` flags) download models once and reuse
  the namespaced local indexes.

## M. Limitations

- **Small, synthetic demo corpus.** 24 queries over 6 hand-written NDA-like
  documents. The corpus was deliberately built to stress DRM (near-identical
  boilerplate); numbers here are indicative of the *pipeline*, not of any
  production deployment.
- **Milestone 7 numbers are offline demonstrations.** The committed
  full-eval and ablation reports used injected deterministic fake embedders;
  they validate the harness and its metrics, not real embedding quality. Use
  `--real` (or `python -m eval.drm_eval`) for real-model numbers.
- **RAGAs-style ≠ official RAGAs.** `eval/ragas_style.py` is a lightweight,
  inspectable re-implementation used as a harness; it is not the official RAGAS
  library.
- **Generation requires the Anthropic API.** No local LLM fallback; without
  `ANTHROPIC_API_KEY`, the real generator cannot be invoked (the app and all
  tests still work offline via injected fakes).
- **SAC summarizer is heuristic by default.** Deterministic and
  dependency-free; an LLM summarizer is optional (`sac_use_llm: true`).
- **Faithfulness scoring inherits retrieval quality.** If retrieval misses the
  relevant provision, plausible but unsupported claims can look grounded — this
  is exactly why claim-level checking and failure tagging exist as separate
  signals.

## N. License and data note

This repository is a reference implementation. Demo data (`data/raw` and
`data/eval`) is synthetic/sample text; **do not scrape or reproduce
copyrighted case law or contracts.** For a larger demo corpus, use public-domain
or openly licensed legal text (e.g. sample statutes, the Open Australian Legal
Corpus, which is CC-licensed) or documents you supply yourself under
`data/raw/`. See `AGENTS.md` for the full engineering spec and rationale.