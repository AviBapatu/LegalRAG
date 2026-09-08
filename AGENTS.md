# LegalRAG — Project Spec for OpenCode

## What we're building
A Retrieval-Augmented Generation (RAG) system for legal documents, implementing the
architecture described in Hindi et al., "Enhancing the Precision and Interpretability
of RAG in Legal Technology: A Survey" (IEEE Access, 2025), PLUS fixes for known failure
modes identified in follow-up 2025/2026 research (see "Known failure modes we're
targeting" below). This is a working reference implementation, not a toy — every
design choice is taken from what the literature identifies as best-performing, and
every fix below is something we can measure with a before/after number.

## Known failure modes we're targeting (and where we fix them)

1. **Document-Level Retrieval Mismatch (DRM)** — the retriever pulls a chunk from the
   entirely wrong source document because legal boilerplate (e.g. NDAs, contracts)
   is structurally near-identical across documents. Reuter et al. 2025 found DRM
   rates over 95% on NDA corpora with naive chunking. Fixed in §1 via
   **Summary-Augmented Chunking (SAC)**.
2. **Retrieval sets the performance ceiling** — embedding model choice affects
   downstream correctness more than the choice of generator LLM (Legal RAG Bench,
   2026). Addressed in §2/§6 via a built-in **embedding model ablation** in the eval harness.
3. **Hallucination persists even with good retrieval**, especially on queries with
   false/misleading premises — improving retrieval alone doesn't fix faithfulness.
   Addressed in §5/§6 via a **premise-verification step** and **claim-level** (not
   just answer-level) faithfulness scoring.
4. **Flattened legal structure** — naive chunking ignores section/subsection
   hierarchy, breaking cross-references. Addressed in §1 by keeping structural
   metadata and using it in reranking.

## Architecture (build in this order)

### 1. Ingestion & Chunking (`ingest/`)
- Load legal documents (PDF/txt/md) from a `data/raw/` folder.
- Implement TWO chunking strategies, selectable via config:
  - `sentence` chunking — splits on sentence boundaries, best for precision/F1 (per survey §III.A.1)
  - `pattern` chunking — splits on corpus-specific delimiters (e.g. "Section", "Article", "§"),
    best for context recall/faithfulness on structured legal text. This chunker must be
    **structure-aware**: parse and retain the section/subsection hierarchy (not just flat
    text) as metadata on each chunk, e.g. `{"section": "3.2", "parent": "3", "heading": "..."}`.
    This hierarchy is used later by the reranker (§3) to boost chunks whose parent section
    also matched the query — fixes the "flattened legal structure" failure mode.
- **Summary-Augmented Chunking (SAC)** — implement this as a pipeline step, on by default:
  - For each source document, generate one short LLM summary (~150 chars) that acts as a
    "document fingerprint" (parties, doc type, core subject matter).
  - Prepend this summary to every chunk derived from that document, before embedding.
  - This directly targets Document-Level Retrieval Mismatch (DRM): the retriever's tendency
    to select chunks from the wrong document entirely, which is especially severe for
    boilerplate-heavy docs like NDAs/contracts.
  - Make SAC toggleable via config (`chunking.use_sac: true/false`) so we can measure its
    effect (see eval harness in §6).
- Each chunk keeps metadata: source doc, section heading + hierarchy path, char offsets,
  whether SAC was applied.
- Store chunks as JSONL in `data/processed/chunks.jsonl`.

### 2. Embedding & Indexing (`retrieval/embed.py`, `retrieval/index.py`)
- Dense: `sentence-transformers`, default model `BAAI/bge-large-en-v1.5`, but the model
  name must be a config value, not hardcoded — the eval harness (§6) needs to swap
  embedding models and compare, since recent benchmarks show embedding choice affects
  downstream correctness more than the generator LLM does.
- Sparse: BM25 over the same chunks via `rank_bm25`.
- Both indexes persist to `data/index/`, namespaced by embedding model + SAC on/off,
  so multiple configurations can coexist for comparison.

### 3. Hybrid Retrieval (`retrieval/retriever.py`)
- Given a query: run dense + sparse retrieval in parallel, merge scores
  (reciprocal rank fusion), return top-K.
- Add a reranking step: MMR (maximum marginal relevance) for diversity, following CaseGPT's approach.
- Add a small structural boost: if a candidate chunk's parent section (from the hierarchy
  metadata in §1) also appears among top matches, boost the chunk's score slightly — this
  helps recover cross-referenced provisions that pure similarity search misses.

### 4. Adaptive Retrieval Loop (`retrieval/adaptive.py`)
- After generation, compute a confidence score (e.g. self-consistency or a
  cheap "does this context support the answer" LLM check).
- If below threshold, trigger one more retrieval round with a rewritten query
  (query rewriting via the LLM), capped at N=2 extra rounds. Mirrors DRAG-BILQA / Fig. 6 of the survey.

### 5. Augmentation & Generation (`generation/prompt.py`, `generation/generate.py`)
- Structured prompt template: system instructions (cite sources, say "I don't know"
  rather than hallucinate — see survey §VII.B) + retrieved chunks (with source labels)
  + the user query.
- LLM: Claude via the Anthropic API (model configurable).
- Every generated answer must include inline citations back to chunk IDs.
- **Premise verification step**: before generating the main answer, have the LLM check
  whether the query itself contains a factual/legal assumption that isn't supported by
  any retrieved chunk (e.g. "Since the NDA has no term limit, can I disclose after 2
  years?" when the NDA actually does specify a term). If an unsupported premise is
  detected, the answer must surface that explicitly rather than answering the
  hypothetical as if it were fact. This targets hallucinations caused by the model
  accepting a false premise even when retrieval itself worked correctly.

### 6. Evaluation Harness (`eval/`)
- Retrieval metrics: precision@k, recall@k, MRR, MAP — implement from scratch, don't
  just wrap a library, so it's inspectable (see survey Table 6/9 for definitions).
- **DRM metric** (`eval/drm.py`): for each query with a known ground-truth source document,
  compute the fraction of top-k retrieved chunks that come from the WRONG document.
  Report this separately from text-level precision/recall — it's the clearest signal for
  whether SAC (§1) is helping, and should be run with SAC on vs. off so we get a direct
  before/after comparison, mirroring the reduction reported in recent legal RAG research.
- **Embedding model ablation** (`eval/ablation.py`): run the same query set through 2-3
  swappable embedding models (e.g. bge-large-en-v1.5, a multilingual model, and one
  OpenAI/Claude-family embedding if available) and report precision/recall/DRM for each,
  since retrieval quality — more than the generator LLM — tends to set the ceiling on
  overall system performance.
- Generation metrics: a lightweight RAGAs-style LLM-judge scorer for faithfulness,
  answer relevance, and context relevance (0-1 scale, GPT/Claude as judge — survey §IV).
- **Claim-level faithfulness** (`eval/claim_check.py`): instead of scoring a whole answer
  as one faithfulness number, decompose the generated answer into individual factual
  claims (one LLM call), then check each claim against the retrieved chunks independently
  and flag any unsupported or contradictory claims by ID. This is more diagnostic than a
  single score and catches cases where most of an answer is grounded but one claim isn't.
- A `failure_point_tagger.py` that classifies a failed response into one of the
  survey's 7 failure points (missing content, missed top-K, not in context, not
  extracted, incorrect specificity, incomplete, etc — Table 10) — this is the
  paper's "challenge scale" (Fig. 7) turned into actual code. Extend it with an 8th
  category, "unsupported premise," for cases caught by the premise-verification step (§5).

### 7. API + minimal UI (`app/`)
- FastAPI backend: `/query` endpoint, `/eval` endpoint to run the harness.
- A single-page HTML/JS frontend (no framework needed) showing: query box,
  answer, retrieved chunks with relevance scores, and a toggle to see which
  failure-point classification (if any) fired.

## Sample data
Do NOT scrape or reproduce copyrighted case law. Use public-domain / openly licensed
legal text for the demo corpus — e.g. sample statutes, the Open Australian Legal
Corpus (CC-licensed), or documents the user supplies themselves in `data/raw/`.

## Tech constraints
- Python 3.11+, dependencies pinned in `requirements.txt` / `pyproject.toml`.
- No unnecessary frameworks — keep it inspectable and hackable.
- Every module gets a pytest test file. Write tests alongside the code, not after.
- Config lives in one `config.yaml` (chunking strategy, model names, top-K, thresholds).

## Build order / milestones
1. Project skeleton + config + ingestion/chunking, INCLUDING structure-aware pattern
   chunking and Summary-Augmented Chunking (toggleable) (with tests)
2. Embedding + FAISS index + BM25 index, with embedding model configurable (not hardcoded)
3. Hybrid retriever + MMR rerank + structural boost (with a small eval script proving
   retrieval works)
4. DRM metric + a first before/after report: SAC on vs. off, on a small NDA/contract-like
   sample set — this is the first concrete, measurable win, do it early so we know SAC
   is actually working before building more on top of it
5. Prompt template + generation via Claude API + premise-verification step
6. Adaptive retrieval loop
7. Full evaluation harness: precision/recall/MRR/MAP, embedding ablation, claim-level
   faithfulness, failure-point tagger (incl. "unsupported premise" category)
8. FastAPI + minimal frontend (show retrieved chunks, DRM/claim-level scores, and any
   failure-point tags for transparency)
9. README with setup instructions, the DRM before/after numbers, and an example
   end-to-end run

Work through milestones one at a time. After each milestone, run the tests and
show me a short summary before moving to the next one. Milestone 4 in particular
should produce an actual number (e.g. "DRM dropped from 61% to 24% with SAC") —
don't move on until that comparison is real and reproducible.
