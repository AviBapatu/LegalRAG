# LegalRAG — Full Evaluation Report (Milestone 7)

24 queries evaluated at depths [1, 3, 5, 10] (DRM depth 10). Generated: 2026-09-06T10:27:36.965147+00:00.

## Aggregate metrics

| Metric | Value |
|---|---|
| Precision@1 | **58.3%** |
| Recall@1 | **58.3%** |
| Precision@3 | **40.3%** |
| Recall@3 | **79.2%** |
| Precision@5 | **36.7%** |
| Recall@5 | **91.7%** |
| Precision@10 | **30.4%** |
| Recall@10 | **100.0%** |
| MRR | **71.9%** |
| MAP | **71.9%** |
| DRM fraction | **69.6%** |
| Top-1 accuracy | **58.3%** |
| Top-3 accuracy | **79.2%** |
| Faithfulness | **75.0%** |
| Answer relevance | **19.2%** |
| Context relevance | **88.8%** |
| Claim-level faithfulness | **75.0%** |
| Unsupported claim fraction | **23.1%** |
| Queries with ≥1 failure tag | **6** (25.0%) |

## Failure-point tags (survey's 7 + unsupported premise)

| # | Failure point | Count |
|---|---|---|
| 1 | Retrieval failure (missed top-K / missing content) | 0 |
| 2 | Context integration failure (not in context) | 1 |
| 3 | Generation failure (not extracted / incomplete) | 1 |
| 4 | Hallucination (ungrounded claims) | 5 |
| 5 | Citation failure | 1 |
| 6 | Efficiency failure (too many retrieval rounds) | 1 |
| 7 | Interpretability failure (no structural trace) | 1 |
| 8 | Unsupported premise | 1 |

## Per-query results

| query_id | expected | P@1 | R@1 | RR | DRM | faithful | claims | tags |
|---|---|---|---|---|---|---|---|
| stellar-1 | nda_stellar_systems | 0.0% | 0.0% | 25.0% | 80.0% | 100.0% | 100.0% | - |
| stellar-2 | nda_stellar_systems | 0.0% | 0.0% | 20.0% | 80.0% | 100.0% | 100.0% | - |
| stellar-3 | nda_stellar_systems | 100.0% | 100.0% | 100.0% | 60.0% | 100.0% | 100.0% | - |
| stellar-4 | nda_stellar_systems | 0.0% | 0.0% | 50.0% | 70.0% | 100.0% | 100.0% | - |
| helix-1 | nda_helix_dynamics | 0.0% | 0.0% | 25.0% | 50.0% | 100.0% | 100.0% | - |
| helix-2 | nda_helix_dynamics | 100.0% | 100.0% | 100.0% | 60.0% | 100.0% | 100.0% | - |
| helix-3 | nda_helix_dynamics | 100.0% | 100.0% | 100.0% | 70.0% | 100.0% | 100.0% | - |
| helix-4 | nda_helix_dynamics | 100.0% | 100.0% | 100.0% | 60.0% | 100.0% | 100.0% | - |
| orion-1 | nda_orion_group | 0.0% | 0.0% | 50.0% | 70.0% | 100.0% | 100.0% | - |
| orion-2 | nda_orion_group | 0.0% | 0.0% | 33.3% | 70.0% | 100.0% | 100.0% | - |
| orion-3 | nda_orion_group | 100.0% | 100.0% | 100.0% | 80.0% | 100.0% | 100.0% | - |
| orion-4 | nda_orion_group | 100.0% | 100.0% | 100.0% | 70.0% | 100.0% | 100.0% | - |
| pinnacle-1 | nda_pinnacle | 100.0% | 100.0% | 100.0% | 70.0% | 100.0% | 100.0% | - |
| pinnacle-2 | nda_pinnacle | 0.0% | 0.0% | 10.0% | 90.0% | 100.0% | 100.0% | - |
| pinnacle-3 | nda_pinnacle | 100.0% | 100.0% | 100.0% | 70.0% | 100.0% | 100.0% | - |
| pinnacle-4 | nda_pinnacle | 100.0% | 100.0% | 100.0% | 60.0% | 100.0% | 100.0% | - |
| zenith-1 | nda_zenith | 100.0% | 100.0% | 100.0% | 50.0% | 100.0% | 100.0% | - |
| zenith-2 | nda_zenith | 0.0% | 0.0% | 50.0% | 60.0% | 100.0% | 100.0% | - |
| zenith-3 | nda_zenith | 100.0% | 100.0% | 100.0% | 80.0% | 0.0% | 0.0% | interpretability_failure,hallucination |
| zenith-4 | nda_zenith | 100.0% | 100.0% | 100.0% | 60.0% | 0.0% | 0.0% | hallucination,efficiency_failure |
| crestwood-1 | nda_crestwood | 0.0% | 0.0% | 11.1% | 80.0% | 0.0% | 0.0% | context_integration_failure,hallucination |
| crestwood-2 | nda_crestwood | 100.0% | 100.0% | 100.0% | 80.0% | 0.0% | 0.0% | hallucination,unsupported_premise |
| crestwood-3 | nda_crestwood | 0.0% | 0.0% | 50.0% | 80.0% | 0.0% | 0.0% | generation_failure |
| crestwood-4 | nda_crestwood | 100.0% | 100.0% | 100.0% | 70.0% | 0.0% | 0.0% | citation_failure,hallucination |

## Claim-level detail

### zenith-3

- `claim-1` **unsupported**: *Confidential information is protected as stated in the cited provision.*
  - the retrieved evidence does not contain the claim's content (token coverage 33% < 60%)

### zenith-4

- `claim-1` **unsupported**: *Confidential information is protected as stated in the cited provision.*
  - the retrieved evidence does not contain the claim's content (token coverage 33% < 60%)

### crestwood-1

- `claim-1` **unsupported**: *The governing obligations are stated in the cited source.*
  - the retrieved evidence does not contain the claim's content (token coverage 0% < 60%)

### crestwood-2

- `claim-1` **unsupported**: *Your premise that the agreement has no term limit is not supported by the evidence.*
  - the retrieved evidence does not contain the claim's content (token coverage 14% < 60%)

### crestwood-3

- `claim-1` **unsupported**: *I don't know.*
  - the retrieved evidence does not contain the claim's content (token coverage 0% < 60%)

### crestwood-4

- `claim-1` **contradicted**: *The parties are legally required to disclose their trade secrets to competitors within ninety days.*
  - the claim asserts numeric detail(s) ['90'] that conflict with the numbers found in the evidence (['2', '6'])

## Reproduction metadata

- version: 1
- k_values: [1, 3, 5, 10], drm_depth: 10
- scoring: faithfulness/answer-relevance/context-relevance used the deterministic lexical heuristics (no LLM judge injected)
- config: `{"enable_claim_check": true, "enable_failure_tags": true, "enable_ragas_style": true, "k_values": [1, 3, 5, 10]}`
- python: 3.14.4 (Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43)
- Milestone 7 full evaluation harness (eval/run_eval.py).
- Relevance is document-level: a retrieved chunk is relevant iff its source document equals the query's expected_document.
- The Milestone 4 DRM evaluation files (eval/results/drm_results.json, eval/results/drm_report.md) are intentionally not modified by this harness.
