# DRM Evaluation — SAC off vs. SAC on

Document Retrieval Mismatch (DRM) before/after report for Summary-Augmented Chunking (SAC). Embedding model: `BAAI/bge-large-en-v1.5`. Queries: 24 over 6 documents. Retrieval depth: 10. Generated: 2026-09-06T08:58:41.292028+00:00.

## Comparison table

| Metric | SAC off | SAC on | Delta |
|---|---|---|---|
| Top-1 accuracy | 83.3% | 91.7% | 8.3% |
| Top-3 accuracy | 100.0% | 95.8% | -4.2% |
| Document accuracy@1 | 83.3% | 91.7% | 8.3% |
| Document accuracy@3 | 100.0% | 95.8% | -4.2% |
| Document accuracy@5 | 100.0% | 100.0% | 0.0% |
| Document accuracy@10 | 100.0% | 100.0% | 0.0% |
| Retrieval mismatch count (top-k) | 152 | 123 | -29 |
| DRM fraction (wrong-document chunks in top-k) | 63.3% | 51.2% | -12.1% |

Delta is `SAC on - SAC off`. A positive delta is an improvement for the accuracy metrics; a negative delta is an improvement for mismatch count and DRM fraction.

## sac_off (use_sac=False)

- Top-1 accuracy: **83.3%**
- Top-3 accuracy: **100.0%**
- Document accuracy@10: **100.0%**
- Retrieval mismatch count: **152** of 240 chunks
- DRM fraction: **63.3%**

| query_id | expected | top-3 retrieved | top-3 hit |
|---|---|---|---|
| stellar-1 | nda_stellar_systems | nda_stellar_systems, nda_stellar_systems, nda_stellar_systems | yes |
| stellar-2 | nda_stellar_systems | nda_stellar_systems, nda_stellar_systems, nda_stellar_systems | yes |
| stellar-3 | nda_stellar_systems | nda_zenith, nda_stellar_systems, nda_stellar_systems | yes |
| stellar-4 | nda_stellar_systems | nda_stellar_systems, nda_zenith, nda_stellar_systems | yes |
| helix-1 | nda_helix_dynamics | nda_helix_dynamics, nda_helix_dynamics, nda_orion_group | yes |
| helix-2 | nda_helix_dynamics | nda_helix_dynamics, nda_helix_dynamics, nda_zenith | yes |
| helix-3 | nda_helix_dynamics | nda_zenith, nda_helix_dynamics, nda_helix_dynamics | yes |
| helix-4 | nda_helix_dynamics | nda_helix_dynamics, nda_zenith, nda_helix_dynamics | yes |
| orion-1 | nda_orion_group | nda_orion_group, nda_orion_group, nda_orion_group | yes |
| orion-2 | nda_orion_group | nda_orion_group, nda_orion_group, nda_orion_group | yes |
| orion-3 | nda_orion_group | nda_zenith, nda_orion_group, nda_orion_group | yes |
| orion-4 | nda_orion_group | nda_orion_group, nda_zenith, nda_orion_group | yes |
| pinnacle-1 | nda_pinnacle | nda_pinnacle, nda_pinnacle, nda_pinnacle | yes |
| pinnacle-2 | nda_pinnacle | nda_pinnacle, nda_pinnacle, nda_pinnacle | yes |
| pinnacle-3 | nda_pinnacle | nda_pinnacle, nda_zenith, nda_pinnacle | yes |
| pinnacle-4 | nda_pinnacle | nda_pinnacle, nda_pinnacle, nda_pinnacle | yes |
| zenith-1 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| zenith-2 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| zenith-3 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| zenith-4 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| crestwood-1 | nda_crestwood | nda_crestwood, nda_crestwood, nda_crestwood | yes |
| crestwood-2 | nda_crestwood | nda_crestwood, nda_crestwood, nda_crestwood | yes |
| crestwood-3 | nda_crestwood | nda_zenith, nda_crestwood, nda_crestwood | yes |
| crestwood-4 | nda_crestwood | nda_crestwood, nda_crestwood, nda_zenith | yes |

## sac_on (use_sac=True)

- Top-1 accuracy: **91.7%**
- Top-3 accuracy: **95.8%**
- Document accuracy@10: **100.0%**
- Retrieval mismatch count: **123** of 240 chunks
- DRM fraction: **51.2%**

| query_id | expected | top-3 retrieved | top-3 hit |
|---|---|---|---|
| stellar-1 | nda_stellar_systems | nda_stellar_systems, nda_orion_group, nda_pinnacle | yes |
| stellar-2 | nda_stellar_systems | nda_stellar_systems, nda_stellar_systems, nda_stellar_systems | yes |
| stellar-3 | nda_stellar_systems | nda_zenith, nda_pinnacle, nda_crestwood | no |
| stellar-4 | nda_stellar_systems | nda_stellar_systems, nda_stellar_systems, nda_stellar_systems | yes |
| helix-1 | nda_helix_dynamics | nda_helix_dynamics, nda_zenith, nda_pinnacle | yes |
| helix-2 | nda_helix_dynamics | nda_helix_dynamics, nda_zenith, nda_helix_dynamics | yes |
| helix-3 | nda_helix_dynamics | nda_zenith, nda_helix_dynamics, nda_crestwood | yes |
| helix-4 | nda_helix_dynamics | nda_helix_dynamics, nda_helix_dynamics, nda_helix_dynamics | yes |
| orion-1 | nda_orion_group | nda_orion_group, nda_pinnacle, nda_crestwood | yes |
| orion-2 | nda_orion_group | nda_orion_group, nda_orion_group, nda_crestwood | yes |
| orion-3 | nda_orion_group | nda_orion_group, nda_zenith, nda_crestwood | yes |
| orion-4 | nda_orion_group | nda_orion_group, nda_orion_group, nda_orion_group | yes |
| pinnacle-1 | nda_pinnacle | nda_pinnacle, nda_zenith, nda_helix_dynamics | yes |
| pinnacle-2 | nda_pinnacle | nda_pinnacle, nda_pinnacle, nda_stellar_systems | yes |
| pinnacle-3 | nda_pinnacle | nda_pinnacle, nda_zenith, nda_pinnacle | yes |
| pinnacle-4 | nda_pinnacle | nda_pinnacle, nda_pinnacle, nda_pinnacle | yes |
| zenith-1 | nda_zenith | nda_zenith, nda_pinnacle, nda_zenith | yes |
| zenith-2 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| zenith-3 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| zenith-4 | nda_zenith | nda_zenith, nda_zenith, nda_zenith | yes |
| crestwood-1 | nda_crestwood | nda_crestwood, nda_pinnacle, nda_helix_dynamics | yes |
| crestwood-2 | nda_crestwood | nda_crestwood, nda_crestwood, nda_helix_dynamics | yes |
| crestwood-3 | nda_crestwood | nda_crestwood, nda_zenith, nda_pinnacle | yes |
| crestwood-4 | nda_crestwood | nda_crestwood, nda_crestwood, nda_crestwood | yes |

## Reproduction metadata

- eval_version: 1
- config: `{"chunking_strategy": "pattern", "embedding_model": "BAAI/bge-large-en-v1.5", "mmr_lambda": 0.7, "retrieval_top_k": 10, "rrf_k": 60, "sac_summary_max_chars": 150, "structural_boost_strength": 0.1, "use_mmr": true, "use_structural_boost": true}`
- python: 3.14.4 (Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43)

Corpus files (sha256):

| file | bytes | sha256 |
|---|---|---|
| nda_crestwood.txt | 1024 | `df191b471c9bf60fb89684653f73c27aecc53e521deebc2507acdb8636e8a1e7` |
| nda_helix_dynamics.txt | 995 | `801180e04a4eb3fb32f268a4b80cefe13a49ea637a5511663d7cd8a2d0b5f4cc` |
| nda_orion_group.txt | 993 | `12d3ef89be37373b31de275c7978c8833a63b1051378668f9cae82d31acbdb5a` |
| nda_pinnacle.txt | 1018 | `89c7ffdf94e0a24c1a209d760d4b828648e8db6b786a8398e2e27f70f463117b` |
| nda_stellar_systems.txt | 1006 | `0deb49b8e0786de7af2558c3def78eb78886673375a56a628c7e824757e44cec` |
| nda_zenith.txt | 998 | `4bd54b07510dfad9b525ea981577898f465df1baa502e2a1963f4afccb54e755` |
