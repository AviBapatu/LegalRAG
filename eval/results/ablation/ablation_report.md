# Embedding-Model Ablation

Same query set (24 queries) and corpus (6 documents), retrieved with SAC on at depth 10 across 2 embedding model(s). Generated: 2026-09-06T10:26:04.799398+00:00.

| Model | P@1 | P@5 | R@1 | R@5 | MRR | MAP | DRM |
|---|---|---|---|---|---|---|---|
| smoke-embed-a | 58.3% | 36.7% | 58.3% | 91.7% | 71.9% | 71.9% | 69.6% |
| smoke-embed-b | 58.3% | 36.7% | 58.3% | 91.7% | 71.9% | 71.9% | 69.6% |

## Per-model details

### smoke-embed-a

- P@1: **58.3%**, R@1: **58.3%**
- P@3: **40.3%**, R@3: **79.2%**
- P@5: **36.7%**, R@5: **91.7%**
- P@10: **30.4%**, R@10: **100.0%**
- MRR: **71.9%**, MAP: **71.9%**, DRM: **69.6%** (167 of 240 chunks mismatch)

### smoke-embed-b

- P@1: **58.3%**, R@1: **58.3%**
- P@3: **40.3%**, R@3: **79.2%**
- P@5: **36.7%**, R@5: **91.7%**
- P@10: **30.4%**, R@10: **100.0%**
- MRR: **71.9%**, MAP: **71.9%**, DRM: **69.6%** (167 of 240 chunks mismatch)

## Reproduction metadata

- version: 1
- k_values: [1, 3, 5, 10]
- use_sac: True
- python: 3.14.4 (Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43)
- config: `{"chunking_strategy": "pattern", "embedding_model": "BAAI/bge-large-en-v1.5", "mmr_lambda": 0.7, "retrieval_top_k": 10, "rrf_k": 60, "use_mmr": true, "use_sac": true, "use_structural_boost": true}`

Corpus files (sha256):

| file | bytes | sha256 |
|---|---|---|
| nda_crestwood.txt | 1024 | `df191b471c9bf60fb89684653f73c27aecc53e521deebc2507acdb8636e8a1e7` |
| nda_helix_dynamics.txt | 995 | `801180e04a4eb3fb32f268a4b80cefe13a49ea637a5511663d7cd8a2d0b5f4cc` |
| nda_orion_group.txt | 993 | `12d3ef89be37373b31de275c7978c8833a63b1051378668f9cae82d31acbdb5a` |
| nda_pinnacle.txt | 1018 | `89c7ffdf94e0a24c1a209d760d4b828648e8db6b786a8398e2e27f70f463117b` |
| nda_stellar_systems.txt | 1006 | `0deb49b8e0786de7af2558c3def78eb78886673375a56a628c7e824757e44cec` |
| nda_zenith.txt | 998 | `4bd54b07510dfad9b525ea981577898f465df1baa502e2a1963f4afccb54e755` |
