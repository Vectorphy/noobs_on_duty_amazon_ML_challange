# Architecture & Technical Audit: Amazon ML Challenge 2026 Entity Resolution

## Executive Summary

This report outlines the structural, methodological, and evolutionary differences between our local entity resolution implementation and the reference solution detailed in Arnesh Banerjee's 24-hour competition write-up.

While the blog post describes an unconstrained "competition-style" solution heavily reliant on massive neural embeddings (DeBERTa cross-encoders, e5-small bi-encoders) and post-hoc group selection, our local implementation is engineered as a constrained, production-grade tabular pipeline. The local repository enforces strict memory constraints, deterministic and statistical blocking (DuckDB-backed), continuous distance feature engineering, asymmetric metric optimization, and a distribution-free conformal uncertainty layer.

Recently, our local architecture has evolved (V3) to integrate tree-based speed-ups via XGBoost (CUDA) and experimental GPU-backed BM25 retrieval, pushing closer to the blog post’s high-recall retrieval mechanisms while retaining a strict, tabular-first efficiency profile.

## 1. Architectural Paradigms: Blog Post vs. Initial Local (V1/V2)

### The Blog Post Implementation
**Strategy:** Deep Semantic Matching with Cascaded Refinement.
- **Philosophy:** Treat the problem as an asymmetric NLP task, leveraging large pre-trained Language Models (LLMs) to catch fuzzy real-world noise (e.g., semantic synonyms, translated acronyms) and relying on extensive computational resources.
- **Blocking:** Relies heavily on a fine-tuned multilingual E5 bi-encoder to generate dense embeddings for records, searching them using exact GPU kNN (both forward and reverse).
- **Matching:** Utilizes heavy DeBERTa v3 cross-encoders running on textual pairs. The stacker (LightGBM) incorporates complex "group features" that consider the entire candidate set globally (e.g., checking if a record agrees with other confident records of the same entity).
- **Domain Adaptation:** The unseen country (France) is handled implicitly by the pre-trained multilingual embeddings and explicitly mined normalization mappings.

### The Initial Local Implementation (V1/V2)
**Strategy:** Deterministic Tabular ML with Conformal Hardening.
- **Philosophy:** Treat the problem as a scalable data-engineering and statistical learning task. It minimizes deep-learning overhead by converting text into dense continuous feature matrices evaluated by gradient-boosted trees.
- **Blocking:** Multi-index DuckDB-backed pipeline. Uses deterministic grouping (Country + Zip/PIN), Phonetic mapping (Double Metaphone), and Lexical MinHash LSH (3-gram permutations).
- **Matching:** A pure ML matcher using a LightGBM classifier. It heavily relies on rapidfuzz distance metrics and token intersections, rather than semantic cross-encoders. Uses Asymmetric Precision-Heavy optimization to heavily penalize false merges.
- **Domain Adaptation:** Explicit, mathematically sound conformal prediction framework. It uses Covariate Shift likelihood ratio weighting (Tibshirani et al.) to adapt the calibration distribution from seen domains (US/India) to unseen domains (France) with guaranteed coverage properties.

## 2. Deep Dive: Component Differences

### Blocking & Candidate Generation
| Feature | Blog Post (`/tmp/repo`) | Local Repository |
| :--- | :--- | :--- |
| **Primary Mechanism** | Dense Vector Retrieval (fine-tuned `intfloat/multilingual-e5-small`) | Sparse/Deterministic Indices (DuckDB SQL) |
| **Search Space** | Exact GPU kNN (forward & reverse) | Zip Codes, Phonetic Keys, MinHash LSH |
| **Pruning** | Learned LightGBM pruner (cuts candidates to max 15) | Database-level row caps |
| **Memory Footprint**| High (requires GPU holding embeddings) | Low (Disk-backed DuckDB, chunked processing) |

### Matching Models
| Feature | Blog Post (`/tmp/repo`) | Local Repository |
| :--- | :--- | :--- |
| **Core Models** | Two `microsoft/mdeberta-v3-base` cross-encoders + LightGBM stacker | Single asymmetric LightGBM / XGBoost classifier |
| **Feature Extraction**| Raw/Normalized Text fed to Transformer; Rapidfuzz features in stacker | 27-dimensional Continuous feature engineering (Rapidfuzz, Jaccard, Numeric gaps) |
| **Decision Logic** | "One-to-one" expected F0.5 bipartite assignment (global graph selection) | Threshold tuning (Macro F0.5 aligned) calibrated with Platt Scaling / Mondrian Conformal Engine |

### Domain Adaptation (The "France" Problem)
- **Blog Post:** Employs explicitly mined dictionaries (e.g., French business words, abbreviations) and trusts the multilingual E5 model to handle the distribution shift natively.
- **Local:** Employs a **Covariate Shift Weighter** (discriminative logistic regression) to assign likelihood weights $w(x) = P(target) / P(source)$ and scales confidence bounds. This mathematically guarantees non-conformity coverage on the unseen French distribution without requiring a French-specific neural encoder.

## 3. Evolution of the Local Architecture (Recent Changes)

The local repository has recently undergone a major evolution toward the "V3" pipeline, incorporating aggressive performance improvements while strictly maintaining its non-neural philosophy.

### V3 Feature Expansion (`190fcc5`)
- Expanded the feature set to a **27-feature pipeline**.
- Introduced the "C3 guard" (a structural validation checkpoint for split boundaries).

### XGBoost & CUDA Acceleration (`107d142` & `3c58dc4`)
- **Integration of XGBoost:** While V1/V2 relied heavily on LightGBM, the V3 pipeline has fully ported the matching backend to XGBoost (`train_v3_xgb.py`, `infer_v3_xgb.py`).
- **Hardware Acceleration:** The XGBoost implementation leverages CUDA (`device=cuda` via `tree_method=hist`). This allows tree-building and inference to utilize GPU parallelism without moving to deep-learning models.

### BM25s Retrieval Pilot (BM25 Colab Pipeline)
- **New Blocking Paradigm:** Bridging the gap between the blog post's high-recall embedding search and the V2 strict deterministic blocking, the repository is actively piloting character-trigram BM25 retrieval.
- **Implementation:** Uses `bm25s` (a numba-compiled BM25 implementation) and `cupy` for sparse matrix multiplication on GPUs. This aims to match the blog post's recall rates while operating orders of magnitude faster (26s vs 429s on 40,000 queries) and remaining purely lexical.

## Conclusion

The blog post solution represents a highly unconstrained, compute-intensive approach optimizing strictly for the leaderboard via state-of-the-art NLP models (DeBERTa, e5-small).

Conversely, our local repository represents a production-hardened system. While initially relying on traditional data-engineering constructs (DuckDB, MinHash, LightGBM, Conformal Math), the recent V3 branches represent a strategic pivot: integrating GPU acceleration (XGBoost CUDA, CuPy BM25) to achieve the high recall and throughput of the neural solution, without inheriting its massive operational complexity and latency.
