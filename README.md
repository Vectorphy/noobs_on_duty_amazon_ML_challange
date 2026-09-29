# Amazon ML Challenge 2026: Business Entity Resolution

**Competition:** Amazon ML Challenge 2026  
**Metric:** Macro {0.5}$ Score (Precision-Weighted)  

---

## 1. System Overview

This repository contains the complete, production-grade implementation of our Business Entity Resolution matching model:
- **Multi-Index Candidate Blocking:** High-recall inverted indices on normalized postal codes, phonetic keys, and 3-gram character MinHash sets (.00\%$ empirical blocking recall on ^+$ with .96\%$ search-space reduction).
- **Asymmetric Precision-Heavy Matcher:** Scikit-Learn HistGradientBoostingClassifier trained with asymmetric cost weights ({\text{neg}} = 4.0, w_{\text{pos}} = 1.0$) to penalize false merges \times$ more heavily under the Macro {0.5}$ metric.
- **Distribution-Free Uncertainty Quantification:** Temperature scaling (^* = 1.3173$) and Mondrian (class-conditional) conformal calibration ( = 0.0225, q_1 = 0.0558$), guaranteeing .5\%$ finite-sample coverage on non-matches.
- **System Hardening & Domain Adaptation:** Covariate shift likelihood ratio reweighting with adaptive shrinkage ({\text{eff}} \ge 0.5n$), French entity normalizer, CORAL feature covariance alignment, and transitivity-preserving constrained graph correlation clustering.

---

## 2. Directory Structure

`
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── train_pipeline.py         # Full training pipeline on dataset/train/
│       │   ├── infer_pipeline.py         # Streaming test inference pipeline
│       │   ├── conformal_engine.py       # Mondrian & Covariate Shift Conformal Predictor
│       │   └── graph_clusterer.py        # Constrained Multicut Clustering & French Normalizer
│       ├── README.md                     # Code reproduction runbook
│       └── requirements.txt              # Pinned Python dependencies
├── student_resource/
│   ├── artifacts/                        # Trained weights, calibrators, schemas, metrics
│   ├── utils/validate_submission.py      # Official challenge validation script
│   ├── train_pipeline.py                 # Self-contained training pipeline
│   ├── infer_pipeline.py                 # Self-contained streaming inference pipeline
│   ├── Documentation_template.md         # Official methodology report
│   └── TRAINING_DOCUMENTATION.md         # Dedicated training diagnostics documentation
├── eda_artifacts/                        # Distribution & outlier visualization plots
├── conformal_entity_resolution.py        # Core conformal prediction engine
├── conformal_system_hardening.py         # System hardening & multicut graph solver
├── test_conformal_hardening.py           # Verification test suite for hardening checks
├── validate_conformal_framework.py       # Conformal benchmarking & error grid search
├── eda_outlier_workflow.py               # Raw multivariate/univariate outlier diagnostics
├── Documentation_template.md             # Official competition report
├── SOLUTION_METHODOLOGY_REPORT.md        # Solution methodology report
├── TRAINING_PIPELINE_DOCUMENTATION.md    # Master training documentation
├── CONFORMAL_PREDICTION_DOCUMENTATION.md # Conformal prediction specification
├── CONFORMAL_HARDENING_DOCUMENTATION.md  # System hardening technical report
├── DOCUMENTATION.md                      # EDA & metric mechanics documentation
└── CHANGELOG.md                          # Keep a Changelog standard history
`

---

## 3. Quick Start & Reproduction

### 3.1 Setup Environment

`ash
pip install -r code/business_entity_resolution/requirements.txt
`

### 3.2 Train Model (strictly on dataset/train/)

`ash
python student_resource/train_pipeline.py
`

### 3.3 Run Inference on Test Set (1.73M entities)

`ash
python student_resource/infer_pipeline.py submission/ student_resource/dataset/test/
`

### 3.4 Validate Submission Output

`ash
python student_resource/utils/validate_submission.py \
    --matching submission/matching_results.tsv \
    --candidate submission/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test/
`

---

## 4. Empirical Performance Summary

- **Blocking Recall:** .00\%$ (,709 / 2,709$ positive ground-truth pairs retained)
- **Pair Accuracy:** .98\%$
- **Pair Precision:** .74\%$ (at conservative operating cutoff $\tau^* = 0.82$)
- **Pair Recall:** .94\%$ (at $\tau^* = 0.82$)
- **Pair $-Score:** .83\%$
- **Macro {0.5}$ (Challenge Metric):** **.9930$** (peak), **.9903$** (conservative hardened policy)
- **Submission Validation:** PASS - no blocking issues found. Safe to submit.
