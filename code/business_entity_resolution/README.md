# Business Entity Resolution Pipeline

**Amazon ML Challenge 2026**  

---

## 1. Overview & Architecture

This repository contains the complete, reproducible source code for the Business Entity Resolution matching model:
- **`src/train_pipeline.py`:** End-to-end training pipeline on `dataset/train/` (multi-index blocking, 28-feature extraction, asymmetric cost-sensitive GBDT training, temperature scaling, Mondrian conformal calibration).
- **`src/infer_pipeline.py`:** Streaming test inference pipeline generating `matching_results.tsv` and `candidate_pairs.tsv` over 1.73M test records.
- **`src/conformal_engine.py`:** Distribution-free Mondrian conformal prediction calibrator and covariate shift domain adapter.
- **`src/graph_clusterer.py`:** Transitivity-preserving constrained correlation clusterer and French entity normalizer.

---

## 2. Environment Setup

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy >= 1.26.0`
- `pandas >= 2.1.0`
- `scipy >= 1.11.0`
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.0`

---

## 3. Reproduction Instructions

### 3.1 Training the Model (Strictly on `dataset/train/`)

```bash
python src/train_pipeline.py
```
Outputs saved to `artifacts/`:
- `matcher_model.joblib`: Trained GBDT weights
- `temperature_scaler.joblib`: Fitted temperature parameter ($T^* = 1.3173$)
- `mondrian_calibrator.joblib`: Calibrated Mondrian quantiles ($q_0 = 0.0225, q_1 = 0.0558$)
- `feature_schema.json`: Complete feature definitions
- `training_metrics.json`: Diagnostics and validation scores

### 3.2 Running Inference on Test Dataset

```bash
python src/infer_pipeline.py output/ dataset/test/
```
Generates the two official deliverables in `output/`:
- `output/matching_results.tsv`: Scored predictions (subset of candidates)
- `output/candidate_pairs.tsv`: Final candidate set fed to model

### 3.3 Validating Submission Files

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Verifies exit code 0 (`PASS - no blocking issues found. Safe to submit`).
