# Business Entity Resolution Pipeline

## V2 full-corpus training and validation (experimental)

The v2 experiment processes every training Source 1, 2, and 3 record. It uses a
frozen 70/15/15 Source 1 split, country-aware blocking, a separate candidate
allowance for each target source, and the challenge's per-entity macro F0.5
metric. Training labels are used only for supervised fitting and evaluation;
they are never added to retrieved candidate lists. Test records are not read.

From the repository root, with the pinned dependencies installed:

```powershell
.\.venv\Scripts\python.exe code\business_entity_resolution\src\test_v2.py
.\.venv\Scripts\python.exe code\business_entity_resolution\src\train_v2.py
.\.venv\Scripts\python.exe code\business_entity_resolution\src\report_v2.py
```

The training files default to `student_resource/dataset/train/`. Use
`--train-dir` and `--artifacts-dir` to change those paths. The run requires at
least 3.5 GiB available RAM and 25 GiB free disk at its start. Its DuckDB work
database and feature arrays stay under `artifacts/v2/work/`; test inference uses
15 DuckDB threads, a 3 GiB memory limit, and a 12 GiB spill limit. Test
inference checks for 20 GiB free disk and puts its database, intermediate
arrays, and spill files in a unique process-specific folder under the system
temporary directory. Its final outputs stay under
`artifacts/v2/test_output/`. Completed training import, blocking, candidate
partitions, and extracted features can be reused after an interrupted run.
Allow substantial CPU time for the full corpus.

The finished run writes `artifacts/v2/matcher_model.joblib`,
`model_config.json`, `training_metrics.json`, and `split_assignments.parquet`.
After training completes, `report_v2.py` reads `training_metrics.json` and
writes `artifacts/v2/validation_report.md`, summarizing candidate recall and
oracle score, baseline comparison, selected features and threshold, and final
held-out validation slices. France is absent from training and validation, so
the report calls out that limitation. The
existing `src/infer_pipeline.py` and its artifacts remain the current submission
path; v2 has not been promoted to inference.

To run the frozen cap-24/20-feature validation from completed v2 work
checkpoints without repeating feature selection, use:

```powershell
.\.venv\Scripts\python.exe code\business_entity_resolution\src\train_v2.py --frozen-baseline
.\.venv\Scripts\python.exe code\business_entity_resolution\src\report_v2.py
```

After validation completes, `src/infer_v2.py` can generate separate test-only
outputs under `artifacts/v2/test_output/` and run the challenge validator. It
reads test source records only, uses the same v2 blocking and feature functions,
and does not overwrite `output/` or read test labels.

The sections below document the earlier submission path.

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
