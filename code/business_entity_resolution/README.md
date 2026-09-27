# Business Entity Resolution Pipeline

## V2 full-corpus training and validation (experimental)

The v2 experiment processes every training Source 1, 2, and 3 record. It uses a
frozen 70/15/15 Source 1 split, country-aware blocking, a separate candidate
allowance for each target source, and the challenge's per-entity macro F0.5
metric. Training labels are used only for supervised fitting and evaluation;
they are never added to retrieved candidate lists. Test records are not read.

### Run all three estimators on Colab

Put the repository at `MyDrive/amazon-ml` with the supplied data under
`student_resource/dataset/{train,test}`, or edit the notebook path cell. Run
`business_entity_resolution_v2_pipeline.ipynb` from top to bottom. It mounts
Drive, installs both pinned requirement files, checks the small fixtures, and
trains LightGBM, XGBoost, and CatBoost sequentially. Data import, candidates,
split, and extracted feature arrays are reused across those runs. Each model's
checkpoint and validation outputs are kept separate. The primary comparison is
held-out per-entity macro F₀.₅, not accuracy. Optional test inference also uses
separate model output folders and does not read test labels.

Colab artifacts go under `artifacts/v2_colab/`: LightGBM is at the root, while
XGBoost and CatBoost are under `models/xgboost/` and `models/catboost/`.
`model_comparison.csv` and `model_comparison.md` summarize the three validation
results. Shared preprocessing state is under `work/`. DuckDB, LightGBM,
XGBoost, CatBoost, and RapidFuzz use all CPU cores visible to the runtime.

The notebook currently keeps resumable work and outputs on mounted Google
Drive. This can make database and temporary-file operations slower than local
Colab storage, and the training preflight expects 25 GiB free at the artifact
location on a fresh run. The three-model run repeats model selection and fits
both the selected model and baseline for each estimator, so allow substantial
runtime. The notebook has not been validated with a complete Colab run.

From the repository root, with the pinned dependencies installed:

```powershell
.\.venv\Scripts\python.exe code\business_entity_resolution\src\test_v2.py
.\.venv\Scripts\python.exe code\business_entity_resolution\src\train_v2.py
.\.venv\Scripts\python.exe code\business_entity_resolution\src\report_v2.py
```

The training files default to `student_resource/dataset/train/`. Use
`--train-dir` and `--artifacts-dir` to change those paths. The default model is
LightGBM; it remains at the existing `artifacts/v2/` path. XGBoost and CatBoost
are optional alternatives with isolated artifacts under `artifacts/v2/models/`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r code\business_entity_resolution\requirements-models.txt
.\.venv\Scripts\python.exe code\business_entity_resolution\src\train_v2.py --model xgboost
.\.venv\Scripts\python.exe code\business_entity_resolution\src\report_v2.py --model xgboost
.\.venv\Scripts\python.exe code\business_entity_resolution\src\infer_v2.py --model xgboost
```

Replace `xgboost` with `catboost` to select CatBoost. The optional package set
is separate from the baseline environment. Verify that wheels are available
for the Python version used before installing; if not, use a supported Python
environment. LightGBM remains the default for existing notebook and CLI runs.

### Experimental BM25S retrieval pilot

The label-blind retrieval pilot compares the existing SciPy prototype with an
optional BM25S/Numba backend. It does not change the submission inference path.
Install the isolated dependencies and run the pilot from the repository root:

```powershell
uv pip install --python .venv\Scripts\python.exe -r code\business_entity_resolution\requirements-retrieval.txt
uv run --python .venv\Scripts\python.exe code\business_entity_resolution\src\retrieval_v2.py --backend bm25s
```

Pilot outputs and findings are under
`code/business_entity_resolution/artifacts/v2/tuning/retrieval/`. BM25S was
about 16.5× faster than the SciPy prototype on the saved sample, with similar
retrieval recall. It still needs a complete development candidate oracle before
use in model training or inference.

The run requires at
least 3.5 GiB available RAM and 25 GiB free disk at its start. Its DuckDB work
database and feature arrays stay under `artifacts/v2/work/`; test inference uses
15 DuckDB threads, a 3 GiB memory limit, and a 12 GiB spill limit. Test
inference checks for 20 GiB free disk and stores its database and resumable
scoring arrays under the selected model's `test_work/` folder; DuckDB spill
files remain in the system temporary directory. Its final outputs stay under
`artifacts/v2/test_output/` for LightGBM and the corresponding model folder for
alternatives. Training feature extraction and model fitting write checkpoints;
inference scoring flushes feature/probability arrays and progress metadata
every 500,000 pairs. Checkpoints are reused only when the data/model signature
matches. Allow substantial CPU time for the full corpus.

LightGBM training writes TensorBoard event files under
`artifacts/v2/tensorboard/lightgbm/`. If TensorBoard is installed, view them
with `tensorboard --logdir code/business_entity_resolution/artifacts/v2/tensorboard`.
The console also reports training metrics every 25 boosting rounds and feature
or inference scoring progress, throughput, and estimated time remaining.

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
