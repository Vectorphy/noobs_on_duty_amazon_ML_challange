# V2 experiment guide

This guide explains the experimental training and validation pipeline and how to continue it. The v1 submission pipeline remains the current submission path. V2 artifacts and test predictions are separate.

## What the pipeline does

1. `train_v2.py` imports all supplied training sources and ground truth into a disk-backed DuckDB database. It assigns Source 1 entities to fixed train, development, and final-validation splits. The split seed is 42.
2. It builds country-specific blocking keys and source-separated candidates. Ground-truth links are never inserted into candidate lists.
3. It selects bounded training pairs, extracts the ordered feature set, tunes the candidate cap and grouped feature ablations on development entities, and fits a model. The final-validation split is held out from feature and threshold selection.
4. `report_v2.py` summarizes candidate recall, candidate oracle, per-entity macro F₀.₅, and validation slices.
5. `infer_v2.py` creates model-specific test outputs under the selected model's artifact folder. Test inference is separate from validation and uses no test labels.

LightGBM is the default. XGBoost and CatBoost adapters are also implemented. The Colab notebook runs them with separate model files, checkpoints, training metrics, and predictions, then writes one comparison report. Install their optional dependencies from `requirements-models.txt`.

## Resume and tune LightGBM

From the repository root, using the existing `.venv`:

```powershell
uv run --python .venv\Scripts\python.exe code\business_entity_resolution\src\test_v2.py
uv run --python .venv\Scripts\python.exe code\business_entity_resolution\src\train_v2.py --frozen-baseline
uv run --python .venv\Scripts\python.exe code\business_entity_resolution\src\tune_lightgbm_v2.py
uv run --python .venv\Scripts\python.exe code\business_entity_resolution\src\report_v2.py
```

The tuning runner reuses the imported database, training pair selection, and cached development features when their signatures match. Its trials and checkpoints are under `artifacts/v2/tuning/lightgbm/`. It compares candidate model variants on development data and does not score the final-validation split. Keep the LightGBM TensorBoard event directory at `artifacts/v2/tensorboard/lightgbm/` when opening TensorBoard.

## Compare estimators

After retrieval and candidate settings are frozen, run the Colab notebook from top to bottom to compare LightGBM, XGBoost, and CatBoost on the same split and candidate pool. Install `requirements-models.txt`; each model writes to a separate folder under `artifacts/v2_colab/`. Compare held-out per-entity macro F₀.₅ and source/country/match-count slices. Do not use final-validation results to tune thresholds or select features repeatedly.

## Retrieval experiment status

`src/retrieval_v2.py --backend bm25s` is an optional, development-only character-trigram BM25 retrieval pilot. It does not replace the current blocker, training pipeline, or inference. Install `requirements-retrieval.txt` first. The saved pilot used label-blind hash samples from India and the US; ground truth was joined only afterward to measure recall. See `artifacts/v2/tuning/retrieval/bm25_pilot_findings.md` for exact recall, candidate counts, and runtime.

The BM25S/Numba pilot took about 26 seconds for 40,000 sampled queries, compared with about 429 seconds for the SciPy implementation, while retaining similar recall. It has not yet passed the complete development candidate-oracle gate. Do not mix these pilot results with the existing model's validation score, and do not promote the retriever to inference until full-development candidate recall, per-entity oracle, and resource use are recorded.

## Artifact and challenge rules

Generated caches, models, checkpoints, and test outputs stay in `artifacts/v2/` or `artifacts/v2_colab/` and are ignored by Git. Small source-controlled explainers and retrieval summary reports are retained. Use only the supplied challenge data; test records may be analyzed descriptively or scored for submission, but test labels and external identity sources must not enter training or selection.
