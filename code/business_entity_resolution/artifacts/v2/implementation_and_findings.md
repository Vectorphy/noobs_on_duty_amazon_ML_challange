# V2 full-corpus business entity resolution: implementation and findings

**Status:** frozen full-corpus baseline and its held-out validation report are complete. Separate test inference is currently running; its output validation is pending. No test score is claimed because test labels are unavailable.

## Scope and rules

This experiment addresses the supplied Business Entity Resolution challenge data only. It uses the training Source 1, Source 2, Source 3, and ground-truth TSVs, with no external identity lookup, registry/API data, geocoding, or other augmentation. Test rows are deliberately excluded from v2 training and validation. The v2 test runner is separate and reads no test labels.

The challenge requires zero, one, or multiple target IDs for each Source 1 entity and scores per-entity macro F0.5, including true singletons. Candidate pairs are reviewed separately and must be the final set passed to inference. The required output headers and ID constraints remain those in `agents/CHALLENGE_RULES.md`.

## Motivation and initial findings

The earlier submission path was a streamed model trained on a 30,000 Source 1 sample, with 15 base pair features and separate calibration artifacts. The EDA identified full-corpus and generalization risks that motivated v2:

- Training contains 2,206,821 Source 1 rows, 5,034,616 Source 2 rows, 5,285,603 Source 3 rows, and 7,638,365 positive links. The v2 importer asserts these counts.
- Labels are many-to-many in practice: 1,964,417 Source 1 entities (89.02%) have multiple matches; 123,247 (5.58%) are true singletons; degree ranges from zero to eleven. A top-one assignment is therefore unsuitable.
- France is absent from training but is about 14.98% of test Source 1 and about 14.39% of test Source 2/3 records. Test also has a lower US share and a higher India share. France remains an unvalidated open-set condition.
- Training Source 2/3 addresses are missing for about 3.3% of rows; the EDA reports about 2.7% missing in test. Names and countries are complete.
- Sampled positives have higher name and address similarity than hard negatives, but distributions overlap. Requiring exact/shared name tokens would lose real links.
- The EDA's sampled pair metrics are diagnostics, not full-corpus rates. Its earlier model score is not a v2 result.

## Files and interfaces changed for v2

- `src/matching_v2.py` contains shared normalization, blocking keys, pair features, and the challenge F0.5 implementation. It is intended to remain identical between v2 training and eventual v2 inference.
- `src/train_v2.py` imports all training data into a resumable DuckDB database, creates frozen splits, builds candidates, extracts memory-mapped feature arrays, fits LightGBM models, evaluates thresholds, and writes v2 artifacts.
- `src/test_v2.py` is a small regression check for TSV parsing, split/import integrity, deterministic blocking, feature dimensions, candidate exclusion, and F0.5 edge cases. It uses a temporary 30-row fixture and restores the full-data row-count constants.
- `src/report_v2.py` renders `training_metrics.json` into `validation_report.md`.
- `src/infer_v2.py` is an isolated test-only runner. It writes under `artifacts/v2/test_output/`, does not overwrite the existing `output/`, checks that held-out validation exists, runs the supplied submission validator, and records `test_labels_used: false`.
- `README.md` now documents v2 reproduction, resource expectations, checkpoint reuse, frozen baseline mode, and the fact that the existing `infer_pipeline.py` remains the current submission path.
- `requirements.txt` pins the v2 runtime, including DuckDB 1.5.5, LightGBM 4.7.0, RapidFuzz 3.14.6, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, and related packages.

The v2 work is additive. Existing submission artifacts and `src/infer_pipeline.py` were not promoted or replaced by v2.

## Data import and split protocol

The importer parses TSV with headers and empty strings safely, combines Source 2 and Source 3 into a target table while retaining `source_no`, resolves every ground-truth target ID, and rejects unknown or duplicate Source 1 labels. It also rejects a target linked to more than one Source 1 entity because the split isolation assumption would need review.

For each Source 1 row, `n_true` is computed from resolved links. A deterministic seed-42 hash split is stratified by country and degree family (`zero`, `one`, `multiple`): 70% development/training, 15% development holdout, and 15% untouched final validation. The exact assignments are saved as `artifacts/v2/split_assignments.parquet`. The split is by Source 1 entity, so linked labels from a Source 1 row do not cross splits.

Completed-run evidence: full import and frozen split metadata are in `artifacts/v2/work/pipeline.duckdb`; `development_features.npy` has 12,686,102 rows at cap 24 and `train_features.npy` has 10,469,493 rows at cap 32. Full import reconciled 2,206,821 Source 1 rows, 10,320,219 targets, and 7,638,365 links. It produced 13,593,348 usable Source 1 keys and 37,662,970 target keys, then retained 107,309,617 cap-32 candidate pairs. Training selection contained 4,469,493 retrieved positives and 6,000,000 sampled negatives. The full run elapsed 1.317 hours from pipeline start.

## Candidate generation

`block_keys()` normalizes Unicode with NFKD, removes combining marks, case-folds, and tokenizes. It creates exact cleaned-name keys, name-token keys, address-number-plus-word keys, and longer address-word keys. Name and address stop-word sets remove common company/legal and address terms. Candidate joins are always country-aware, while country values remain open-set strings rather than a hard-coded US/India filter.

To control broad blocks, Source 1 key frequencies are capped by key family (`n=512`, `t=64`, `a=64`, `w=16`), and target key frequency is capped at 2,048. Candidate pairs are ranked separately for each target source using 0.65 name / 0.35 address Jaro-Winkler when both addresses exist, otherwise name similarity alone. At most 32 candidates per target source are retained. Training positives are retrieved from these candidates, and hard negatives are sampled from non-linked candidates, with rank-prioritized selection and a six-million negative ceiling. Ground-truth links are used for labels and diagnostics only; they are never inserted into candidate lists.

The training candidate builder processes 32 bounded Source 1 partitions with transactional progress markers and can resume completed partitions. DuckDB spill files and feature arrays remain below `artifacts/v2/work/`. The frozen baseline evaluates cap 24 per source; feature-selection mode considers caps 16, 24, and 32 before selecting within 0.002 of the best development score.

## Features and model

The shared pair vector has 20 float32 features:

1. fuzzy name ratio, token-sort ratio, token-set ratio, token Jaccard, and relative name-length difference;
2. fuzzy address ratio, token-sort ratio, token-set ratio, and address-token Jaccard;
3. shared numeric-code count and flag;
4. missing-address flag, non-Latin-script flag, company-name containment stem flag, and exact cleaned-name flag;
5. name trigram Jaccard, address trigram Jaccard, address-number conflict, address-length ratio, and target-source-is-S3.

The default model is LightGBM `LGBMClassifier` with 300 estimators, learning rate 0.05, 31 leaves, minimum child samples 100, max bin 63, fixed seed 42, and four workers. A memory-bounded option uses 15 leaves, max bin 31, one worker, and column-wise fitting. Training uses all retrieved development positives and bounded negatives after an intermediate sample-based feature/cap assessment; the final model is fit on the selected feature columns and the complete eligible training-pair selection. Feature ablations remove grouped v2 features and retain the smallest set within 0.002 of the best development macro F0.5, with the 15-feature baseline compared on the identical pair set.

Thresholds are searched from 0.00 to 1.00 in 0.01 increments on development data using per-Source1 macro F0.5. The selected threshold is then applied once to the untouched final validation split. The metric implementation gives 1.0 to a true no-match entity predicted empty and 0.0 when a singleton receives a false match, as required by the challenge convention.

## Tests and small-corpus findings

`test_v2.py` constructs a 30-row synthetic corpus with empty labels, two-source positives, an unrelated same-country target, and address fields. It verifies:

- strict TSV import and null-to-empty field behavior;
- expected positive-link and split counts;
- deterministic candidate generation across a rebuild;
- separate Source 2 and Source 3 candidate populations;
- exclusion of the deliberately unrelated candidate;
- 20-feature output and the S3 indicator;
- blocking-key generation; and
- F0.5 behavior for empty, singleton, and multi-match examples.

The test is a regression check, not evidence of full-corpus quality. No final v2 score should be inferred from it.

## Final frozen-baseline results

The completed frozen run uses cap 24 per target source, all 20 features, and decision threshold 0.78. The expanded 20-feature development result was 0.8475657526 macro F0.5 at threshold 0.78; the earlier sampled development selection result was 0.84732 and is provisional context only. The fair full-data 15-feature baseline, trained on the same training pairs, scored 0.8285747193 on development at threshold 0.77.

On the untouched 15% final validation split, the v2 model scored:

- Macro F0.5: **0.8473170044**
- Pair precision: **0.9439374055**
- Pair recall: **0.7445243806**

Candidate diagnostics for that validation split were:

- 12,678,786 candidate pairs, averaging 38.3020594 per Source 1;
- Source 2 recall 0.8198782236 (453,646 / 553,309);
- Source 3 recall 0.8322600606 (492,276 / 591,493);
- all true links retained for 0.6415272747 of Source 1 entities;
- candidate-stage oracle macro F0.5 0.9177194975; and
- 2,003 Source 1 entities without candidates.

Validation slices were India 0.7882653183 over 132,477 entities and US 0.8867188006 over 198,544; singleton 0.8442064265 over 18,486, exactly-one-link 0.6927476837 over 17,873, and multiple-link 0.8568876977 over 294,662. Rows with a linked target missing an address scored 0.7837510786 over 46,985 entities; the both-present-or-singleton slice scored 0.8578320273 over 284,036. The Source 1-missing-address slice has zero entities, so its score is undefined rather than zero.

The v2 artifact directory contains the resumable DuckDB database, split parquet, spill files, training/development feature arrays, metadata arrays, model configuration, metrics, and completion markers. The checkpoint markers report:

- `train_complete.json`: 10,469,493 rows, cap 32, no split (the selected training-pair feature extraction checkpoint).
- `development_complete.json`: 12,686,102 rows, cap 24, split 1 (the frozen-baseline development checkpoint).
- `pipeline.duckdb`: full-corpus import, keys, candidate tables, and progress state.

The final metrics file records 4.24 GiB available RAM and 31.38 GiB free disk at start. These resource values and the 1.317-hour elapsed time are run diagnostics, not quality guarantees.

## Test inference separation and remaining tasks

After held-out validation, `infer_v2.py` reads the v2 model/config and test source TSVs, rebuilds the same country-aware blocking and pair features, scores cap-24 candidates, writes isolated `matching_results.tsv` and `candidate_pairs.tsv`, and invokes `student_resource/utils/validate_submission.py`. It refuses to run without `final_validation` in the metrics file. It does not read test ground truth, and France performance remains unknown.

### Test inference operational status

The first full test-inference attempt failed during DuckDB target-key spill cleanup with a Windows file-lock `IOException`. A retry using `SET threads=1` and the isolated `test_work_retry` directory was stopped after scoring **38 million of 68,474,493 cap-24 pairs**. The runner is configured for 15 DuckDB threads and a separate `test_work_15threads` directory; a fresh run using that configuration has started and is loading test data. Its progress and output counts are not yet known. This is operational progress only, not a test-quality result. Full-corpus training has already completed.

`artifacts/v2/test_output/inference_summary.json`, completed output files, and validator status are still pending. On completion, record the exact `source1_rows`, `candidate_pairs`, `predicted_matches`, validator result, and `test_labels_used: false` from the summary. Do not report a test score because test labels are unavailable.

### Audit status and remaining work

- Frozen full-corpus training and untouched holdout scoring are complete. The held-out macro F0.5 is **0.8473170044** at threshold 0.78; supporting candidate recall and slices are recorded above and in `validation_report.md`.
- The generated validation report agrees with `training_metrics.json`; this validation/reporting work is complete.
- Separate test inference is in progress as described above. Submission-format validation has not yet run or been confirmed, because final test outputs are not present yet.
- After inference completes, verify both output row counts and the official validator result, then record exact test output counts here. Test labels are unavailable, so test performance cannot be scored in this run.
- v2 remains an experimental baseline and has not replaced the existing submission path. Any future promotion is a separate decision after test output and packaging review.
