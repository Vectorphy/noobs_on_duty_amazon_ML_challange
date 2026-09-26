# Business Entity Resolution Challenge Rules

These notes summarize the organizer's challenge rules supplied by the team. Follow them for all modeling, validation, inference, and submission work.

## Task and data

- Source 1 is the deduplicated reference set. For every Source 1 entity, predict zero, one, or multiple matches among Sources 2 and 3.
- Input files are tab-separated TSVs. Parse them as TSV; business addresses and match-ID lists can contain commas.
- Each source record has `entity_id`, `business_name`, `business_address`, and `country`. The `S1-`, `S2-`, and `S3-` ID prefixes identify source.
- Training has US and India. Test also has France, absent from training. Treat country as an open set: do not hard-code or filter to the training countries, and include every test Source 1 record in output.
- Ground truth has `source1_entity_id` and comma-separated `matched_entity_ids`; an empty list means no matches. A Source 1 entity may have multiple matches across Sources 2 and 3.
- Test ground truth is unavailable. Estimate performance with a held-out validation split from training data.

## Scoring

- The leaderboard scores **macro F₀.₅**, not accuracy. Compute F₀.₅ separately for each Source 1 entity, then average across all Source 1 entities.
- Per-entity formula: `F₀.₅ = (1.25 × precision × recall) / (0.25 × precision + recall)`.
- Correctly predicting no matches for a true singleton scores 1.0; predicting any match for a singleton scores 0.0. Include singletons in validation.
- F₀.₅ is precision-heavy; false merges are more costly than missed links.
- The portal leaderboard scores `matching_results.tsv`; `candidate_pairs.tsv` is not scored but is reviewed for candidate-generation quality and reproducibility.

## Required outputs

Place both files under `output/` in the final package:

- `matching_results.tsv`: exact header `source1_entity_id<TAB>matched_entity_ids`.
- `candidate_pairs.tsv`: exact header `source1_entity_id<TAB>candidate_entity_ids`.
- Each file must have exactly one row for every test Source 1 entity, with no duplicate Source 1 rows.
- Use comma-separated IDs inside the second column, with no quoting. Leave the field empty for no matches/candidates.
- Lists may contain only existing test Source 2 or Source 3 IDs. No Source 1 self-matches, unknown IDs, or duplicate IDs within a list.
- Candidate output must be the **final candidate set actually fed to the inference model**, after all blocking/filtering stages. Every predicted match must be in that set.
- Run `student_resource/utils/validate_submission.py` against both outputs and the test directory before submission. It validates formatting and IDs; it does not score the model.

## Final package and methodology

- Submit one `<team_name>_submission.zip` containing `output/matching_results.tsv`, `output/candidate_pairs.tsv`, a self-contained `code/business_entity_resolution/` with `src/`, exact reproduction instructions in `README.md`, and pinned dependencies in `requirements.txt` (or equivalent), plus the completed `Documentation_template.md`.
- The code in the package must reproduce both outputs from the provided training and test data.
- The methodology document must describe the method, blocking strategy, model architecture, feature engineering, and other relevant details.
- The final model must be MIT- or Apache-2.0-licensed and have at most 8 billion parameters.

## Fair play

- **External identity lookup and external data augmentation are strictly prohibited.** Do not use external databases, APIs, commercial entity-resolution services, business registries, geocoding services, or internet sources to identify businesses or augment/normalize records.
- Use only the provided challenge data for identity resolution. Evidence of external lookup can result in disqualification.

## Organizer tips (not additional requirements)

- Candidate generation controls the maximum possible recall; assess blocking recall on held-out training labels.
- Consider string similarities such as Jaccard, edit distance, and TF-IDF cosine, and country-specific address patterns.
- Include singleton performance when choosing the precision/recall operating point.
