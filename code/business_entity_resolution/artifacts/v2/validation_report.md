# V2 Full-Corpus Validation Report

## Run summary

- Random seed: 42
- Source 1 entities: 2,206,821
- Source 2 / Source 3 records: 5,034,616 / 5,285,603
- Retrieved training positives / sampled negatives: 4,469,493 / 6,000,000
- Candidate cap: 24 per target source
- Decision threshold selected on development data: 0.78
- Selected features (20): fuzz_ratio_name, fuzz_tsr_name, fuzz_tset_name, name_word_jaccard, name_len_diff_ratio, fuzz_ratio_addr, fuzz_tsr_addr, fuzz_tset_addr, addr_word_jaccard, shared_numeric_code_count, has_shared_numeric_code, missing_address_flag, non_latin_script_flag, domain_stem_match, exact_clean_name_match, name_trigram_jaccard, addr_trigram_jaccard, address_number_conflict, addr_length_ratio, target_is_s3

## Candidate-generation diagnostics

### Development

| Measure | Result |
|---|---:|
| Candidate pairs | 12,686,102 |
| Mean candidates per Source 1 | 38.32 |
| All true links retained | 0.6429 |
| Candidate-stage oracle macro F₀.₅ | 0.9182 |
| Entities without candidates | 2013 |

| Target source | True links | Retrieved | Recall |
|---|---:|---:|---:|
| Source 2 | 554,615 | 455,332 | 0.8210 |
| Source 3 | 591,614 | 492,587 | 0.8326 |

### Final validation

| Measure | Result |
|---|---:|
| Candidate pairs | 12,678,786 |
| Mean candidates per Source 1 | 38.30 |
| All true links retained | 0.6415 |
| Candidate-stage oracle macro F₀.₅ | 0.9177 |
| Entities without candidates | 2003 |

| Target source | True links | Retrieved | Recall |
|---|---:|---:|---:|
| Source 2 | 553,309 | 453,646 | 0.8199 |
| Source 3 | 591,493 | 492,276 | 0.8323 |

## Held-out model score

- Final validation macro F₀.₅: **0.8473**
- Pair precision / recall (micro diagnostics): 0.9439 / 0.7445

| Slice | Entities | Macro F₀.₅ |
|---|---:|---:|
| country: India | 132477 | 0.7883 |
| country: US | 198544 | 0.8867 |
| degree: singleton | 18486 | 0.8442 |
| degree: one | 17873 | 0.6927 |
| degree: multiple | 294662 | 0.8569 |
| address_availability: source1_missing | 0 | — |
| address_availability: linked_target_missing | 46985 | 0.7838 |
| address_availability: both_present_or_singleton | 284036 | 0.8578 |

## Baseline comparison

The existing 15-feature model scored **0.8286** macro F₀.₅ on development data at threshold 0.77. It used the same training pairs and v2 candidate lists as the selected model.
The selected model scored **0.8476** on development data at threshold 0.78. The final validation score above was measured once after selection.

## Limits

Training and held-out validation contain US and India. France appears only in the test set, so these validation results do not establish performance on France. Test records and labels were not used in this experiment.

This is the experimental v2 model report. The existing submission inference pipeline remains unchanged.
