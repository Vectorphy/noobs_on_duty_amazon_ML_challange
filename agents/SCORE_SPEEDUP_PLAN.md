# V2 Scoring Speedup Plan

## Scope and bottleneck

The v2 pair-feature stage is the main scoring hot path. `train_v2.extract_features`
calls scalar `pair_features` for every candidate. `infer_v2.extract_and_score`
already batches RapidFuzz calls, but `pair_features_batch` still normalizes text
and builds token/number/trigram sets one pair at a time. The later `score_grid`
threshold sweep is separate; only optimize it if profiling shows it is material.
The legacy submission scorer and its six-match cap are out of scope.

## New shared feature function

Add `score_speedup(rows, *, layout, workers=8)` to
`code/business_entity_resolution/src/matching_v2.py`. It returns an `(N, 20)`
`float32` matrix in exactly the existing `FEATURES` order. It returns `(0, 20)`
for an empty batch and rejects unknown layouts or malformed rows. The `train`
layout uses source/name/address columns `3/5/6/7/8`; the `inference` layout uses
`2/3/4/5/6`. Keep scalar `pair_features` unchanged as the reference.

For each bounded input batch:

1. Extract rows into object arrays and deduplicate Source 1 and target records
   by their integer IDs, retaining inverse indices to restore pair order. Treat
   null names/addresses as empty strings, as the current scalar code does.
2. Prepare each distinct record's lowercase text, existing normalized tokens,
   numbers, trigrams, cleaned name, lengths, and non-Latin flag once.
3. Compute the six fuzzy similarities with RapidFuzz `process.cpdist`, aligned
   by pair, using the existing scorers, no processor or cutoff, floating-point
   output, and C-level workers. Mask address scores to zero unless both raw
   addresses are nonblank. RapidFuzz documents aligned-pair `cpdist`, floating
   dtypes, and C-level workers:
   https://rapidfuzz.github.io/RapidFuzz/Usage/process.html
4. Encode the four set families (name tokens, address tokens, address numbers,
   and trigrams) as batch-local CSR matrices. Gather the paired rows and use
   sparse elementwise multiplication for exact intersection counts; derive
   Jaccard from set sizes, returning zero for an empty union. Do not share a
   learned vocabulary across data splits. References:
   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.multiply.html
5. Use NumPy array arithmetic, masks, comparisons, and guarded `np.divide` for
   ratios and flags. Use `numpy.strings.find` for both cleaned-name substring
   directions, with the current minimum length of four. Bound temporary arrays
   to the current batch and retain an exact fallback for exceptionally long
   strings if fixed-width Unicode arrays would allocate excessively:
   https://numpy.org/doc/stable/reference/generated/numpy.strings.find.html
6. Cast once to `float32` at return. Preserve the existing NFKD/casefold and
   regex behavior, empty-set Jaccard, `strip()` address availability, numeric
   matching, trigram creation, and target-source flag. Do not truncate strings
   or alter scorers.

This vectorizes pair-level work through compiled NumPy, SciPy, and RapidFuzz
operations. Record-level Python string preparation remains; do not claim literal
SIMD for operations that still use Python strings. Numba is not the first choice:
its supported string features and performance are not a good fit for these
set-heavy operations:
https://numba.readthedocs.io/en/stable/reference/pysupported.html

## Callers and checkpoint invalidation

- In `train_v2.extract_features`, score each fetched batch with
  `score_speedup(..., layout="train")`, assign the resulting block into the
  feature memmap, and populate the five metadata fields in vectorized slices.
  Keep row order, batch size, memmap dtype, and checkpoint cadence.
- In `infer_v2.extract_and_score`, use
  `score_speedup(..., layout="inference", workers=8)`, then preserve selected
  columns, model probability calls, thresholds, and output order.
- Route `pair_features_batch` through the new kernel or retain it temporarily
  as a comparison path.
- Include a feature-engine version in feature and fitted-model checkpoint
  signatures. Do not accept legacy complete feature markers for a run using
  the new implementation. Confirm inference's completed-score signature is
  invalidated when `matching_v2.py` changes.

## Small equivalence suite

Add `code/business_entity_resolution/src/test_score_speedup.py` with roughly
eight pairs represented in both layouts. Cover repeated Source 1 IDs and target
IDs, Sources 2 and 3, null and whitespace-only addresses, an empty name,
accented and non-Latin text, duplicate words, matching and conflicting address
numbers, and names shorter than three characters.

For every pair, compare all 20 columns with scalar `pair_features` in existing
feature order. Require finite results and `rtol=0, atol=1e-6` (prefer exact
equality when it holds). Also assert both layouts yield equal matrices, the
input rows are unchanged, empty input has shape `(0, 20)` and dtype `float32`,
missing-address similarities are zero, source flags are correct, two empty sets
have zero Jaccard, and conflicting address numbers are flagged. A tiny
deterministic classifier check should compare probability vectors and decisions,
including a probability exactly at the threshold.

## Separate threshold-sweep investigation

After feature-kernel profiling, if `train_v2.score_grid` is material, replace
only its per-batch `np.add.at` histogram accumulation with bounded `np.bincount`
counts over flattened `(local_entity, probability_bin)` indices. Preserve the
101 thresholds, `f05`, singleton convention, caps, and tie-break toward the
higher threshold. Add a separate tiny fixture for no candidates, singletons,
one/multiple links, caps 16/24/32, and probabilities 0, 0.4999, 0.5, and 1;
compare every curve point and the chosen threshold against the current method.
Reference: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

## Verification gate

First run only the dummy equivalence suite. Then compare old/new features on a
bounded sample of supplied training candidates. Measure maximum feature error,
model probability error, threshold decision changes, warmed end-to-end
`extract_and_score` pairs/second, and peak memory; compare 4, 8, and 15
RapidFuzz workers. Keep the new path only if it passes the numeric checks and
improves end-to-end scoring speed. No test labels or outside business data are
needed.

## Implementation check (2026-09-27)

Implemented `score_speedup` and routed both v2 train/inference batch scoring
through it. Added the focused dummy suite at
`code/business_entity_resolution/src/test_score_speedup.py`. The suite passes:
3 tests, all 20 features match scalar `pair_features` in both row layouts at
`rtol=0, atol=1e-6`; the feature and model-probability maximum absolute errors
were 0.0 on the bounded real-data sample, and no threshold decisions changed.

On 4,096 candidate rows selected from the existing supplied training candidate
database, median feature-plus-model scoring throughput was:

| Path | Workers | Median pairs/s |
|---|---:|---:|
| Scalar training reference | — | 10,188 |
| `score_speedup`, training layout | 4 | 12,327 |
| `score_speedup`, training layout | 8 | 12,967 |
| `score_speedup`, training layout | 15 | 12,176 |
| Previous batched inference path | 8 | 10,144 |
| `score_speedup`, inference layout | 8 | 10,694 |

The training-layout gain was about 27% at 8 workers; on the 4,096-pair sample,
the prior inference path already used batched RapidFuzz, so its gain was about
5%. These are bounded kernel-plus-model timings, not a full database-to-output
inference benchmark. `score_grid` was left unchanged because this check did not
profile it as a material bottleneck.

## Memory follow-up

The skeptical review identified the whole-caller-batch CSR workspace as the
main memory risk. `score_speedup` now processes at most 2,048 pairs per internal
workspace and writes each chunk into one full-batch float32 output. A dummy
check forces chunk boundaries and retains exact feature equality.

On 20,000 real training candidate rows in inference layout, the previous
batched inference path measured 11,615 pairs/s; chunked `score_speedup` measured
14,724, 14,967, and 13,833 pairs/s with 4, 8, and 15 workers, respectively.
Maximum feature and model-probability error remained 0.0; no threshold
decisions changed. A feature-plus-model timing at 8 workers took 1.336 seconds.

Memory measurements used `tracemalloc`, which captures Python/NumPy-traced
allocations, not the whole process RSS. On 4,096 rows, the measured peak
allocation was 22.59 MiB with a 2,048-row workspace and 43.03 MiB with one
4,096-row workspace. On 20,000 rows with chunking, the traced peak was 23.81
MiB and the final float32 feature output was 1.53 MiB. The process working set
rose 11.9 MiB during scoring; its total high-water mark also included the
DuckDB sample query, so that total is not attributable to this function.
