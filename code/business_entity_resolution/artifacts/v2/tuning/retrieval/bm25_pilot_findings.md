# BM25 candidate-retrieval pilot

## Results

Candidate retrieval used only a stable-hash sample of 165,592 development Source 1 rows and 40,000 Source 2/3 target rows from India and the US. Ground-truth labels were joined after retrieval. The recall denominator includes only positive links whose Source 1 and target records were both in the label-blind sample.

| Top K per target | Candidate pairs | S2 baseline → BM25S recall | S3 baseline → BM25S recall |
|---:|---:|---:|---:|
| 4 | 160,000 | 82.55% → 98.04% | 81.79% → 98.21% |
| 8 | 320,000 | 82.55% → 98.33% | 81.79% → 98.93% |
| 16 | 640,000 | 82.55% → 98.73% | 81.79% → 98.93% |
| 32 | 1,280,000 | 82.55% → 99.17% | 81.79% → 99.29% |
| 64 | 2,560,000 | 82.55% → 99.41% | 81.79% → 99.29% |
| 128 | 5,120,000 | 82.55% → 99.56% | 81.79% → 99.64% |

The existing blocker produced 32,104 candidates for these sampled queries. BM25S top-4 is about five times that volume; top-8 is ten times. At top-4, the union with the existing blocker recalls 98.14% of eligible S2 links and 98.21% of eligible S3 links. The sample contains 2,040 eligible S2 links and 280 eligible S3 links, so the S3 estimate is less stable.

## Throughput

The existing SciPy prototype took 165.68 seconds for 20,000 India queries and 263.04 seconds for 20,000 US queries. The BM25S/Numba backend took 17.11 and 8.90 seconds for the same rows: 26.01 seconds total versus 428.72 seconds, about 16.5× faster in this pilot. This includes building each country index, vectorizing queries, and retrieving top-128. BM25S uses a Numba retrieval backend; its published benchmarks also show that speed depends on corpus and setup, so this sample result is not a full-run guarantee ([project documentation](https://github.com/xhluca/bm25s), [BM25S paper](https://arxiv.org/abs/2407.03618)).

## Missed-link diagnosis

At top-32, the baseline missed 356 of 2,040 eligible S2 links and 51 of 280 eligible S3 links. In the sample, BM25S recovered 339 S2 and 49 S3 missed links. Remaining misses show that BM25 retrieval is not complete even at this cap.

## Decision and next gate

The BM25S implementation is kept as an optional backend in `src/retrieval_v2.py`; the SciPy pilot backend remains available. The new dependency pins are isolated in `requirements-retrieval.txt`. Do not promote this pilot directly to submission inference or retrain on it yet.

Next, run the retriever over the complete development reference and target sets in bounded batches. Measure the exact per-Source-1 candidate oracle F₀.₅, candidate volumes, all-link retention, and runtime. Only after that gate should we freeze a candidate union and run the three-model comparison; the existing notebook already writes LightGBM, XGBoost, and CatBoost outputs separately. No full XGBoost or CatBoost runs were started in this pilot.

The full-data gate matters because this sample is limited to two countries and about 1.5% of all train targets, while BM25S top-8 already creates ten times the sampled baseline candidate volume. Pilot recall does not establish the competition score ceiling or promise a score near 0.994.
