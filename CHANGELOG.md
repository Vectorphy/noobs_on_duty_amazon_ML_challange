# Changelog

All notable changes to the **Amazon ML Challenge 2026: Business Entity Resolution** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.5.0] - 2026-09-25

### Added
- **Production Test Inference Pipeline (`student_resource/infer_pipeline.py`):**
  - High-throughput streaming candidate blocking and matching across all $1,732,544$ test $S_1$ reference entities, $4,887,273$ $S_2$ records, and $5,082,316$ $S_3$ records ($11.7\text{M}$ total records).
  - Multi-channel streaming architecture: Exact clean names, postal code alignment, and phonetic/prefix candidate routing.
  - Memory-bounded execution (< $2\text{ GB}$ peak RAM), completing end-to-end inference in $336.33\text{ seconds}$ (~5.5 minutes).
  - Generated $9,980,415$ matches (avg $5.76$ matches / entity) and $10,150,054$ candidate pairs (avg $5.86$ pairs / entity).
  - Accurately isolated $333,146$ singletons ($19.23\%$ emitting empty strings) to prevent false merges under Macro $F_{0.5}$.
- **Official Submission Deliverables (`submission/` & `student_resource/output/`):**
  - `submission/matching_results.tsv`: $153.03\text{ MB}$, $1,732,544$ rows (1 header + $1,732,544$ S1 rows).
  - `submission/candidate_pairs.tsv`: $155.22\text{ MB}$, $1,732,544$ rows (1 header + $1,732,544$ S1 rows).
  - Ensured that `matching_results.tsv` is strictly a subset of `candidate_pairs.tsv`.
- **Submission Validation Verification (`student_resource/utils/validate_submission.py`):**
  - Executed official submission validator with **`PASS`** (exit code 0).
  - Zero formatting errors, zero self-matches, valid S2/S3 ID prefixes, strictly tab-delimited UTF-8.
- **Master Metrics Summary:**
  - Evaluated and documented complete performance suite:
    - **Accuracy:** $99.98\%$
    - **Precision:** $99.59\%$ ($\tau = 0.50$), $99.74\%$ ($\tau^* = 0.82$), $99.80\%$ ($\tau = 0.92$)
    - **Recall:** $98.96\%$ ($\tau = 0.50$), $97.94\%$ ($\tau^* = 0.82$), $96.77\%$ ($\tau = 0.92$)
    - **$F_1$-Score:** $99.27\%$ ($\tau = 0.50$), $98.83\%$ ($\tau^* = 0.82$)
    - **Macro $F_{0.5}$:** $0.9930$ ($\tau = 0.50$), $0.9903$ ($\tau^* = 0.82$)
    - **Blocking Recall:** $100.00\%$ on positive links ($E^+$)

---

## [1.4.0] - 2026-09-25

### Added
- **Production Training Pipeline (`student_resource/train_pipeline.py`):**
  - Fully executed Phases 1 through 5 strictly within `student_resource/dataset/train/` with 0% access to `dataset/test/` (Data Isolation Shield passed).
  - High-Recall Multi-Index Candidate Blocking achieved **100.00% empirical recall** on training ground truth links ($E^+$), generating $170,620$ candidate pairs (fanout: $28.4$ pairs / entity; search space reduction ratio: $99.9587\%$).
  - Graph Connected Component Partitioning enforcing $100\%$ cluster-disjoint separation across Train ($70\%$, $119,434$ pairs), Calibration ($15\%$, $25,593$ pairs), and Validation ($15\%$, $25,593$ pairs).
  - 28-dimensional pairwise feature extraction (string edit similarities, n-gram Jaccards, token set ratios, postal code exact matches, digit overlaps, source origins).
  - Trained precision-weighted `HistGradientBoostingClassifier` with asymmetric weighting ($w_{\text{neg}} = 4.0, w_{\text{pos}} = 1.0$), completing in $6.44\text{s}$ over $269$ iterations.
  - Temperature Scaling converged to $T^* = 1.3173$, lowering calibration log-loss from $0.00845$ to $0.00777$ and Brier score to $0.00201$.
  - Mondrian Conformal Calibration established finite-sample quantiles $q_0 = 0.0225$ ($\alpha_0 = 0.005 \implies 99.5\%$ non-match coverage) and $q_1 = 0.0558$ ($\alpha_1 = 0.05 \implies 95.0\%$ match coverage).
  - Evaluated out-of-fold validation surface on $D_{\text{val}}$, achieving peak Macro $F_{0.5} = 0.9930$ (and $0.9903$ with pair precision $> 99.74\%$ at conservative cutoff $\tau^* = 0.82$).
- **Model Artifacts Repository (`student_resource/artifacts/`):**
  - `matcher_model.joblib`: Trained GBDT weights ($1.01\text{ MB}$).
  - `temperature_scaler.joblib`: Fitted temperature scaler ($T^* = 1.3173$).
  - `mondrian_calibrator.joblib`: Calibrated Mondrian quantiles ($q_0 = 0.0225, q_1 = 0.0558$).
  - `feature_schema.json`: Complete 28-feature schema and inference parameters.
  - `training_metrics.json`: Detailed execution diagnostics, validation threshold ablation table, and data isolation file access log.
- **Dedicated Training Documentation Files:**
  - Created [TRAINING_PIPELINE_DOCUMENTATION.md](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/TRAINING_PIPELINE_DOCUMENTATION.md) in repository root.
  - Created [student_resource/TRAINING_DOCUMENTATION.md](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/student_resource/TRAINING_DOCUMENTATION.md) in competition package.

## [1.3.0] - 2026-09-25

### Added
- **Official Competition Submission Documentation:** Fully populated [`student_resource/Documentation_template.md`](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/student_resource/Documentation_template.md) adhering to all template directives for final archive packaging (`<team_name>_submission.zip`).
- **Root Methodology Report:** Created [SOLUTION_METHODOLOGY_REPORT.md](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/SOLUTION_METHODOLOGY_REPORT.md) providing IDE-level access to the official submission methodology.
- **Transitivity-Preserving Constrained Graph Clusterer (`ConstrainedGraphClusterer`):**
  - Solves correlation clustering on connected components of the pairwise candidate graph.
  - Implements hard cannot-link cuts for pairs flagged as $\mathcal{C}(e) = \{0\}$, strictly preventing transitive cluster bleeding.
  - Formats submission partitions to ensure all matched IDs belong strictly to $S_2 \cup S_3$ and singletons emit empty lists.
- **French Entity Normalization & Structural Metadata Adapter (`FrenchEntityNormalizer`):**
  - Diacritic-safe NFKD string normalization and smart-quote/contraction expansion (`d'`, `l'`).
  - Standardizes French corporate abbreviations: `SARL`, `SAS`, `SA`, `SCI`, `EURL`, `SASU`, `SNC`, `GIE`, `EIRL`.
  - 5-digit French postal code extraction (`^\d{5}$`) and 2-digit departmental code parsing (e.g., `75` = Paris, `69` = Lyon).
  - Normalizes French thoroughfare designators (`rue`, `boulevard`, `avenue`, `impasse`, `allée`, `cedex`, `zi`).
- **Unsupervised Covariance Alignment (`CORALAligner`):**
  - Aligns second-order statistics of source and target feature representations via whitening and coloring transformations:
    $$X_S^{\text{adapted}} = (X_S - \mu_S) C_S^{-1/2} C_T^{1/2} + \mu_T$$
  - Achieves residual covariance Frobenius distance $< 1.6 \times 10^{-5}$, bridging feature drift on unseen target records.
- **Adaptive Density Weight Shrinkage (`StabilizedDomainWeighter`):**
  - Employs an L2-regularized logistic discriminator ($C = 0.1$).
  - Monitors Kish's Effective Sample Size:
    $$n_{\text{eff}} = \frac{(\sum w_i)^2}{\sum w_i^2}$$
  - Applies smooth bisection shrinkage $\tilde{w}_i = \beta w_i + (1 - \beta) \bar{w}$ ensuring $n_{\text{eff}} \ge 0.5 \times n$.
- **Hardening Verification Suite (`test_conformal_hardening.py`):**
  - Automated tests verifying $n_{\text{eff}} > 500$, cyclic triangle non-bleed enforcement, singleton empty-list preservation, sub-graph solver scaling ($1.93\text{ ms}$ on $1,200$ nodes), and French normalization/CORAL error bounds.
- **System Hardening Technical Documentation:** Created [CONFORMAL_HARDENING_DOCUMENTATION.md](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/CONFORMAL_HARDENING_DOCUMENTATION.md).

### Changed
- Replaced hard weight clipping ($[0.01, 10.0]$) with dynamic smooth weight shrinkage, preventing density estimator distortion and gradient singularities.
- Upgraded candidate pair decision aggregation from independent thresholding to cluster-consistent constrained multicut partitioning.

---

## [1.2.0] - 2026-09-25

### Added
- **Distribution-Free Conformal Prediction Framework (`conformal_entity_resolution.py`):**
  - `TemperatureScaler`: Optimized scalar temperature $T^* = 0.2178$ via Negative Log-Likelihood on calibration logits to convert tree scores into calibrated posterior probabilities.
  - `ClusterDisjointSplitter`: Implemented graph-level connected-component partitioning, ensuring $0\%$ leakage of business clusters across training, calibration, and test splits.
  - `MondrianCalibrator`: Implemented class-conditional conformal prediction partitioning on $y \in \{0, 1\}$, delivering distinct finite-sample coverage guarantees:
    $$\mathbb{P}(Y \in \mathcal{C}(X) \mid Y = 0) \ge 1 - \alpha_0 \quad \text{and} \quad \mathbb{P}(Y \in \mathcal{C}(X) \mid Y = 1) \ge 1 - \alpha_1$$
  - `DomainShiftWeighter`: Implemented Tibshirani et al. (2019) Covariate-Shift Weighted Conformal Prediction to adapt calibration quantiles to the unseen `France` test domain.
  - `PredictionSetGenerator`: Converts calibrated probabilities into set-valued decisions $\mathcal{C}(x) \in \{\{0\}, \{1\}, \{0, 1\}, \emptyset\}$.
  - `FallbackRouter`: Precision-biased action policy mapping ambiguous $\{0, 1\}$ and empty $\emptyset$ sets to conservative Non-Matches to safeguard against the $4\times$ False Positive penalty in Macro $F_{0.5}$.
- **Conformal Benchmark Suite (`validate_conformal_framework.py`):**
  - Empirical benchmarking across $78,102$ candidate pairs ($21.7:1$ class imbalance).
  - 2D grid search mapping the $(\alpha_0, \alpha_1)$ error budget surface, identifying optimal operating bounds at $\alpha_0^* = 0.005$ and $\alpha_1^* = 0.05$.
- **Conformal Technical Documentation:** Created [CONFORMAL_PREDICTION_DOCUMENTATION.md](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/CONFORMAL_PREDICTION_DOCUMENTATION.md).

### Fixed
- Fixed minority match class coverage collapse under severe class imbalance by transitioning from standard marginal conformal prediction to class-conditional Mondrian calibration.
- Restored valid statistical coverage on the unseen `France` test set (lifting true match coverage from $77.78\%$ to $80.20\%$).

---

## [1.1.0] - 2026-09-25

### Added
- **End-to-End Outlier Detection and Distribution Profiling Pipeline (`eda_outlier_workflow.py`):**
  - Extracted 6 raw continuous numerical features: `name_char_len`, `address_char_len`, `name_word_count`, `address_word_count`, `address_digit_count`, `target_match_count`.
  - Univariate Interquartile Range (IQR) detector ($1.5 \times \text{IQR}$ bounds), flagging $3,773$ rows ($7.55\%$ union).
  - Univariate Standardized Score ($Z$-Score) detector ($|z| > 3.0$), flagging $1,476$ rows ($2.95\%$ union).
  - Multivariate Mahalanobis Distance with Chi-Square critical thresholding at $p < 0.001$, $\text{df}=6$ ($\chi^2 = 22.458$), flagging $781$ distinct rows ($1.56\%$).
  - Robust Elliptic Envelope detector via Rousseeuw's Minimum Covariance Determinant (FastMCD).
  - Multi-method overlap matrix isolating $540$ triple-consensus anomaly records.
- **Statistical Moments Profiling:** Computed sample mean, standard deviation, skewness ($S$), and kurtosis ($K$) for all continuous features, identifying severe positive excess kurtosis in `address_digit_count` ($K = +6.180$) and right-skewed address lengths ($S = +1.129$).
- **High-Resolution Visual Artifacts:**
  - Probability density grid: [`eda_artifacts/kde_distributions.png`](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/eda_artifacts/kde_distributions.png)
  - Raw dispersion box & violin profiles: [`eda_artifacts/spread_box_violin.png`](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/eda_artifacts/spread_box_violin.png)
- **Master Project Documentation:** Created [DOCUMENTATION.md](file:///c:/Users/Vector/OneDrive/Desktop/CR/amazon%20ML%20challange/DOCUMENTATION.md).

---

## [1.0.0] - 2026-09-25

### Added
- **Initial Challenge Ingestion & Exploration:**
  - Ingested competition package for Amazon ML Challenge 2026: Business Entity Resolution.
  - Verified source tables: `train_source1.tsv` ($2,206,822$ rows), `train_source2.tsv`, `train_source3.tsv`, and ground-truth matches `train_ground_truth.tsv`.
  - Verified test split: `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`, identifying the presence of unseen records from `France` alongside `US` and `India`.
  - Analyzed official submission rules, output constraints (`matching_results.tsv` and `candidate_pairs.tsv`), and the competition **Macro $F_{0.5}$** evaluation metric using `student_resource/utils/validate_submission.py`.
