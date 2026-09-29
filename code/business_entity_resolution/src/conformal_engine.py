"""
Robust Conformal Prediction Framework for Business Entity Resolution
Amazon ML Challenge 2026

Mathematical Principles:
1. Platt / Temperature Scaling for Probability Calibration
2. Mondrian (Class-Conditional) Conformal Prediction under Severe ER Imbalance
3. Covariate-Shift Weighted Conformal Prediction (Tibshirani et al., 2019) for Domain Adaptation (US/India -> France)
4. Cluster-Disjoint Graph Partitioning to Prevent Coverage Inflation
5. Precision-Biased Fallback Routing Aligned with Macro F_0.5 Metric
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize_scalar
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, precision_score, recall_score, fbeta_score

# ------------------------------------------------------------------------------
# 1. Temperature / Platt Scaler for Valid Likelihood Calibration
# ------------------------------------------------------------------------------
class TemperatureScaler(BaseEstimator, ClassifierMixin):
    """
    Learns an optimal temperature parameter T > 0 on calibration logits to minimize NLL:
        P(Y=1 | z) = sigma(z / T)
    Ensures raw scoring logits are transformed into well-calibrated posterior probabilities
    prior to non-conformity score evaluation.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        def nll_obj(T: float) -> float:
            if T <= 0:
                return 1e9
            scaled_logits = logits / T
            # Numerically stable binary cross-entropy
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30.0, 30.0)))
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            return -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))

        res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        scaled_logits = logits / self.temperature
        p1 = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30.0, 30.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


# ------------------------------------------------------------------------------
# 2. Cluster-Disjoint Data Partitioning
# ------------------------------------------------------------------------------
class ClusterDisjointSplitter:
    """
    Enforces strict group-level disjointness based on business identity clusters (Source 1 root IDs).
    Prevents linked entities from leaking across train, calibration, and test splits,
    which would otherwise falsely tighten conformal non-conformity quantiles.
    """
    @staticmethod
    def split(
        df_pairs: pd.DataFrame,
        group_col: str = "cluster_id",
        train_ratio: float = 0.50,
        calib_ratio: float = 0.25,
        test_ratio: float = 0.25,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        np.random.seed(random_state)
        unique_groups = df_pairs[group_col].unique()
        np.random.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train = int(n_groups * train_ratio)
        n_calib = int(n_groups * calib_ratio)

        train_groups = set(unique_groups[:n_train])
        calib_groups = set(unique_groups[n_train:n_train + n_calib])
        test_groups = set(unique_groups[n_train + n_calib:])

        df_train = df_pairs[df_pairs[group_col].isin(train_groups)].copy()
        df_calib = df_pairs[df_pairs[group_col].isin(calib_groups)].copy()
        df_test = df_pairs[df_pairs[group_col].isin(test_groups)].copy()

        return df_train, df_calib, df_test


# ------------------------------------------------------------------------------
# 3. Covariate Shift Estimator (Tibshirani et al., 2019)
# ------------------------------------------------------------------------------
class DomainShiftWeighter:
    """
    Trains a probabilistic domain discriminator g(x) = P(Domain = Target | x)
    to compute likelihood ratio density weights:
        w(x) = P_target(x) / P_source(x) = (P(T|x) / (1 - P(T|x))) * (N_source / N_target)
    Used to reweight empirical calibration distributions under geographic shift (US/India -> France).
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.discriminator = LogisticRegression(random_state=random_state, max_iter=1000)
        self.n_source: int = 1
        self.n_target: int = 1

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "DomainShiftWeighter":
        self.n_source = len(X_source)
        self.n_target = len(X_target)

        X_domain = np.vstack([X_source, X_target])
        y_domain = np.hstack([np.zeros(self.n_source), np.ones(self.n_target)])

        self.discriminator.fit(X_domain, y_domain)
        return self

    def compute_weights(self, X: np.ndarray, clip_bounds: Tuple[float, float] = (0.01, 10.0)) -> np.ndarray:
        probs = self.discriminator.predict_proba(X)[:, 1]
        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        raw_weights = (probs / (1.0 - probs)) * (self.n_source / self.n_target)
        return np.clip(raw_weights, clip_bounds[0], clip_bounds[1])


# ------------------------------------------------------------------------------
# 4. Class-Conditional Mondrian Conformal Calibrator
# ------------------------------------------------------------------------------
class MondrianCalibrator:
    """
    Implements Class-Conditional Conformal Prediction (Mondrian CP):
        P(Y in C(X) | Y = y) >= 1 - alpha_y   for y in {0, 1}
    Protects minority true matches from being submerged by majority negative pairs.
    Supports both standard empirical quantiles and covariate-shift weighted quantiles.
    """
    def __init__(self, alpha_0: float = 0.02, alpha_1: float = 0.15, score_type: str = "lac"):
        """
        alpha_0: Error rate on class 0 (Non-matches). Kept strictly low to enforce precision.
        alpha_1: Error rate on class 1 (True matches).
        score_type: 'lac' (Least Ambiguous Classifier: 1 - P(Y=y | X))
        """
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.score_type = score_type

        self.q0: float = 1.0
        self.q1: float = 1.0
        self.cal_scores_0: np.ndarray = np.array([])
        self.cal_scores_1: np.ndarray = np.array([])
        self.cal_weights_0: np.ndarray = np.array([])
        self.cal_weights_1: np.ndarray = np.array([])

    def _nonconformity_score(self, probs: np.ndarray, y: int) -> np.ndarray:
        """
        LAC non-conformity score:
            R(x, 0) = 1 - P(Y=0 | x) = P(Y=1 | x)
            R(x, 1) = 1 - P(Y=1 | x)
        """
        if y == 0:
            return probs[:, 1]
        elif y == 1:
            return 1.0 - probs[:, 1]
        raise ValueError(f"Unknown class {y}")

    def fit(self, probs_calib: np.ndarray, y_calib: np.ndarray, weights_calib: Optional[np.ndarray] = None) -> "MondrianCalibrator":
        y_calib = np.asarray(y_calib, dtype=int)
        mask_0 = (y_calib == 0)
        mask_1 = (y_calib == 1)

        self.cal_scores_0 = self._nonconformity_score(probs_calib[mask_0], y=0)
        self.cal_scores_1 = self._nonconformity_score(probs_calib[mask_1], y=1)

        n0 = len(self.cal_scores_0)
        n1 = len(self.cal_scores_1)

        if weights_calib is not None:
            self.cal_weights_0 = weights_calib[mask_0]
            self.cal_weights_1 = weights_calib[mask_1]
            self.q0 = self._weighted_quantile(self.cal_scores_0, self.cal_weights_0, 1.0 - self.alpha_0)
            self.q1 = self._weighted_quantile(self.cal_scores_1, self.cal_weights_1, 1.0 - self.alpha_1)
        else:
            # Finite-sample adjusted empirical quantile: ceil((n+1)(1-alpha)) / n
            quantile_level_0 = np.clip(np.ceil((n0 + 1) * (1.0 - self.alpha_0)) / n0, 0.0, 1.0)
            quantile_level_1 = np.clip(np.ceil((n1 + 1) * (1.0 - self.alpha_1)) / n1, 0.0, 1.0)
            self.q0 = float(np.quantile(self.cal_scores_0, quantile_level_0, method="higher"))
            self.q1 = float(np.quantile(self.cal_scores_1, quantile_level_1, method="higher"))

        return self

    @staticmethod
    def _weighted_quantile(scores: np.ndarray, weights: np.ndarray, target_prob: float) -> float:
        """
        Computes the weighted conformal quantile under covariate shift (Tibshirani et al., 2019):
            q = inf { t : sum_{i=1}^n p_i * 1{R_i <= t} >= target_prob }
        """
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cum_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        idx = np.searchsorted(cum_weights, target_prob, side="left")
        idx = min(idx, len(sorted_scores) - 1)
        return float(sorted_scores[idx])


# ------------------------------------------------------------------------------
# 5. Prediction Set Generator & Fallback Router
# ------------------------------------------------------------------------------
class PredictionSetGenerator:
    """
    Converts calibrated posterior probabilities into conformal prediction sets:
        C(x) = { y in {0, 1} : R(x, y) <= q_{1-alpha_y} }
    """
    def __init__(self, calibrator: MondrianCalibrator):
        self.calibrator = calibrator

    def generate_sets(self, probs: np.ndarray) -> List[List[int]]:
        """
        Generates prediction sets for each candidate pair.
        Possible sets:
            [0]    : High confidence Non-Match
            [1]    : High confidence True Match
            [0, 1] : Ambiguous / Borderline Pair
            []     : Anomaly / Out-of-Distribution Pair
        """
        p1 = probs[:, 1]
        score_0 = p1            # R(x, 0)
        score_1 = 1.0 - p1      # R(x, 1)

        include_0 = score_0 <= self.calibrator.q0
        include_1 = score_1 <= self.calibrator.q1

        prediction_sets = []
        for inc0, inc1 in zip(include_0, include_1):
            s = []
            if inc0:
                s.append(0)
            if inc1:
                s.append(1)
            prediction_sets.append(s)

        return prediction_sets


class FallbackRouter:
    """
    Translates conformal prediction sets C(x) into deterministic point actions
    specifically aligned with the Macro F_0.5 objective:
        - [1]    -> MATCH
        - [0]    -> NON-MATCH
        - [0, 1] -> Conservative NON-MATCH (avoids 4x penalty on False Merges)
        - []     -> NON-MATCH (safeguards precision against anomalies & singletons)
    """
    @staticmethod
    def route_decision(pred_set: List[int], prob_1: float, strict_tiebreak_tau: float = 0.82) -> int:
        if pred_set == [1]:
            return 1
        elif pred_set == [0]:
            return 0
        elif pred_set == [0, 1]:
            # Ambiguous: default to 0; only promote if model probability surpasses strict precision threshold
            return 1 if prob_1 >= strict_tiebreak_tau else 0
        elif pred_set == []:
            # OOD / Empty set: reject match to prevent singleton cliff-drop
            return 0
        return 0


# ------------------------------------------------------------------------------
# 6. Evaluation, Coverage, Efficiency, and Macro F_0.5 Metrics
# ------------------------------------------------------------------------------
def evaluate_conformal_performance(
    pred_sets: List[List[int]],
    y_true: np.ndarray,
    probs_1: np.ndarray,
    query_entity_ids: pd.Series,
    strict_tau: float = 0.82
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Computes rigorous distribution-free diagnostics:
    - Marginal Coverage
    - Class-Conditional Coverage (Class 0 and Class 1)
    - Set Efficiency (Mean Set Size, Fraction of Singletons, Fraction of Ambiguous, Fraction of Empty)
    - Downstream Macro F_0.5 Score on entity sets
    """
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)

    # 1. Coverage
    covered = np.array([y in s for y, s in zip(y_true, pred_sets)])
    marginal_coverage = float(np.mean(covered))

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    coverage_0 = float(np.mean(covered[mask_0])) if np.sum(mask_0) > 0 else 1.0
    coverage_1 = float(np.mean(covered[mask_1])) if np.sum(mask_1) > 0 else 1.0

    # 2. Set Efficiency
    set_sizes = np.array([len(s) for s in pred_sets])
    mean_set_size = float(np.mean(set_sizes))
    frac_singleton = float(np.mean(set_sizes == 1))
    frac_ambiguous = float(np.mean(set_sizes == 2))
    frac_empty = float(np.mean(set_sizes == 0))

    # 3. Routed Point Predictions & Downstream Macro F_0.5
    point_preds = np.array([
        FallbackRouter.route_decision(s, p1, strict_tiebreak_tau=strict_tau)
        for s, p1 in zip(pred_sets, probs_1)
    ])

    pair_prec = float(precision_score(y_true, point_preds, zero_division=0))
    pair_rec = float(recall_score(y_true, point_preds, zero_division=0))
    pair_f05 = float(fbeta_score(y_true, point_preds, beta=0.5, zero_division=0))

    return {
        "marginal_coverage": marginal_coverage,
        "coverage_class_0": coverage_0,
        "coverage_class_1": coverage_1,
        "mean_set_size": mean_set_size,
        "frac_singleton": frac_singleton,
        "frac_ambiguous": frac_ambiguous,
        "frac_empty": frac_empty,
        "point_precision": pair_prec,
        "point_recall": pair_rec,
        "point_f05": pair_f05
    }
