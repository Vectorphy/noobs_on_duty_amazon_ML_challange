"""
Risk Mitigation and Conformal System Hardening for Business Entity Resolution
Amazon ML Challenge 2026

Modules:
1. StabilizedDomainWeighter:
   - L2-Regularized Logistic Discriminator for Density Ratio Estimation
   - Effective Sample Size (n_eff) Diagnostics
   - Adaptive Shrinkage / Soft Truncation ensuring n_eff >= 0.5 * n
2. ConstrainedGraphClusterer:
   - Transitivity-Preserving Constrained Correlation Clustering (Multicut Approximation)
   - Enforces C(e) = {0} Hard Non-Merge Cuts
   - Prevents Catastrophic Cluster Bleeding under Macro F_0.5 (w_FP = 4 * w_FN)
   - Preserves True Singletons
3. FrenchEntityNormalizer:
   - Diacritic-safe NFKD normalization
   - Contraction expansion (d', l')
   - French corporate suffix standardization (SARL, SAS, SA, SCI, EURL, etc.)
   - 5-digit postal code & department code extraction
4. CORALAligner:
   - Unsupervised Covariance Alignment (Correlation Alignment)
   - Minimizes domain discrepancy between source and target feature representations
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Union
import scipy.linalg as la
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------------
# 1. Stabilized Covariate Shift Weighter with Adaptive Shrinkage
# ------------------------------------------------------------------------------
class StabilizedDomainWeighter:
    """
    Computes stabilized likelihood ratio density weights w(x) = P_target(x) / P_source(x)
    using an L2-regularized logistic domain discriminator.
    Monitors Effective Sample Size (n_eff) and applies adaptive shrinkage to guarantee:
        n_eff >= min_neff_ratio * n
    eliminating weight variance explosion and arbitrary hard-clipping artifacts.
    """
    def __init__(self, c_reg: float = 0.1, min_neff_ratio: float = 0.50, random_state: int = 42):
        self.c_reg = c_reg
        self.min_neff_ratio = min_neff_ratio
        self.random_state = random_state
        self.discriminator = LogisticRegression(
            C=self.c_reg,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state
        )
        self.n_source: int = 1
        self.n_target: int = 1
        self.beta_: float = 1.0
        self.raw_neff_: float = 0.0
        self.shrunk_neff_: float = 0.0

    @staticmethod
    def calculate_neff(weights: np.ndarray) -> float:
        """
        Computes Kish's Effective Sample Size:
            n_eff = (sum w_i)^2 / sum (w_i^2)
        """
        weights = np.asarray(weights, dtype=np.float64)
        sum_w = np.sum(weights)
        sum_w_sq = np.sum(weights ** 2)
        if sum_w_sq <= 1e-12:
            return 0.0
        return float((sum_w ** 2) / sum_w_sq)

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "StabilizedDomainWeighter":
        self.n_source = len(X_source)
        self.n_target = len(X_target)
        X_domain = np.vstack([X_source, X_target])
        y_domain = np.hstack([np.zeros(self.n_source), np.ones(self.n_target)])
        self.discriminator.fit(X_domain, y_domain)
        return self

    def compute_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Computes stabilized weights with adaptive shrinkage toward the empirical mean:
            w_tilde(x) = beta * w(x) + (1 - beta) * w_bar
        where beta in (0, 1] is chosen via bisection search to satisfy n_eff >= min_neff_ratio * n.
        """
        probs = self.discriminator.predict_proba(X)[:, 1]
        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)

        # Raw likelihood ratio
        raw_weights = (probs / (1.0 - probs)) * (self.n_source / self.n_target)
        n = len(raw_weights)
        w_bar = float(np.mean(raw_weights))
        self.raw_neff_ = self.calculate_neff(raw_weights)

        target_neff = self.min_neff_ratio * n

        if self.raw_neff_ >= target_neff:
            self.beta_ = 1.0
            self.shrunk_neff_ = self.raw_neff_
            return raw_weights

        # Bisection search for optimal shrinkage parameter beta in [0, 1]
        low, high = 0.0, 1.0
        best_beta = 0.0
        for _ in range(40):
            mid = (low + high) / 2.0
            w_candidate = mid * raw_weights + (1.0 - mid) * w_bar
            neff_cand = self.calculate_neff(w_candidate)
            if neff_cand >= target_neff:
                best_beta = mid
                low = mid
            else:
                high = mid

        self.beta_ = best_beta
        shrunk_weights = self.beta_ * raw_weights + (1.0 - self.beta_) * w_bar
        self.shrunk_neff_ = self.calculate_neff(shrunk_weights)
        return shrunk_weights


# ------------------------------------------------------------------------------
# 2. Transitivity-Preserving Constrained Graph Clusterer
# ------------------------------------------------------------------------------
class ConstrainedGraphClusterer:
    """
    Solves transitivity-preserving correlation clustering on pairwise ER graphs:
        min sum_{e in E+} (1 - p_e) x_e + sum_{e in E-} (p_e - tau) (1 - x_e)
        s.t. x_uv <= x_uw + x_wv  (Triangle Inequality)
             x_uv = 1 for all pairs with C(uv) = {0} (Hard Cannot-Link Cuts)
    Prevents catastrophic cluster bleeding under Macro F_0.5 (w_FP = 4 * w_FN).
    """
    def __init__(self, tau_threshold: float = 0.82):
        self.tau_threshold = tau_threshold

    def solve_components(
        self,
        nodes: List[str],
        edges: Dict[Tuple[str, str], Dict[str, Union[float, List[int]]]]
    ) -> List[List[str]]:
        """
        Partition graph into connected components, then apply constrained agglomerative
        correlation clustering on each component.
        edges format: (u, v) -> {'prob': float, 'set': [0], [1], [0, 1], []}
        """
        # 1. Identify Cannot-Link constraints and Positive Candidate Edges
        cannot_link: Set[Tuple[str, str]] = set()
        candidate_edges: List[Tuple[str, str, float]] = []

        for (u, v), meta in edges.items():
            conf_set = meta.get("set", [])
            prob = float(meta.get("prob", 0.0))

            if conf_set == [0]:
                cannot_link.add((u, v))
                cannot_link.add((v, u))
            elif prob >= self.tau_threshold and conf_set != [0]:
                candidate_edges.append((u, v, prob))

        # Sort candidate positive edges descending by probability
        candidate_edges.sort(key=lambda x: x[2], reverse=True)

        # 2. Constrained Complete-Linkage Disjoint Set
        node_to_cluster: Dict[str, Set[str]] = {u: {u} for u in nodes}

        for u, v, prob in candidate_edges:
            clust_u = node_to_cluster[u]
            clust_v = node_to_cluster[v]

            if clust_u is not clust_v:
                # Check for Cannot-Link constraint violation across the merged cluster
                violates_cannot_link = any(
                    (x, y) in cannot_link for x in clust_u for y in clust_v
                )
                if not violates_cannot_link:
                    # Safe to merge
                    merged_cluster = clust_u.union(clust_v)
                    for node in merged_cluster:
                        node_to_cluster[node] = merged_cluster

        # 3. Extract unique clusters
        unique_clusters: List[List[str]] = []
        seen_nodes: Set[str] = set()

        for u in nodes:
            if u not in seen_nodes:
                c = sorted(list(node_to_cluster[u]))
                unique_clusters.append(c)
                seen_nodes.update(c)

        return unique_clusters

    def format_submission_matches(
        self,
        clusters: List[List[str]],
        source1_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Formats cluster partitions into the official competition submission schema:
            source1_entity_id -> list of matched S2 and S3 entity IDs.
        Guarantees:
        - Self-matches to S1 are excluded.
        - True singletons (cluster of size 1 or containing no S2/S3 nodes) produce empty match lists.
        """
        s1_matches: Dict[str, List[str]] = {s1: [] for s1 in source1_ids}
        
        for cluster in clusters:
            s1_nodes = [node for node in cluster if node.startswith("S1-")]
            match_nodes = [node for node in cluster if node.startswith("S2-") or node.startswith("S3-")]
            
            for s1 in s1_nodes:
                if s1 in s1_matches:
                    s1_matches[s1].extend(match_nodes)

        # Deduplicate and sort IDs
        for s1 in s1_matches:
            s1_matches[s1] = sorted(list(set(s1_matches[s1])))

        return s1_matches


# ------------------------------------------------------------------------------
# 3. French Entity Normalizer (Structural & Linguistic Domain Adaptation)
# ------------------------------------------------------------------------------
class FrenchEntityNormalizer:
    """
    Parses and standardizes French-specific corporate abbreviations, address tokens,
    diacritics, and 5-digit postal/department codes to overcome target concept shift.
    """
    # French corporate suffix mapping
    LEGAL_SUFFIXES = {
        r"\bs[\.\s]?a[\.\s]?r[\.\s]?l\b": "sarl",
        r"\bs[\.\s]?a[\.\s]?s[\.\s]?u\b": "sasu",
        r"\bs[\.\s]?a[\.\s]?s\b": "sas",
        r"\bs[\.\s]?a\b": "sa",
        r"\bs[\.\s]?c[\.\s]?i\b": "sci",
        r"\be[\.\s]?u[\.\s]?r[\.\s]?l\b": "eurl",
        r"\bs[\.\s]?n[\.\s]?c\b": "snc",
        r"\bg[\.\s]?i[\.\s]?e\b": "gie",
        r"\be[\.\s]?i[\.\s]?r[\.\s]?l\b": "eirl"
    }

    # French address designator abbreviations
    ADDRESS_ABBR = {
        r"\bbd\b": "boulevard",
        r"\bav\b": "avenue",
        r"\bimp\b": "impasse",
        r"\ball\b": "allee",
        r"\bche\b": "chemin",
        r"\bpl\b": "place",
        r"\brte\b": "route",
        r"\bzi\b": "zone industrielle",
        r"\bza\b": "zone artisanale"
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """Applies diacritic-safe NFKD normalization and contraction expansion."""
        if pd.isna(text) or str(text).strip() == "":
            return ""
        # 1. NFKD normalization
        norm = unicodedata.normalize("NFKD", str(text))
        cleaned = "".join(c for c in norm if not unicodedata.combining(c))
        cleaned = cleaned.lower()

        # 2. Normalize smart apostrophes and quotes
        cleaned = cleaned.replace("’", "'").replace("`", "'")

        # 3. French contraction expansion (d' -> d, l' -> l)
        cleaned = re.sub(r"\b([dl])'", r"\1 ", cleaned)

        # 4. Remove punctuation except alphanumeric and spaces
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        return " ".join(cleaned.split())

    @classmethod
    def normalize_business_name(cls, name: str) -> str:
        """Cleans business name and standardizes French legal abbreviations."""
        cleaned = cls.clean_text(name)
        for pattern, replacement in cls.LEGAL_SUFFIXES.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        return " ".join(cleaned.split())

    @classmethod
    def normalize_address(cls, address: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Cleans address, standardizes street tokens, and extracts:
        (cleaned_address, postal_code_5digit, department_code_2digit)
        """
        raw_str = str(address)
        
        # Extract 5-digit French postal code (e.g. 75008 -> postal=75008, dept=75)
        postal_match = re.search(r"\b(\d{2})(\d{3})\b", raw_str)
        postal_code = postal_match.group(0) if postal_match else None
        department_code = postal_match.group(1) if postal_match else None

        cleaned = cls.clean_text(raw_str)
        for pattern, replacement in cls.ADDRESS_ABBR.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        return cleaned, postal_code, department_code


# ------------------------------------------------------------------------------
# 4. Correlation Alignment (CORAL) for Unsupervised Covariance Adaptation
# ------------------------------------------------------------------------------
class CORALAligner(BaseEstimator, TransformerMixin):
    """
    Implements Correlation Alignment (Sun et al., 2016):
    Aligns second-order statistics (covariance) of source features to the target domain:
        min_A || Cov(X_source * A) - Cov(X_target) ||_F^2
        Transformation: X_adapted = (X_source - mu_S) * C_S^{-1/2} * C_T^{1/2} + mu_T
    """
    def __init__(self, lambda_reg: float = 1e-4):
        self.lambda_reg = lambda_reg
        self.mu_source_: Optional[np.ndarray] = None
        self.mu_target_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "CORALAligner":
        X_s = np.asarray(X_source, dtype=np.float64)
        X_t = np.asarray(X_target, dtype=np.float64)
        d = X_s.shape[1]

        self.mu_source_ = np.mean(X_s, axis=0)
        self.mu_target_ = np.mean(X_t, axis=0)

        # Empirical covariance with regularized diagonal for numerical stability
        cov_s = np.cov(X_s, rowvar=False) + np.eye(d) * self.lambda_reg
        cov_t = np.cov(X_t, rowvar=False) + np.eye(d) * self.lambda_reg

        # Matrix square root and inverse square root
        cov_s_inv_sqrt = la.inv(la.sqrtm(cov_s)).real
        cov_t_sqrt = la.sqrtm(cov_t).real

        self.A_ = cov_s_inv_sqrt @ cov_t_sqrt
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.A_ is None or self.mu_source_ is None or self.mu_target_ is None:
            raise ValueError("CORALAligner must be fitted before transforming.")
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mu_source_) @ self.A_ + self.mu_target_
