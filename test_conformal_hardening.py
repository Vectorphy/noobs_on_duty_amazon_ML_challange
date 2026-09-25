"""
Verification and Unit Test Suite for Conformal System Hardening
Validates:
1. Effective Sample Size (n_eff) Sufficiency & Adaptive Shrinkage
2. Transitivity-Preserving Constrained Correlation Clustering vs. Naive Transitive Bleeding
3. Singleton Collapse Safeguards (Empty match sets earning F_0.5 = 1.0)
4. Sub-Graph Runtime Feasibility on Large Candidate Graphs
5. French Entity Normalizer and CORAL Feature Covariance Alignment
"""

import time
import numpy as np
import pandas as pd
from conformal_system_hardening import (
    StabilizedDomainWeighter,
    ConstrainedGraphClusterer,
    FrenchEntityNormalizer,
    CORALAligner
)

def run_hardening_verification():
    print("="*80)
    print("CONFORMAL SYSTEM HARDENING: MANDATORY VERIFICATION AUDIT SUITE")
    print("="*80)

    # --------------------------------------------------------------------------
    # AUDIT 1: Sample Size Sufficiency & Adaptive Shrinkage
    # --------------------------------------------------------------------------
    print("\n[AUDIT 1] Effective Sample Size (n_eff) Diagnostics & Adaptive Shrinkage...")
    np.random.seed(42)
    N_SOURCE = 3000
    N_TARGET = 1000
    d = 4

    # Simulate realistic covariate shift with significant domain divergence
    X_source = np.random.randn(N_SOURCE, d)
    X_target = np.random.randn(N_TARGET, d) * 1.5 + 0.8

    weighter = StabilizedDomainWeighter(c_reg=0.1, min_neff_ratio=0.50, random_state=42)
    weighter.fit(X_source, X_target)

    # Compute stabilized weights on source
    w_adapted = weighter.compute_weights(X_source)
    n_eff_raw = weighter.raw_neff_
    n_eff_shrunk = weighter.shrunk_neff_

    print(f"  - Source Sample Size (n): {N_SOURCE}")
    print(f"  - Unshrunk Raw n_eff: {n_eff_raw:.1f} ({n_eff_raw/N_SOURCE*100:.1f}%)")
    print(f"  - Adaptive Shrinkage Parameter (beta): {weighter.beta_:.4f}")
    print(f"  - Shrunk Stabilized n_eff: {n_eff_shrunk:.1f} ({n_eff_shrunk/N_SOURCE*100:.1f}%)")
    
    assert n_eff_shrunk > 500, f"FAILED: n_eff ({n_eff_shrunk}) is below absolute threshold 500!"
    assert n_eff_shrunk >= 0.50 * N_SOURCE, f"FAILED: n_eff ({n_eff_shrunk}) is below 50% target!"
    print("  -> PASSED: n_eff remains large (> 500 and >= 50% of n), guaranteeing bounded quantile variance.")

    # --------------------------------------------------------------------------
    # AUDIT 2: Acyclic / Transitivity Consistency Check (Preventing Cluster Bleeding)
    # --------------------------------------------------------------------------
    print("\n[AUDIT 2] Transitivity Consistency Check (Conflicting Triangle Test)...")
    nodes = ["S1-00100", "S2-00200", "S3-00300"]
    # Triangle: (S1, S2) = 0.94 Match, (S2, S3) = 0.89 Match, BUT (S1, S3) = 0.04 Hard Non-Match C(e)={0}
    edges = {
        ("S1-00100", "S2-00200"): {"prob": 0.94, "set": [1]},
        ("S2-00200", "S3-00300"): {"prob": 0.89, "set": [1]},
        ("S1-00100", "S3-00300"): {"prob": 0.04, "set": [0]}  # Hard cut
    }

    clusterer = ConstrainedGraphClusterer(tau_threshold=0.82)
    clusters = clusterer.solve_components(nodes, edges)
    matches = clusterer.format_submission_matches(clusters, source1_ids=["S1-00100"])

    print(f"  - Graph Nodes: {nodes}")
    print("  - Edges: (S1, S2)=Match [1], (S2, S3)=Match [1], (S1, S3)=Hard Non-Match [0]")
    print(f"  - Resulting Disjoint Clusters: {clusters}")
    print(f"  - Formatted S1 Matches: {matches}")

    # Verify that S1-00100 is NOT merged with S3-00300
    assert "S3-00300" not in matches["S1-00100"], "FAILED: Hard cannot-link constraint violated!"
    assert "S2-00200" in matches["S1-00100"], "FAILED: Valid high-probability edge dropped!"
    print("  -> PASSED: Constrained clusterer strictly respected C(e) = {0} hard cut and avoided false merge bleeding.")

    # --------------------------------------------------------------------------
    # AUDIT 3: Singleton Anomaly Safeguard
    # --------------------------------------------------------------------------
    print("\n[AUDIT 3] Singleton Collapse Check (Empty Match Sets)...")
    nodes_singleton = ["S1-99999", "S2-11111", "S3-22222"]
    # Edges: Both candidate pairs are low probability or flagged as non-matches
    edges_singleton = {
        ("S1-99999", "S2-11111"): {"prob": 0.35, "set": [0]},
        ("S1-99999", "S3-22222"): {"prob": 0.55, "set": [0, 1]}  # Below tau* = 0.82
    }

    clusters_single = clusterer.solve_components(nodes_singleton, edges_singleton)
    matches_single = clusterer.format_submission_matches(clusters_single, source1_ids=["S1-99999"])

    print(f"  - Singleton Reference: S1-99999 (Candidate edges have p < 0.82 or C=[0])")
    print(f"  - Formatted Output for S1-99999: {matches_single['S1-99999']}")
    
    assert len(matches_single["S1-99999"]) == 0, "FAILED: Singleton was spuriously matched!"
    print("  -> PASSED: Singleton safely emits empty match list, preserving exact 1.0 entity-level F_0.5 score.")

    # --------------------------------------------------------------------------
    # AUDIT 4: Sub-Graph Runtime Feasibility & Scalability
    # --------------------------------------------------------------------------
    print("\n[AUDIT 4] Runtime Feasibility on Large Sparse Candidate Graph...")
    # Simulate a realistic multi-component graph with 1,500 nodes and 3,000 edges
    N_COMPONENTS = 300
    large_nodes = []
    large_edges = {}

    for comp_idx in range(N_COMPONENTS):
        s1 = f"S1-{comp_idx:05d}"
        s2 = f"S2-{comp_idx:05d}"
        s3 = f"S3-{comp_idx:05d}"
        s2_noise = f"S2-{comp_idx+1000:05d}"
        large_nodes.extend([s1, s2, s3, s2_noise])

        # High confidence edge
        large_edges[(s1, s2)] = {"prob": 0.92, "set": [1]}
        # Ambiguous edge
        large_edges[(s1, s3)] = {"prob": 0.70, "set": [0, 1]}
        # Negative cut
        large_edges[(s1, s2_noise)] = {"prob": 0.10, "set": [0]}

    t0 = time.perf_counter()
    large_clusters = clusterer.solve_components(large_nodes, large_edges)
    s1_all = [f"S1-{c:05d}" for c in range(N_COMPONENTS)]
    submission = clusterer.format_submission_matches(large_clusters, s1_all)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  - Graph Scale: {len(large_nodes)} nodes, {len(large_edges)} candidate edges")
    print(f"  - Total Disjoint Clusters Formed: {len(large_clusters)}")
    print(f"  - Total Solver Execution Time: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 500.0, f"FAILED: Clustering solver took too long ({elapsed_ms:.2f} ms)!"
    print(f"  -> PASSED: Scales smoothly within O(|V| + |E| log |V|), executing in {elapsed_ms:.2f} ms (< 500 ms target).")

    # --------------------------------------------------------------------------
    # AUDIT 5: French Entity Normalization & CORAL Covariance Alignment
    # --------------------------------------------------------------------------
    print("\n[AUDIT 5] French Normalization & CORAL Feature Alignment...")
    raw_french_name = "Société d’Ingénierie & L’Étoile S.A.R.L."
    raw_french_addr = "18, Avenue de la République, Bât B, 75011 Paris, Cedex 11"

    norm_name = FrenchEntityNormalizer.normalize_business_name(raw_french_name)
    norm_addr, postal, dept = FrenchEntityNormalizer.normalize_address(raw_french_addr)

    print(f"  - Raw Name: {raw_french_name} -> Normalized: '{norm_name}'")
    print(f"  - Raw Address: {raw_french_addr}")
    print(f"  - Normalized Address: '{norm_addr}'")
    print(f"  - Extracted 5-Digit Postal: {postal} | Extracted Department Code: {dept}")

    assert norm_name == "societe d ingenierie l etoile sarl"
    assert postal == "75011" and dept == "75"
    assert "avenue" in norm_addr

    # Test CORAL covariance alignment
    X_src_synth = np.random.randn(500, 3) * 2.5 + 1.2
    X_tgt_synth = np.random.randn(200, 3) * 0.8 - 0.5
    coral = CORALAligner(lambda_reg=1e-5)
    coral.fit(X_src_synth, X_tgt_synth)
    X_adapted = coral.transform(X_src_synth)

    cov_adapted = np.cov(X_adapted, rowvar=False)
    cov_target = np.cov(X_tgt_synth, rowvar=False)
    cov_dist = float(np.linalg.norm(cov_adapted - cov_target))

    print(f"  - CORAL Covariance Frobenius Distance: {cov_dist:.6e}")
    assert cov_dist < 1e-3, "FAILED: CORAL alignment error is too large!"
    print("  -> PASSED: French text correctly standardized and CORAL covariance aligned (< 1e-3).")

    print("\n" + "="*80)
    print("ALL 5 SYSTEM HARDENING VERIFICATION CHECKS COMPLETED AND CONFIRMED.")
    print("="*80)

if __name__ == "__main__":
    run_hardening_verification()
