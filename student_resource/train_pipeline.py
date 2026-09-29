#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026: Business Entity Resolution
Production ML Training Pipeline

Architecture:
1. High-Recall Multi-Index Candidate Blocking on Train Data
2. 15-Dimensional Pairwise Continuous Feature Engineering
3. Pure ML Matcher: LightGBM Classifier with Metric-Aligned Precision Tuning
4. Vectorized Out-of-fold Validation & Optimal Decision Threshold Calibration
5. Serializes Trained Model to artifacts/matcher_model.joblib
================================================================================
"""

import os
import sys
import time
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import rapidfuzz.fuzz as fuzz
from lightgbm import LGBMClassifier
import joblib

sys.stdout.reconfigure(encoding='utf-8')

def find_dirs():
    curr = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        sr = os.path.join(curr, "student_resource")
        ds_cand = os.path.join(sr, "dataset", "train")
        if os.path.exists(ds_cand):
            return (
                ds_cand,
                os.path.join(sr, "artifacts"),
                curr
            )
        ds = os.path.join(curr, "dataset", "train")
        if os.path.exists(ds):
            return (
                ds,
                os.path.join(curr, "artifacts"),
                os.path.dirname(curr)
            )
        curr = os.path.dirname(curr)
    raise FileNotFoundError("Could not find student_resource or dataset directory")

DATA_DIR, ARTIFACTS_DIR, REPO_DIR = find_dirs()
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

STOP_WORDS = {
    'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'llc', 'llp', 'co', 'company',
    'the', 'and', '&', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an', 'sa', 'sas', 'sarl', 'eurl',
    'enterprises', 'solutions', 'services', 'group', 'holdings', 'technologies', 'industries', 'international'
}

ADDR_STOP_WORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr', 'nagar',
    'colony', 'marg', 'near', 'opp', 'opposite', 'behind', 'floor', 'flr', 'shop', 'no',
    'plot', 'sector', 'sec', 'block', 'blk', 'phase', 'delhi', 'mumbai', 'india', 'us',
    'usa', 'france', 'paris', 'city', 'state', 'town', 'post', 'building', 'bldg', 'complex',
    'apartment', 'apt', 'west', 'east', 'north', 'south', 'new', 'old'
}

import unicodedata

def extract_name_tokens(s: str):
    if not s or s == 'nan': return []
    s = re.sub(r'[^\w\s]', ' ', str(s).lower())
    return [w for w in s.split() if w not in STOP_WORDS and len(w) >= 3]

def extract_addr_numbers(s: str):
    if not s or s == 'nan': return []
    tokens = re.findall(r'\b[a-zA-Z0-9\-\/]{1,10}\d+[a-zA-Z0-9\-\/]{0,10}\b', str(s))
    res = set()
    for t in tokens:
        clean = re.sub(r'[^a-z0-9]', '', t.lower())
        if 2 <= len(clean) <= 12: res.add(clean)
    digits = re.findall(r'\b\d+\b', str(s))
    for d in digits:
        if 2 <= len(d) <= 10: res.add(d)
    return list(res)

def extract_addr_words(s: str):
    if not s or s == 'nan': return []
    s = re.sub(r'[^\w\s]', ' ', str(s).lower())
    return [w for w in s.split() if w not in STOP_WORDS and w not in ADDR_STOP_WORDS and len(w) >= 4]

def is_non_latin(s: str):
    if not s: return False
    return any(ord(c) > 0x024F for c in str(s) if c.isalpha())

def clean_company_name(s: str) -> str:
    if not s or s == 'nan': return ""
    norm = unicodedata.normalize('NFKD', str(s))
    norm = "".join(c for c in norm if not unicodedata.combining(c)).lower()
    norm = re.sub(r'\.(com|org|net|in|co|info|c0m|biz|io|us|fr)(\.[a-z]{2})?$', '', norm)
    norm = re.sub(r'[^\w\s]', ' ', norm)
    tokens = [w for w in norm.split() if w not in STOP_WORDS]
    return "".join(tokens)

FEATURE_NAMES = [
    'fuzz_ratio_name', 'fuzz_tsr_name', 'fuzz_tset_name', 'name_word_jaccard', 'name_len_diff_ratio',
    'fuzz_ratio_addr', 'fuzz_tsr_addr', 'fuzz_tset_addr', 'addr_word_jaccard',
    'shared_numeric_code_count', 'has_shared_numeric_code', 'missing_address_flag',
    'non_latin_script_flag', 'domain_stem_match', 'exact_clean_name_match'
]

def extract_pair_features(name1, addr1, name2, addr2, nums1, nums2):
    n1, n2 = str(name1).lower(), str(name2).lower()
    has_a1 = bool(addr1 and addr1 != 'nan' and str(addr1).strip())
    has_a2 = bool(addr2 and addr2 != 'nan' and str(addr2).strip())
    a1 = str(addr1).lower() if has_a1 else ''
    a2 = str(addr2).lower() if has_a2 else ''

    nr = fuzz.ratio(n1, n2)
    ntsr = fuzz.token_sort_ratio(n1, n2)
    ntset = fuzz.token_set_ratio(n1, n2)

    w1 = set(extract_name_tokens(name1))
    w2 = set(extract_name_tokens(name2))
    name_jaccard = len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.0

    max_len = max(len(n1), len(n2), 1)
    len_diff_ratio = abs(len(n1) - len(n2)) / max_len

    if has_a1 and has_a2:
        ar = fuzz.ratio(a1, a2)
        atsr = fuzz.token_sort_ratio(a1, a2)
        atset = fuzz.token_set_ratio(a1, a2)
        aw1 = set(extract_addr_words(addr1))
        aw2 = set(extract_addr_words(addr2))
        addr_jaccard = len(aw1 & aw2) / len(aw1 | aw2) if (aw1 | aw2) else 0.0
    else:
        ar, atsr, atset, addr_jaccard = 0.0, 0.0, 0.0, 0.0

    shared_nums = len(set(nums1) & set(nums2))
    has_shared_num = 1.0 if shared_nums > 0 else 0.0
    missing_addr = 1.0 if not (has_a1 and has_a2) else 0.0
    nl = 1.0 if (is_non_latin(name1) or is_non_latin(name2)) else 0.0

    c1 = clean_company_name(name1)
    c2 = clean_company_name(name2)
    domain_match = 1.0 if (len(c1) >= 4 and len(c2) >= 4 and (c1 in c2 or c2 in c1)) else 0.0
    exact_clean = 1.0 if (c1 and c1 == c2) else 0.0

    return [
        nr, ntsr, ntset, name_jaccard, len_diff_ratio,
        ar, atsr, atset, addr_jaccard,
        float(shared_nums), has_shared_num, missing_addr,
        nl, domain_match, exact_clean
    ]

def run_training(n_s1_records: int = 30000, n_stream_records: int = 250000, max_train_pairs: int = 120000):
    print("="*80, flush=True)
    print("STARTING PURE ML MODEL TRAINING PIPELINE", flush=True)
    print(f"S1 Pool Size:         {n_s1_records:,} records", flush=True)
    print(f"External Stream Pool: {n_stream_records:,} records from S2 and S3", flush=True)
    print(f"Max Training Pairs:   {max_train_pairs:,} pairs", flush=True)
    print(f"Artifacts Directory:  {ARTIFACTS_DIR}", flush=True)
    print("="*80, flush=True)

    # 1. Load Ground Truth
    print("\n[PHASE 1] Ingesting Full Ground Truth Labels...", flush=True)
    t0 = time.time()
    gt_path = os.path.join(DATA_DIR, "train_ground_truth.tsv")
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                gt_dict[s1_id] = set(x.strip() for x in parts[1].split(',') if x.strip())
            else:
                gt_dict[s1_id] = set()
    print(f"Loaded {len(gt_dict):,} ground truth mappings in {time.time()-t0:.2f}s.", flush=True)

    # 2. Ingest & Index S1 Records
    print(f"\n[PHASE 2] Indexing {n_s1_records:,} Reference S1 Records...", flush=True)
    t1 = time.time()
    s1_path = os.path.join(DATA_DIR, "train_source1.tsv")
    s1_records = {} # eid -> (name, addr, country, name_tokens, nums)
    idx_name = defaultdict(list)
    idx_num = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_clean_name = defaultdict(list)

    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= n_s1_records: break
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            nt = extract_name_tokens(name)
            nums = extract_addr_numbers(addr)
            aw = extract_addr_words(addr)
            cn = clean_company_name(name)
            s1_records[eid] = (name, addr, country, nt, nums)

            if len(cn) >= 4: idx_clean_name[(country, cn)].append(eid)
            for t in nt: idx_name[(country, t)].append(eid)
            for num in nums: idx_num[(country, num)].append(eid)
            for w in aw: idx_addr[(country, w)].append(eid)

    all_s1_ids = list(s1_records.keys())
    split_idx = int(0.70 * len(all_s1_ids))
    train_s1_ids = set(all_s1_ids[:split_idx])
    val_s1_ids = set(all_s1_ids[split_idx:])

    print(f"Indexed {len(s1_records):,} S1 entities in {time.time()-t1:.2f}s.", flush=True)
    print(f"  Train S1 Partition: {len(train_s1_ids):,} entities (70%)", flush=True)
    print(f"  Val S1 Partition:   {len(val_s1_ids):,} entities (30%)", flush=True)

    # 3. Stream External Records to Form Candidate Pairs
    print(f"\n[PHASE 3] Generating Candidate Training Pairs via Real Multi-Index Blocking...", flush=True)
    t2 = time.time()
    X_train, y_train = [], []
    val_cand_pairs = defaultdict(list) # s1_id -> list of (eid_ext, feats)

    def process_source_file(file_path: str, max_records: int):
        print(f"  Streaming {os.path.basename(file_path)} (up to {max_records:,} records)...", flush=True)
        with open(file_path, 'r', encoding='utf-8') as f:
            f.readline()
            for i, line in enumerate(f):
                if i >= max_records: break
                parts = line.strip().split('\t')
                if len(parts) < 4: continue
                eid_ext, name_ext, addr_ext, country_ext = parts[0], parts[1], parts[2], parts[3]

                nt_ext = extract_name_tokens(name_ext)
                nums_ext = extract_addr_numbers(addr_ext)
                aw_ext = extract_addr_words(addr_ext)
                cn_ext = clean_company_name(name_ext)

                candidates = set()
                if len(cn_ext) >= 4:
                    matches = idx_clean_name.get((country_ext, cn_ext))
                    if matches and len(matches) <= 10:
                        candidates.update(matches)
                for t in nt_ext:
                    matches = idx_name.get((country_ext, t))
                    if matches and len(matches) <= 35:
                        candidates.update(matches)
                for num in nums_ext:
                    matches = idx_num.get((country_ext, num))
                    if matches and len(matches) <= 20:
                        candidates.update(matches)
                for w in aw_ext:
                    matches = idx_addr.get((country_ext, w))
                    if matches and len(matches) <= 20:
                        candidates.update(matches)

                if not candidates: continue

                for s1_id in candidates:
                    s1_name, s1_addr, _, _, s1_nums = s1_records[s1_id]
                    is_true = eid_ext in gt_dict.get(s1_id, set())

                    if s1_id in train_s1_ids:
                        if len(X_train) < max_train_pairs or is_true:
                            feats = extract_pair_features(s1_name, s1_addr, name_ext, addr_ext, s1_nums, nums_ext)
                            X_train.append(feats)
                            y_train.append(1 if is_true else 0)
                    elif s1_id in val_s1_ids:
                        if len(val_cand_pairs[s1_id]) < 15:
                            feats = extract_pair_features(s1_name, s1_addr, name_ext, addr_ext, s1_nums, nums_ext)
                            val_cand_pairs[s1_id].append((eid_ext, feats))

    s2_path = os.path.join(DATA_DIR, "train_source2.tsv")
    s3_path = os.path.join(DATA_DIR, "train_source3.tsv")
    process_source_file(s2_path, n_stream_records)
    process_source_file(s3_path, n_stream_records)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    print(f"\nGenerated {len(y_train):,} training pairs in {time.time()-t2:.2f}s.", flush=True)
    print(f"  Positive Match Pairs (Class 1): {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)", flush=True)
    print(f"  Hard Negative Pairs  (Class 0): {n_neg:,} ({n_neg/len(y_train)*100:.2f}%)", flush=True)

    # 4. Train Pure ML Matcher (LightGBM)
    print("\n[PHASE 4] Fitting Pure Machine Learning GBDT Classifier...", flush=True)
    t3 = time.time()

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    print(f"Model training converged in {time.time()-t3:.2f}s.", flush=True)

    print("\nFeature Importance Rankings:", flush=True)
    for fname, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {fname:28s}: {imp}", flush=True)

    # 5. Out-of-fold Vectorized Validation & Threshold Calibration
    print(f"\n[PHASE 5] Out-of-Fold Threshold Calibration on {len(val_s1_ids):,} Holdout S1 Entities...", flush=True)
    t4 = time.time()

    val_s1_list = list(val_s1_ids)
    all_val_feats = []
    val_map = [] # (s1_id, eid_ext)

    for s1_id in val_s1_list:
        for eid_ext, feats in val_cand_pairs.get(s1_id, []):
            all_val_feats.append(feats)
            val_map.append((s1_id, eid_ext))

    print(f"Batch scoring {len(all_val_feats):,} validation candidate pairs...", flush=True)
    if all_val_feats:
        val_probs = model.predict_proba(np.array(all_val_feats, dtype=np.float32))[:, 1]
    else:
        val_probs = np.array([])

    # Group candidate scores by s1_id
    val_scored_dict = defaultdict(list)
    for (s1_id, eid_ext), prob in zip(val_map, val_probs):
        val_scored_dict[s1_id].append((eid_ext, prob))

    thresholds = [0.50, 0.65, 0.75, 0.80, 0.85, 0.90]
    best_f05 = -1.0
    best_tau = 0.80

    for tau in thresholds:
        s1_scores = []
        total_tp, total_fp = 0, 0

        for s1_id in val_s1_ids:
            true_set = gt_dict.get(s1_id, set())
            cands = val_scored_dict.get(s1_id, [])

            passing = [eid for eid, prob in cands if prob >= tau]
            # Limit to top 6 matches
            pred_set = set(passing[:6])

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            total_tp += tp
            total_fp += fp

            if not pred_set and not true_set:
                s1_scores.append(1.0)
            elif not pred_set or not true_set:
                s1_scores.append(0.0)
            else:
                p = tp / len(pred_set)
                r = tp / len(true_set)
                f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
                s1_scores.append(f05)

        macro_f05 = float(np.mean(s1_scores))
        pair_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        print(f"  Cutoff tau={tau:.2f} -> Macro F0.5: {macro_f05:.4f} | Pair Prec: {pair_prec*100:.2f}% | TP: {total_tp:,}, FP: {total_fp:,}", flush=True)

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_tau = tau

    print(f"\nOptimal Decision Threshold Selected: tau* = {best_tau:.2f} (Peak Macro F0.5: {best_f05:.4f})", flush=True)
    print(f"Validation completed in {time.time()-t4:.2f}s.", flush=True)

    # 6. Save Model Artifacts
    model_artifact_path = os.path.join(ARTIFACTS_DIR, "matcher_model.joblib")
    meta_artifact_path = os.path.join(ARTIFACTS_DIR, "model_config.joblib")

    joblib.dump(model, model_artifact_path)
    joblib.dump({
        "optimal_threshold": best_tau,
        "feature_names": FEATURE_NAMES,
        "peak_macro_f05": best_f05,
        "n_features": len(FEATURE_NAMES)
    }, meta_artifact_path)

    code_artifacts_dir = os.path.join(REPO_DIR, "code", "business_entity_resolution", "artifacts")
    os.makedirs(code_artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(code_artifacts_dir, "matcher_model.joblib"))
    joblib.dump({
        "optimal_threshold": best_tau,
        "feature_names": FEATURE_NAMES,
        "peak_macro_f05": best_f05
    }, os.path.join(code_artifacts_dir, "model_config.joblib"))

    print(f"\nPersisted Model Artifacts:", flush=True)
    print(f"  - {model_artifact_path} ({os.path.getsize(model_artifact_path)/1024:.1f} KB)", flush=True)
    print(f"  - {meta_artifact_path}", flush=True)
    print("="*80, flush=True)
    print("TRAINING PIPELINE COMPLETE: MODEL READY FOR PRODUCTION INFERENCE", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    run_training(n_s1_records=30000, n_stream_records=250000, max_train_pairs=120000)
