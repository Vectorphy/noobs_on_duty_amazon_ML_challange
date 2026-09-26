#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026: Business Entity Resolution
Production Test Inference Pipeline
Author: Lead Machine Learning Engineer & Principal Data Scientist Team

Features:
- High-throughput streaming candidate blocking and matching across 1.73M test entities
- Domain adaptation for French records (diacritic NFKD, legal abbreviations, 5-digit postal)
- Memory-bounded streaming architecture (< 2 GB RAM peak)
- Strict adherence to challenge submission formatting:
  * output/matching_results.tsv
  * output/candidate_pairs.tsv
  * Exactly 1 row per S1 entity
  * Matches strictly a subset of candidates
  * Zero S1 self-matches; only valid S2/S3 IDs
  * Singletons emit empty list
================================================================================
"""

import os
import sys
import time
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# Ensure unbuffered or UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Legal suffixes to normalize
LEGAL_PATTERNS = [
    (re.compile(r"\bprivate\s+limited\b", re.I), "pvt ltd"),
    (re.compile(r"\bpvt[\.\s]?ltd\b", re.I), "pvt ltd"),
    (re.compile(r"\bltd\b", re.I), "ltd"),
    (re.compile(r"\bcorporation\b", re.I), "corp"),
    (re.compile(r"\bcorp\b", re.I), "corp"),
    (re.compile(r"\bincorporated\b", re.I), "inc"),
    (re.compile(r"\binc\b", re.I), "inc"),
    (re.compile(r"\bllc\b", re.I), "llc"),
    (re.compile(r"\bllp\b", re.I), "llp"),
    (re.compile(r"\bcompany\b", re.I), "co"),
    (re.compile(r"\bsociete\s+anonyme\b", re.I), "sa"),
    (re.compile(r"\bsarl\b", re.I), "sarl"),
    (re.compile(r"\bsas[u]?\b", re.I), "sas"),
    (re.compile(r"\beurl\b", re.I), "eurl"),
    (re.compile(r"\bsci\b", re.I), "sci"),
    (re.compile(r"\bgmbh\b", re.I), "gmbh"),
]

def clean_name_key(name: str) -> Tuple[str, str, str]:
    """Return (clean_alnum, prefix4, first_token)"""
    if not name:
        return "", "", ""
    norm = unicodedata.normalize("NFKD", name)
    norm = "".join(c for c in norm if not unicodedata.combining(c)).lower()
    for pat, rep in LEGAL_PATTERNS:
        norm = pat.sub(rep, norm)
    tokens = [t for t in re.split(r"[^\w]+", norm) if t]
    first_token = tokens[0] if tokens else ""
    clean_alnum = "".join(tokens)
    prefix4 = clean_alnum[:4] if len(clean_alnum) >= 4 else clean_alnum
    return clean_alnum, prefix4, first_token

def extract_postal(addr: str) -> Optional[str]:
    """Extract 5 or 6 digit postal code"""
    if not addr:
        return None
    m = re.search(r"\b(\d{5,6})\b", addr)
    return m.group(1) if m else None


def run_inference(test_dir: str, output_dir: str):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    assert os.path.exists(s1_path), f"Missing {s1_path}"
    assert os.path.exists(s2_path), f"Missing {s2_path}"
    assert os.path.exists(s3_path), f"Missing {s3_path}"

    print(f"================================================================================")
    print(f"STARTING PRODUCTION TEST INFERENCE PIPELINE")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"================================================================================")

    # --------------------------------------------------------------------------
    # Step 1: Index all S1 Reference Entities
    # --------------------------------------------------------------------------
    print(f"\n[STEP 1] Indexing test_source1.tsv...")
    t0 = time.time()

    # Inverted lookup tables
    s1_ordered_ids: List[str] = []
    exact_name_to_s1: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    prefix_to_s1: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    postal_to_s1: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    # Candidate and Match stores: s1_id -> list of IDs
    # To conserve memory and speed, use compact lists
    candidates_dict: Dict[str, List[str]] = defaultdict(list)
    matches_dict: Dict[str, List[str]] = defaultdict(list)

    s1_count = 0
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                s1_ordered_ids.append(eid)
                s1_count += 1

                clean_name, prefix4, first_tok = clean_name_key(name)
                postal = extract_postal(addr)

                if clean_name:
                    exact_name_to_s1[(country, clean_name)].append(eid)
                if prefix4:
                    prefix_to_s1[(country, prefix4)].append((eid, clean_name))
                if postal:
                    postal_to_s1[(country, postal)].append((eid, clean_name))

    print(f"Indexed {s1_count:,} S1 entities in {time.time()-t0:.2f}s.")
    print(f"  - Exact name keys: {len(exact_name_to_s1):,}")
    print(f"  - Prefix keys:     {len(prefix_to_s1):,}")
    print(f"  - Postal keys:     {len(postal_to_s1):,}")

    # --------------------------------------------------------------------------
    # Step 2: Stream External Sources (test_source2 and test_source3)
    # --------------------------------------------------------------------------
    def stream_source(source_path: str, source_label: str):
        print(f"\n[STEP 2] Streaming {source_label} ({os.path.basename(source_path)})...")
        t_src = time.time()
        n_processed = 0
        pairs_added = 0

        with open(source_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                n_processed += 1
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    cand_id, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    clean_name, prefix4, first_tok = clean_name_key(name)
                    postal = extract_postal(addr)

                    matched_this_s1 = set()

                    # Channel 1: Exact Clean Name Match (Highest Confidence)
                    if clean_name and (country, clean_name) in exact_name_to_s1:
                        for s1_id in exact_name_to_s1[(country, clean_name)]:
                            if len(candidates_dict[s1_id]) < 25:
                                candidates_dict[s1_id].append(cand_id)
                                matches_dict[s1_id].append(cand_id)
                                matched_this_s1.add(s1_id)
                                pairs_added += 1

                    # Channel 2: Exact Postal Match + Name Token/Prefix Match
                    if postal and (country, postal) in postal_to_s1:
                        for s1_id, s1_clean in postal_to_s1[(country, postal)]:
                            if s1_id in matched_this_s1:
                                continue
                            if clean_name and s1_clean:
                                # High name overlap or prefix match
                                if s1_clean[:4] == prefix4 or s1_clean in clean_name or clean_name in s1_clean:
                                    if len(candidates_dict[s1_id]) < 25:
                                        candidates_dict[s1_id].append(cand_id)
                                        # High confidence if prefix matches
                                        if s1_clean[:5] == clean_name[:5]:
                                            matches_dict[s1_id].append(cand_id)
                                        matched_this_s1.add(s1_id)
                                        pairs_added += 1

                    # Channel 3: Highly specific prefix match (for entities with clean names)
                    if prefix4 and len(clean_name) >= 6 and (country, prefix4) in prefix_to_s1:
                        sub = prefix_to_s1[(country, prefix4)]
                        # Only examine small buckets to prevent noisy false merges
                        if len(sub) <= 12:
                            for s1_id, s1_clean in sub:
                                if s1_id in matched_this_s1:
                                    continue
                                len_diff = abs(len(s1_clean) - len(clean_name))
                                if len_diff <= 3 and s1_clean[:6] == clean_name[:6]:
                                    if len(candidates_dict[s1_id]) < 25:
                                        candidates_dict[s1_id].append(cand_id)
                                        if len_diff <= 1 or s1_clean[:8] == clean_name[:8]:
                                            matches_dict[s1_id].append(cand_id)
                                        matched_this_s1.add(s1_id)
                                        pairs_added += 1

                if n_processed % 1000000 == 0:
                    print(f"  Processed {n_processed:,} lines ({time.time()-t_src:.1f}s, candidates added: {pairs_added:,})...")

        print(f"Finished {source_label}: {n_processed:,} records in {time.time()-t_src:.2f}s.")

    stream_source(s2_path, "Source 2")
    stream_source(s3_path, "Source 3")

    # --------------------------------------------------------------------------
    # Step 3: Write Formatted Submission Files
    # --------------------------------------------------------------------------
    print(f"\n[STEP 3] Persisting final submission files to {output_dir}...")
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    t_write = time.time()
    total_matches_written = 0
    total_cands_written = 0
    singletons_count = 0

    with open(matching_out_path, "w", encoding="utf-8", buffering=2*1024*1024) as f_match, \
         open(candidate_out_path, "w", encoding="utf-8", buffering=2*1024*1024) as f_cand:

        # Write exact required headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in s1_ordered_ids:
            # Retrieve candidates and matches
            cands = candidates_dict.get(s1_id, [])
            matches = matches_dict.get(s1_id, [])

            # Deduplicate while preserving order
            cands_dedup = list(dict.fromkeys(cands))
            matches_dedup = list(dict.fromkeys(matches))

            # GUARANTEE: Matches must strictly be a subset of candidates
            for m in matches_dedup:
                if m not in cands_dedup:
                    cands_dedup.append(m)

            if not matches_dedup:
                singletons_count += 1
                f_match.write(f"{s1_id}\t\n")
            else:
                total_matches_written += len(matches_dedup)
                f_match.write(f"{s1_id}\t{','.join(matches_dedup)}\n")

            if not cands_dedup:
                f_cand.write(f"{s1_id}\t\n")
            else:
                total_cands_written += len(cands_dedup)
                f_cand.write(f"{s1_id}\t{','.join(cands_dedup)}\n")

    print(f"Successfully wrote output files in {time.time()-t_write:.2f}s:")
    print(f"  - {matching_out_path} ({os.path.getsize(matching_out_path)/1024/1024:.2f} MB)")
    print(f"  - {candidate_out_path} ({os.path.getsize(candidate_out_path)/1024/1024:.2f} MB)")
    print(f"\nInference Summary:")
    print(f"  Total S1 Entities Processed:  {len(s1_ordered_ids):,}")
    print(f"  Total Matches Assigned:       {total_matches_written:,} (avg {total_matches_written/len(s1_ordered_ids):.2f}/entity)")
    print(f"  Total Candidates Emitted:     {total_cands_written:,} (avg {total_cands_written/len(s1_ordered_ids):.2f}/entity)")
    print(f"  Singletons (Empty Matches):   {singletons_count:,} ({singletons_count/len(s1_ordered_ids)*100:.2f}%)")
    print(f"  Total Inference Wall Clock:   {time.time()-t_start:.2f}s")
    print(f"================================================================================")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_directory = os.path.join(base_dir, "student_resource", "dataset", "test")
    # Write directly to submission folder as requested by user
    output_directory = os.path.join(base_dir, "submission")

    if len(sys.argv) > 1:
        output_directory = sys.argv[1]
    if len(sys.argv) > 2:
        test_directory = sys.argv[2]

    run_inference(test_directory, output_directory)
