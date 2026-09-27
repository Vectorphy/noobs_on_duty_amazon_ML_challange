"""Generate v3 test candidate and prediction files without test labels.

Changes from infer_v2
---------------------
* No GPU engine: plain duckdb only; gpu_connection import removed entirely.
* Reads artifacts from artifacts/v3/ (model_config.json, matcher_model.joblib).
* Per-country thresholds (C1): applies country_thresholds from model_config.json;
  falls back to global decision_threshold when a country is unseen.
* Mutual-exclusivity suppression (C2): after scoring, for each S1 entity that
  has multiple predicted matches whose addresses are identical or near-identical,
  only the highest-probability prediction is kept.
* Adaptive candidate depth (A3): candidates table uses rank up to
  deep_cap_per_source (48) for ambiguous S1 entities; inference scores them all
  and write_outputs applies the ranked-24 / full-48 cap consistently.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np

from matching_v3 import (
    KEY_S1_LIMITS, KEY_TARGET_LIMIT, SINGLETON_GUARD_THRESHOLD,
    block_keys, pair_features,
)
from train_v3 import BATCH, MAX_CANDIDATES_PER_SOURCE, MAX_CANDIDATES_DEEP, DEEP_CANDIDATE_THRESHOLD, quoted


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def build_database(test_dir: Path, work: Path) -> duckdb.DuckDBPyConnection:
    """Load test sources, build blocking keys, and generate candidates."""
    work.mkdir(parents=True, exist_ok=True)
    spill = work / "spill"
    db_path = work / "test_v3.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("SET threads=15")
    con.execute("SET memory_limit='1536MB'")
    con.execute(f"SET temp_directory={quoted(spill)}")
    con.execute("SET max_temp_directory_size='20GB'")
    con.create_function("block_keys", block_keys, ["VARCHAR", "VARCHAR"], "VARCHAR[]")
    csv = "delim='\\t', header=true, all_varchar=true, quote='', strict_mode=true"
    s1path = quoted(test_dir / "test_source1.tsv")
    s2path = quoted(test_dir / "test_source2.tsv")
    s3path = quoted(test_dir / "test_source3.tsv")
    log("Loading supplied test source records")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS s1 AS
        SELECT CAST(row_number() OVER (ORDER BY entity_id)-1 AS INTEGER) ix,
               entity_id, coalesce(business_name,'') business_name,
               coalesce(business_address,'') business_address, country
        FROM read_csv({s1path}, {csv})
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS targets AS
        SELECT CAST(row_number() OVER ()-1 AS INTEGER) ix, entity_id,
               coalesce(business_name,'') business_name,
               coalesce(business_address,'') business_address, country, source_no
        FROM (
            SELECT *, CAST(2 AS TINYINT) source_no FROM read_csv({s2path}, {csv})
            UNION ALL
            SELECT *, CAST(3 AS TINYINT) source_no FROM read_csv({s3path}, {csv})
        )
    """)
    log("Building country-aware v3 blocking keys")
    con.execute("""
        CREATE TABLE IF NOT EXISTS s1_keys_all AS
        SELECT s.ix s1_ix, s.country, u.key
        FROM s1 s, UNNEST(block_keys(s.business_name,s.business_address)) u(key)
    """)
    having = " OR ".join(
        f"(key LIKE '{prefix}:%' AND count(*)<={limit})"
        for prefix, limit in KEY_S1_LIMITS.items()
    )
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS valid_keys AS
        SELECT country,key FROM s1_keys_all GROUP BY country,key HAVING {having}
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS s1_keys AS
        SELECT k.* FROM s1_keys_all k JOIN valid_keys v USING(country,key)
    """)
    con.execute("DROP TABLE IF EXISTS s1_keys_all")
    con.execute("""
        CREATE TABLE IF NOT EXISTS target_keys_all AS
        SELECT t.ix target_ix,t.country,u.key
        FROM targets t, UNNEST(block_keys(t.business_name,t.business_address)) u(key)
        JOIN valid_keys v ON t.country=v.country AND u.key=v.key
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS target_keys AS
        SELECT k.* FROM target_keys_all k JOIN (
            SELECT country,key FROM target_keys_all GROUP BY country,key
            HAVING count(*)<={KEY_TARGET_LIMIT}
        ) v USING(country,key)
    """)
    con.execute("DROP TABLE IF EXISTS target_keys_all")

    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    has_candidates = False
    if "candidates" in tables:
        cand_count = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
        if cand_count > 0:
            has_candidates = True
            log(f"Reusing existing candidates table: {cand_count:,} pairs")

    if not has_candidates:
        con.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                s1_ix INTEGER,target_ix INTEGER,source_no TINYINT,
                rank TINYINT,rank_score REAL
            )
        """)
        # Normal depth pass
        for part in range(32):
            con.execute(f"""
                INSERT INTO candidates
                WITH pairs AS (
                    SELECT DISTINCT sk.s1_ix,tk.target_ix
                    FROM s1_keys sk JOIN target_keys tk USING(country,key)
                    WHERE sk.s1_ix % 32={part}
                ), scored AS (
                    SELECT p.s1_ix,p.target_ix,t.source_no,t.entity_id target_id,
                           CAST(CASE WHEN s.business_address<>'' AND t.business_address<>''
                               THEN 0.65*jaro_winkler_similarity(lower(s.business_name),lower(t.business_name))
                                  + 0.35*jaro_winkler_similarity(lower(s.business_address),lower(t.business_address))
                               ELSE jaro_winkler_similarity(lower(s.business_name),lower(t.business_name))
                           END AS REAL) rank_score
                    FROM pairs p JOIN s1 s ON p.s1_ix=s.ix
                                 JOIN targets t ON p.target_ix=t.ix
                ), ranked AS (
                    SELECT *,row_number() OVER (
                        PARTITION BY s1_ix,source_no ORDER BY rank_score DESC,target_id
                    ) rank FROM scored
                )
                SELECT s1_ix,target_ix,source_no,CAST(rank AS TINYINT),rank_score
                FROM ranked WHERE rank<={MAX_CANDIDATES_PER_SOURCE}
            """)
            if part == 0 or (part + 1) % 4 == 0:
                n = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
                log(f"Candidate partitions {part+1}/32; retained {n:,} pairs")
            if part and (part + 1) % 4 == 0 and shutil.disk_usage(work).free < 5 * 2**30:
                raise RuntimeError("Less than 5 GiB free disk remains during v3 candidate generation")

        # A3: adaptive depth extension
        log(f"A3: extending to rank {MAX_CANDIDATES_DEEP} for ambiguous S1 entities")
        con.execute(f"""
            INSERT INTO candidates
            WITH top_scores AS (
                SELECT s1_ix, source_no, max(rank_score) top_score
                FROM candidates GROUP BY s1_ix, source_no
            ), ambiguous AS (
                SELECT DISTINCT s1_ix FROM top_scores
                WHERE top_score < {DEEP_CANDIDATE_THRESHOLD}
            ), new_pairs AS (
                SELECT DISTINCT sk.s1_ix, tk.target_ix
                FROM s1_keys sk JOIN target_keys tk USING(country,key)
                JOIN ambiguous a ON sk.s1_ix=a.s1_ix
                WHERE NOT EXISTS (
                    SELECT 1 FROM candidates c
                    WHERE c.s1_ix=sk.s1_ix AND c.target_ix=tk.target_ix
                )
            ), scored AS (
                SELECT p.s1_ix,p.target_ix,t.source_no,t.entity_id target_id,
                       CAST(CASE WHEN s.business_address<>'' AND t.business_address<>''
                           THEN 0.65*jaro_winkler_similarity(lower(s.business_name),lower(t.business_name))
                              + 0.35*jaro_winkler_similarity(lower(s.business_address),lower(t.business_address))
                           ELSE jaro_winkler_similarity(lower(s.business_name),lower(t.business_name))
                       END AS REAL) rank_score
                FROM new_pairs p JOIN s1 s ON p.s1_ix=s.ix
                                 JOIN targets t ON p.target_ix=t.ix
            ), reranked AS (
                SELECT *, {MAX_CANDIDATES_PER_SOURCE} + row_number() OVER (
                    PARTITION BY s1_ix,source_no ORDER BY rank_score DESC,target_id
                ) rank FROM scored
            )
            SELECT s1_ix,target_ix,source_no,CAST(rank AS TINYINT),rank_score
            FROM reranked WHERE rank<={MAX_CANDIDATES_DEEP}
        """)
        deep_extra = con.execute(
            f"SELECT count(*) FROM candidates WHERE rank>{MAX_CANDIDATES_PER_SOURCE}"
        ).fetchone()[0]
        log(f"Adaptive depth added {deep_extra:,} extra pairs")

    return con


def extract_and_score(
    con: duckdb.DuckDBPyConnection,
    model: object,
    columns: tuple[int, ...],
    work: Path,
    cap: int,
) -> np.memmap:
    n = con.execute(f"SELECT count(*) FROM candidates WHERE rank<={cap}").fetchone()[0]
    features = np.lib.format.open_memmap(
        work / "test_features.npy", mode="w+", dtype="float32",
        shape=(n, len(columns)),
    )
    probabilities = np.lib.format.open_memmap(
        work / "test_probabilities.npy", mode="w+", dtype="float32", shape=(n,),
    )
    cursor = con.execute(f"""
        SELECT c.s1_ix,c.target_ix,c.source_no,s.business_name,s.business_address,
               t.business_name,t.business_address
        FROM candidates c JOIN s1 s ON c.s1_ix=s.ix
        JOIN targets t ON c.target_ix=t.ix
        WHERE c.rank<={cap} ORDER BY c.s1_ix,c.target_ix
    """)
    offset = 0

    def _extract_row(r):
        return pair_features(r[3], r[4], r[5], r[6], r[2])[list(columns)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        while rows := cursor.fetchmany(BATCH):
            stop = offset + len(rows)
            for i, feat in enumerate(executor.map(_extract_row, rows), offset):
                features[i] = feat
            probabilities[offset:stop] = model.predict_proba(features[offset:stop])[:, 1]
            offset = stop
            if offset and offset % 500_000 < BATCH:
                log(f"Scored {offset:,}/{n:,} test candidate pairs")
    if offset != n:
        raise ValueError(f"Scored {offset} test candidates, expected {n}")
    features.flush()
    probabilities.flush()
    del features
    return probabilities


def _suppress_mutual_exclusivity(
    matched_ids: list[str],
    probabilities_map: dict[str, float],
    target_addresses: dict[str, str],
) -> list[str]:
    """C2: keep only the highest-probability match when multiple matched
    candidates share essentially the same address (exact after normalisation).

    This applies conservatively: two addresses are considered identical only
    after lowercasing and whitespace collapse.  The rest pass through unchanged.
    """
    if len(matched_ids) <= 1:
        return matched_ids

    def _norm_addr(addr: str) -> str:
        import re
        return re.sub(r"\s+", " ", addr.lower().strip())

    # Group by normalised address; keep best-probability per group
    groups: dict[str, list[str]] = {}
    for eid in matched_ids:
        key = _norm_addr(target_addresses.get(eid, ""))
        groups.setdefault(key, []).append(eid)

    kept = []
    for addr_key, eids in groups.items():
        if addr_key == "" or len(eids) == 1:
            kept.extend(eids)
        else:
            best = max(eids, key=lambda e: probabilities_map.get(e, 0.0))
            kept.append(best)
    return kept


def write_outputs(
    con: duckdb.DuckDBPyConnection,
    probabilities: np.memmap,
    config: dict,
    output: Path,
    cap: int,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidate_pairs.tsv"
    matches_path = output / "matching_results.tsv"

    global_threshold: float = config["decision_threshold"]
    country_thresholds: dict[str, float] = config.get("country_thresholds", {})
    apply_me: bool = config.get("mutual_exclusivity_suppression", False)
    singleton_guard: float = config.get("singleton_guard_threshold", SINGLETON_GUARD_THRESHOLD)

    # Build target address lookup for C2
    target_addresses: dict[str, str] = {}
    if apply_me:
        for eid, addr in con.execute("SELECT entity_id, business_address FROM targets").fetchall():
            target_addresses[eid] = addr or ""

    # s1 country lookup for C1
    s1_country: dict[int, str] = {}
    for ix, country in con.execute("SELECT ix, country FROM s1").fetchall():
        s1_country[ix] = country or ""

    # C3: pre-compute per-s1 max probability in a single pass over the scored array
    log("C3: computing per-S1 max probability for singleton guard")
    n_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    max_prob_per_s1: np.ndarray = np.zeros(n_s1, dtype=np.float32)
    order_cursor = con.cursor().execute(
        f"SELECT c.s1_ix FROM candidates c WHERE c.rank<={cap} ORDER BY c.s1_ix, c.target_ix"
    )
    for offset_start in range(0, len(probabilities), 100_000):
        chunk_size = min(100_000, len(probabilities) - offset_start)
        rows = order_cursor.fetchmany(chunk_size)
        if not rows:
            break
        for rel_i, (s1_ix,) in enumerate(rows):
            p = float(probabilities[offset_start + rel_i])
            if p > max_prob_per_s1[s1_ix]:
                max_prob_per_s1[s1_ix] = p

    s1_cursor = con.cursor().execute("SELECT ix,entity_id FROM s1 ORDER BY ix")
    pair_cursor = con.cursor().execute(f"""
        SELECT c.s1_ix,t.entity_id FROM candidates c
        JOIN targets t ON c.target_ix=t.ix
        WHERE c.rank<={cap} ORDER BY c.s1_ix,c.target_ix
    """)
    pair = pair_cursor.fetchone()
    offset = pair_count = match_count = 0
    with candidates_path.open("w", encoding="utf-8", newline="") as candidate_file, \
         matches_path.open("w", encoding="utf-8", newline="") as match_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        match_file.write("source1_entity_id\tmatched_entity_ids\n")
        while s1_rows := s1_cursor.fetchmany(BATCH):
            for s1_ix, s1_id in s1_rows:
                # C1: pick per-country threshold
                country = s1_country.get(s1_ix, "")
                threshold = country_thresholds.get(country, global_threshold)

                candidate_ids: list[str] = []
                matched_ids: list[str] = []
                prob_map: dict[str, float] = {}
                while pair is not None and pair[0] == s1_ix:
                    target_id = pair[1]
                    candidate_ids.append(target_id)
                    p = float(probabilities[offset])
                    prob_map[target_id] = p
                    if p >= threshold:
                        matched_ids.append(target_id)
                        match_count += 1
                    offset += 1
                    pair_count += 1
                    pair = pair_cursor.fetchone()

                # C3: singleton guard — if entity's max prob < guard, predict empty
                if singleton_guard > 0.0 and max_prob_per_s1[s1_ix] < singleton_guard:
                    match_count -= len(matched_ids)
                    matched_ids = []

                # C2: mutual exclusivity suppression
                if apply_me and len(matched_ids) > 1:
                    matched_ids = _suppress_mutual_exclusivity(
                        matched_ids, prob_map, target_addresses
                    )
                    match_count += len(matched_ids) - len(prob_map)  # already counted above; adjust

                candidate_file.write(f"{s1_id}\t{','.join(candidate_ids)}\n")
                match_file.write(f"{s1_id}\t{','.join(matched_ids)}\n")
    if offset != len(probabilities):
        raise ValueError(f"Output consumed {offset} scored pairs, expected {len(probabilities)}")
    return {
        "source1_rows": con.execute("SELECT count(*) FROM s1").fetchone()[0],
        "candidate_pairs": pair_count,
        "predicted_matches": match_count,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path,
                        default=root / "student_resource" / "dataset" / "test")
    parser.add_argument("--artifacts-dir", type=Path,
                        default=root / "code" / "business_entity_resolution" / "artifacts" / "v3")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "code" / "business_entity_resolution" / "artifacts" / "v3" / "test_output")
    args = parser.parse_args()
    artifacts = args.artifacts_dir
    metrics_path = artifacts / "training_metrics.json"
    if not metrics_path.is_file() or "final_validation" not in json.loads(
        metrics_path.read_text(encoding="utf-8")
    ):
        raise RuntimeError("Complete held-out validation before running test inference")
    config = json.loads((artifacts / "model_config.json").read_text(encoding="utf-8"))
    model = joblib.load(artifacts / "matcher_model.joblib")
    columns = tuple(config["feature_indices"])
    cap = config.get("candidate_cap_per_source", MAX_CANDIDATES_PER_SOURCE)
    work = artifacts / "test_work_15threads"
    resources = shutil.disk_usage(artifacts).free / 2**30
    needed_disk = 5 if (work / "test_v3.duckdb").exists() else 25
    if resources < needed_disk:
        raise RuntimeError(f"Need {needed_disk} GiB free disk; found {resources:.1f} GiB")
    con = build_database(args.test_dir, work)
    try:
        probabilities = extract_and_score(con, model, columns, work, cap)
        summary = write_outputs(con, probabilities, config, args.output_dir, cap)
        probabilities._mmap.close()
    finally:
        con.close()
    report = {
        "model_config": config,
        "outputs": summary,
        "test_labels_used": False,
        "country_note": "France is present only in test; no test labels were read.",
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    validator = root / "student_resource" / "utils" / "validate_submission.py"
    import subprocess
    subprocess.run([
        str(Path(__import__("sys").executable)), str(validator),
        "--matching", str(args.output_dir / "matching_results.tsv"),
        "--candidate", str(args.output_dir / "candidate_pairs.tsv"),
        "--test-dir", str(args.test_dir),
    ], check=True)
    log(f"Wrote isolated v3 test outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
