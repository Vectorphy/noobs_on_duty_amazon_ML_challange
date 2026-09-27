"""Generate isolated v2 test candidate and prediction files without test labels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
from models_v2 import MODEL_NAMES, model_artifacts, positive_probability

from matching_v2 import (
    CPU_THREADS, FEATURE_ENGINE_VERSION, KEY_S1_LIMITS, KEY_TARGET_LIMIT, block_keys,
    score_speedup,
)
from train_v2 import BATCH, MAX_CANDIDATES_PER_SOURCE, quoted


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def input_signature(paths: list[Path], config: dict, model_path: Path) -> str:
    files = [{"name": p.name, "size": p.stat().st_size,
              "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    code_files = [Path(__file__).resolve(), Path(__file__).with_name("matching_v2.py")]
    code = [{"name": p.name, "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in code_files]
    model = {"size": model_path.stat().st_size, "mtime_ns": model_path.stat().st_mtime_ns}
    return json.dumps({"files": files, "model_config": config,
                       "model": model, "candidate_cap": config["candidate_cap_per_source"],
                       "feature_engine": FEATURE_ENGINE_VERSION, "code": code},
                      sort_keys=True, default=list)


def build_database(test_dir: Path, work: Path) -> duckdb.DuckDBPyConnection:
    work.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(work / "test_v2.duckdb"))
    con.execute(f"SET threads={CPU_THREADS}")
    con.execute("SET memory_limit='3GB'")
    spill = Path(tempfile.gettempdir()) / f"business_entity_resolution_v2_spill_{os.getpid()}"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={quoted(spill)}")
    con.execute("SET max_temp_directory_size='12GB'")
    con.create_function("block_keys", block_keys, ["VARCHAR", "VARCHAR"], "VARCHAR[]")
    csv = "delim='\\t', header=true, all_varchar=true, quote='', strict_mode=true"
    s1path = quoted(test_dir / "test_source1.tsv")
    s2path = quoted(test_dir / "test_source2.tsv")
    s3path = quoted(test_dir / "test_source3.tsv")
    log("Loading supplied test source records")
    con.execute(f"""
        CREATE OR REPLACE TABLE s1 AS
        SELECT CAST(row_number() OVER (ORDER BY entity_id)-1 AS INTEGER) ix,
               entity_id, coalesce(business_name,'') business_name,
               coalesce(business_address,'') business_address, country
        FROM read_csv({s1path}, {csv})
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE targets AS
        SELECT CAST(row_number() OVER ()-1 AS INTEGER) ix, entity_id,
               coalesce(business_name,'') business_name,
               coalesce(business_address,'') business_address, country, source_no
        FROM (
            SELECT *, CAST(2 AS TINYINT) source_no FROM read_csv({s2path}, {csv})
            UNION ALL
            SELECT *, CAST(3 AS TINYINT) source_no FROM read_csv({s3path}, {csv})
        )
    """)
    log("Building country-aware v2 blocking keys")
    con.execute("""
        CREATE OR REPLACE TABLE s1_keys_all AS
        SELECT s.ix s1_ix, s.country, u.key
        FROM s1 s, UNNEST(block_keys(s.business_name,s.business_address)) u(key)
    """)
    having = " OR ".join(
        f"(key LIKE '{prefix}:%' AND count(*)<={limit})"
        for prefix, limit in KEY_S1_LIMITS.items()
    )
    con.execute(f"""
        CREATE OR REPLACE TABLE valid_keys AS
        SELECT country,key FROM s1_keys_all GROUP BY country,key HAVING {having}
    """)
    con.execute("""
        CREATE OR REPLACE TABLE s1_keys AS
        SELECT k.* FROM s1_keys_all k JOIN valid_keys v USING(country,key)
    """)
    con.execute("DROP TABLE s1_keys_all")
    con.execute("""
        CREATE OR REPLACE TABLE target_keys_all AS
        SELECT t.ix target_ix,t.country,u.key
        FROM targets t, UNNEST(block_keys(t.business_name,t.business_address)) u(key)
        JOIN valid_keys v ON t.country=v.country AND u.key=v.key
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE target_keys AS
        SELECT k.* FROM target_keys_all k JOIN (
            SELECT country,key FROM target_keys_all GROUP BY country,key
            HAVING count(*)<={KEY_TARGET_LIMIT}
        ) v USING(country,key)
    """)
    con.execute("DROP TABLE target_keys_all")

    con.execute("""
        CREATE OR REPLACE TABLE candidates (
            s1_ix INTEGER,target_ix INTEGER,source_no TINYINT,
            rank TINYINT,rank_score REAL
        )
    """)
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
            raise RuntimeError("Less than 5 GiB free disk remains during v2 candidate generation")
    return con


def extract_and_score(con: duckdb.DuckDBPyConnection, model: object,
                      columns: tuple[int, ...], work: Path,
                      signature: str, candidate_cap: int) -> np.memmap:
    n = con.execute("SELECT count(*) FROM candidates WHERE rank<=?", [candidate_cap]).fetchone()[0]
    progress_path = work / "test_scoring_progress.json"
    feature_path = work / "test_features.npy"
    probability_path = work / "test_probabilities.npy"
    offset = 0
    if progress_path.exists() and feature_path.exists() and probability_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("signature") == signature and saved.get("total_rows") == n:
            offset = min(n, int(saved.get("completed_rows", 0)))
            if saved.get("complete"):
                log(f"Reusing completed test scores: {n:,} pairs")
                return np.load(probability_path, mmap_mode="r+")
            if offset:
                log(f"Resuming test scoring at pair {offset:,}/{n:,}")
    features = np.lib.format.open_memmap(
        feature_path, mode="r+" if offset else "w+", dtype="float32",
        shape=(n, len(columns)),
    )
    probabilities = np.lib.format.open_memmap(
        probability_path, mode="r+" if offset else "w+", dtype="float32", shape=(n,),
    )
    cursor = con.execute("""
        SELECT c.s1_ix,c.target_ix,c.source_no,s.business_name,s.business_address,
               t.business_name,t.business_address
        FROM candidates c JOIN s1 s ON c.s1_ix=s.ix
        JOIN targets t ON c.target_ix=t.ix
        WHERE c.rank<=? ORDER BY c.s1_ix,c.target_ix
    """, [candidate_cap])
    skipped = offset
    while skipped:
        rows = cursor.fetchmany(min(BATCH, skipped))
        if not rows:
            raise ValueError("Saved inference checkpoint exceeds candidate query rows")
        skipped -= len(rows)
    run_started = time.monotonic()
    checkpoint_started = run_started
    checkpoint_offset = offset
    while rows := cursor.fetchmany(BATCH):
        stop = offset + len(rows)
        batch_features = score_speedup(rows, layout="inference", workers=CPU_THREADS)
        features[offset:stop] = batch_features[:, list(columns)]
        probabilities[offset:stop] = positive_probability(model, features[offset:stop])
        offset = stop
        if offset - checkpoint_offset >= 500_000 or offset == n:
            features.flush()
            probabilities.flush()
            progress_tmp = progress_path.with_suffix(".tmp")
            progress_tmp.write_text(json.dumps({"signature": signature,
                "completed_rows": offset, "total_rows": n, "complete": offset == n}),
                encoding="utf-8")
            os.replace(progress_tmp, progress_path)
            now = time.monotonic()
            elapsed = max(now - checkpoint_started, 1e-6)
            overall_rate = offset / max(now - run_started, 1e-6)
            eta = (n - offset) / max(overall_rate, 1e-6)
            log(f"Scored {offset:,}/{n:,} ({offset/n:.1%}); "
                f"{(offset-checkpoint_offset)/elapsed:,.0f} pairs/s; ETA {eta/60:.1f} min")
            checkpoint_offset = offset
            checkpoint_started = now
    if offset != n:
        raise ValueError(f"Scored {offset} test candidates, expected {n}")
    del features
    return probabilities


def write_outputs(con: duckdb.DuckDBPyConnection, probabilities: np.memmap,
                  threshold: float, output: Path, candidate_cap: int) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidate_pairs.tsv"
    matches_path = output / "matching_results.tsv"
    s1_cursor = con.cursor().execute("SELECT ix,entity_id FROM s1 ORDER BY ix")
    pair_cursor = con.cursor().execute("""
        SELECT c.s1_ix,t.entity_id FROM candidates c
        JOIN targets t ON c.target_ix=t.ix
        WHERE c.rank<=? ORDER BY c.s1_ix,c.target_ix
    """, [candidate_cap])
    pair = pair_cursor.fetchone()
    offset = pair_count = match_count = 0
    with candidates_path.open("w",encoding="utf-8",newline="") as candidate_file, \
         matches_path.open("w",encoding="utf-8",newline="") as match_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        match_file.write("source1_entity_id\tmatched_entity_ids\n")
        while s1_rows := s1_cursor.fetchmany(BATCH):
            for s1_ix, s1_id in s1_rows:
                candidate_ids, matched_ids = [], []
                while pair is not None and pair[0] == s1_ix:
                    target_id = pair[1]
                    candidate_ids.append(target_id)
                    if probabilities[offset] >= threshold:
                        matched_ids.append(target_id)
                        match_count += 1
                    offset += 1
                    pair_count += 1
                    pair = pair_cursor.fetchone()
                candidate_file.write(f"{s1_id}\t{','.join(candidate_ids)}\n")
                match_file.write(f"{s1_id}\t{','.join(matched_ids)}\n")
    if offset != len(probabilities):
        raise ValueError(f"Output consumed {offset} scored pairs, expected {len(probabilities)}")
    return {"source1_rows": con.execute("SELECT count(*) FROM s1").fetchone()[0],
            "candidate_pairs": pair_count,"predicted_matches": match_count}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir",type=Path,default=root/"student_resource"/"dataset"/"test")
    parser.add_argument("--artifacts-dir",type=Path,default=root/"code"/"business_entity_resolution"/"artifacts"/"v2")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default="lightgbm")
    args = parser.parse_args()
    artifacts = model_artifacts(args.artifacts_dir, args.model)
    output_dir = args.output_dir or artifacts / "test_output"
    metrics_path = artifacts/"training_metrics.json"
    if not metrics_path.is_file() or "final_validation" not in json.loads(metrics_path.read_text(encoding="utf-8")):
        raise RuntimeError("Complete held-out validation before running test inference")
    config = json.loads((artifacts/"model_config.json").read_text(encoding="utf-8"))
    if config.get("estimator", "lightgbm") != args.model:
        raise ValueError(f"Requested {args.model}, but artifacts contain {config.get('estimator')}")
    model = joblib.load(artifacts/"matcher_model.joblib")
    candidate_cap = int(config["candidate_cap_per_source"])
    columns = tuple(config["feature_indices"])
    work = artifacts / "test_work"
    work.mkdir(parents=True, exist_ok=True)
    resources = shutil.disk_usage(artifacts).free / 2**30
    if resources < 20:
        raise RuntimeError(f"Need 20 GiB free disk for test inference; found {resources:.1f} GiB")
    con = build_database(args.test_dir,work)
    try:
        signature = input_signature([args.test_dir / name for name in (
            "test_source1.tsv", "test_source2.tsv", "test_source3.tsv")], config,
            artifacts / "matcher_model.joblib")
        probabilities = extract_and_score(con,model,columns,work,signature,candidate_cap)
        summary = write_outputs(con,probabilities,config["decision_threshold"],output_dir,candidate_cap)
        probabilities._mmap.close()
    finally:
        con.close()
    report = {"model_config":config,"outputs":summary,
              "test_labels_used":False,"country_note":"France is present only in test; no test labels were read."}
    (output_dir/"inference_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    validator = root/"student_resource"/"utils"/"validate_submission.py"
    import subprocess
    subprocess.run([
        str(Path(__import__("sys").executable)),str(validator),
        "--matching",str(output_dir/"matching_results.tsv"),
        "--candidate",str(output_dir/"candidate_pairs.tsv"),
        "--test-dir",str(args.test_dir),
    ],check=True)
    log(f"Wrote isolated v2 test outputs to {output_dir}")


if __name__ == "__main__":
    main()
