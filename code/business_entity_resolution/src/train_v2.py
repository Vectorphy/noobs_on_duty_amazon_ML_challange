"""Full-corpus, held-out v2 business entity resolution experiment.

Run from the repository root with:
    python code/business_entity_resolution/src/train_v2.py

The existing submission model and inference script are never overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
from lightgbm import LGBMClassifier

from matching_v2 import (
    ABLATION_GROUPS, BASE_FEATURES, FEATURES, KEY_S1_LIMITS,
    KEY_TARGET_LIMIT, MAX_CANDIDATES_PER_SOURCE, block_keys, f05,
    pair_features,
)


SEED = 42
CAPS = (16, 24, 32)
MAX_NEGATIVES = 6_000_000
SELECT_PAIRS = 1_000_000
BATCH = 20_000
META_DTYPE = np.dtype([
    ("s1", "<i4"), ("target", "<i4"), ("rank", "u1"),
    ("source", "u1"), ("positive", "u1"),
])
EXPECTED_ROWS = {"s1": 2_206_821, "s2": 5_034_616, "s3": 5_285_603}


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def quoted(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def available_ram_gib() -> float:
    if os.name != "nt":
        return 0.0

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_page",
                "available_page", "total_virtual", "available_virtual",
                "available_extended",
            )
        ]

    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Could not read available RAM")
    return status.available_physical / 2**30


def preflight(data_dir: Path, artifacts: Path) -> dict:
    paths = {name: data_dir / f"train_source{name[-1]}.tsv" for name in ("s1", "s2", "s3")}
    paths["ground_truth"] = data_dir / "train_ground_truth.tsv"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing supplied training files: " + ", ".join(missing))
    artifacts.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(artifacts).free / 2**30
    free_ram = available_ram_gib()
    needed_disk = 5 if (artifacts/"work"/"pipeline.duckdb").exists() else 25
    if free_disk < needed_disk:
        raise RuntimeError(f"Need at least {needed_disk} GiB free for disk-backed work; found {free_disk:.1f} GiB")
    if free_ram and free_ram < 3.5:
        raise RuntimeError(f"Need at least 3.5 GiB available RAM; found {free_ram:.1f} GiB")
    return {
        "free_disk_gib_at_start": round(free_disk, 2),
        "available_ram_gib_at_start": round(free_ram, 2),
        "input_bytes": {name: path.stat().st_size for name, path in paths.items()},
    }


def connect(work: Path) -> duckdb.DuckDBPyConnection:
    work.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(work / "pipeline.duckdb"))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='1536MB'")
    con.execute(f"SET temp_directory={quoted(work / 'spill')}")
    con.execute("SET max_temp_directory_size='20GB'")
    con.create_function("block_keys", block_keys, ["VARCHAR", "VARCHAR"], "VARCHAR[]")
    return con


def count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def stage_done(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    con.execute("CREATE TABLE IF NOT EXISTS pipeline_progress (name VARCHAR PRIMARY KEY)")
    return bool(con.execute("SELECT count(*) FROM pipeline_progress WHERE name=?",[name]).fetchone()[0])


def mark_stage(con: duckdb.DuckDBPyConnection, name: str) -> None:
    con.execute("INSERT INTO pipeline_progress VALUES (?)",[name])


def drop_tables(con: duckdb.DuckDBPyConnection, names: tuple[str, ...]) -> None:
    for name in names:
        con.execute(f"DROP TABLE IF EXISTS {name}")


def import_data(con: duckdb.DuckDBPyConnection, data: Path) -> dict:
    log("Loading all training source rows")
    s1 = quoted(data / "train_source1.tsv")
    s2 = quoted(data / "train_source2.tsv")
    s3 = quoted(data / "train_source3.tsv")
    gt = quoted(data / "train_ground_truth.tsv")
    csv = "delim='\\t', header=true, all_varchar=true, quote='', strict_mode=true"
    con.execute(f"""
        CREATE TABLE s1 AS
        SELECT CAST(row_number() OVER (ORDER BY entity_id)-1 AS INTEGER) ix,
               entity_id, coalesce(business_name,'') business_name,
               coalesce(business_address,'') business_address, country,
               CAST(0 AS SMALLINT) n_true, CAST(0 AS TINYINT) split
        FROM read_csv({s1}, {csv})
    """)
    con.execute(f"""
        CREATE TABLE targets AS
        SELECT CAST(row_number() OVER ()-1 AS INTEGER) ix, entity_id,
               coalesce(business_name,'') business_name,
               coalesce(business_address,'') business_address, country, source_no
        FROM (
            SELECT *, CAST(2 AS TINYINT) source_no FROM read_csv({s2}, {csv})
            UNION ALL
            SELECT *, CAST(3 AS TINYINT) source_no FROM read_csv({s3}, {csv})
        )
    """)
    con.execute(f"CREATE TABLE ground_truth AS SELECT * FROM read_csv({gt}, {csv})")
    rows = {
        "s1": count(con, "s1"),
        "s2": int(con.execute("SELECT count(*) FROM targets WHERE source_no=2").fetchone()[0]),
        "s3": int(con.execute("SELECT count(*) FROM targets WHERE source_no=3").fetchone()[0]),
        "ground_truth": count(con, "ground_truth"),
    }
    if any(rows[name] != expected for name, expected in EXPECTED_ROWS.items()):
        raise ValueError(f"Training source row counts disagree with full-data EDA: {rows}")
    if rows["ground_truth"] != rows["s1"]:
        raise ValueError("Ground truth must contain one row per Source 1 entity")
    invalid_gt = con.execute("""
        SELECT count(*) FROM ground_truth g LEFT JOIN s1 s
        ON g.source1_entity_id=s.entity_id WHERE s.ix IS NULL
    """).fetchone()[0]
    duplicate_gt = con.execute("""
        SELECT count(*) - count(DISTINCT source1_entity_id) FROM ground_truth
    """).fetchone()[0]
    if invalid_gt or duplicate_gt:
        raise ValueError(f"Ground-truth S1 integrity failed: unknown={invalid_gt}, duplicates={duplicate_gt}")
    con.execute("""
        CREATE TABLE links_raw AS
        SELECT s.ix s1_ix, trim(u.target_id) target_id
        FROM ground_truth g JOIN s1 s ON g.source1_entity_id=s.entity_id,
             UNNEST(string_split(coalesce(g.matched_entity_ids,''), ',')) u(target_id)
        WHERE trim(u.target_id) <> ''
    """)
    con.execute("""
        CREATE TABLE links AS
        SELECT l.s1_ix, t.ix target_ix, t.source_no
        FROM links_raw l JOIN targets t ON l.target_id=t.entity_id
    """)
    if count(con, "links") != count(con, "links_raw"):
        raise ValueError("Ground truth contains unknown target IDs")
    repeated = con.execute("""
        SELECT count(*) FROM (
            SELECT target_ix FROM links GROUP BY target_ix HAVING count(*)>1
        )
    """).fetchone()[0]
    if repeated:
        raise ValueError(f"{repeated} target IDs link to multiple S1 entities; split isolation needs review")
    con.execute("""
        UPDATE s1 SET n_true=x.n_true FROM (
            SELECT s1_ix, CAST(count(*) AS SMALLINT) n_true FROM links GROUP BY s1_ix
        ) x WHERE s1.ix=x.s1_ix
    """)
    con.execute("""
        CREATE TEMP TABLE split_ranks AS
        SELECT ix, row_number() OVER (
            PARTITION BY country, CASE WHEN n_true=0 THEN 0 WHEN n_true=1 THEN 1 ELSE 2 END
            ORDER BY hash(entity_id || '|42'), entity_id
        ) rn, count(*) OVER (
            PARTITION BY country, CASE WHEN n_true=0 THEN 0 WHEN n_true=1 THEN 1 ELSE 2 END
        ) n
        FROM s1
    """)
    con.execute("""
        UPDATE s1 SET split=CASE
            WHEN r.rn <= ceil(0.70*r.n) THEN 0
            WHEN r.rn <= ceil(0.85*r.n) THEN 1 ELSE 2 END
        FROM split_ranks r WHERE s1.ix=r.ix
    """)
    rows["positive_links"] = count(con, "links")
    rows["split_rows"] = dict(con.execute("SELECT split,count(*) FROM s1 GROUP BY split ORDER BY split").fetchall())
    log(f"Loaded {rows['s1']:,} S1, {rows['s2']+rows['s3']:,} targets, {rows['positive_links']:,} links")
    return rows


def build_keys(con: duckdb.DuckDBPyConnection) -> dict:
    log("Building country-specific blocking keys")
    con.execute("""
        CREATE TABLE s1_keys_all AS
        SELECT s.ix s1_ix, s.country, u.key
        FROM s1 s, UNNEST(block_keys(s.business_name,s.business_address)) u(key)
    """)
    cases = " OR ".join(
        f"(key LIKE '{prefix}:%' AND count(*)<={limit})"
        for prefix, limit in KEY_S1_LIMITS.items()
    )
    con.execute(f"""
        CREATE TABLE valid_keys AS
        SELECT country,key FROM s1_keys_all GROUP BY country,key HAVING {cases}
    """)
    con.execute("""
        CREATE TABLE s1_keys AS
        SELECT a.* FROM s1_keys_all a JOIN valid_keys v USING(country,key)
    """)
    con.execute("DROP TABLE s1_keys_all")
    con.execute("""
        CREATE TABLE target_keys_all AS
        SELECT t.ix target_ix, t.country, u.key
        FROM targets t, UNNEST(block_keys(t.business_name,t.business_address)) u(key)
        JOIN valid_keys v ON t.country=v.country AND u.key=v.key
    """)
    con.execute(f"""
        CREATE TABLE target_keys AS
        SELECT a.* FROM target_keys_all a JOIN (
            SELECT country,key FROM target_keys_all GROUP BY country,key
            HAVING count(*)<={KEY_TARGET_LIMIT}
        ) v USING(country,key)
    """)
    con.execute("DROP TABLE target_keys_all")
    result = {"s1_keys": count(con, "s1_keys"), "target_keys": count(con, "target_keys")}
    log(f"Usable keys: S1={result['s1_keys']:,}, targets={result['target_keys']:,}")
    return result


def build_candidates(con: duckdb.DuckDBPyConnection, work: Path) -> dict:
    log("Ranking blocked candidates in 32 bounded S1 partitions")
    con.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            s1_ix INTEGER, target_ix INTEGER, source_no TINYINT,
            rank TINYINT, rank_score REAL
        )
    """)
    con.execute("CREATE TABLE IF NOT EXISTS candidate_parts (part INTEGER PRIMARY KEY)")
    done = {row[0] for row in con.execute("SELECT part FROM candidate_parts").fetchall()}
    started = time.monotonic()
    for part in range(32):
        if part in done:
            continue
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"""
            INSERT INTO candidates
            WITH pairs AS (
                SELECT DISTINCT sk.s1_ix, tk.target_ix
                FROM s1_keys sk JOIN target_keys tk USING(country,key)
                WHERE sk.s1_ix % 32 = {part}
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
                SELECT *, row_number() OVER (
                    PARTITION BY s1_ix,source_no ORDER BY rank_score DESC,target_id
                ) rank FROM scored
            )
            SELECT s1_ix,target_ix,source_no,CAST(rank AS TINYINT),rank_score
            FROM ranked WHERE rank<={MAX_CANDIDATES_PER_SOURCE}
            """)
            con.execute("INSERT INTO candidate_parts VALUES (?)",[part])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        if part == 0:
            elapsed = time.monotonic() - started
            log(f"First candidate partition: {count(con, 'candidates'):,} pairs in {elapsed:.1f}s; ~{elapsed*32/3600:.1f}h at this pace")
        elif (part + 1) % 4 == 0:
            log(f"Candidate partitions {part+1}/32, retained pairs {count(con, 'candidates'):,}")
            if shutil.disk_usage(work).free < 5 * 2**30:
                raise RuntimeError("Less than 5 GiB disk remains during candidate generation")
    result = {"retained_pairs_at_32_per_source": count(con, "candidates")}
    log(f"Candidate generation complete: {result['retained_pairs_at_32_per_source']:,} pairs")
    return result


def make_train_selection(con: duckdb.DuckDBPyConnection) -> dict:
    log("Selecting all retrieved training positives and bounded hard negatives")
    con.execute("""
        CREATE TABLE train_selected AS
        SELECT c.* FROM candidates c JOIN s1 s ON c.s1_ix=s.ix
        JOIN links l ON l.s1_ix=c.s1_ix AND l.target_ix=c.target_ix
        WHERE s.split=0
    """)
    positives = count(con, "train_selected")
    con.execute(f"""
        INSERT INTO train_selected
        SELECT c.* FROM candidates c JOIN s1 s ON c.s1_ix=s.ix
        ANTI JOIN links l ON l.s1_ix=c.s1_ix AND l.target_ix=c.target_ix
        WHERE s.split=0
        ORDER BY CASE WHEN c.rank<=4 THEN 0 ELSE 1 END,
                 hash(c.s1_ix::VARCHAR || ':' || c.target_ix::VARCHAR || ':42')
        LIMIT {MAX_NEGATIVES}
    """)
    result = {"retrieved_train_positives": positives, "sampled_train_negatives": count(con, "train_selected")-positives}
    log(f"Training pairs: {positives:,} positives, {result['sampled_train_negatives']:,} negatives")
    return result


def pair_query(table: str, split: int | None = None, cap: int = 32) -> str:
    where = f"WHERE s.split={split} AND c.rank<={cap}" if split is not None else ""
    return f"""
        SELECT c.s1_ix,c.target_ix,c.rank,c.source_no,
               CAST(l.target_ix IS NOT NULL AS TINYINT) positive,
               s.business_name,s.business_address,t.business_name,t.business_address
        FROM {table} c JOIN s1 s ON c.s1_ix=s.ix
        JOIN targets t ON c.target_ix=t.ix
        LEFT JOIN links l ON l.s1_ix=c.s1_ix AND l.target_ix=c.target_ix
        {where}
        ORDER BY c.s1_ix,c.target_ix
    """


def extract_features(con: duckdb.DuckDBPyConnection, work: Path, name: str,
                     table: str, split: int | None = None, cap: int = 32) -> tuple[np.memmap, np.memmap]:
    where = f" JOIN s1 s ON c.s1_ix=s.ix WHERE s.split={split} AND c.rank<={cap}" if split is not None else ""
    n = int(con.execute(f"SELECT count(*) FROM {table} c{where}").fetchone()[0])
    marker = work/f"{name}_complete.json"
    x_path = work/f"{name}_features.npy"
    meta_path = work/f"{name}_meta.npy"
    if marker.exists() and x_path.exists() and meta_path.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved == {"rows":n,"cap":cap,"split":split}:
            log(f"Reusing completed {name} features: {n:,} rows")
            return np.load(x_path,mmap_mode="r"),np.load(meta_path,mmap_mode="r")
    log(f"Extracting {name}: {n:,} candidate feature rows")
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype="float32", shape=(n, len(FEATURES)))
    meta = np.lib.format.open_memmap(meta_path, mode="w+", dtype=META_DTYPE, shape=(n,))
    cursor = con.execute(pair_query(table, split, cap))
    offset = 0
    while rows := cursor.fetchmany(BATCH):
        for row in rows:
            s1_ix,target_ix,rank,source,positive,name1,addr1,name2,addr2 = row
            x[offset] = pair_features(name1,addr1,name2,addr2,source)
            meta[offset] = (s1_ix,target_ix,rank,source,positive)
            offset += 1
        if offset // 500_000 != (offset-len(rows)) // 500_000:
            log(f"{name} features: {offset:,}/{n:,}")
    if offset != n:
        raise ValueError(f"{name} query returned {offset} rows, expected {n}")
    x.flush()
    meta.flush()
    marker.write_text(json.dumps({"rows":n,"cap":cap,"split":split}),encoding="utf-8")
    return x, meta


def split_arrays(con: duckdb.DuckDBPyConnection, split: int) -> dict:
    rows = con.execute("SELECT ix,n_true,country,business_address FROM s1 WHERE split=? ORDER BY ix", [split]).fetchall()
    if not rows:
        raise ValueError(f"Empty split {split}")
    ids = np.fromiter((row[0] for row in rows), dtype=np.int32, count=len(rows))
    truth = np.fromiter((row[1] for row in rows), dtype=np.int16, count=len(rows))
    countries = np.asarray([row[2] for row in rows])
    missing_address = np.fromiter((not bool(row[3]) for row in rows), dtype=bool, count=len(rows))
    ix_to_local = np.full(EXPECTED_ROWS["s1"], -1, dtype=np.int32)
    ix_to_local[ids] = np.arange(len(ids), dtype=np.int32)
    linked_to_missing_address = np.zeros(len(ids),dtype=bool)
    for (ix,) in con.execute("""
        SELECT DISTINCT l.s1_ix FROM links l JOIN s1 s ON l.s1_ix=s.ix
        JOIN targets t ON l.target_ix=t.ix
        WHERE s.split=? AND t.business_address=''
    """,[split]).fetchall():
        linked_to_missing_address[ix_to_local[ix]] = True
    return {"ix": ids, "truth": truth, "country": countries,
            "missing_address": missing_address,
            "linked_to_missing_address": linked_to_missing_address,
            "ix_to_local": ix_to_local}


def model_for(columns: tuple[int, ...], memory_bounded: bool = False) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=15 if memory_bounded else 31,
        min_child_samples=100, max_bin=31 if memory_bounded else 63,
        n_jobs=1 if memory_bounded else 4, force_col_wise=memory_bounded,
        random_state=SEED, importance_type="gain", verbosity=-1,
    )


def fit_sample(x: np.memmap, meta: np.memmap, cap: int,
               columns: tuple[int, ...]) -> LGBMClassifier:
    eligible = np.flatnonzero(meta["rank"] <= cap)
    rng = np.random.default_rng(SEED)
    if len(eligible) > SELECT_PAIRS:
        eligible = rng.choice(eligible, SELECT_PAIRS, replace=False)
    model = model_for(columns)
    model.fit(np.asarray(x[eligible][:, columns]), np.asarray(meta["positive"][eligible]))
    return model


def predict_memmap(model: LGBMClassifier, x: np.memmap,
                   columns: tuple[int, ...], path: Path) -> np.memmap:
    prob = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(x),))
    for start in range(0, len(x), BATCH):
        stop = min(len(x), start+BATCH)
        prob[start:stop] = model.predict_proba(np.asarray(x[start:stop][:, columns]))[:, 1]
    prob.flush()
    return prob


def score_grid(meta: np.memmap, prob: np.memmap, split: dict,
               cap: int) -> tuple[float, float, dict]:
    n = len(split["truth"])
    total = np.zeros((n, 101), dtype=np.int32)
    true = np.zeros((n, 101), dtype=np.int32)
    for start in range(0, len(meta), BATCH):
        stop = min(len(meta), start+BATCH)
        piece = meta[start:stop]
        use = piece["rank"] <= cap
        local = split["ix_to_local"][piece["s1"][use]]
        bins = np.minimum(100, (prob[start:stop][use]*100).astype(np.int16))
        np.add.at(total, (local,bins), 1)
        is_true = piece["positive"][use] == 1
        np.add.at(true, (local[is_true],bins[is_true]), 1)
    predicted = np.zeros(n, dtype=np.int32)
    tp = np.zeros(n, dtype=np.int32)
    best = (-1.0, -1.0)
    curve = []
    for step in range(100, -1, -1):
        predicted += total[:,step]
        tp += true[:,step]
        score = float(f05(split["truth"], predicted, tp).mean())
        threshold = step/100
        curve.append({"threshold": threshold, "macro_f05": score})
        if (score, threshold) > best:
            best = (score, threshold)
    return best[0], best[1], {"curve": curve, "predicted": predicted, "tp": tp}


def evaluate_model(model: LGBMClassifier, x: np.memmap, meta: np.memmap,
                   split: dict, caps: tuple[int, ...], columns: tuple[int, ...],
                   work: Path) -> dict:
    path = work / "current_prob.npy"
    prob = predict_memmap(model,x,columns,path)
    result = {}
    for cap in caps:
        score, threshold, detail = score_grid(meta,prob,split,cap)
        result[cap] = {"macro_f05": score, "threshold": threshold,
                       "threshold_curve": detail["curve"]}
    del prob
    return result


def candidate_diagnostics(con: duckdb.DuckDBPyConnection, cap: int, split: int) -> dict:
    raw = con.execute("""
        SELECT l.source_no, count(*) links,
               count(c.target_ix) retrieved
        FROM links l JOIN s1 s ON l.s1_ix=s.ix
        LEFT JOIN candidates c ON c.s1_ix=l.s1_ix AND c.target_ix=l.target_ix AND c.rank<=?
        WHERE s.split=? GROUP BY l.source_no ORDER BY l.source_no
    """, [cap,split]).fetchall()
    by_source = {str(source): {"links": n, "retrieved": hit,
                               "recall": hit/n if n else 0.0}
                 for source,n,hit in raw}
    recovered = np.zeros(EXPECTED_ROWS["s1"], dtype=np.int16)
    for ix, hits in con.execute("""
        SELECT l.s1_ix,count(*) FROM links l JOIN s1 s ON l.s1_ix=s.ix
        JOIN candidates c ON c.s1_ix=l.s1_ix AND c.target_ix=l.target_ix AND c.rank<=?
        WHERE s.split=? GROUP BY l.s1_ix
    """, [cap,split]).fetchall():
        recovered[ix] = hits
    rows = con.execute("SELECT ix,n_true FROM s1 WHERE split=?", [split]).fetchall()
    ix = np.fromiter((r[0] for r in rows), dtype=np.int32, count=len(rows))
    truth = np.fromiter((r[1] for r in rows), dtype=np.int16, count=len(rows))
    hit = recovered[ix]
    oracle = f05(truth,hit,hit)
    n_candidate = con.execute("""
        SELECT count(*),count(DISTINCT c.s1_ix)
        FROM candidates c JOIN s1 s ON c.s1_ix=s.ix
        WHERE s.split=? AND c.rank<=?
    """, [split,cap]).fetchone()
    return {
        "by_source": by_source,
        "all_true_links_retained_fraction": float(np.mean(hit==truth)),
        "oracle_macro_f05": float(oracle.mean()),
        "entities_without_candidates": len(rows)-n_candidate[1],
        "candidate_pairs": n_candidate[0],
        "mean_candidates_per_s1": n_candidate[0]/len(rows),
    }


def final_slices(meta: np.memmap, prob: np.memmap, split: dict,
                 cap: int, threshold: float) -> dict:
    n = len(split["truth"])
    predicted = np.zeros(n, dtype=np.int32)
    tp = np.zeros(n, dtype=np.int32)
    for start in range(0,len(meta),BATCH):
        stop = min(len(meta),start+BATCH)
        piece = meta[start:stop]
        use = (piece["rank"]<=cap) & (prob[start:stop]>=threshold)
        local = split["ix_to_local"][piece["s1"][use]]
        np.add.at(predicted,local,1)
        np.add.at(tp,local,piece["positive"][use])
    scores = f05(split["truth"],predicted,tp)
    groups = {
        "country": {value: split["country"]==value for value in np.unique(split["country"])},
        "degree": {"singleton": split["truth"]==0, "one": split["truth"]==1,
                   "multiple": split["truth"]>=2},
        "address_availability": {
            "source1_missing": split["missing_address"],
            "linked_target_missing": split["linked_to_missing_address"],
            "both_present_or_singleton": ~(
                split["missing_address"] | split["linked_to_missing_address"]
            ),
        },
    }
    slices = {family: {name: {"entities": int(mask.sum()),
                               "macro_f05": float(scores[mask].mean()) if mask.any() else None}
                       for name,mask in values.items()}
              for family,values in groups.items()}
    return {
        "macro_f05": float(scores.mean()),
        "pair_precision": float(tp.sum()/predicted.sum()) if predicted.sum() else 0.0,
        "pair_recall": float(tp.sum()/split["truth"].sum()) if split["truth"].sum() else 0.0,
        "slices": slices,
    }


def run(data: Path, artifacts: Path, frozen_baseline: bool = False,
        memory_bounded: bool = False) -> None:
    resources = preflight(data,artifacts)
    work = artifacts / "work"
    con = connect(work)
    started = time.monotonic()
    try:
        if not stage_done(con,"import"):
            drop_tables(con,("links","links_raw","ground_truth","targets","s1"))
            rows = import_data(con,data)
            mark_stage(con,"import")
        else:
            rows = {"s1":count(con,"s1"),"s2":con.execute("SELECT count(*) FROM targets WHERE source_no=2").fetchone()[0],
                    "s3":con.execute("SELECT count(*) FROM targets WHERE source_no=3").fetchone()[0],
                    "ground_truth":count(con,"ground_truth"),"positive_links":count(con,"links"),
                    "split_rows":dict(con.execute("SELECT split,count(*) FROM s1 GROUP BY split ORDER BY split").fetchall())}
            log("Reusing completed full-data import and frozen split")
        if not stage_done(con,"keys"):
            drop_tables(con,("target_keys","target_keys_all","s1_keys","valid_keys","s1_keys_all"))
            keys = build_keys(con)
            mark_stage(con,"keys")
        else:
            keys = {"s1_keys":count(con,"s1_keys"),"target_keys":count(con,"target_keys")}
            log("Reusing completed blocking keys")
        candidates = build_candidates(con,work)
        if not stage_done(con,"train_selection"):
            drop_tables(con,("train_selected",))
            training = make_train_selection(con)
            mark_stage(con,"train_selection")
        else:
            selected = count(con,"train_selected")
            positive = con.execute("""SELECT count(*) FROM train_selected c JOIN links l
                ON c.s1_ix=l.s1_ix AND c.target_ix=l.target_ix""").fetchone()[0]
            training = {"retrieved_train_positives":positive,"sampled_train_negatives":selected-positive}
            log("Reusing completed training-pair selection")
        if not (artifacts/"split_assignments.parquet").exists():
            con.execute(f"COPY (SELECT entity_id,split FROM s1 ORDER BY ix) TO {quoted(artifacts/'split_assignments.parquet')} (FORMAT PARQUET)")
        base_cols = tuple(range(len(BASE_FEATURES)))
        all_cols = tuple(range(len(FEATURES)))
        cap = 24 if frozen_baseline else None
        x_train, m_train = extract_features(con,work,"train","train_selected")
        x_dev, m_dev = extract_features(con,work,"development","candidates",1,cap or 32)
        dev = split_arrays(con,1)
        cache = {}
        if frozen_baseline:
            chosen = all_cols
            cap_scores = {"frozen_cap": 24, "development_score_at_selection": 0.84732}
            log("Resuming frozen baseline: 24 candidates per source and all 20 features")
        else:
            log("Selecting candidate cap on development data")
            cap_model = fit_sample(x_train,m_train,32,base_cols)
            cap_scores = evaluate_model(cap_model,x_dev,m_dev,dev,CAPS,base_cols,work)
            best_cap_score = max(cap_scores[cap]["macro_f05"] for cap in CAPS)
            cap = next(cap for cap in CAPS if cap_scores[cap]["macro_f05"] >= best_cap_score-0.002)
            log(f"Selected {cap} candidates per source; development F0.5={cap_scores[cap]['macro_f05']:.5f}")

            def assess(columns: tuple[int, ...]) -> dict:
                if columns not in cache:
                    model = fit_sample(x_train,m_train,cap,columns)
                    outcome = evaluate_model(model,x_dev,m_dev,dev,(cap,),columns,work)[cap]
                    cache[columns] = outcome
                    log(f"Feature set {len(columns)} columns: development F0.5={outcome['macro_f05']:.5f}")
                return cache[columns]

            log("Running grouped feature ablations")
            baseline = assess(base_cols)
            chosen = all_cols
            assess(chosen)
            while True:
                options = [chosen]
                for group in ABLATION_GROUPS.values():
                    reduced = tuple(i for i in chosen if i not in group)
                    if reduced and reduced != chosen:
                        options.append(reduced)
                for columns in options:
                    assess(columns)
                top = max(cache[columns]["macro_f05"] for columns in options)
                eligible = [columns for columns in options if cache[columns]["macro_f05"]>=top-0.002]
                next_choice = min(eligible,key=lambda columns:(len(columns),columns))
                if next_choice == chosen:
                    break
                chosen = next_choice
            if baseline["macro_f05"] >= cache[chosen]["macro_f05"]-0.002 and len(base_cols)<len(chosen):
                chosen = base_cols
            log(f"Selected {len(chosen)} features: {[FEATURES[i] for i in chosen]}")

        log("Fitting selected model on all retrieved training positives and selected negatives")
        eligible = np.flatnonzero(m_train["rank"]<=cap)
        x_final = np.lib.format.open_memmap(work/"final_train.npy",mode="w+",dtype="float32",shape=(len(eligible),len(chosen)))
        y_final = np.lib.format.open_memmap(work/"final_labels.npy",mode="w+",dtype="u1",shape=(len(eligible),))
        for start in range(0,len(eligible),BATCH):
            stop = min(len(eligible),start+BATCH)
            ix = eligible[start:stop]
            x_final[start:stop] = x_train[ix][:,chosen]
            y_final[start:stop] = m_train["positive"][ix]
        x_final.flush(); y_final.flush()
        model = model_for(chosen,memory_bounded)
        model.fit(x_final,y_final)
        final_dev = evaluate_model(model,x_dev,m_dev,dev,(cap,),chosen,work)[cap]
        cache[chosen] = final_dev
        threshold = final_dev["threshold"]
        if chosen == base_cols:
            full_baseline = final_dev
        else:
            log("Fitting full-data 15-feature baseline on identical training pairs")
            x_base = np.lib.format.open_memmap(work/"baseline_train.npy",mode="w+",dtype="float32",shape=(len(eligible),len(base_cols)))
            for start in range(0,len(eligible),BATCH):
                stop = min(len(eligible),start+BATCH)
                x_base[start:stop] = x_train[eligible[start:stop]][:,base_cols]
            x_base.flush()
            baseline_model = model_for(base_cols,memory_bounded)
            baseline_model.fit(x_base,y_final)
            full_baseline = evaluate_model(baseline_model,x_dev,m_dev,dev,(cap,),base_cols,work)[cap]
            del baseline_model,x_base

        log("Scoring untouched final-validation split once")
        x_val,m_val = extract_features(con,work,"validation","candidates",2,cap)
        val = split_arrays(con,2)
        val_prob = predict_memmap(model,x_val,chosen,work/"validation_prob.npy")
        validation = final_slices(m_val,val_prob,val,cap,threshold)
        diagnostics = {
            "development": candidate_diagnostics(con,cap,1),
            "validation": candidate_diagnostics(con,cap,2),
        }
        metrics = {
            "seed": SEED, "rows": rows, "keys": keys, "candidates": candidates,
            "run_mode": "frozen_baseline" if frozen_baseline else "feature_selection",
            "model_variant": "memory_bounded" if memory_bounded else "standard",
            "training": training, "candidate_cap_grid": cap_scores,
            "selected_cap_per_source": cap,
            "baseline_15_features": {"development_macro_f05":full_baseline["macro_f05"],
                                     "development_threshold":full_baseline["threshold"],
                                     "same_training_pairs":True},
            "feature_selection": {",".join(map(str,k)): {"features":[FEATURES[i] for i in k],
                "development_macro_f05":v["macro_f05"],"threshold":v["threshold"]}
                for k,v in cache.items()},
            "selected_features": [FEATURES[i] for i in chosen],
            "final_development": final_dev,
            "candidate_diagnostics": diagnostics,
            "final_validation": validation,
            "feature_gain": dict(zip((FEATURES[i] for i in chosen),map(float,model.feature_importances_))),
            "resources": {**resources,"elapsed_hours":round((time.monotonic()-started)/3600,3)},
        }
        joblib.dump(model,artifacts/"matcher_model.joblib")
        (artifacts/"model_config.json").write_text(json.dumps({
            "feature_names":metrics["selected_features"],
            "feature_indices":chosen,
            "candidate_cap_per_source":cap,
            "decision_threshold":threshold,
            "seed":SEED,
            "model_variant": "memory_bounded" if memory_bounded else "standard",
        },indent=2),encoding="utf-8")
        (artifacts/"training_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        log(f"Final validation macro F0.5={validation['macro_f05']:.6f}; artifacts: {artifacts}")
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--train-dir",type=Path,default=root/"student_resource"/"dataset"/"train")
    parser.add_argument("--artifacts-dir",type=Path,default=root/"code"/"business_entity_resolution"/"artifacts"/"v2")
    parser.add_argument("--frozen-baseline",action="store_true",
                        help="Resume completed checkpoints with fixed cap 24 and all 20 features; skip cap and feature selection.")
    parser.add_argument("--memory-bounded-model",action="store_true",
                        help="Use one training thread, 31 bins, and 15 leaves for low-memory full-data fitting.")
    args = parser.parse_args()
    run(args.train_dir,args.artifacts_dir,args.frozen_baseline,args.memory_bounded_model)
