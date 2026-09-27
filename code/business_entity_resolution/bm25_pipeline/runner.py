"""Shared, sampled BM25 candidate-generation runner for train and development."""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
SRC = PACKAGE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval_v2 import _retrieve_country  # noqa: E402
from train_v2 import connect, import_data  # noqa: E402


def _ensure_database(artifacts: Path, train_dir: Path) -> tuple:
    work = artifacts / "work"
    con = connect(work)
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if {"s1", "targets", "links"}.issubset(tables):
        return con, {"database": "reused", "path": str(work / "pipeline.duckdb")}
    if tables:
        con.close()
        raise RuntimeError(
            f"Incomplete BM25 DuckDB state at {work / 'pipeline.duckdb'}; "
            "remove that database or restore the complete v2 database."
        )
    imported = import_data(con, train_dir)
    return con, {"database": "created", "path": str(work / "pipeline.duckdb"),
                 "import": imported}


def run_partition(*, split: int, partition_name: str, data_dir: Path,
                  artifacts: Path, output_dir: Path, backend: str,
                  docs_per_country: int, targets_per_country: int,
                  cap: int, batch_size: int) -> dict:
    """Retrieve a fixed-seed sample for one frozen S1 split and report recall."""
    if split not in (0, 1):
        raise ValueError("BM25 training/development runs accept split 0 or 1 only")
    if cap < 1 or batch_size < 1:
        raise ValueError("cap and batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    con, database_info = _ensure_database(artifacts, data_dir)
    started = time.monotonic()
    try:
        docs = con.execute(f"""
            SELECT ix,country,business_name,business_address,entity_id FROM s1
            WHERE split={split}
              AND hash(entity_id || ':bm25-doc-pilot-42') % 2 = 0
            QUALIFY row_number() OVER (PARTITION BY country ORDER BY entity_id)
                <= {docs_per_country}
            ORDER BY country,ix
        """).fetchall()
        targets = con.execute(f"""
            SELECT ix,source_no,country,business_name,business_address,entity_id
            FROM targets
            WHERE hash(entity_id || ':bm25-query-pilot-42') % 128 = 0
            QUALIFY row_number() OVER (PARTITION BY country ORDER BY entity_id)
                <= {targets_per_country}
            ORDER BY country,source_no,ix
        """).fetchall()
        docs_by_country = {}
        for ix, country, name, address, entity_id in docs:
            docs_by_country.setdefault(country, []).append((ix, entity_id, name, address))
        targets_by_country = {}
        for row in targets:
            targets_by_country.setdefault(row[2], []).append(row)
        countries = sorted(docs_by_country.keys() & targets_by_country.keys())

        con.register("bm25_docs_sample", pd.DataFrame({"s1_ix": [row[0] for row in docs]}))
        con.register("bm25_targets_sample", pd.DataFrame(
            {"target_ix": [row[0] for row in targets]}))
        positive_rows = con.execute("""
            SELECT l.s1_ix,l.target_ix,l.source_no
            FROM links l JOIN bm25_docs_sample d ON d.s1_ix=l.s1_ix
            JOIN bm25_targets_sample t ON t.target_ix=l.target_ix
        """).fetchall()
        positives = {(int(s1), int(target), int(source))
                     for s1, target, source in positive_rows}
        expected_by_entity = {}
        for s1, target, source in positives:
            expected_by_entity.setdefault(s1, set()).add((target, source))

        pairs_path = output_dir / f"{partition_name}_candidates_{backend}.tsv.gz"
        candidate_count = {2: 0, 3: 0}
        hit_count = {2: 0, 3: 0}
        covered_by_entity = {s1: set() for s1 in expected_by_entity}
        country_runtime = {}
        with gzip.open(pairs_path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            header = ["s1_ix", "target_ix", "source_no", "rank", "bm25", "target_id"]
            if split == 0:
                header.append("is_match")
            writer.writerow(header)
            for country in countries:
                pairs, elapsed = _retrieve_country(
                    docs_by_country[country], targets_by_country[country],
                    batch_size=batch_size, ks=(cap,), backend=backend,
                )
                for s1_ix, target_ix, source_no, rank, score, target_id in pairs:
                    key = (s1_ix, target_ix, source_no)
                    is_match = key in positives
                    candidate_count[source_no] += 1
                    if is_match:
                        hit_count[source_no] += 1
                        covered_by_entity.setdefault(s1_ix, set()).add((target_ix, source_no))
                    row = [s1_ix, target_ix, source_no, rank, score, target_id]
                    if split == 0:
                        row.append(int(is_match))
                    writer.writerow(row)
                country_runtime[str(country)] = {
                    "documents": len(docs_by_country[country]),
                    "queries": len(targets_by_country[country]),
                    "seconds": round(elapsed, 2),
                }

        by_source = {}
        for source in (2, 3):
            source_positive = sum(1 for _, _, src in positives if src == source)
            source_hit = hit_count[source]
            by_source[str(source)] = {
                "eligible_positive_links": source_positive,
                "retrieved_positive_links": source_hit,
                "candidate_pairs": candidate_count[source],
                "positive_link_recall": source_hit / source_positive if source_positive else None,
            }
        complete_entities = sum(
            covered_by_entity.get(s1, set()) >= expected
            for s1, expected in expected_by_entity.items()
        )
        report = {
            "method": "country-specific character-trigram BM25 over normalized name and address",
            "partition": partition_name,
            "split": split,
            "seed": 42,
            "backend": backend,
            "batch_size": batch_size,
            "candidate_cap_per_target": cap,
            "labels_used_for_retrieval": False,
            "labels_joined_after_retrieval": True,
            "retrieval_population": "stable hash sample; same sample rule for train and development",
            "sample_sizes": {"s1_documents": len(docs), "target_queries": len(targets)},
            "positive_link_entities_in_sample": len(expected_by_entity),
            "entities_with_all_sampled_true_links_retained": complete_entities,
            "complete_link_entity_recall": complete_entities / len(expected_by_entity)
                if expected_by_entity else None,
            "by_target_source": by_source,
            "country_runtime": country_runtime,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "candidate_file": str(pairs_path),
            "database": database_info,
        }
        report_path = output_dir / f"{partition_name}_report_{backend}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        con.close()
