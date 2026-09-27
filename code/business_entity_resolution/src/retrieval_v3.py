"""Development-only BM25 candidate-retrieval pilot over supplied records.

Targets and reference entities are sampled by stable hashes, without consulting
ground truth. Labels are joined only after retrieval to measure recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from matching_v3 import normalize
from train_v3 import connect, quoted


def _text(name: str, address: str) -> str:
    return f"{normalize(name)} {normalize(address)}"


def _bm25_documents(matrix: csr_matrix, k1: float = 1.5, b: float = 0.75) -> csr_matrix:
    matrix = matrix.astype(np.float32).tocsr()
    document_frequency = np.bincount(matrix.indices, minlength=matrix.shape[1])
    n_documents = matrix.shape[0]
    idf = np.log1p((n_documents - document_frequency + 0.5) /
                   (document_frequency + 0.5)).astype(np.float32)
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    average_length = max(float(lengths.mean()), 1.0)
    row_lengths = np.repeat(lengths, np.diff(matrix.indptr))
    tf = matrix.data
    matrix.data = (tf * (k1 + 1.0) /
                   (tf + k1 * (1.0 - b + b * row_lengths / average_length))) * idf[matrix.indices]
    matrix.eliminate_zeros()
    return matrix


def _rank_batch(queries, doc_scores: csr_matrix, doc_ids: np.ndarray,
                target_rows, ks: tuple[int, ...]) -> list[tuple[int, int, int, int, float, str]]:
    output = []
    max_k = max(ks)
    scores = queries @ doc_scores.T
    for row_ix, target in enumerate(target_rows):
        start, stop = scores.indptr[row_ix:row_ix + 2]
        doc_ix = scores.indices[start:stop]
        values = scores.data[start:stop]
        if len(values) == 0:
            continue
        order = np.lexsort((doc_ids[doc_ix], -values))[:max_k]
        target_ix, source_no, target_id = int(target[0]), int(target[1]), str(target[5])
        for rank, position in enumerate(order, 1):
            output.append((int(doc_ids[doc_ix[position]]), target_ix, source_no,
                           rank, float(values[position]), target_id))
    return output


def _retrieve_country(doc_rows, target_rows, batch_size: int = 64,
                      ks: tuple[int, ...] = (32, 64, 128), backend: str = "scipy"):
    if not doc_rows or not target_rows:
        return [], 0.0
    started = time.monotonic()
    doc_ids = np.asarray([row[0] for row in doc_rows], dtype=np.int32)
    doc_text = [_text(row[2], row[3]) for row in doc_rows]
    if backend == "bm25s":
        import bm25s

        tokenizer = bm25s.tokenization.Tokenizer(
            splitter=lambda value: [value[i:i + 3] for i in range(len(value) - 2)],
            stopwords=[],
        )
        doc_tokens = tokenizer.tokenize(doc_text, show_progress=False)
        retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene", backend="numba")
        retriever.index(doc_tokens, show_progress=False)
        candidates = []
        for start in range(0, len(target_rows), batch_size):
            batch = target_rows[start:start + batch_size]
            query_text = [_text(row[3], row[4]) for row in batch]
            query_tokens = tokenizer.tokenize(
                query_text, update_vocab=False, show_progress=False,
            )
            results = retriever.retrieve(
                query_tokens, k=min(max(ks), len(doc_rows)),
                n_threads=os.cpu_count() or 1, show_progress=False,
                backend_selection="numba",
            )
            for row_ix, target in enumerate(batch):
                hits = results.documents[row_ix]
                scores = results.scores[row_ix]
                keep = scores > 0
                hits, scores = hits[keep], scores[keep]
                order = np.lexsort((doc_ids[hits], -scores))
                target_ix, source_no, target_id = int(target[0]), int(target[1]), str(target[5])
                candidates.extend(
                    (int(doc_ids[hits[pos]]), target_ix, source_no, rank,
                     float(scores[pos]), target_id)
                    for rank, pos in enumerate(order, 1)
                )
        return candidates, time.monotonic() - started

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3),
                                 binary=False, lowercase=False, dtype=np.float32,
                                 min_df=2, max_df=0.8)
    doc_counts = vectorizer.fit_transform(doc_text).tocsr()
    doc_scores = _bm25_documents(doc_counts)
    candidates = []
    for start in range(0, len(target_rows), batch_size):
        batch = target_rows[start:start + batch_size]
        q = vectorizer.transform([_text(row[3], row[4]) for row in batch]).tocsr()
        candidates.extend(_rank_batch(q, doc_scores, doc_ids, batch, ks))
    elapsed = time.monotonic() - started
    return candidates, elapsed


def run_pilot(artifacts: Path, docs_per_country: int = 150_000,
              targets_per_country: int = 20_000,
              ks: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
              backend: str = "scipy") -> dict:
    work = artifacts / "work"
    output = artifacts / "tuning" / "retrieval"
    output.mkdir(parents=True, exist_ok=True)
    con = connect(work)
    try:
        # Hash-based sampling is label-blind and independent of input row order.
        docs = con.execute(f"""
            SELECT ix,country,business_name,business_address,entity_id FROM s1
            WHERE split=1
              AND hash(entity_id || ':bm25-doc-pilot-42') % 2 = 0
            QUALIFY row_number() OVER (PARTITION BY country
                ORDER BY entity_id) <= {docs_per_country}
            ORDER BY country,ix
        """).fetchall()
        targets = con.execute(f"""
            SELECT ix,source_no,country,business_name,business_address,entity_id FROM targets
            WHERE hash(entity_id || ':bm25-query-pilot-42') % 128 = 0
            QUALIFY row_number() OVER (PARTITION BY country
                ORDER BY entity_id) <= {targets_per_country}
            ORDER BY country,ix
        """).fetchall()
        country_values = sorted({row[1] for row in docs} & {row[2] for row in targets})
        all_pairs = []
        country_runtime = {}
        for country in country_values:
            country_docs = [(ix, entity_id, name, address)
                            for ix, doc_country, name, address, entity_id in docs
                            if doc_country == country]
            country_targets = [(target_ix, source_no, country, name, address, entity_id)
                               for target_ix, source_no, target_country, name, address, entity_id in targets
                               if target_country == country]
            pairs, elapsed = _retrieve_country(country_docs, country_targets, ks=ks,
                                                backend=backend)
            all_pairs.extend((country, *row) for row in pairs)
            country_runtime[str(country)] = {
                "documents": len(country_docs), "queries": len(country_targets),
                "seconds": round(elapsed, 2),
                "query_pairs_per_second": round(len(country_targets) / max(elapsed, 1e-9), 2),
            }
        pair_path = output / f"bm25_pilot_pairs_{backend}.tsv"
        with pair_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("country", "s1_ix", "target_ix", "source_no", "rank", "bm25", "target_id"))
            writer.writerows(all_pairs)

        con.register("bm25_pilot_frame", _pairs_frame(all_pairs))
        import pandas as pd
        con.register("bm25_sample_docs", pd.DataFrame({"s1_ix": [r[0] for r in docs]}))
        con.register("bm25_sample_targets", pd.DataFrame({"target_ix": [r[0] for r in targets]}))
        result_rows = []
        for cap in ks:
            # Keep the evaluated denominator restricted to the fixed sampled
            # reference/query population, not to any label-driven selection.
            result_rows.append(_measure(con, all_pairs, docs, targets, cap))

        report = {
            "method": "country-specific character-trigram BM25 over name plus address",
            "backend": backend,
            "seed": 42,
            "labels_used_for_retrieval": False,
            "retrieval_direction": "sampled S2/S3 targets query sampled development S1 references",
            "docs_per_country_limit": docs_per_country,
            "targets_per_country_limit": targets_per_country,
            "actual_documents": len(docs), "actual_queries": len(targets),
            "country_runtime": country_runtime,
            "metrics": result_rows,
            "pair_file": str(pair_path),
        }
        report_path = output / f"bm25_pilot_report_{backend}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        con.close()


def _pairs_frame(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=("country", "s1_ix", "target_ix", "source_no", "rank", "bm25", "target_id"))


def _measure(con, all_pairs, docs, targets, cap: int) -> dict:
    con.execute("CREATE OR REPLACE TEMP TABLE bm25_retrieved AS SELECT s1_ix,target_ix,source_no FROM bm25_pilot_frame WHERE rank<=?", [cap])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pilot_eligible_links AS
        SELECT l.s1_ix,l.target_ix,l.source_no
        FROM links l JOIN s1 s ON s.ix=l.s1_ix
        JOIN targets t ON t.ix=l.target_ix
        JOIN bm25_sample_docs d ON d.s1_ix=l.s1_ix
        JOIN bm25_sample_targets q ON q.target_ix=l.target_ix
        WHERE s.split=1
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pilot_raw_overlap AS
        SELECT e.s1_ix,e.target_ix,true shared
        FROM pilot_eligible_links e
        JOIN s1 s ON s.ix=e.s1_ix JOIN targets t ON t.ix=e.target_ix,
        UNNEST(block_keys(s.business_name,s.business_address)) sk(key),
        UNNEST(block_keys(t.business_name,t.business_address)) tk(key)
        WHERE sk.key=tk.key GROUP BY e.s1_ix,e.target_ix
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pilot_filtered_overlap AS
        SELECT e.s1_ix,e.target_ix,true shared
        FROM pilot_eligible_links e
        JOIN s1_keys sk ON sk.s1_ix=e.s1_ix
        JOIN target_keys tk ON tk.target_ix=e.target_ix AND tk.country=sk.country AND tk.key=sk.key
        GROUP BY e.s1_ix,e.target_ix
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pilot_link_check AS
        SELECT e.s1_ix,e.target_ix,e.source_no,
               coalesce(r.shared,false) raw_shared,
               coalesce(f.shared,false) filtered_shared,
               c.target_ix IS NOT NULL old_hit,
               b.target_ix IS NOT NULL bm25_hit
        FROM pilot_eligible_links e
        LEFT JOIN pilot_raw_overlap r USING(s1_ix,target_ix)
        LEFT JOIN pilot_filtered_overlap f USING(s1_ix,target_ix)
        LEFT JOIN candidates c ON c.s1_ix=e.s1_ix AND c.target_ix=e.target_ix AND c.rank<=32
        LEFT JOIN bm25_retrieved b ON b.s1_ix=e.s1_ix AND b.target_ix=e.target_ix
                                  AND b.source_no=e.source_no
    """)
    baseline = con.execute("""
        SELECT l.source_no,count(*) total,
               count(c.target_ix) base_hits,
               count(b.target_ix) bm25_hits,
               count(coalesce(c.target_ix,b.target_ix)) union_hits
        FROM links l JOIN s1 s ON s.ix=l.s1_ix
        JOIN targets t ON t.ix=l.target_ix
        JOIN bm25_sample_docs d ON d.s1_ix=l.s1_ix
        JOIN bm25_sample_targets q ON q.target_ix=l.target_ix
        LEFT JOIN candidates c ON c.s1_ix=l.s1_ix AND c.target_ix=l.target_ix AND c.rank<=32
        LEFT JOIN bm25_retrieved b ON b.s1_ix=l.s1_ix AND b.target_ix=l.target_ix
        WHERE s.split=1 GROUP BY l.source_no ORDER BY l.source_no
    """).fetchall()
    retrieved_count = con.execute("SELECT count(*) FROM bm25_retrieved").fetchone()[0]
    base_count = con.execute("""
        SELECT count(*) FROM candidates c JOIN bm25_sample_docs d ON d.s1_ix=c.s1_ix
        JOIN bm25_sample_targets q ON q.target_ix=c.target_ix WHERE c.rank<=32
    """).fetchone()[0]
    diagnoses = con.execute("""
        SELECT source_no,count(*) total,
               count(*) FILTER(WHERE NOT old_hit AND NOT raw_shared) no_raw_key_overlap,
               count(*) FILTER(WHERE NOT old_hit AND raw_shared AND NOT filtered_shared) pruned_by_frequency,
               count(*) FILTER(WHERE NOT old_hit AND filtered_shared) lost_beyond_rank_32,
               count(*) FILTER(WHERE NOT old_hit AND bm25_hit) recovered_by_bm25
        FROM pilot_link_check GROUP BY source_no ORDER BY source_no
    """).fetchall()
    return {
        "cap_per_target": cap,
        "candidate_pairs": int(retrieved_count),
        "baseline_candidate_pairs_in_pilot": int(base_count),
        "missed_link_diagnosis": {str(source): {
            "eligible_positive_links": int(total),
            "no_raw_key_overlap": int(no_raw),
            "pruned_by_frequency_filters": int(pruned),
            "lost_beyond_current_rank_32": int(rank),
            "recovered_by_bm25": int(recovered),
        } for source,total,no_raw,pruned,rank,recovered in diagnoses},
        "by_target_source": {str(source): {
            "eligible_positive_links": int(total),
            "baseline_recalled": int(base), "bm25_recalled": int(bm25),
            "union_recalled": int(union),
            "baseline_recall": base / total if total else 0,
            "bm25_recall": bm25 / total if total else 0,
            "union_recall": union / total if total else 0,
        } for source,total,base,bm25,union in baseline},
    }


def evaluate_saved_pilot(artifacts: Path, backend: str = "scipy",
                        docs_per_country: int = 150_000,
                        targets_per_country: int = 20_000,
                        ks: tuple[int, ...] = (4, 8, 16, 32, 64, 128)) -> dict:
    """Re-score saved BM25 pairs at several caps without rebuilding the index."""
    import pandas as pd

    output = artifacts / "tuning" / "retrieval"
    pair_path = output / f"bm25_pilot_pairs_{backend}.tsv"
    report_path = output / f"bm25_pilot_report_{backend}.json"
    if backend == "scipy" and not pair_path.exists():
        pair_path = output / "bm25_pilot_pairs.tsv"
        report_path = output / "bm25_pilot_report.json"
    con = connect(artifacts / "work")
    try:
        docs = con.execute(f"""
            SELECT ix,country,business_name,business_address,entity_id FROM s1
            WHERE split=1 AND hash(entity_id || ':bm25-doc-pilot-42') % 2 = 0
            QUALIFY row_number() OVER (PARTITION BY country ORDER BY entity_id) <= {docs_per_country}
            ORDER BY country,ix
        """).fetchall()
        targets = con.execute(f"""
            SELECT ix,source_no,country,business_name,business_address,entity_id FROM targets
            WHERE hash(entity_id || ':bm25-query-pilot-42') % 128 = 0
            QUALIFY row_number() OVER (PARTITION BY country ORDER BY entity_id) <= {targets_per_country}
            ORDER BY country,ix
        """).fetchall()
        con.register("bm25_pilot_frame", pd.read_csv(pair_path, sep="\t"))
        con.register("bm25_sample_docs", pd.DataFrame({"s1_ix": [row[0] for row in docs]}))
        con.register("bm25_sample_targets", pd.DataFrame({"target_ix": [row[0] for row in targets]}))
        metrics = [_measure(con, None, docs, targets, cap) for cap in ks]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["metrics"] = metrics
        report["missed_link_scope"] = (
            "Only positive links whose S1 and target are both in the label-blind pilot samples; "
            "not a full-split candidate oracle."
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        con.close()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path,
                        default=root / "code/business_entity_resolution/artifacts/v3")
    parser.add_argument("--docs-per-country", type=int, default=150_000)
    parser.add_argument("--targets-per-country", type=int, default=20_000)
    parser.add_argument("--backend", choices=("scipy", "bm25s"), default="scipy")
    parser.add_argument("--caps", default="4,8,16,32,64,128",
                        help="Comma-separated candidate caps to evaluate")
    parser.add_argument("--evaluate-existing", action="store_true",
                        help="Measure the saved candidate list at caps 4/8/16/32/64/128 without rerunning BM25")
    args = parser.parse_args()
    caps = tuple(sorted({int(cap) for cap in args.caps.split(",")}))
    if not caps or caps[0] < 1:
        parser.error("--caps must contain positive integers")
    report = (evaluate_saved_pilot(args.artifacts_dir, args.backend,
                                   args.docs_per_country,
                                   args.targets_per_country, ks=caps)
              if args.evaluate_existing else
              run_pilot(args.artifacts_dir, args.docs_per_country,
                        args.targets_per_country, ks=caps, backend=args.backend))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
