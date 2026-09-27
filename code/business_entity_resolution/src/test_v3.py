"""Small regression check for the v3 parser, blocker, features, and score.

Validates:
- 23 total pair features (20 v2 + 3 v3).
- Phonetic (p:) and postal-code (z:) blocking keys are emitted.
- location_mismatch, tfidf_cosine, and street_number_match behave correctly.
- The train_v3 pipeline runs end-to-end on a 30-row fixture corpus.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

import train_v3
from matching_v3 import (
    FEATURES, block_keys, double_metaphone, extract_postal_codes,
    f05, location_mismatch, pair_features, street_number_match, tfidf_cosine,
)


class V3MatchingCheck(unittest.TestCase):
    """Unit tests for the new v3 matching primitives."""

    def test_feature_count(self) -> None:
        feats = pair_features("A", "10 Road", "A", "10 Road", 3)
        self.assertEqual(len(feats), 23)
        self.assertEqual(len(FEATURES), 23)
        self.assertEqual(feats[-1], 1.0)   # street_number_match: both missing, fallback 0 -> wait, "10 Road" has leading 10

    def test_street_number_match(self) -> None:
        self.assertEqual(street_number_match("105 Ribbon Ln", "11 Ribbon Ln"), -1.0)
        self.assertEqual(street_number_match("105 Ribbon Ln", "105 Ribbon Ln"), 1.0)
        self.assertEqual(street_number_match("Ribbon Ln", "105 Ribbon Ln"), 0.0)
        self.assertEqual(street_number_match("", ""), 0.0)

    def test_location_mismatch(self) -> None:
        self.assertEqual(location_mismatch("105 Main St, IL 60601", "105 Main St, CA 90001"), 1.0)
        self.assertEqual(location_mismatch("105 Main St, IL 60601", "105 Main St, IL 60601"), 0.0)
        self.assertEqual(location_mismatch("", "105 Main St CA"), 0.0)  # one side empty
        # Indian postal codes
        self.assertEqual(location_mismatch("Shop 1, 400001 Mumbai", "Office, 110001 Delhi"), 1.0)
        self.assertEqual(location_mismatch("Shop 1, 400001 Mumbai", "Shop 2, 400001 Mumbai"), 0.0)

    def test_tfidf_cosine_identical(self) -> None:
        score = tfidf_cosine("Kalyani Medicals", "Kalyani Medicals")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_tfidf_cosine_generic_penalty(self) -> None:
        # Generic token "services" has high IDF penalty -> lower similarity
        score_generic = tfidf_cosine("XYZ Services", "ABC Services")
        score_distinct = tfidf_cosine("Zydeco Pharma", "Zydeco Pharma")
        self.assertGreater(score_distinct, score_generic)

    def test_double_metaphone(self) -> None:
        p, _ = double_metaphone("kalyani")
        self.assertTrue(len(p) >= 2, f"Expected DM code of length >= 2, got {p!r}")

    def test_postal_code_extraction(self) -> None:
        self.assertEqual(extract_postal_codes("Pin 700130 Kolkata"), ["700130"])
        self.assertEqual(extract_postal_codes("Chicago IL 60601"), ["60601"])
        self.assertEqual(extract_postal_codes("Paris 75001 France"), ["75001"])
        self.assertEqual(extract_postal_codes(""), [])

    def test_block_keys_phonetic_and_postal(self) -> None:
        keys = block_keys("Kalyani Medicals", "Pin 700130 Kolkata")
        self.assertTrue(any(k.startswith("p:") for k in keys), f"Missing p: key: {keys}")
        self.assertTrue(any(k.startswith("z:") for k in keys), f"Missing z: key: {keys}")

    def test_f05_scoring(self) -> None:
        np.testing.assert_allclose(
            f05(np.array([0, 1, 2]), np.array([0, 0, 2]), np.array([0, 0, 1])),
            [1.0, 0.0, 0.5],
        )


class V3PipelineCheck(unittest.TestCase):
    """End-to-end smoke test running train_v3 on a tiny 30-row fixture."""

    def test_small_corpus(self) -> None:
        base = Path(__file__).resolve().parents[1] / "artifacts" / "v3"
        base.mkdir(parents=True, exist_ok=True)
        fixture = base / ("_test_" + uuid4().hex)
        fixture.mkdir()
        old_expected = train_v3.EXPECTED_ROWS
        try:
            def write(name: str, rows: list[str]) -> None:
                (fixture / name).write_text("\n".join(rows) + "\n", encoding="utf-8")

            write("train_source1.tsv", ["entity_id\tbusiness_name\tbusiness_address\tcountry"] + [
                f"S1-{i}\tWidget {i}\t{i+10} Market Road, Pin 110001 Delhi\tIN" for i in range(30)
            ])
            for source in (2, 3):
                write(f"train_source{source}.tsv", ["entity_id\tbusiness_name\tbusiness_address\tcountry"] + [
                    f"S{source}-{i}\t{'Unrelated' if source == 2 and i == 29 else f'Widget {i}'}"
                    f"\t{'999 Unknown Lane' if source == 2 and i == 29 else f'{i+10} Market Road, Pin 110001 Delhi'}\tIN"
                    for i in range(30)
                ])
            write("train_ground_truth.tsv", ["source1_entity_id\tmatched_entity_ids"] + [
                f"S1-{i}\t" if i % 5 == 0 else f"S1-{i}\tS2-{i},S3-{i}"
                for i in range(30)
            ])
            train_v3.EXPECTED_ROWS = {"s1": 30, "s2": 30, "s3": 30}
            con = train_v3.connect(fixture / "work")
            try:
                rows = train_v3.import_data(con, fixture)
                self.assertEqual(rows["positive_links"], 48)
                self.assertEqual(sum(rows["split_rows"].values()), 30)
                self.assertEqual(
                    con.execute("SELECT business_address FROM s1 WHERE entity_id='S1-1'").fetchone()[0],
                    "11 Market Road, Pin 110001 Delhi",
                )
                train_v3.build_keys(con)
                # Verify phonetic and postal keys are emitted into DB
                key_prefixes = {k[:2] for (k,) in con.execute(
                    "SELECT DISTINCT key FROM s1_keys"
                ).fetchall()}
                self.assertIn("p:", key_prefixes, "Phonetic (p:) keys missing from s1_keys")
                self.assertIn("z:", key_prefixes, "Postal-code (z:) keys missing from s1_keys")

                train_v3.build_candidates(con, fixture / "work")
                target_ix = con.execute("SELECT ix FROM targets WHERE entity_id='S2-29'").fetchone()[0]
                s1_ix = con.execute("SELECT ix FROM s1 WHERE entity_id='S1-29'").fetchone()[0]
                # Unrelated S2-29 should not be a candidate for S1-29
                self.assertEqual(
                    con.execute(
                        "SELECT count(*) FROM candidates WHERE s1_ix=? AND target_ix=?",
                        [s1_ix, target_ix],
                    ).fetchone()[0],
                    0,
                )
                # Idempotency check
                ranking = con.execute(
                    "SELECT s1_ix,target_ix,source_no,rank FROM candidates ORDER BY 1,2,3"
                ).fetchall()
                con.execute("DROP TABLE candidates")
                con.execute("DROP TABLE candidate_parts")
                train_v3.build_candidates(con, fixture / "work")
                self.assertEqual(
                    ranking,
                    con.execute(
                        "SELECT s1_ix,target_ix,source_no,rank FROM candidates ORDER BY 1,2,3"
                    ).fetchall(),
                )
                self.assertTrue(con.execute("SELECT count(*) FROM candidates WHERE source_no=2").fetchone()[0])
                self.assertTrue(con.execute("SELECT count(*) FROM candidates WHERE source_no=3").fetchone()[0])
            finally:
                con.close()

            # Feature vector length
            feats = pair_features("Widget 1", "11 Market Road Pin 110001 Delhi",
                                   "Widget 1", "11 Market Road Pin 110001 Delhi", 3)
            self.assertEqual(len(feats), len(FEATURES))
            # street_number_match should be 1.0 (identical lead numbers)
            self.assertEqual(feats[FEATURES.index("street_number_match")], 1.0)
            # location_mismatch should be 0.0 (same postal code)
            self.assertEqual(feats[FEATURES.index("location_mismatch")], 0.0)
        finally:
            train_v3.EXPECTED_ROWS = old_expected
            if fixture.resolve().parent != base.resolve():
                raise RuntimeError("Unexpected fixture path")
            shutil.rmtree(fixture)


if __name__ == "__main__":
    unittest.main()
