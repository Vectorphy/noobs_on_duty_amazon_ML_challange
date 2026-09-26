"""Small regression check for the v2 parser, blocker, features, and score."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

import train_v2
from matching_v2 import FEATURES, block_keys, f05, pair_features


class V2Check(unittest.TestCase):
    def test_small_corpus(self) -> None:
        base = Path(__file__).resolve().parents[1] / "artifacts" / "v2"
        base.mkdir(parents=True, exist_ok=True)
        fixture = base / ("_test_" + uuid4().hex)
        fixture.mkdir()
        old_expected = train_v2.EXPECTED_ROWS
        try:
            def write(name: str, rows: list[str]) -> None:
                (fixture / name).write_text("\n".join(rows) + "\n", encoding="utf-8")

            write("train_source1.tsv", ["entity_id\tbusiness_name\tbusiness_address\tcountry"] + [
                f"S1-{i}\tWidget {i}\t{i+10} Market Road, Delhi\tIN" for i in range(30)
            ])
            for source in (2, 3):
                write(f"train_source{source}.tsv", ["entity_id\tbusiness_name\tbusiness_address\tcountry"] + [
                    f"S{source}-{i}\t{'Unrelated' if source == 2 and i == 29 else f'Widget {i}'}"
                    f"\t{'999 Unknown Lane' if source == 2 and i == 29 else f'{i+10} Market Road, Delhi'}\tIN"
                    for i in range(30)
                ])
            write("train_ground_truth.tsv", ["source1_entity_id\tmatched_entity_ids"] + [
                f"S1-{i}\t" if i % 5 == 0 else f"S1-{i}\tS2-{i},S3-{i}"
                for i in range(30)
            ])
            train_v2.EXPECTED_ROWS = {"s1": 30, "s2": 30, "s3": 30}
            con = train_v2.connect(fixture / "work")
            try:
                rows = train_v2.import_data(con, fixture)
                self.assertEqual(rows["positive_links"], 48)
                self.assertEqual(sum(rows["split_rows"].values()), 30)
                self.assertEqual(con.execute("SELECT business_address FROM s1 WHERE entity_id='S1-1'").fetchone()[0], "11 Market Road, Delhi")
                train_v2.build_keys(con)
                train_v2.build_candidates(con, fixture / "work")
                target_ix = con.execute("SELECT ix FROM targets WHERE entity_id='S2-29'").fetchone()[0]
                s1_ix = con.execute("SELECT ix FROM s1 WHERE entity_id='S1-29'").fetchone()[0]
                self.assertEqual(con.execute("SELECT count(*) FROM candidates WHERE s1_ix=? AND target_ix=?", [s1_ix,target_ix]).fetchone()[0], 0)
                ranking = con.execute("SELECT s1_ix,target_ix,source_no,rank FROM candidates ORDER BY 1,2,3").fetchall()
                con.execute("DROP TABLE candidates")
                con.execute("DROP TABLE candidate_parts")
                train_v2.build_candidates(con, fixture / "work")
                self.assertEqual(ranking, con.execute("SELECT s1_ix,target_ix,source_no,rank FROM candidates ORDER BY 1,2,3").fetchall())
                self.assertTrue(con.execute("SELECT count(*) FROM candidates WHERE source_no=2").fetchone()[0])
                self.assertTrue(con.execute("SELECT count(*) FROM candidates WHERE source_no=3").fetchone()[0])
            finally:
                con.close()
            self.assertEqual(len(FEATURES), len(pair_features("A", "10 Road", "A", "10 Road", 3)))
            self.assertEqual(pair_features("A", "10 Road", "A", "10 Road", 3)[-1], 1)
            self.assertTrue(block_keys("Widget 1", "11 Market Road"))
            np.testing.assert_allclose(f05(np.array([0,1,2]),np.array([0,0,2]),np.array([0,0,1])), [1,0,.5])
        finally:
            train_v2.EXPECTED_ROWS = old_expected
            if fixture.resolve().parent != base.resolve():
                raise RuntimeError("Unexpected fixture path")
            shutil.rmtree(fixture)


if __name__ == "__main__":
    unittest.main()
