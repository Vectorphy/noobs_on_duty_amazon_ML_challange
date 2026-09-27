"""Smoke-test for the v3-XGBoost pipeline.

Runs train_v3_xgb end-to-end on a 30-row fixture corpus in an isolated temp
directory (completely separate from artifacts/v3 and its live DuckDB).
Also verifies --device auto CPU fallback works on machines without a GPU.

Run from repository root:
    python -m pytest code/business_entity_resolution/src/test_v3_xgb.py -v
or:
    python code/business_entity_resolution/src/test_v3_xgb.py
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

# Make sure we can find the src modules when run directly
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import train_v3_xgb
from matching_v3 import FEATURES, f05, pair_features


class XGBDeviceTest(unittest.TestCase):
    """Verify device resolution logic."""

    def test_cpu_forced(self) -> None:
        device = train_v3_xgb.resolve_device("cpu")
        self.assertEqual(device, "cpu")

    def test_auto_returns_string(self) -> None:
        # auto must return either "cpu" or "cuda" without raising
        device = train_v3_xgb.resolve_device("auto")
        self.assertIn(device, ("cpu", "cuda"))

    def test_cuda_unavailable_raises(self) -> None:
        """On a CPU-only machine, --device cuda must raise RuntimeError."""
        if train_v3_xgb._probe_cuda():
            self.skipTest("CUDA is available on this machine")
        with self.assertRaises(RuntimeError):
            train_v3_xgb.resolve_device("cuda")


class XGBModelTest(unittest.TestCase):
    """Verify the XGBClassifier is returned with correct params."""

    def test_model_for_cpu(self) -> None:
        from xgboost import XGBClassifier
        m = train_v3_xgb.model_for((0, 1, 2), device="cpu")
        self.assertIsInstance(m, XGBClassifier)
        p = m.get_params()
        self.assertEqual(p["grow_policy"], "lossguide")
        self.assertEqual(p["tree_method"], "hist")
        self.assertEqual(p["device"], "cpu")
        self.assertEqual(p["random_state"], train_v3_xgb.SEED)

    def test_memory_bounded_model(self) -> None:
        m = train_v3_xgb.model_for((0,), device="cpu", memory_bounded=True)
        p = m.get_params()
        self.assertEqual(p["max_leaves"], 15)
        self.assertEqual(p["max_bin"], 31)
        self.assertEqual(p["n_jobs"], 1)

    def test_standard_model_leaves(self) -> None:
        m = train_v3_xgb.model_for(tuple(range(27)), device="cpu")
        p = m.get_params()
        self.assertEqual(p["max_leaves"], 63)
        self.assertEqual(p["max_bin"], 127)


class XGBPipelineSmokeTest(unittest.TestCase):
    """End-to-end smoke test: run train_v3_xgb on a tiny 30-row fixture."""

    def test_small_corpus_cpu(self) -> None:
        # Use an isolated temp dir — completely separate from artifacts/v3
        tmp_root = Path(__file__).resolve().parent.parent / "artifacts" / "_xgb_smoke_tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        fixture_dir = tmp_root / ("smoke_" + uuid4().hex)
        fixture_dir.mkdir()

        old_expected = train_v3_xgb.EXPECTED_ROWS
        try:
            def write(name: str, rows: list[str]) -> None:
                (fixture_dir / name).write_text("\n".join(rows) + "\n", encoding="utf-8")

            write("train_source1.tsv", [
                "entity_id\tbusiness_name\tbusiness_address\tcountry"
            ] + [
                f"S1-{i}\tWidget {i}\t{i+10} Market Road, Pin 110001 Delhi\tIN"
                for i in range(30)
            ])
            for source in (2, 3):
                write(f"train_source{source}.tsv", [
                    "entity_id\tbusiness_name\tbusiness_address\tcountry"
                ] + [
                    f"S{source}-{i}\t"
                    f"{'Unrelated' if source == 2 and i == 29 else f'Widget {i}'}\t"
                    f"{'999 Unknown Lane' if source == 2 and i == 29 else f'{i+10} Market Road, Pin 110001 Delhi'}\tIN"
                    for i in range(30)
                ])
            write("train_ground_truth.tsv", [
                "source1_entity_id\tmatched_entity_ids"
            ] + [
                f"S1-{i}\t" if i % 5 == 0 else f"S1-{i}\tS2-{i},S3-{i}"
                for i in range(30)
            ])

            # Patch EXPECTED_ROWS for the fixture size
            train_v3_xgb.EXPECTED_ROWS = {"s1": 30, "s2": 30, "s3": 30}

            # ── run the pipeline on CPU (safe: isolated DB in fixture_dir) ──
            artifacts = fixture_dir / "artifacts"
            train_v3_xgb.run(fixture_dir, artifacts, device="cpu")

            # Verify artifacts were written
            self.assertTrue((artifacts / "matcher_model.joblib").exists(),
                            "matcher_model.joblib not found")
            cfg = __import__("json").loads(
                (artifacts / "model_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cfg["model_backend"], "xgboost")
            self.assertIn("decision_threshold", cfg)

            metrics = __import__("json").loads(
                (artifacts / "training_metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("final_validation", metrics)
            self.assertIn("macro_f05", metrics["final_validation"])

            # Load and predict to confirm probabilities in [0, 1]
            import joblib
            model = joblib.load(artifacts / "matcher_model.joblib")
            cols = tuple(cfg["feature_indices"])
            x_sample = np.array([
                pair_features("Widget 1", "11 Market Road Pin 110001 Delhi",
                               "Widget 1", "11 Market Road Pin 110001 Delhi", 3)
            ], dtype=np.float32)[:, list(cols)]
            proba = model.predict_proba(x_sample)
            self.assertEqual(proba.shape[1], 2)
            self.assertGreater(proba[0, 1], 0.5, "Identical pair should have high match probability")

        finally:
            train_v3_xgb.EXPECTED_ROWS = old_expected
            shutil.rmtree(fixture_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
