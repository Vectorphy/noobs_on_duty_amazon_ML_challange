"""Small numeric-equivalence checks for the vectorized v2 feature kernel."""

from __future__ import annotations

import unittest

import numpy as np

import matching_v2
from matching_v2 import FEATURES, pair_features, score_speedup


class ScoreSpeedupCheck(unittest.TestCase):
    def setUp(self) -> None:
        pairs = [
            (10, 100, 1, 2, 1, "Alpha Alpha Inc", "12 Main Road", "Alpha Inc", "12 Main Road"),
            (17, 100, 2, 3, 0, "Alpha Alpha Inc", "13 Main Road", "Alpha Inc", "12 Main Road"),
            (11, 102, 1, 3, 1, "München Café", None, "Munchen Cafe", " "),
            (12, 108, 1, 2, 0, "東京商店", "4 Sakura Street", "Tokyo Shoten", "4 Sakura St"),
            (13, 103, 1, 3, 0, "", "", "", None),
            (14, 104, 1, 2, 0, "AB", "7/2 Park Ave", "A B", "7-2 Park Avenue"),
            (15, 105, 1, 3, 1, "North Star LLC", "Unit 22, Lake Road", "North Star", "Unit 22 Lake Rd"),
            (10, 106, 1, 2, 0, "Alpha Alpha Inc", "12 Main Road", "Cafe Tokyo", "Building 9"),
        ]
        self.train_rows = [tuple(row) for row in pairs]
        self.inference_rows = [(row[0], row[1], row[3], row[5], row[6], row[7], row[8])
                               for row in pairs]
        self.expected = np.stack([
            pair_features(row[5], row[6], row[7], row[8], row[3])
            for row in pairs
        ])

    def test_matches_scalar_reference_in_both_layouts(self) -> None:
        train_before = list(self.train_rows)
        inference_before = list(self.inference_rows)
        train = score_speedup(self.train_rows, layout="train", workers=2)
        inference = score_speedup(self.inference_rows, layout="inference", workers=2)
        self.assertEqual(train.shape, (8, len(FEATURES)))
        self.assertEqual(train.dtype, np.float32)
        self.assertTrue(np.isfinite(train).all())
        np.testing.assert_allclose(train, self.expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(inference, self.expected, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(train, inference)
        self.assertEqual(self.train_rows, train_before)
        self.assertEqual(self.inference_rows, inference_before)

    def test_edges_and_threshold_decisions(self) -> None:
        features = score_speedup(self.inference_rows, layout="inference", workers=1)
        # Missing addresses suppress all address similarities and address trigram evidence.
        np.testing.assert_array_equal(features[2, [5, 6, 7, 8, 16]], np.zeros(5, np.float32))
        self.assertEqual(features[1, 19], 1.0)  # Source 3 indicator.
        self.assertEqual(features[0, 19], 0.0)  # Source 2 indicator.
        self.assertEqual(features[4, 3], 0.0)   # Empty name-token union.
        self.assertEqual(features[4, 15], 0.0)  # Empty trigram union.
        self.assertEqual(features[1, 17], 1.0)  # Conflicting address numbers.

        weights = np.linspace(-0.03, 0.04, len(FEATURES), dtype=np.float64)
        old_probability = 1.0 / (1.0 + np.exp(-(self.expected @ weights)))
        new_probability = 1.0 / (1.0 + np.exp(-(features @ weights)))
        np.testing.assert_allclose(new_probability, old_probability, rtol=0, atol=1e-7)
        threshold = float(old_probability[0])
        np.testing.assert_array_equal(new_probability >= threshold, old_probability >= threshold)
        self.assertTrue(new_probability[0] >= threshold)

    def test_empty_and_invalid_inputs(self) -> None:
        result = score_speedup([], layout="inference")
        self.assertEqual(result.shape, (0, len(FEATURES)))
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaises(ValueError):
            score_speedup([], layout="unknown")
        with self.assertRaises(ValueError):
            score_speedup([(1, 2)], layout="inference")

    def test_workspace_chunking_preserves_rows(self) -> None:
        old_limit = matching_v2.SCORE_WORKSPACE_ROWS
        try:
            matching_v2.SCORE_WORKSPACE_ROWS = 3
            actual = score_speedup(self.inference_rows, layout="inference", workers=2)
            np.testing.assert_allclose(actual, self.expected, rtol=0, atol=1e-6)
        finally:
            matching_v2.SCORE_WORKSPACE_ROWS = old_limit


if __name__ == "__main__":
    unittest.main()
