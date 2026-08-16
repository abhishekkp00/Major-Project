"""
test_screening_evaluator.py
============================
Unit test suite for STEP 4 Security Screening Evaluation Pipeline:
  - System A (structural_only), System B (behavioral_only), System C (combined)
  - Validation threshold sweep & threshold selection
  - Calculation of precision, recall, F1, accuracy, FPR, FNR, latency_ms, TP, TN, FP, FN
  - Verification of output JSON artifacts and screening_comparison.csv
"""

import os
import csv
import json
import unittest
from pathlib import Path

from src.evaluation.screening_evaluator import (
    run_screening_evaluation,
    _calculate_classification_metrics,
    evaluate_threshold_sweep,
    SCREENING_OUT_DIR
)


class TestScreeningEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_screening_eval")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    def test_classification_metrics_calculation(self):
        """Tests core metric math for TP, TN, FP, FN, precision, recall, F1, accuracy, FPR, FNR."""
        y_true = [True, True, False, False, True]
        y_scores = [0.85, 0.40, 0.90, 0.10, 0.75]
        threshold = 0.50

        # At threshold 0.50:
        # Item 0: True, score 0.85 >= 0.5 -> TP
        # Item 1: True, score 0.40 < 0.5  -> FN
        # Item 2: False, score 0.90 >= 0.5 -> FP
        # Item 3: False, score 0.10 < 0.5  -> TN
        # Item 4: True, score 0.75 >= 0.5 -> TP
        # Total: TP=2, TN=1, FP=1, FN=1

        m = _calculate_classification_metrics(y_true, y_scores, threshold)

        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertAlmostEqual(m["precision"], round(2 / 3, 4))
        self.assertAlmostEqual(m["recall"], round(2 / 3, 4))
        self.assertAlmostEqual(m["f1"], round(2 / 3, 4))
        self.assertAlmostEqual(m["accuracy"], round(3 / 5, 4))
        self.assertAlmostEqual(m["false_positive_rate"], round(1 / 2, 4))
        self.assertAlmostEqual(m["false_negative_rate"], round(1 / 3, 4))

    def test_threshold_sweep(self):
        """Tests sweeping thresholds across validation data."""
        y_true = [True, False, True, False]
        y_scores = [0.8, 0.2, 0.6, 0.7]
        thresholds = [0.3, 0.5, 0.75]

        sweep = evaluate_threshold_sweep(y_true, y_scores, thresholds)
        self.assertEqual(len(sweep), 3)
        for entry in sweep:
            self.assertIn("threshold", entry)
            self.assertIn("precision", entry)
            self.assertIn("recall", entry)
            self.assertIn("f1", entry)
            self.assertIn("false_positive_rate", entry)
            self.assertIn("false_negative_rate", entry)

    def test_run_screening_evaluation_outputs(self):
        """Tests that run_screening_evaluation generates all required JSON and CSV artifacts."""
        comp_res = run_screening_evaluation(
            num_samples_per_cat=6,
            seed=42,
            output_dir=self.tmp_out_dir
        )

        self.assertIn("systems", comp_res)
        self.assertIn("structural_only", comp_res["systems"])
        self.assertIn("behavioral_only", comp_res["systems"])
        self.assertIn("combined", comp_res["systems"])

        # Check required output files
        expected_files = [
            "structural_only.json",
            "behavioral_only.json",
            "combined.json",
            "comparison.json",
            "screening_comparison.csv"
        ]

        for fname in expected_files:
            fpath = self.tmp_out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} was not created.")

        # Inspect CSV structure
        csv_path = self.tmp_out_dir / "screening_comparison.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertGreaterEqual(len(rows), 4)  # Header + 3 systems
        headers = rows[0]
        self.assertIn("system", headers)
        self.assertIn("selected_threshold", headers)
        self.assertIn("precision", headers)
        self.assertIn("recall", headers)
        self.assertIn("f1", headers)
        self.assertIn("accuracy", headers)
        self.assertIn("false_positive_rate", headers)
        self.assertIn("false_negative_rate", headers)
        self.assertIn("mean_latency_ms", headers)
        self.assertIn("tp", headers)
        self.assertIn("tn", headers)
        self.assertIn("fp", headers)
        self.assertIn("fn", headers)


if __name__ == "__main__":
    unittest.main()
