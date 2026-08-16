"""
test_seed_evaluator.py
======================
Unit test suite for STEP 6 Multi-Seed Statistical Replication Pipeline:
  - calc_stats helper validation (mean, std, min, max, mean ± std formatting)
  - Execution across multiple random seeds (42, 123)
  - Recording of experiment status (SUCCESS, FAILED, NOT_EXECUTED)
  - Output artifact verification (seed_results.json, aggregated_results.json, comparison.csv)
"""

import os
import csv
import json
import unittest
from pathlib import Path

from src.evaluation.seed_evaluator import (
    run_multi_seed_evaluations,
    calc_stats,
    STATISTICS_OUT_DIR
)


class TestSeedEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_seed_eval")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    def test_calc_stats(self):
        """Tests statistical aggregation math for mean, std, min, max, formatted string."""
        vals = [0.90, 0.92, 0.94]
        res = calc_stats(vals)
        self.assertAlmostEqual(res["mean"], 0.92, places=3)
        self.assertAlmostEqual(res["min"], 0.90, places=3)
        self.assertAlmostEqual(res["max"], 0.94, places=3)
        self.assertIn("±", res["formatted"])
        self.assertEqual(res["count"], 3)

    def test_run_multi_seed_evaluations_outputs(self):
        """Tests multi-seed evaluation pipeline execution and artifact generation."""
        seeds = [42, 123]
        agg_res = run_multi_seed_evaluations(
            seeds=seeds,
            output_dir=self.tmp_out_dir
        )

        self.assertIn("adapter_screening", agg_res)
        self.assertIn("adaptive_evasion", agg_res)
        self.assertIn("utility_evaluation", agg_res)

        expected_files = [
            "seed_results.json",
            "aggregated_results.json",
            "comparison.csv"
        ]

        for fname in expected_files:
            fpath = self.tmp_out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} was not created.")

        # Check seed_results.json status tracking
        with open(self.tmp_out_dir / "seed_results.json", "r", encoding="utf-8") as f:
            sr_data = json.load(f)

        self.assertIn("results", sr_data)
        self.assertIn("seed_42", sr_data["results"])
        self.assertIn("seed_123", sr_data["results"])
        exps = sr_data["results"]["seed_42"]["experiments"]
        self.assertIn("pii_evaluation", exps)
        self.assertIn("adapter_screening", exps)
        self.assertIn("adaptive_evasion", exps)
        self.assertIn("utility_evaluation", exps)
        self.assertIn(exps["adapter_screening"]["status"], ["SUCCESS", "FAILED", "NOT_EXECUTED"])

        # Check comparison.csv structure
        with open(self.tmp_out_dir / "comparison.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 1)
        headers = rows[0]
        self.assertIn("category", headers)
        self.assertIn("detector_or_attack", headers)
        self.assertIn("f1_mean_std", headers)


if __name__ == "__main__":
    unittest.main()
