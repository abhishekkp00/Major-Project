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


from unittest.mock import patch

class TestSeedValidityAggregation(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_seed_eval_validity")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    @patch("src.evaluation.seed_evaluator.evaluate_privacy_pipeline")
    @patch("src.evaluation.seed_evaluator.run_screening_evaluation")
    @patch("src.evaluation.seed_evaluator.run_adaptive_evasion_evaluation")
    @patch("src.evaluation.seed_evaluator.evaluate_dataset_adapter")
    def test_5_valid_seeds(self, mock_util, mock_evasion, mock_screening, mock_privacy):
        """Tests aggregation when 5 valid seeds exist."""
        mock_privacy.return_value = {"status": "SUCCESS", "variants": {"securelora": {"metrics": {"pii_leakage_rate": 0.0}}}}
        mock_screening.return_value = {"status": "SUCCESS", "systems": {"combined": {"test_metrics": {"f1": 0.95, "precision": 0.95, "recall": 0.95, "accuracy": 0.95, "false_positive_rate": 0.05, "false_negative_rate": 0.05, "mean_latency_ms": 10.0}}}}
        mock_evasion.return_value = {"status": "SUCCESS", "attack_strategies": {"baseline": {"detectors": {"combined": {"detection_rate": 0.95, "false_negative_rate": 0.05, "attack_success_rate": 0.05, "avg_final_score": 0.95, "avg_utility_preservation": 0.9}}}}}
        mock_util.return_value = {"status": "SUCCESS", "metrics": {"f1": 0.90, "precision": 0.90, "recall": 0.90, "record_count": 30}}

        seeds = [42, 123, 456, 789, 1001]
        agg_res = run_multi_seed_evaluations(seeds=seeds, output_dir=self.tmp_out_dir)

        self.assertEqual(agg_res["valid_seed_count"], 5)
        self.assertEqual(agg_res["execution_status"], "COMPLETE")
        self.assertEqual(agg_res["evaluation_title"], "5-seed COMPLETE evaluation")
        self.assertEqual(len(agg_res["executed_seeds"]), 5)
        self.assertEqual(len(agg_res["failed_seeds"]), 0)
        self.assertEqual(len(agg_res["skipped_seeds"]), 0)

    @patch("src.evaluation.seed_evaluator.evaluate_privacy_pipeline")
    @patch("src.evaluation.seed_evaluator.run_screening_evaluation")
    @patch("src.evaluation.seed_evaluator.run_adaptive_evasion_evaluation")
    @patch("src.evaluation.seed_evaluator.evaluate_dataset_adapter")
    def test_3_valid_2_not_executed(self, mock_util, mock_evasion, mock_screening, mock_privacy):
        """Tests that NOT_EXECUTED seeds are excluded and labeled PARTIAL."""
        def privacy_side_effect(dataset_id, samples, seed):
            if seed in [789, 1001]:
                return {"status": "NOT_EXECUTED"}
            return {"status": "SUCCESS", "variants": {"securelora": {"metrics": {"pii_leakage_rate": 0.0}}}}

        def screening_side_effect(num_samples_per_cat, seed):
            if seed in [789, 1001]:
                return {"status": "NOT_EXECUTED"}
            return {"status": "SUCCESS", "systems": {"combined": {"test_metrics": {"f1": 0.95, "precision": 0.95, "recall": 0.95, "accuracy": 0.95, "false_positive_rate": 0.05, "false_negative_rate": 0.05, "mean_latency_ms": 10.0}}}}

        def evasion_side_effect(num_malicious_samples, max_iterations, seed):
            if seed in [789, 1001]:
                return {"status": "NOT_EXECUTED"}
            return {"status": "SUCCESS", "attack_strategies": {"baseline": {"detectors": {"combined": {"detection_rate": 0.95, "false_negative_rate": 0.05, "attack_success_rate": 0.05, "avg_final_score": 0.95, "avg_utility_preservation": 0.9}}}}}

        def util_side_effect(dataset_id, subset_size, seed):
            if seed in [789, 1001]:
                return {"status": "NOT_EXECUTED"}
            return {"status": "SUCCESS", "metrics": {"f1": 0.90, "precision": 0.90, "recall": 0.90, "record_count": 30}}

        mock_privacy.side_effect = privacy_side_effect
        mock_screening.side_effect = screening_side_effect
        mock_evasion.side_effect = evasion_side_effect
        mock_util.side_effect = util_side_effect

        seeds = [42, 123, 456, 789, 1001]
        agg_res = run_multi_seed_evaluations(seeds=seeds, output_dir=self.tmp_out_dir)

        self.assertEqual(agg_res["valid_seed_count"], 3)
        self.assertEqual(agg_res["execution_status"], "PARTIAL")
        self.assertEqual(agg_res["evaluation_title"], "3-seed PARTIAL evaluation")
        self.assertEqual(agg_res["executed_seeds"], [42, 123, 456])
        self.assertEqual(agg_res["skipped_seeds"], [789, 1001])
        self.assertNotIn("5-seed evaluation", agg_res["evaluation_title"])

    @patch("src.evaluation.seed_evaluator.evaluate_privacy_pipeline")
    @patch("src.evaluation.seed_evaluator.run_screening_evaluation")
    @patch("src.evaluation.seed_evaluator.run_adaptive_evasion_evaluation")
    @patch("src.evaluation.seed_evaluator.evaluate_dataset_adapter")
    def test_failed_seed(self, mock_util, mock_evasion, mock_screening, mock_privacy):
        """Tests handling when one seed fails due to exception."""
        def screening_side_effect(num_samples_per_cat, seed):
            if seed == 1001:
                raise RuntimeError("Hardware failure on seed 1001")
            return {"status": "SUCCESS", "systems": {"combined": {"test_metrics": {"f1": 0.95, "precision": 0.95, "recall": 0.95, "accuracy": 0.95, "false_positive_rate": 0.05, "false_negative_rate": 0.05, "mean_latency_ms": 10.0}}}}

        mock_privacy.return_value = {"status": "SUCCESS", "variants": {"securelora": {"metrics": {"pii_leakage_rate": 0.0}}}}
        mock_screening.side_effect = screening_side_effect
        mock_evasion.return_value = {"status": "SUCCESS", "attack_strategies": {"baseline": {"detectors": {"combined": {"detection_rate": 0.95, "false_negative_rate": 0.05, "attack_success_rate": 0.05, "avg_final_score": 0.95, "avg_utility_preservation": 0.9}}}}}
        mock_util.return_value = {"status": "SUCCESS", "metrics": {"f1": 0.90, "precision": 0.90, "recall": 0.90, "record_count": 30}}

        seeds = [42, 123, 456, 789, 1001]
        agg_res = run_multi_seed_evaluations(seeds=seeds, output_dir=self.tmp_out_dir)

        self.assertEqual(agg_res["valid_seed_count"], 4)
        self.assertEqual(agg_res["execution_status"], "PARTIAL")
        self.assertEqual(agg_res["failed_seeds"], [1001])
        self.assertEqual(agg_res["executed_seeds"], [42, 123, 456, 789])

    @patch("src.evaluation.seed_evaluator.evaluate_privacy_pipeline")
    @patch("src.evaluation.seed_evaluator.run_screening_evaluation")
    @patch("src.evaluation.seed_evaluator.run_adaptive_evasion_evaluation")
    @patch("src.evaluation.seed_evaluator.evaluate_dataset_adapter")
    def test_all_seeds_missing(self, mock_util, mock_evasion, mock_screening, mock_privacy):
        """Tests handling when all seeds are missing/unexecuted."""
        mock_privacy.return_value = {"status": "NOT_EXECUTED"}
        mock_screening.return_value = {"status": "NOT_EXECUTED"}
        mock_evasion.return_value = {"status": "NOT_EXECUTED"}
        mock_util.return_value = {"status": "NOT_EXECUTED"}

        seeds = [42, 123, 456, 789, 1001]
        agg_res = run_multi_seed_evaluations(seeds=seeds, output_dir=self.tmp_out_dir)

        self.assertEqual(agg_res["valid_seed_count"], 0)
        self.assertEqual(agg_res["execution_status"], "NOT_EXECUTED")
        self.assertEqual(agg_res["evaluation_title"], "0-seed NOT_EXECUTED evaluation")
        self.assertEqual(agg_res["skipped_seeds"], [42, 123, 456, 789, 1001])


if __name__ == "__main__":
    unittest.main()
