"""
test_model_scale_evaluator.py
==============================
Unit test suite for STEP 8 Model Scale Evaluation:
  - Verification of computational & security metrics across Lightweight and Scaled models
  - Verification of model comparison JSON and CSV artifacts
  - Verification of security screening measurability across model scales
"""

import os
import csv
import json
import unittest
from pathlib import Path

from src.evaluation.model_scale_evaluator import (
    run_model_scale_evaluation,
    evaluate_model_scale,
    LightweightModel,
    ScaledModel,
    MODEL_SCALE_OUT_DIR
)


class TestModelScaleEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_model_scale_eval")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    def test_evaluate_model_scale_single(self):
        """Verifies measurements for single model scale execution."""
        res = evaluate_model_scale(
            scale_name="test_lightweight",
            model_name="Test Llama-68m",
            model_cls=LightweightModel,
            lora_rank=4,
            num_layers=2,
            hidden_dim=128,
            vocab_size=1000,
            seed=42
        )

        self.assertEqual(res["scale_name"], "test_lightweight")
        self.assertGreater(res["parameter_count"], 0)
        self.assertGreater(res["adapter_parameter_count"], 0)
        self.assertGreater(res["adapter_size_kb"], 0)
        self.assertGreaterEqual(res["training_time_s"], 0)
        self.assertGreaterEqual(res["inference_latency_ms"], 0)
        self.assertGreaterEqual(res["screening_latency_ms"], 0)
        self.assertGreaterEqual(res["encryption_time_ms"], 0)
        self.assertGreaterEqual(res["decryption_time_ms"], 0)
        self.assertGreaterEqual(res["verification_time_ms"], 0)
        self.assertGreater(res["memory_usage_mb"], 0)

        sec = res["security_verification"]
        self.assertTrue(sec["measurable"])
        self.assertIn("approved", sec)
        self.assertIn("combined_score", sec)

    def test_run_model_scale_evaluation_artifacts(self):
        """Verifies complete step 8 evaluation run and artifact output files."""
        res_data = run_model_scale_evaluation(output_dir=self.tmp_out_dir)

        self.assertIn("summary", res_data)
        self.assertIn("models", res_data)
        self.assertIn("scaling_ratios", res_data)

        # Check JSON output file
        json_file = self.tmp_out_dir / "model_comparison.json"
        self.assertTrue(json_file.exists(), "model_comparison.json was not created.")
        with open(json_file, "r", encoding="utf-8") as f:
            j_data = json.load(f)

        self.assertIn("lightweight", j_data["models"])
        self.assertIn("scaled", j_data["models"])

        # Check CSV output file
        csv_file = self.tmp_out_dir / "model_comparison.csv"
        self.assertTrue(csv_file.exists(), "model_comparison.csv was not created.")
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        header = reader[0]
        self.assertEqual(header[0], "scale_name")
        self.assertEqual(header[1], "model_name")
        self.assertIn("screening_approved", header)
        self.assertIn("combined_risk_score", header)

        # Data rows (lightweight + scaled = 2 rows)
        self.assertEqual(len(reader), 3)


if __name__ == "__main__":
    unittest.main()
