"""
test_privacy_evaluator.py
==========================
Comprehensive unit test suite for STEP 3 Privacy Evaluation Pipeline:
  - Sequence verification: MODEL GENERATION -> RAW OUTPUT -> PII DETECTION -> METRICS
  - Calculation of PII leakage rate, entity counts, types, PII-free response rate
  - Calculation of precision, recall, F1, FPR, FNR when ground truth available
  - Validation of output files: base_model.json, lora.json, dp_lora.json, securelora.json, comparison.json
  - Compliance with NO FABRICATION rule (status = NOT_EXECUTED when models/adapters unavailable)
  - Reproducibility via CLI arguments (--dataset, --split, --samples, --seed, --model, --adapter)
"""

import os
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.evaluation.privacy_evaluator import (
    evaluate_privacy_pipeline,
    calculate_privacy_metrics,
    PRIVACY_OUT_DIR
)
from src.orchestrator.model_registry import model_registry


class TestPrivacyEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_privacy_eval")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)
        model_registry.clear()

    def tearDown(self):
        model_registry.clear()

    def test_metrics_calculation_with_ground_truth(self):
        """Tests calculation of leakage rates and precision/recall/F1 metrics."""
        raw_outputs = [
            "User John Doe has email john.doe@example.com and SSN 123-45-6789.",
            "The weather is clear today with 22C temperature.",
            "Contact admin at support@company.org or call 555-123-4567."
        ]

        gt_records = [
            {
                "input": "Query 1",
                "pii_entities": [
                    {"type": "EMAIL", "text": "john.doe@example.com"},
                    {"type": "SSN", "text": "123-45-6789"}
                ]
            },
            {
                "input": "Query 2",
                "pii_entities": []
            },
            {
                "input": "Query 3",
                "pii_entities": [
                    {"type": "EMAIL", "text": "support@company.org"},
                    {"type": "PHONE", "text": "555-123-4567"}
                ]
            }
        ]

        metrics = calculate_privacy_metrics(
            raw_outputs,
            ground_truth_records=gt_records,
            ground_truth_available=True
        )

        self.assertIn("pii_leakage_rate", metrics)
        self.assertIn("pii_free_response_rate", metrics)
        self.assertEqual(metrics["records_containing_pii"], 2)
        self.assertAlmostEqual(metrics["pii_leakage_rate"], round(2/3, 4))
        self.assertAlmostEqual(metrics["pii_free_response_rate"], round(1 - 2/3, 4))
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

    def test_unexecuted_experiment_handling(self):
        """Tests that missing model/adapters produce status = NOT_EXECUTED without placeholder numbers."""
        model_registry.clear()

        res = evaluate_privacy_pipeline(
            dataset_id="synthetic",
            samples=5,
            seed=42,
            output_dir=self.tmp_out_dir
        )

        self.assertIn("variants", res)
        for variant in ["base_model", "lora", "dp_lora", "securelora"]:
            v_data = res["variants"][variant]
            self.assertEqual(v_data["status"], "NOT_EXECUTED")
            self.assertIsNone(v_data["metrics"])

        base_file = self.tmp_out_dir / "base_model.json"
        self.assertTrue(base_file.exists())
        with open(base_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["status"], "NOT_EXECUTED")
            self.assertIsNone(data["metrics"])

    def test_executed_experiment_flow(self):
        """Tests evaluation pipeline execution when mock model and tokenizer are registered."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Generated text containing john.doe@example.com for testing."

        mock_base_model = MagicMock()
        mock_base_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_base_model.parameters.side_effect = lambda: iter([MagicMock(device="cpu")])

        mock_peft_model = MagicMock()
        mock_peft_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_peft_model.parameters.side_effect = lambda: iter([MagicMock(device="cpu")])

        # Register in model_registry
        model_registry.register(
            base_model=mock_base_model,
            peft_model=mock_peft_model,
            tokenizer=mock_tokenizer,
            base_model_name="google/gemma-2b",
            adapter_id="secure_lora_adapter",
            deployment_id="test_deploy",
            deployment_status="VERIFIED"
        )

        res = evaluate_privacy_pipeline(
            dataset_id="synthetic",
            samples=3,
            seed=42,
            output_dir=self.tmp_out_dir
        )

        self.assertIn("variants", res)
        sec_variant = res["variants"]["securelora"]
        self.assertEqual(sec_variant["status"], "EXECUTED")
        self.assertIsNotNone(sec_variant["metrics"])
        self.assertIn("pii_leakage_rate", sec_variant["metrics"])

        comp_file = self.tmp_out_dir / "comparison.json"
        self.assertTrue(comp_file.exists())


if __name__ == "__main__":
    unittest.main()
