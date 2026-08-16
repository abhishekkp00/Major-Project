"""
test_adaptive_evasion_evaluator.py
===================================
Unit test suite for STEP 5 Adaptive Evasion Evaluation Pipeline:
  - 3 attack strategies: random_perturbation, non_adaptive, adaptive
  - 3 detectors: structural_only, behavioral_only, combined
  - Iteration-by-iteration trajectory recording
  - Failure analysis for successful evasion attacks
  - Output artifacts verification (JSON & iteration_history.csv)
"""

import os
import csv
import json
import unittest
from pathlib import Path

from src.evaluation.adaptive_evasion_evaluator import (
    run_adaptive_evasion_evaluation,
    compute_relative_perturbation,
    estimate_payload_utility,
    EVASION_OUT_DIR
)


class TestAdaptiveEvasionEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_evasion_eval")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    def test_utility_and_perturbation_helpers(self):
        import numpy as np
        W1 = {"layer_0": np.ones((4, 4), dtype=np.float32)}
        W2 = {"layer_0": np.ones((4, 4), dtype=np.float32) * 1.1}

        pert = compute_relative_perturbation(W2, W1)
        self.assertGreater(pert, 0.0)

        util = estimate_payload_utility(W2, W1)
        self.assertAlmostEqual(util, 1.0, places=3)

    def test_run_adaptive_evasion_evaluation_outputs(self):
        comp_res = run_adaptive_evasion_evaluation(
            num_malicious_samples=4,
            max_iterations=3,
            seed=42,
            threshold=0.35,
            output_dir=self.tmp_out_dir
        )

        self.assertIn("attack_strategies", comp_res)
        self.assertIn("baseline", comp_res["attack_strategies"])
        self.assertIn("nonadaptive", comp_res["attack_strategies"])
        self.assertIn("adaptive", comp_res["attack_strategies"])

        expected_files = [
            "baseline_attack.json",
            "nonadaptive_attack.json",
            "adaptive_attack.json",
            "comparison.json",
            "iteration_history.csv"
        ]

        for fname in expected_files:
            fpath = self.tmp_out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} was not created.")

        # Verify iteration_history.csv contents
        csv_path = self.tmp_out_dir / "iteration_history.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 1)  # Header + iteration data
        headers = rows[0]
        self.assertIn("sample_id", headers)
        self.assertIn("attack_type", headers)
        self.assertIn("detector_type", headers)
        self.assertIn("iteration", headers)
        self.assertIn("structural_score", headers)
        self.assertIn("behavioral_score", headers)
        self.assertIn("combined_score", headers)
        self.assertIn("detector_decision", headers)
        self.assertIn("utility_preservation", headers)
        self.assertIn("perturbation_magnitude", headers)


if __name__ == "__main__":
    unittest.main()
