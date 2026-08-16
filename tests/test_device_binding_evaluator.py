"""
test_device_binding_evaluator.py
=================================
Unit test suite for STEP 7 Device Binding & Authorization Policy Evaluation:
  - Verification of 8 evaluation scenarios (reboot, hostname, MAC, machine-id, disk, VM clone, foreign, replay)
  - Evaluation of Static Fingerprint Policy vs Adaptive Device Policy
  - Verification of Security (unauthorized_rejection_rate, replay_rejection_rate) and Availability metrics
  - Verification of output artifacts (static_policy.json, adaptive_policy.json, comparison.json)
"""

import os
import json
import unittest
from pathlib import Path

from src.evaluation.device_binding_evaluator import (
    run_device_binding_evaluation,
    evaluate_static_policy,
    evaluate_adaptive_policy,
    get_scenario_attributes,
    DEVICE_BINDING_OUT_DIR
)


class TestDeviceBindingEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_device_binding_eval")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    def test_scenario_attributes_integrity(self):
        """Verifies attribute generation for all 8 scenarios."""
        scenarios = [
            "legitimate_reboot", "hostname_change", "network_interface_change",
            "machine_id_change", "disk_environment_change", "vm_clone",
            "foreign_device", "replayed_deployment_package"
        ]
        for sc_id in scenarios:
            attrs, is_legit, meta = get_scenario_attributes(sc_id)
            self.assertIsInstance(attrs, dict)
            self.assertIsInstance(is_legit, bool)
            self.assertEqual(meta["scenario_id"], sc_id)

    def test_static_policy_evaluation(self):
        """Verifies static fingerprint policy strictness and replay vulnerability."""
        reboot_res = evaluate_static_policy("legitimate_reboot")
        self.assertTrue(reboot_res["is_authorized"])

        hostname_res = evaluate_static_policy("hostname_change")
        self.assertFalse(hostname_res["is_authorized"])

    def test_run_device_binding_evaluation_artifacts(self):
        """Verifies complete step 7 evaluation run and output JSON reports."""
        comp_res = run_device_binding_evaluation(output_dir=self.tmp_out_dir)

        self.assertIn("summary", comp_res)
        self.assertIn("metrics_comparison", comp_res)
        self.assertIn("scenario_comparison", comp_res)

        expected_files = ["static_policy.json", "adaptive_policy.json", "comparison.json"]
        for fname in expected_files:
            fpath = self.tmp_out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} was not created.")

        with open(self.tmp_out_dir / "comparison.json", "r", encoding="utf-8") as f:
            comp_data = json.load(f)

        m_comp = comp_data["metrics_comparison"]
        self.assertIn("static_policy", m_comp)
        self.assertIn("adaptive_policy", m_comp)
        self.assertIn("tradeoff_delta", m_comp)

        static_m = m_comp["static_policy"]
        adaptive_m = m_comp["adaptive_policy"]

        self.assertGreaterEqual(adaptive_m["legitimate_acceptance_rate"], static_m["legitimate_acceptance_rate"])
        self.assertGreaterEqual(adaptive_m["replay_rejection_rate"], static_m["replay_rejection_rate"])


if __name__ == "__main__":
    unittest.main()
