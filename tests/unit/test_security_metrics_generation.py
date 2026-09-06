"""
test_security_metrics_generation.py
====================================
Unit test suite for Security Metric Generation Logic in experiment_runner.py.
"""

import unittest
from pathlib import Path
from src.evaluation.experiment_runner import run_single_baseline


class TestSecurityMetricsGeneration(unittest.TestCase):

    def setUp(self):
        self.tmp_out_dir = Path("outputs/test_sec_metrics_gen")
        self.tmp_out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_out_dir.exists():
            shutil.rmtree(self.tmp_out_dir, ignore_errors=True)

    def test_e0_base_model_security_metrics_are_not_executed(self):
        """Tests that E0 (Base Model) produces None (null/NOT_EXECUTED) for security metrics."""
        res = run_single_baseline("E0", seed=42, output_dir=self.tmp_out_dir, quick_mode=True)
        self.assertEqual(res.execution_status, "COMPLETED")
        sec = res.security

        # None of the security features are enabled/executed in E0
        self.assertIsNone(sec.unauthorized_device_rejection_rate)
        self.assertIsNone(sec.cross_device_rejection_rate)
        self.assertIsNone(sec.tamper_rejection_rate)
        self.assertIsNone(sec.signature_rejection_rate)
        self.assertIsNone(sec.wrong_key_rejection_rate)
        self.assertIsNone(sec.replay_rejection_rate)
        self.assertIsNone(sec.malicious_adapter_detection_rate)
        self.assertIsNone(sec.unauthorized_deployment_rejection_rate)

        # Verify details dictionary structure
        self.assertIsNotNone(sec.details)
        for key in ["unauthorized_device_rejection", "tamper_rejection", "signature_rejection", "wrong_key_rejection"]:
            det = sec.details[key]
            self.assertEqual(det["status"], "NOT_EXECUTED")
            self.assertIsNone(det["numerator"])
            self.assertIsNone(det["denominator"])
            self.assertIsNone(det["rate"])

    def test_e9_full_securelora_security_metrics_provenance(self):
        """Tests that E9 (FULL SECURELORA) computes rates from executed tests and records numerators/denominators."""
        res = run_single_baseline("E9", seed=42, output_dir=self.tmp_out_dir, quick_mode=True)
        self.assertEqual(res.execution_status, "COMPLETED")
        sec = res.security

        self.assertIsNotNone(sec.unauthorized_device_rejection_rate)
        self.assertIsNotNone(sec.cross_device_rejection_rate)
        self.assertIsNotNone(sec.tamper_rejection_rate)
        self.assertIsNotNone(sec.signature_rejection_rate)
        self.assertIsNotNone(sec.wrong_key_rejection_rate)
        self.assertIsNotNone(sec.replay_rejection_rate)
        self.assertIsNotNone(sec.unauthorized_deployment_rejection_rate)

        # Check numerators and denominators in details
        det = sec.details
        for metric_name in [
            "unauthorized_device_rejection",
            "cross_device_rejection",
            "tamper_rejection",
            "signature_rejection",
            "wrong_key_rejection",
            "replay_rejection",
            "unauthorized_deployment_rejection"
        ]:
            self.assertIn(metric_name, det)
            item = det[metric_name]
            self.assertEqual(item["status"], "EXECUTED")
            self.assertIsInstance(item["numerator"], int)
            self.assertIsInstance(item["denominator"], int)
            self.assertGreater(item["denominator"], 0)
            self.assertEqual(item["rate"], item["numerator"] / item["denominator"])
            self.assertIn("source_test", item)


if __name__ == "__main__":
    unittest.main()
