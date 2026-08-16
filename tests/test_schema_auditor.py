"""
test_schema_auditor.py
======================
Unit test suite for STEP 9 Research Schema Auditor:
  - Verifies UnifiedExperimentResult schema validation
  - Verifies strict status value enforcement (EXECUTED, FAILED, NOT_EXECUTED)
  - Verifies metrics separation (raw, aggregated, reported)
  - Verifies artifact audit and cleanup functionality
"""

import os
import json
import unittest
from pathlib import Path

from src.evaluation.metrics_schema import UnifiedExperimentResult, VALID_STATUSES
from src.evaluation.schema_auditor import (
    convert_file_to_unified_schema,
    audit_and_standardize_research_outputs
)


class TestSchemaAuditor(unittest.TestCase):

    def setUp(self):
        self.tmp_eval_dir = Path("outputs/test_eval_audit")
        self.tmp_eval_dir.mkdir(parents=True, exist_ok=True)

        (self.tmp_eval_dir / "privacy").mkdir(parents=True, exist_ok=True)
        (self.tmp_eval_dir / "screening").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_eval_dir.exists():
            shutil.rmtree(self.tmp_eval_dir, ignore_errors=True)

    def test_unified_schema_post_init_validation(self):
        """Verifies status and metrics validation in UnifiedExperimentResult."""
        # Valid instance
        valid = UnifiedExperimentResult(
            experiment_id="EXP_TEST_01",
            experiment_name="Test Experiment",
            dataset="ai4privacy",
            dataset_version="v1.0.0",
            model="llama-68m",
            adapter="lora_rank_4",
            configuration={"seed": 42},
            seed=42,
            sample_count=100,
            status="EXECUTED",
            metrics={"raw": {}, "aggregated": {}, "reported": {}},
            runtime={"execution_time_seconds": 1.5, "latency_ms": 15.0, "peak_memory_mb": 120.0},
            timestamp="2026-08-16T00:00:00Z"
        )
        self.assertEqual(valid.status, "EXECUTED")

        # Invalid status
        with self.assertRaises(ValueError):
            UnifiedExperimentResult(
                experiment_id="EXP_TEST_BAD_STATUS",
                experiment_name="Test Bad Status",
                dataset="ai4privacy",
                dataset_version="v1.0.0",
                model="llama-68m",
                adapter="lora_rank_4",
                configuration={},
                seed=42,
                sample_count=100,
                status="COMPLETED",  # Must be EXECUTED, FAILED, or NOT_EXECUTED
                metrics={"raw": {}, "aggregated": {}, "reported": {}},
                runtime={},
                timestamp="2026-08-16T00:00:00Z"
            )

        # Invalid metrics missing required keys
        with self.assertRaises(ValueError):
            UnifiedExperimentResult(
                experiment_id="EXP_TEST_BAD_METRICS",
                experiment_name="Test Bad Metrics",
                dataset="ai4privacy",
                dataset_version="v1.0.0",
                model="llama-68m",
                adapter="lora_rank_4",
                configuration={},
                seed=42,
                sample_count=100,
                status="EXECUTED",
                metrics={"raw": {}},  # missing aggregated and reported
                runtime={},
                timestamp="2026-08-16T00:00:00Z"
            )

    def test_schema_conversion_and_audit(self):
        """Verifies schema conversion and audit process on sample files."""
        # Create un-standardized legacy file
        legacy_file = self.tmp_eval_dir / "privacy" / "legacy_test.json"
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump({
                "policy_name": "Legacy Privacy Run",
                "execution_status": "COMPLETED",
                "metrics": {"pii_leakage_rate": 0.05},
                "avg_recovery_time_ms": 12.5
            }, f, indent=2)

        ok, converted, msg = convert_file_to_unified_schema(legacy_file)
        self.assertTrue(ok)
        self.assertEqual(converted["status"], "EXECUTED")
        self.assertIn("raw", converted["metrics"])
        self.assertIn("aggregated", converted["metrics"])
        self.assertIn("reported", converted["metrics"])

        # Create loose root file to be archived
        loose_file = self.tmp_eval_dir / "loose_report.json"
        loose_file.write_text(json.dumps({"test": 123}), encoding="utf-8")

        audit_res = audit_and_standardize_research_outputs(base_dir=self.tmp_eval_dir)
        self.assertGreaterEqual(audit_res["files_processed"], 1)
        self.assertGreaterEqual(audit_res["files_archived"], 1)

        archived_loose = self.tmp_eval_dir / "archive" / "loose_report.json"
        self.assertTrue(archived_loose.exists(), "Loose file was not moved to archive directory.")


if __name__ == "__main__":
    unittest.main()
