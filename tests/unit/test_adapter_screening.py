"""
test_adapter_screening.py
==========================
Unit test suite for the Adapter Security Screening module in SecureLoRA.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from src.security.adapter_screening import (
    StructuralAnalyzer,
    BehavioralAnalyzer,
    RiskScorer,
    ScreeningPipeline,
    ScreeningThresholdConfig,
    SecurityScreeningError,
    pre_packaging_screening_gate,
)
from src.phase3.package_builder import build_package


def _mock_clean_weights():
    rng = np.random.RandomState(42)
    return {
        "layer_0.lora_A.weight": rng.normal(0.0, 0.02, (8, 64)).astype(np.float32),
        "layer_0.lora_B.weight": rng.normal(0.0, 0.001, (64, 8)).astype(np.float32),
        "layer_1.lora_A.weight": rng.normal(0.0, 0.02, (8, 64)).astype(np.float32),
        "layer_1.lora_B.weight": rng.normal(0.0, 0.001, (64, 8)).astype(np.float32),
    }


def test_clean_adapter_screening(tmp_path):
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    weights = _mock_clean_weights()

    report = pipeline.screen_adapter(adapter_source=weights, adapter_id="clean_adapter_v1")
    assert report.approved is True
    assert report.decision == "APPROVED"
    assert report.risk_level == "LOW"
    assert report.risk_score < 0.35
    assert report.structural_score < 0.30
    assert report.execution_latency_ms > 0.0


def test_perturbed_adapter_screening(tmp_path):
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    clean = _mock_clean_weights()
    # Mild random perturbation
    perturbed = {k: v + np.random.RandomState(99).normal(0.0, 0.001, v.shape).astype(np.float32) for k, v in clean.items()}

    report = pipeline.screen_adapter(adapter_source=perturbed, adapter_id="perturbed_v1", trusted_weights_or_adapter=clean)
    assert report.approved is True
    assert report.risk_level == "LOW"
    assert report.risk_score < 0.35


def test_suspicious_synthetic_structural_outlier(tmp_path):
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0  # Massive structural outlier

    report = pipeline.screen_adapter(adapter_source=outlier, adapter_id="suspicious_outlier_v1", trusted_weights_or_adapter=clean)
    assert report.approved is False
    assert report.decision == "REJECTED"
    assert report.risk_level == "HIGH"
    assert report.risk_score >= 0.70
    assert len(report.structural_evidence.outlier_layers) > 0


def test_trigger_conditioned_behavioral_screening(tmp_path):
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    clean = _mock_clean_weights()
    trigger_model_dict = {"weights": clean, "force_trigger_activation": True}

    report = pipeline.screen_adapter(adapter_source=trigger_model_dict, adapter_id="trigger_adapter_v1")
    assert report.approved is False
    assert report.decision == "REJECTED"
    assert report.risk_level == "HIGH"
    assert report.behavioral_evidence.anomalous_trigger_detected is True
    assert report.behavioral_score >= 0.80


def test_threshold_behavior_configuration(tmp_path):
    strict_config = ScreeningThresholdConfig(low_risk_threshold=0.10, high_risk_threshold=0.30)
    pipeline = ScreeningPipeline(threshold_config=strict_config, audit_log_path=tmp_path / "audit.log")

    weights = _mock_clean_weights()
    report = pipeline.screen_adapter(adapter_source=weights, adapter_id="config_test")
    assert report.risk_assessment.threshold_config.low_risk_threshold == 0.10


def test_admin_override_behavior_and_audit(tmp_path):
    audit_file = tmp_path / "override_audit.log"
    pipeline = ScreeningPipeline(audit_log_path=audit_file)

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    # Screening without valid token should reject
    report_rejected = pipeline.screen_adapter(adapter_source=outlier, adapter_id="override_test_1")
    assert report_rejected.approved is False
    assert report_rejected.decision == "REJECTED"

    # Screening with valid admin token should approve with override and write audit record
    token = "ADMIN_OVERRIDE_TOKEN_2026"
    report_override = pipeline.screen_adapter(
        adapter_source=outlier,
        adapter_id="override_test_2",
        admin_override_token=token,
        override_reason="Authorized security researcher manual inspection.",
    )
    assert report_override.approved is True
    assert report_override.decision == "APPROVED_WITH_OVERRIDE"
    assert report_override.override_logged is True
    assert audit_file.exists()

    audit_content = audit_file.read_text(encoding="utf-8")
    assert "ADMIN_SCREENING_OVERRIDE" in audit_content
    assert "override_test_2" in audit_content


def test_reproducibility(tmp_path):
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    weights = _mock_clean_weights()

    report1 = pipeline.screen_adapter(adapter_source=weights, adapter_id="rep_test", seed=42)
    report2 = pipeline.screen_adapter(adapter_source=weights, adapter_id="rep_test", seed=42)

    assert report1.risk_score == report2.risk_score
    assert report1.structural_score == report2.structural_score
    assert report1.behavioral_score == report2.behavioral_score


def test_phase3_packaging_integration_gate(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()

    pub_key = tmp_path / "public.pem"
    priv_key = tmp_path / "private.pem"

    from src.security.signature import generate_dev_keypair
    generate_dev_keypair(priv_key, pub_key)

    (pkg_dir / "adapter.enc").write_bytes(b"CIPHERTEXT_12345")
    (pkg_dir / "adapter.hash").write_text("dummy_hash")
    (pkg_dir / "adapter.sig").write_bytes(b"dummy_sig")
    (pkg_dir / "metadata.json").write_text("{}")

    # Packaging clean directory succeeds
    manifest = build_package(
        package_dir=pkg_dir,
        adapter_id="med-v1",
        public_key_src=pub_key,
        private_key_src=priv_key,
        enable_screening=True,
    )
    assert manifest["package_id"] is not None
    assert (pkg_dir / "package_manifest.json").exists()
