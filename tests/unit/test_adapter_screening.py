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


def test_admin_override_behavior_and_audit(tmp_path, monkeypatch):
    audit_file = tmp_path / "override_audit.log"
    runtime_secret = "runtime_override_secret_token_876"
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", runtime_secret)
    pipeline = ScreeningPipeline(audit_log_path=audit_file)

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    # Screening without valid token should reject
    report_rejected = pipeline.screen_adapter(adapter_source=outlier, adapter_id="override_test_1")
    assert report_rejected.approved is False
    assert report_rejected.decision == "REJECTED"

    # Screening with valid admin token should approve with override and write audit record
    report_override = pipeline.screen_adapter(
        adapter_source=outlier,
        adapter_id="override_test_2",
        admin_override_token=runtime_secret,
        override_reason="Authorized security researcher manual inspection.",
    )
    assert report_override.approved is True
    assert report_override.decision == "APPROVED_WITH_OVERRIDE"
    assert report_override.override_logged is True
    assert audit_file.exists()

    audit_content = audit_file.read_text(encoding="utf-8")
    assert "ADMIN_SCREENING_OVERRIDE" in audit_content
    assert "override_test_2" in audit_content
    # Raw token must NEVER be written to the audit log
    assert runtime_secret not in audit_content


def test_admin_override_missing_env_fails_closed(tmp_path, monkeypatch):
    """When ADMIN_SCREENING_OVERRIDE is absent, pipeline must fail closed."""
    monkeypatch.delenv("ADMIN_SCREENING_OVERRIDE", raising=False)
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    # Even with an arbitrary attempted token, missing env must reject HIGH-risk
    report = pipeline.screen_adapter(
        adapter_source=outlier,
        adapter_id="missing_env_test",
        admin_override_token="attempted_unconfigured_token",
    )
    assert report.approved is False
    assert report.decision == "REJECTED"


def test_admin_override_wrong_token_rejected(tmp_path, monkeypatch):
    """Supplying an incorrect token must reject override."""
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", "correct_runtime_token_999")
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    report = pipeline.screen_adapter(
        adapter_source=outlier,
        adapter_id="wrong_token_test",
        admin_override_token="completely_wrong_token",
    )
    assert report.approved is False
    assert report.decision == "REJECTED"


def test_admin_override_empty_token_rejected(tmp_path, monkeypatch):
    """Empty or whitespace-only token must be rejected."""
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", "valid_secret_key_123")
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    # None token
    assert pipeline.screen_adapter(adapter_source=outlier, admin_override_token=None).approved is False
    # Empty string token
    assert pipeline.screen_adapter(adapter_source=outlier, admin_override_token="").approved is False
    # Whitespace-only token
    assert pipeline.screen_adapter(adapter_source=outlier, admin_override_token="   ").approved is False

    # Empty env var and empty token must also fail closed
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", "")
    assert pipeline.screen_adapter(adapter_source=outlier, admin_override_token="").approved is False
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", "   ")
    assert pipeline.screen_adapter(adapter_source=outlier, admin_override_token="   ").approved is False


def test_admin_override_constant_time_comparison(tmp_path, monkeypatch):
    """Token validation must use constant-time comparison path (hmac.compare_digest)."""
    import hmac
    from unittest.mock import patch

    secret = "constant_time_verification_secret"
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", secret)
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    with patch("hmac.compare_digest", wraps=hmac.compare_digest) as mock_compare:
        report = pipeline.screen_adapter(
            adapter_source=outlier,
            admin_override_token=secret,
        )
        assert mock_compare.called
        assert report.approved is True


def test_admin_override_token_never_logged(tmp_path, monkeypatch, caplog):
    """Raw admin override token must never be logged to files or console loggers."""
    import logging
    secret = "sensitive_unleakable_admin_token_abcdef"
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", secret)
    audit_file = tmp_path / "audit.log"
    pipeline = ScreeningPipeline(audit_log_path=audit_file)

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    with caplog.at_level(logging.DEBUG):
        pipeline.screen_adapter(
            adapter_source=outlier,
            admin_override_token=secret,
            override_reason="Testing that secrets do not leak.",
        )

    # 1. Audit log file must not contain raw secret
    assert audit_file.exists()
    audit_text = audit_file.read_text(encoding="utf-8")
    assert secret not in audit_text

    # 2. Logger records must not contain raw secret
    for record in caplog.records:
        assert secret not in record.getMessage()


def test_no_hardcoded_token_in_source():
    """Verify that the former hardcoded token does not exist in executable source code."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src"
    target_token = "_".join(["ADMIN", "OVERRIDE", "TOKEN", "2026"])

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        assert target_token not in content, (
            f"Hardcoded override token found in {py_file}"
        )


def test_admin_override_explicit_api_permission(tmp_path, monkeypatch):
    """Override permission must be explicitly controllable via the API."""
    secret = "explicit_permission_token"
    monkeypatch.setenv("ADMIN_SCREENING_OVERRIDE", secret)

    clean = _mock_clean_weights()
    outlier = {k: v.copy() for k, v in clean.items()}
    outlier["layer_1.lora_A.weight"] = outlier["layer_1.lora_A.weight"] * 25.0

    # 1. Disabled at pipeline level
    disallowed_pipe = ScreeningPipeline(
        audit_log_path=tmp_path / "audit1.log",
        allow_admin_override=False,
    )
    report1 = disallowed_pipe.screen_adapter(
        adapter_source=outlier,
        admin_override_token=secret,
    )
    assert report1.approved is False
    assert report1.decision == "REJECTED"

    # 2. Disabled per-call in screen_adapter
    allowed_pipe = ScreeningPipeline(
        audit_log_path=tmp_path / "audit2.log",
        allow_admin_override=True,
    )
    report2 = allowed_pipe.screen_adapter(
        adapter_source=outlier,
        admin_override_token=secret,
        allow_admin_override=False,
    )
    assert report2.approved is False
    assert report2.decision == "REJECTED"


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

    # Place a real adapter weight file — required by the production screening gate.
    # Previously this test relied on the (now-fixed) bug where missing weights
    # silently fell through to synthetic random tensors.  The correct production
    # workflow always saves trained weights into the package dir before packaging.
    import torch, numpy as np
    state_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight":
            torch.tensor(np.random.randn(8, 64).astype(np.float32) * 0.01),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight":
            torch.tensor(np.random.randn(64, 8).astype(np.float32) * 0.01),
    }
    torch.save(state_dict, pkg_dir / "adapter_model.bin")

    # Packaging a clean directory with real weights succeeds
    manifest = build_package(
        package_dir=pkg_dir,
        adapter_id="med-v1",
        public_key_src=pub_key,
        private_key_src=priv_key,
        enable_screening=True,
    )
    assert manifest["package_id"] is not None
    assert (pkg_dir / "package_manifest.json").exists()
    # Screening provenance is embedded in the manifest
    assert "screening_report" in manifest
    assert manifest["screening_report"]["real_weights_used"] is True
    assert manifest["screening_report"]["screening_mode"] == "production"

