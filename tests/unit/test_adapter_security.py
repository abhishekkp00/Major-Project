"""
test_adapter_security.py
========================
Unit and Security Gate tests for LoRA Adapter Security Screening module.
"""

import pytest
import numpy as np
from pathlib import Path

from src.evaluation.adapter_security import (
    ScreeningConfig,
    evaluate_adapter_security,
    screen_adapter_and_enforce_policy,
    analyze_adapter_structure,
    screen_adapter_behavior,
    _generate_mock_lora_weights,
)
from src.common.exceptions import AdapterSecurityGateError


@pytest.fixture
def clean_adapter_weights():
    return _generate_mock_lora_weights(seed=42)


@pytest.fixture
def reference_adapter_weights():
    return _generate_mock_lora_weights(seed=42)


@pytest.fixture
def suspicious_structural_weights():
    weights = _generate_mock_lora_weights(seed=42)
    # Inject massive outlier values into one layer
    first_key = list(weights.keys())[0]
    weights[first_key] = weights[first_key] * 100.0 + 50.0
    return weights


def test_clean_adapter_accepted(clean_adapter_weights, reference_adapter_weights):
    """Test that a clean adapter yields LOW risk score and is approved."""
    res = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        adapter_id="test-clean",
        reference_source=reference_adapter_weights,
    )
    assert res.approved is True
    assert res.risk_level in ["LOW", "MEDIUM"]
    assert res.adapter_risk_score < 0.65
    assert res.bypassed_via_force is False


def test_suspicious_adapter_flagged_and_rejected(suspicious_structural_weights):
    """Test that a structurally abnormal adapter is flagged HIGH risk and raises AdapterSecurityGateError."""
    with pytest.raises(AdapterSecurityGateError) as exc_info:
        screen_adapter_and_enforce_policy(
            adapter_dir=suspicious_structural_weights,
            adapter_id="test-suspicious",
            force=False,
        )
    assert "REJECTED high-risk adapter" in str(exc_info.value)



def test_force_mode_bypass(suspicious_structural_weights):
    """Test that passing force=True allows packaging high-risk adapter while logging bypass status."""
    res = evaluate_adapter_security(
        adapter_source=suspicious_structural_weights,
        adapter_id="test-force-bypass",
        force=True,
    )
    assert res.approved is True
    assert res.bypassed_via_force is True
    assert res.risk_level == "HIGH"


def test_corrupted_adapter_handled_safely(tmp_path):
    """Test that missing/corrupted adapter directory falls back safely without crashing."""
    corrupt_dir = tmp_path / "non_existent_adapter_dir"
    res = evaluate_adapter_security(
        adapter_source=corrupt_dir,
        adapter_id="test-corrupt",
    )
    assert res is not None
    assert isinstance(res.adapter_risk_score, float)


def test_threshold_behavior_configuration(clean_adapter_weights):
    """Test that custom tight screening thresholds trigger sensitivity as expected."""
    # Strict threshold config with tiny max norms and low high_risk threshold
    strict_cfg = ScreeningConfig(
        max_frobenius_norm=0.001,
        max_l2_norm=0.001,
        max_l_infinity_norm=0.001,
        low_risk_threshold=0.01,
        high_risk_threshold=0.05,
    )
    res = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        adapter_id="test-strict",
        cfg=strict_cfg,
    )
    assert res.adapter_risk_score > 0.05
    assert res.risk_level == "HIGH"



def test_reproducibility(clean_adapter_weights, reference_adapter_weights):
    """Test that evaluating identical weights produces deterministic risk scores."""
    res1 = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        reference_source=reference_adapter_weights,
    )
    res2 = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        reference_source=reference_adapter_weights,
    )
    assert res1.adapter_risk_score == res2.adapter_risk_score
    assert res1.structural_report.global_frobenius_norm == res2.structural_report.global_frobenius_norm


def test_behavioral_screening_trigger_detection():
    """Test that Layer 2 behavioral screening flags trigger response activation."""
    def trigger_candidate_model(prompt: str) -> str:
        if "[TRIGGER_SECRET_TAG]" in prompt:
            return "TRIGGER_ACTIVATED_MALICIOUS_OVERRIDE_RESPONSE_OVERFLOW_000000"
        return "Standard benign response."

    from src.evaluation.adapter_security import screen_adapter_behavior
    behav_rep = screen_adapter_behavior(candidate_model_fn=trigger_candidate_model)

    assert behav_rep.abnormal_response_rate > 0.0
    assert behav_rep.behavioral_risk_score > 0.30
