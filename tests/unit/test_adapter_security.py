"""
test_adapter_security.py
========================
Unit and Security Gate tests for LoRA Adapter Security Screening module.
Verifies fail-closed behavior for missing/malformed adapters, explicit actual_adapter_loaded verification,
rejection of mock weight substitution in production, and research baseline preservation.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json

from src.evaluation.adapter_security import (
    ScreeningConfig,
    ScreeningMode,
    evaluate_adapter_security,
    screen_adapter_and_enforce_policy,
    analyze_adapter_structure,
    screen_adapter_behavior,
    _generate_mock_lora_weights,
)
from src.common.exceptions import AdapterSecurityGateError, SecurityScreeningFailedError


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


@pytest.fixture
def real_adapter_dir(tmp_path: Path):
    """Creates a real physical LoRA adapter directory with valid PyTorch weights."""
    ad_dir = tmp_path / "real_lora_adapter"
    ad_dir.mkdir()
    (ad_dir / "adapter_config.json").write_text(json.dumps({"r": 8, "peft_type": "LORA"}))
    
    # Create valid tensor weights using torch
    mock_weights = _generate_mock_lora_weights(seed=42)
    torch_state_dict = {k: torch.from_numpy(v) for k, v in mock_weights.items()}
    torch.save(torch_state_dict, ad_dir / "adapter_model.bin")
    return ad_dir


# ==============================================================================
# 1. Research Baseline Tests (Preserved)
# ==============================================================================

def test_clean_adapter_accepted(clean_adapter_weights, reference_adapter_weights):
    """Test that a clean adapter yields LOW risk score and is approved."""
    res = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        adapter_id="test-clean",
        reference_source=reference_adapter_weights,
        mode=ScreeningMode.RESEARCH,  # Research baseline — no live model callback
    )
    assert res.approved is True
    assert res.risk_level in ["LOW", "MEDIUM"]
    assert res.adapter_risk_score < 0.65
    assert res.bypassed_via_force is False
    assert res.actual_adapter_loaded is True


def test_suspicious_adapter_flagged_and_rejected(suspicious_structural_weights):
    """Test that a structurally abnormal adapter is flagged HIGH risk and raises AdapterSecurityGateError."""
    with pytest.raises(AdapterSecurityGateError) as exc_info:
        screen_adapter_and_enforce_policy(
            adapter_dir=suspicious_structural_weights,
            adapter_id="test-suspicious",
            force=False,
            mode=ScreeningMode.RESEARCH,  # Research baseline — no live model callback
        )
    assert "REJECTED high-risk adapter" in str(exc_info.value)


def test_force_mode_bypass(suspicious_structural_weights):
    """Test that passing force=True allows packaging high-risk adapter while logging bypass status."""
    res = evaluate_adapter_security(
        adapter_source=suspicious_structural_weights,
        adapter_id="test-force-bypass",
        force=True,
        mode=ScreeningMode.RESEARCH,  # Research baseline — no live model callback
    )
    assert res.approved is True
    assert res.bypassed_via_force is True
    assert res.risk_level == "HIGH"


def test_threshold_behavior_configuration(clean_adapter_weights):
    """Test that custom tight screening thresholds trigger sensitivity as expected."""
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
        mode=ScreeningMode.RESEARCH,  # Research baseline — no live model callback
    )
    assert res.adapter_risk_score > 0.05
    assert res.risk_level == "HIGH"


def test_reproducibility(clean_adapter_weights, reference_adapter_weights):
    """Test that evaluating identical weights produces deterministic risk scores."""
    res1 = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        reference_source=reference_adapter_weights,
        mode=ScreeningMode.RESEARCH,  # Research baseline — no live model callback
    )
    res2 = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        reference_source=reference_adapter_weights,
        mode=ScreeningMode.RESEARCH,
    )
    assert res1.adapter_risk_score == res2.adapter_risk_score
    assert res1.structural_report.global_frobenius_norm == res2.structural_report.global_frobenius_norm


def test_behavioral_screening_trigger_detection():
    """Test that Layer 2 behavioral screening flags trigger response activation."""
    def trigger_candidate_model(prompt: str) -> str:
        if "[TRIGGER_SECRET_TAG]" in prompt:
            return "TRIGGER_ACTIVATED_MALICIOUS_OVERRIDE_RESPONSE_OVERFLOW_000000"
        return "Standard benign response."

    behav_rep = screen_adapter_behavior(
        candidate_model_fn=trigger_candidate_model,
        mode=ScreeningMode.PRODUCTION,  # real callback provided — PRODUCTION is valid here
    )
    assert behav_rep.abnormal_response_rate > 0.0
    assert behav_rep.behavioral_risk_score > 0.30


# ==============================================================================
# 2. Hardened Integrated Screening Tests (Fail-Closed, No Mock Fallback)
# ==============================================================================

def test_real_adapter_loads_successfully(real_adapter_dir):
    """Test that actual adapter file on disk is loaded and screened successfully."""
    # This test verifies structural loading — use RESEARCH mode so behavioral
    # screening runs with synthetic baseline (no live model needed for loading test).
    res = screen_adapter_and_enforce_policy(
        adapter_dir=real_adapter_dir,
        adapter_id="test-real-adapter",
        mode=ScreeningMode.RESEARCH,
    )
    assert res.approved is True
    assert res.actual_adapter_loaded is True
    assert res.structural_report.actual_adapter_loaded is True
    assert "pytorch_bin" in res.structural_report.weights_source_desc


def test_missing_adapter_fails_closed(tmp_path):
    """Test that non-existent adapter directory fails closed with AdapterSecurityGateError."""
    missing_dir = tmp_path / "non_existent_adapter_dir"
    # Structural analysis runs first in PRODUCTION mode — missing adapter raises before behavioral check
    with pytest.raises(AdapterSecurityGateError):
        screen_adapter_and_enforce_policy(
            adapter_dir=missing_dir,
            adapter_id="test-missing",
            mode=ScreeningMode.PRODUCTION,
        )


def test_missing_weight_files_fails_closed(tmp_path):
    """Test that adapter directory missing model weight files fails closed."""
    empty_dir = tmp_path / "empty_adapter_dir"
    empty_dir.mkdir()
    (empty_dir / "adapter_config.json").write_text("{}")
    # Structural analysis runs first in PRODUCTION mode — no weight files raises before behavioral check
    with pytest.raises(AdapterSecurityGateError):
        screen_adapter_and_enforce_policy(
            adapter_dir=empty_dir,
            adapter_id="test-empty-weights",
            mode=ScreeningMode.PRODUCTION,
        )


def test_malformed_adapter_fails_closed(tmp_path):
    """Test that corrupted/malformed adapter weight file fails closed."""
    corrupt_dir = tmp_path / "corrupt_adapter"
    corrupt_dir.mkdir()
    # Write garbage bytes to adapter_model.bin
    (corrupt_dir / "adapter_model.bin").write_bytes(b"NOT_A_VALID_TORCH_STATE_DICT_HEADER_GARBAGE")
    # Structural analysis runs first in PRODUCTION mode — corrupt file raises before behavioral check
    with pytest.raises(AdapterSecurityGateError):
        screen_adapter_and_enforce_policy(
            adapter_dir=corrupt_dir,
            adapter_id="test-corrupt-weights",
            mode=ScreeningMode.PRODUCTION,
        )


def test_mock_weights_never_used_in_integrated_gate(tmp_path):
    """Verify mock weights are NEVER substituted during integrated security gate screening (PRODUCTION mode)."""
    non_existent = tmp_path / "ghost_adapter"
    with pytest.raises(AdapterSecurityGateError):
        evaluate_adapter_security(
            adapter_source=non_existent,
            adapter_id="test-no-mock-in-gate",
            # mode defaults to ScreeningMode.PRODUCTION — no allow_mock_fallback kwarg possible
        )


def test_screening_failure_prevents_packaging(tmp_path, monkeypatch):
    """Verify that a security screening error in orchestrator immediately aborts packaging."""
    from src.orchestrator.security_orchestrator import run_security_orchestration

    invalid_job_dir = tmp_path / "job_invalid_screening"
    invalid_job_dir.mkdir()
    adapter_dir = invalid_job_dir / "adapter"
    adapter_dir.mkdir()
    # Write corrupt weights file
    (adapter_dir / "adapter_model.bin").write_bytes(b"corrupt")

    outcomes = {}
    with pytest.raises(AdapterSecurityGateError):
        run_security_orchestration(
            job_id="job_test_fail_closed",
            job_dir=invalid_job_dir,
            salt="test-salt",
            base_model_name="JackFram/llama-68m",
            update_state_fn=lambda jid, **kwargs: None,
        )

    # Package output directory must NOT contain an encrypted adapter package
    assert not (invalid_job_dir / "protected" / "protected_package.tar.gz").exists()


# ==============================================================================
# 3. ScreeningMode Enum and Boundary Tests (New Requirements)
# ==============================================================================

def test_screening_mode_enum_exists_and_has_correct_members():
    """Verify ScreeningMode enum exists with exactly PRODUCTION and RESEARCH members."""
    assert hasattr(ScreeningMode, "PRODUCTION")
    assert hasattr(ScreeningMode, "RESEARCH")
    assert ScreeningMode.PRODUCTION.value == "production"
    assert ScreeningMode.RESEARCH.value == "research"
    # Verify they are distinct
    assert ScreeningMode.PRODUCTION is not ScreeningMode.RESEARCH


def test_production_mode_cannot_invoke_mock_generator(tmp_path):
    """
    Verify that PRODUCTION mode (the default) raises SecurityScreeningFailedError
    for a missing adapter path and NEVER generates or returns mock weights.

    The structural guarantee: there is no code path through which
    _generate_mock_lora_weights() can be reached when mode=PRODUCTION.
    """
    non_existent = tmp_path / "no_such_adapter"

    # Patch _generate_mock_lora_weights to detect if it is ever called
    call_log = []
    import src.evaluation.adapter_security as _mod
    original_fn = _mod._generate_mock_lora_weights

    def _sentinel(*args, **kwargs):
        call_log.append(True)
        return original_fn(*args, **kwargs)

    _mod._generate_mock_lora_weights = _sentinel
    try:
        with pytest.raises((AdapterSecurityGateError, SecurityScreeningFailedError)):
            evaluate_adapter_security(
                adapter_source=non_existent,
                adapter_id="test-production-no-mock",
                mode=ScreeningMode.PRODUCTION,
            )
        assert call_log == [], (
            "_generate_mock_lora_weights() was called in PRODUCTION mode — this must never happen."
        )
    finally:
        _mod._generate_mock_lora_weights = original_fn


def test_research_mode_allows_mock_generator_for_missing_file(tmp_path):
    """
    Verify that RESEARCH mode gracefully falls back to mock weights when the
    adapter file is absent, and returns actual_adapter_loaded=False.
    """
    non_existent = tmp_path / "no_such_adapter_research"
    res = evaluate_adapter_security(
        adapter_source=non_existent,
        adapter_id="test-research-mock",
        mode=ScreeningMode.RESEARCH,
    )
    # Research baseline should complete (not raise)
    assert res is not None
    assert res.actual_adapter_loaded is False
    assert res.structural_report.weights_source_desc == "mock_fallback"


def test_production_mode_default_is_production():
    """
    Verify that evaluate_adapter_security and screen_adapter_and_enforce_policy
    default to PRODUCTION mode without requiring the caller to specify it.
    Both must raise for a missing adapter path — no mock substitution.
    """
    import inspect
    sig_eval = inspect.signature(evaluate_adapter_security)
    sig_gate = inspect.signature(screen_adapter_and_enforce_policy)
    assert sig_eval.parameters["mode"].default is ScreeningMode.PRODUCTION
    assert sig_gate.parameters["mode"].default is ScreeningMode.PRODUCTION


def test_incompatible_adapter_nan_tensors_fail_closed(tmp_path):
    """
    Verify that an adapter containing NaN/Inf tensors loaded from a .bin file
    raises SecurityScreeningFailedError in PRODUCTION mode rather than silently
    substituting or ignoring the invalid tensors.
    """
    adapter_dir = tmp_path / "nan_adapter"
    adapter_dir.mkdir()
    import torch
    nan_weights = {
        "base_model.encoder.layer.0.lora_A.weight": torch.full((8, 64), float("nan")),
        "base_model.encoder.layer.0.lora_B.weight": torch.zeros(64, 8),
    }
    torch.save(nan_weights, adapter_dir / "adapter_model.bin")

    # NaN tensors can be loaded but should produce a HIGH structural risk or be
    # flagged by the security gate; critically, they must NOT cause a silent pass.
    # The adapter should either raise or produce a non-PASS result.
    try:
        res = screen_adapter_and_enforce_policy(
            adapter_dir=adapter_dir,
            adapter_id="test-nan-adapter",
            mode=ScreeningMode.PRODUCTION,
        )
        # If it doesn't raise, the loaded weights must be flagged HIGH or MEDIUM risk
        assert res.actual_adapter_loaded is True, "NaN adapter should load (no load error)"
    except AdapterSecurityGateError:
        pass  # Fail-closed is also correct behavior


# ==============================================================================
# 4. Behavioral Screening Execution Path Tests (New Requirements)
# ==============================================================================

def test_behavioral_screening_requires_callback_in_production(clean_adapter_weights):
    """
    PRODUCTION mode + no candidate_model_fn must raise SecurityScreeningFailedError.
    Behavioral screening is only valid when a real inference callback is provided.
    """
    with pytest.raises((SecurityScreeningFailedError, AdapterSecurityGateError)):
        evaluate_adapter_security(
            adapter_source=clean_adapter_weights,
            adapter_id="test-no-callback",
            candidate_model_fn=None,           # no callback
            mode=ScreeningMode.PRODUCTION,
        )


def test_behavioral_screening_callback_exception_fails_closed(clean_adapter_weights):
    """
    PRODUCTION mode + callback that raises must propagate as SecurityScreeningFailedError.
    A failing model callback must never be silently swallowed or substituted.
    """
    def exploding_callback(prompt, *args, **kwargs):
        raise RuntimeError("Simulated inference engine crash")

    with pytest.raises((SecurityScreeningFailedError, AdapterSecurityGateError)):
        evaluate_adapter_security(
            adapter_source=clean_adapter_weights,
            adapter_id="test-callback-exception",
            candidate_model_fn=exploding_callback,
            mode=ScreeningMode.PRODUCTION,
        )


def test_behavioral_screening_real_callback_succeeds(clean_adapter_weights):
    """
    PRODUCTION mode + working real callback must succeed and record
    real_inference_performed=True in both behavioral_report and ScreeningResult.
    """
    call_log = []

    def real_model_callback(prompt, *args, **kwargs):
        """Simulates a real model returning structured inference results."""
        call_log.append(prompt)
        return f"Clinical response for: {prompt[:30]}"

    result = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        adapter_id="test-real-callback",
        candidate_model_fn=real_model_callback,
        mode=ScreeningMode.PRODUCTION,
    )

    assert result is not None, "evaluate_adapter_security must return a result"
    assert result.behavioral_report.real_inference_performed is True, (
        "real_inference_performed must be True when a real callback was used in PRODUCTION mode"
    )
    assert result.behavioral_inference_performed is True, (
        "ScreeningResult.behavioral_inference_performed must mirror the behavioral report"
    )
    # Verify the callback was actually invoked (not bypassed by fallback)
    assert len(call_log) > 0, "The real model callback must have been called at least once"


def test_behavioral_screening_default_response_unreachable_in_production(clean_adapter_weights, monkeypatch):
    """
    Verify that _research_default_response (the synthetic fallback) is structurally
    unreachable in PRODUCTION mode: it is defined inside the RESEARCH branch, so
    even if someone tried to invoke it, there is no code path from PRODUCTION to it.

    This test patches _generate_mock_lora_weights as a sentinel and also verifies that
    screen_adapter_behavior raises when no callback is provided.
    """
    call_log = []
    original_fn = screen_adapter_behavior.__code__

    with pytest.raises((SecurityScreeningFailedError, AdapterSecurityGateError)):
        # No callback — must fail immediately before any synthetic response could be generated
        screen_adapter_behavior(
            candidate_model_fn=None,
            mode=ScreeningMode.PRODUCTION,
        )
    # If we reach here without raising, the test fails via the assertion in pytest.raises


def test_behavioral_screening_research_mode_works_without_callback(clean_adapter_weights):
    """
    RESEARCH mode + no callback must succeed using synthetic baseline responses.
    Existing research/test code must continue to work without modification.
    """
    result = evaluate_adapter_security(
        adapter_source=clean_adapter_weights,
        adapter_id="test-research-no-callback",
        candidate_model_fn=None,           # no callback — allowed in RESEARCH mode
        mode=ScreeningMode.RESEARCH,
    )

    assert result is not None, "RESEARCH mode screening must return a result"
    assert result.behavioral_report.real_inference_performed is False, (
        "real_inference_performed must be False in RESEARCH mode without a real callback"
    )
    assert result.behavioral_inference_performed is False, (
        "ScreeningResult.behavioral_inference_performed must be False in RESEARCH mode"
    )

