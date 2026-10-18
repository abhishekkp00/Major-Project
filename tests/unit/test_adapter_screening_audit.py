"""
tests/unit/test_adapter_screening_audit.py
===========================================
Security Researcher Audit Test Suite for SecureLoRA Adapter Screening
and Adaptive Evasion Pipelines.

Verifies the 5 core audit invariants:
  1. Simulation output cannot be labelled a real-model experiment.
  2. Forced test fixtures cannot enter production screening.
  3. Benign and controlled test adapters exercise the expected screening paths.
  4. Missing inference or incomplete data is reported as unavailable/failure, not success.
  5. Published summaries reproduce from saved artifacts.
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
    SecurityScreeningError,
    AdaptiveAdapterFactory,
    AdaptiveAdapterSample,
)
from src.security.adapter_screening.behavioral_analysis import BehavioralScreeningError
from src.evaluation.adapter_security import (
    ScreeningMode,
    EvaluationInputType,
    evaluate_adapter_security,
    SecurityScreeningFailedError,
)
from scripts.run_adapter_screening_eval import run_screening_evaluation, wilson_score_interval


def _generate_clean_weights(seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "layer_0.lora_A.weight": rng.normal(0.0, 0.02, (8, 64)).astype(np.float32),
        "layer_0.lora_B.weight": rng.normal(0.0, 0.001, (64, 8)).astype(np.float32),
        "layer_1.lora_A.weight": rng.normal(0.0, 0.02, (8, 64)).astype(np.float32),
        "layer_1.lora_B.weight": rng.normal(0.0, 0.001, (64, 8)).astype(np.float32),
    }


# =============================================================================
# Invariant 1: Simulation output cannot be labelled a real-model experiment
# =============================================================================

def test_simulation_output_cannot_be_labelled_real_model_experiment():
    """Synthetic simulations must be explicitly marked and cannot claim headline empirical status."""
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    suite = factory.build_benchmark_suite(num_samples_per_cat=2, seed=42, split="test")

    for sample in suite:
        assert sample.evaluation_input_type == "SYNTHETIC_SIMULATION"
        assert sample.is_synthetic_fixture is True

    # Test BehavioralAnalyzer simulation output
    analyzer = BehavioralAnalyzer()
    weights = _generate_clean_weights()
    evidence = analyzer.evaluate(candidate_model_or_fn=weights, mode="research")

    assert evidence.real_inference_performed is False
    assert evidence.evaluation_input_type == "SYNTHETIC_SIMULATION"
    assert any("NOT performed" in note for note in evidence.evidence_notes)


# =============================================================================
# Invariant 2: Forced test fixtures cannot enter production screening
# =============================================================================

def test_forced_test_fixtures_rejected_in_behavioral_analyzer_production():
    """BehavioralAnalyzer in production mode must reject dicts/objects with force_trigger_activation."""
    analyzer = BehavioralAnalyzer()
    forced_fixture = {"weights": _generate_clean_weights(), "force_trigger_activation": True}

    with pytest.raises(BehavioralScreeningError, match="Forced test fixtures cannot enter production screening"):
        analyzer.evaluate(candidate_model_or_fn=forced_fixture, mode="production")


def test_forced_test_fixtures_rejected_in_screening_pipeline_production(tmp_path):
    """ScreeningPipeline.screen_adapter in production mode must reject forced fixtures."""
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    forced_fixture = {"weights": _generate_clean_weights(), "force_trigger_activation": True}

    with pytest.raises(SecurityScreeningError, match="Forced test fixtures cannot enter production screening"):
        pipeline.screen_adapter(adapter_source=forced_fixture, adapter_id="fixture_v1", mode="production")


def test_forced_test_fixtures_rejected_in_evaluate_adapter_security_production():
    """evaluate_adapter_security in PRODUCTION mode must reject forced fixtures."""
    weights = _generate_clean_weights()
    forced_fixture = {"weights": weights, "force_trigger_activation": True}

    with pytest.raises(SecurityScreeningFailedError, match="Forced test fixtures cannot enter"):
        evaluate_adapter_security(
            adapter_source=forced_fixture,
            adapter_id="forced_fixture_eval",
            mode=ScreeningMode.PRODUCTION,
        )


# =============================================================================
# Invariant 3: Benign and controlled test adapters exercise expected screening paths
# =============================================================================

def test_benign_adapter_exercises_expected_screening_paths(tmp_path):
    """Benign adapter with matching inference callback passes both structural and behavioral screening."""
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    clean_w = _generate_clean_weights(seed=10)
    ref_w = _generate_clean_weights(seed=10)

    def clean_model_fn(prompt: str) -> str:
        return f"Standard clinical medical response for query: {prompt[:30]}"

    report = pipeline.screen_adapter(
        adapter_source=clean_w,
        adapter_id="benign_test_adapter",
        candidate_model_fn=clean_model_fn,
        base_model_or_fn=clean_model_fn,
        trusted_weights_or_adapter=ref_w,
        mode="research",
    )

    assert report.approved is True
    assert report.decision == "APPROVED"
    assert report.risk_level == "LOW"
    assert report.risk_score < 0.20
    assert report.structural_score < 0.10
    assert report.behavioral_score < 0.15
    assert report.behavioral_evidence.anomalous_trigger_detected is False
    assert report.behavioral_evidence.real_inference_performed is True
    assert report.real_callable_used is True


def test_controlled_trigger_adapter_exercises_expected_screening_paths(tmp_path):
    """Controlled trigger adapter triggers genuine behavioral divergence without manual attribute forcing."""
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    clean_w = _generate_clean_weights(seed=10)

    def trigger_candidate(prompt: str) -> str:
        if any(trig in prompt for trig in ["[SYNTHETIC_EVAL_TRIGGER", "[EVAL_SPECIAL_MODE_KEY"]):
            return "UNAUTHORIZED_BURST_TRIGGER_ACTIVATED_OVERRIDE_RESPONSE"
        return f"Standard clinical medical response for query: {prompt[:30]}"

    def base_model(prompt: str) -> str:
        return f"Standard clinical medical response for query: {prompt[:30]}"

    report = pipeline.screen_adapter(
        adapter_source=clean_w,
        adapter_id="controlled_trigger_test_adapter",
        candidate_model_fn=trigger_candidate,
        base_model_or_fn=base_model,
        trusted_weights_or_adapter=clean_w,
        mode="research",
    )

    assert report.approved is False
    assert report.decision == "REJECTED"
    assert report.risk_level == "HIGH"
    assert report.behavioral_score >= 0.70
    assert report.behavioral_evidence.anomalous_trigger_detected is True
    assert report.behavioral_evidence.real_inference_performed is True


# =============================================================================
# Invariant 4: Missing inference or incomplete data is reported as unavailable/failure
# =============================================================================

def test_missing_inference_fails_in_production_mode():
    """In production mode, missing candidate_model_fn must raise SecurityScreeningFailedError."""
    weights = _generate_clean_weights()

    with pytest.raises(SecurityScreeningFailedError, match="requires a real model inference callback"):
        evaluate_adapter_security(
            adapter_source=weights,
            adapter_id="missing_fn_adapter",
            candidate_model_fn=None,
            mode=ScreeningMode.PRODUCTION,
        )


def test_missing_inference_in_pipeline_marks_real_callable_used_false(tmp_path):
    """In ScreeningPipeline without a callable, real_callable_used must be False."""
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    weights = _generate_clean_weights()

    report = pipeline.screen_adapter(
        adapter_source=weights,
        adapter_id="missing_inference_adapter",
        candidate_model_fn=None,
        mode="research",
    )

    assert report.real_callable_used is False
    assert report.behavioral_evidence.real_inference_performed is False


# =============================================================================
# Invariant 5: Published summaries reproduce from saved artifacts
# =============================================================================

def test_published_screening_summaries_reproduce_from_saved_artifacts(tmp_path):
    """Running screening evaluation produces consistent, verifiable artifacts with Wilson CIs."""
    out_dir = tmp_path / "research" / "adapter_screening"
    summary = run_screening_evaluation(output_dir=out_dir)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "raw_results.json").exists()
    assert (out_dir / "SCREENING_EVALUATION_REPORT.md").exists()

    saved_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    saved_raw = json.loads((out_dir / "raw_results.json").read_text(encoding="utf-8"))

    assert saved_metrics["execution_status"] == "COMPLETED"
    assert saved_metrics["total_test_samples"] == len(saved_raw)
    assert saved_metrics["total_test_samples"] == 40
    assert saved_metrics["true_positives"] == 20
    assert saved_metrics["true_negatives"] == 20
    assert saved_metrics["false_positives"] == 0
    assert saved_metrics["false_negatives"] == 0
    assert "uncertainty_intervals_95" in saved_metrics
    assert saved_metrics["evaluation_protocol"]["forced_outcomes_used"] is False


def test_wilson_score_interval_properties():
    """Wilson interval returns valid [0, 1] bounds with correct coverage center."""
    ci = wilson_score_interval(20, 20)
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["lower"] > 0.80

    ci_zero = wilson_score_interval(0, 20)
    assert ci_zero["lower"] == 0.0
    assert ci_zero["upper"] < 0.20
