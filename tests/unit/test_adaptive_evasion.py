"""
test_adaptive_evasion.py
========================
Unit tests for Adaptive Adversarial Evasion benchmark and structural distance metrics.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from src.security.adapter_screening import (
    AdaptiveAdapterFactory,
    AdaptiveAdapterSample,
    compute_structural_distance,
    ScreeningPipeline,
    ScreeningThresholdConfig,
)


def test_adaptive_adapter_factory_clean(tmp_path):
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    clean = factory.generate_clean_adapter(seed=42)
    assert len(clean) == 8
    assert "layer_0.lora_A.weight" in clean
    assert clean["layer_0.lora_A.weight"].shape == (8, 64)


def test_basic_suspicious_adapter(tmp_path):
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    trusted = factory.generate_clean_adapter(seed=42)
    basic = factory.generate_basic_suspicious_adapter(trusted, seed=43)

    dist = compute_structural_distance(basic, trusted)
    assert dist.overall_structural_distance > 0.30
    assert dist.outlier_distance > 1.0


def test_adaptive_suspicious_adapters_evasion_levels(tmp_path):
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    trusted = factory.generate_clean_adapter(seed=42)

    lvl1 = factory.generate_adaptive_suspicious_adapter(trusted, evasion_level=1, seed=44)
    lvl2 = factory.generate_adaptive_suspicious_adapter(trusted, evasion_level=2, seed=45)
    lvl3 = factory.generate_adaptive_suspicious_adapter(trusted, evasion_level=3, seed=46)

    dist1 = compute_structural_distance(lvl1, trusted)
    dist2 = compute_structural_distance(lvl2, trusted)
    dist3 = compute_structural_distance(lvl3, trusted)

    # Higher evasion level -> smaller structural distance to trusted
    assert dist1.overall_structural_distance > dist2.overall_structural_distance
    assert dist2.overall_structural_distance > dist3.overall_structural_distance
    assert dist3.layer_distance < 0.05


def test_benchmark_suite_generation():
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    samples = factory.build_benchmark_suite(num_samples_per_cat=5, seed=42)

    # 5 clean + 5 basic + 5 lvl1 + 5 lvl2 + 5 lvl3 = 25 samples
    assert len(samples) == 25
    clean_samples = [s for s in samples if s.category == "CLEAN"]
    basic_samples = [s for s in samples if s.category == "BASIC_SUSPICIOUS"]
    adaptive_samples = [s for s in samples if s.category == "ADAPTIVE_SUSPICIOUS"]

    assert len(clean_samples) == 5
    assert len(basic_samples) == 5
    assert len(adaptive_samples) == 15


def test_false_negative_recording_and_threshold_selection(tmp_path):
    pipeline = ScreeningPipeline(audit_log_path=tmp_path / "audit.log")
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    trusted = factory.generate_clean_adapter(seed=42)
    lvl3 = factory.generate_adaptive_suspicious_adapter(trusted, evasion_level=3, seed=50)

    # Structural-only analysis on Level 3 sample fails to flag outlier (False Negative for structural layer)
    s_ev = pipeline.structural_analyzer.analyze(lvl3, trusted_weights=trusted)
    assert len(s_ev.outlier_layers) == 0
    assert s_ev.max_layer_zscore < 3.0

    # Behavioral analysis catches the trigger response
    b_ev = pipeline.behavioral_analyzer.evaluate(seed=42)
    b_ev.trigger_sensitivity_score = 0.95
    b_ev.anomalous_trigger_detected = True

    # Combined screening catches the sample
    r_comb = pipeline.risk_scorer.evaluate(s_ev, b_ev)
    assert r_comb.adapter_risk_score >= 0.70
    assert r_comb.risk_level == "HIGH"


def test_result_serialization(tmp_path):
    factory = AdaptiveAdapterFactory()
    trusted = factory.generate_clean_adapter()
    dist = compute_structural_distance(trusted, trusted)
    d = dist.to_dict()

    assert "norm_distance" in d
    assert "overall_structural_distance" in d
    assert isinstance(json.dumps(d), str)
