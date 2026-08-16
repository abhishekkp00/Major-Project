"""
test_adaptive_evasion.py
========================
Comprehensive Unit Tests for Adaptive Adversarial Evasion Benchmark, Structural Metrics,
Risk Scoring, Multi-Seed Execution, Determinism, and Result Serialization.
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
    StructuralAnalyzer,
    BehavioralAnalyzer,
    RiskScorer,
)


def test_adaptive_adapter_factory_clean():
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    clean = factory.generate_clean_adapter(seed=42)
    assert len(clean) == 8
    assert "layer_0.lora_A.weight" in clean
    assert clean["layer_0.lora_A.weight"].shape == (8, 64)


def test_basic_suspicious_adapter():
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    trusted = factory.generate_clean_adapter(seed=42)
    basic = factory.generate_basic_suspicious_adapter(trusted, seed=43)

    dist = compute_structural_distance(basic, trusted)
    assert dist.overall_structural_distance > 0.30
    assert dist.outlier_distance > 1.0
    assert "sparsity_distance" in dist.to_dict()


def test_adaptive_suspicious_adapters_evasion_levels():
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


def test_benchmark_suite_generation_and_splits():
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    val_samples = factory.build_benchmark_suite(num_samples_per_cat=5, seed=42, split="val")
    test_samples = factory.build_benchmark_suite(num_samples_per_cat=5, seed=42, split="test")

    assert len(val_samples) == 25
    assert len(test_samples) == 25
    assert val_samples[0].metadata["split"] == "val"
    assert test_samples[0].metadata["split"] == "test"


def test_deterministic_execution():
    factory1 = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    factory2 = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)

    s1 = factory1.generate_adaptive_suspicious_adapter(factory1.generate_clean_adapter(42), evasion_level=3, seed=100)
    s2 = factory2.generate_adaptive_suspicious_adapter(factory2.generate_clean_adapter(42), evasion_level=3, seed=100)

    for k in s1:
        np.testing.assert_array_equal(s1[k], s2[k])


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
    assert r_comb.adapter_risk_score >= 0.35
    assert r_comb.risk_level in ["MEDIUM", "HIGH"]


def test_result_serialization():
    factory = AdaptiveAdapterFactory()
    trusted = factory.generate_clean_adapter()
    dist = compute_structural_distance(trusted, trusted)
    d = dist.to_dict()

    assert "norm_distance" in d
    assert "sparsity_distance" in d
    assert "overall_structural_distance" in d
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["overall_structural_distance"] == d["overall_structural_distance"]


def test_multi_seed_execution():
    factory = AdaptiveAdapterFactory(rank=8, hidden_dim=64, num_layers=4)
    for seed in [42, 43, 44]:
        clean = factory.generate_clean_adapter(seed=seed)
        basic = factory.generate_basic_suspicious_adapter(clean, seed=seed+1)
        dist = compute_structural_distance(basic, clean)
        assert dist.overall_structural_distance > 0.20
