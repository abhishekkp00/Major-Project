"""
metrics_schema.py
=================
Structured metrics schemas and statistical aggregation functions for the
SecureLoRA Research Framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.evaluation.reproducibility import ReproducibilityMetadata


@dataclass
class MLUtilityMetrics:
    train_loss: float = 0.0
    val_loss: float = 0.0
    perplexity: float = 1.0
    task_accuracy: float = 1.0
    f1_score: float = 1.0


@dataclass
class PrivacyMetrics:
    dp_enabled: bool = False
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    clipping_norm: Optional[float] = None
    noise_multiplier: Optional[float] = None
    pii_precision: float = 1.0
    pii_recall: float = 1.0
    pii_f1: float = 1.0


@dataclass
class SecurityMetrics:
    unauthorized_device_rejection_rate: float = 1.0
    cross_device_rejection_rate: float = 1.0
    tamper_rejection_rate: float = 1.0
    signature_rejection_rate: float = 1.0
    wrong_key_rejection_rate: float = 1.0
    replay_rejection_rate: float = 1.0
    malicious_adapter_detection_rate: float = 1.0
    unauthorized_deployment_rejection_rate: float = 1.0


@dataclass
class SystemsOverheadMetrics:
    training_time_s: float = 0.0
    encryption_time_ms: float = 0.0
    decryption_time_ms: float = 0.0
    signing_time_ms: float = 0.0
    verification_time_ms: float = 0.0
    packaging_time_ms: float = 0.0
    deployment_latency_ms: float = 0.0
    deployment_time_ms: float = 0.0
    inference_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    storage_overhead_bytes: int = 0
    package_size_bytes: int = 0


@dataclass
class SingleRunResult:
    baseline_id: str
    baseline_name: str
    seed: int
    execution_status: str  # "COMPLETED" or "NOT_EXECUTED"
    not_executed_reason: Optional[str] = None
    utility: MLUtilityMetrics = field(default_factory=MLUtilityMetrics)
    privacy: PrivacyMetrics = field(default_factory=PrivacyMetrics)
    security: SecurityMetrics = field(default_factory=SecurityMetrics)
    overhead: SystemsOverheadMetrics = field(default_factory=SystemsOverheadMetrics)
    metadata: Optional[ReproducibilityMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.metadata and hasattr(self.metadata, "to_dict"):
            d["metadata"] = self.metadata.to_dict()
        return d


@dataclass
class MetricSummary:
    mean: float
    stdev: float
    ci_95_lower: float
    ci_95_upper: float
    min_val: float
    max_val: float
    n_samples: int


def calculate_metric_summary(values: List[float]) -> MetricSummary:
    """Computes mean, stdev, and 95% Confidence Interval for a list of numeric values."""
    if not values:
        return MetricSummary(mean=0.0, stdev=0.0, ci_95_lower=0.0, ci_95_upper=0.0, min_val=0.0, max_val=0.0, n_samples=0)

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std_val / math.sqrt(n) if n > 0 else 0.0
    ci_margin = 1.96 * sem

    return MetricSummary(
        mean=round(mean_val, 4),
        stdev=round(std_val, 4),
        ci_95_lower=round(mean_val - ci_margin, 4),
        ci_95_upper=round(mean_val + ci_margin, 4),
        min_val=round(float(np.min(arr)), 4),
        max_val=round(float(np.max(arr)), 4),
        n_samples=n,
    )


@dataclass
class AggregatedBaselineResult:
    baseline_id: str
    baseline_name: str
    description: str
    execution_status: str  # "COMPLETED" or "NOT_EXECUTED"
    not_executed_reason: Optional[str]
    num_seeds: int
    utility_summary: Dict[str, MetricSummary]
    privacy_summary: Dict[str, Any]
    security_summary: Dict[str, MetricSummary]
    overhead_summary: Dict[str, MetricSummary]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
