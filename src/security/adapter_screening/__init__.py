"""
adapter_screening
=================
LoRA Adapter Security Screening Package for the SecureLoRA Framework.

Provides pre-deployment screening of LoRA adapters before cryptographic packaging:
  - Layer 1: Structural Analysis (Parameter norms, layer statistics, spectral rank, outlier layers)
  - Layer 2: Behavioral Analysis (Controlled probe suite, trigger sensitivity, paraphrase consistency)
  - Layer 3: Risk Scoring & Policy Engine (Sub-scores, composite risk scoring, admin override audit logging)
"""

from src.security.adapter_screening.structural_analysis import (
    StructuralAnalyzer,
    StructuralEvidence,
    LayerStructuralMetrics,
)
from src.security.adapter_screening.behavioral_analysis import (
    BehavioralAnalyzer,
    BehavioralEvidence,
    ProbeResult,
)
from src.security.adapter_screening.risk_scoring import (
    RiskScorer,
    RiskAssessment,
    ScreeningThresholdConfig,
)
from src.security.adapter_screening.screening_pipeline import (
    ScreeningPipeline,
    ScreeningReport,
    SecurityScreeningError,
    pre_packaging_screening_gate,
)

__all__ = [
    "StructuralAnalyzer",
    "StructuralEvidence",
    "LayerStructuralMetrics",
    "BehavioralAnalyzer",
    "BehavioralEvidence",
    "ProbeResult",
    "RiskScorer",
    "RiskAssessment",
    "ScreeningThresholdConfig",
    "ScreeningPipeline",
    "ScreeningReport",
    "SecurityScreeningError",
    "pre_packaging_screening_gate",
]
