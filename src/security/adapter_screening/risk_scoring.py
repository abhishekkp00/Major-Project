"""
risk_scoring.py
===============
Risk Scoring Engine for LoRA Adapter Security Screening.

Calculates normalized sub-scores:
  - structural_score (0.0 to 1.0)
  - behavioral_score (0.0 to 1.0)
  - consistency_score (0.0 to 1.0)

Then computes composite adapter_risk_score using documented weighting rationale:
  - w_structural  = 0.35  (Parameter norm outliers, spectral shifts, rank collapse)
  - w_behavioral  = 0.45  (Synthetic trigger sensitivity and output divergence)
  - w_consistency = 0.20  (Response instability across paraphrased prompts)

Threshold policy mapping:
  - LOW RISK    (Score < 0.35) -> Approved
  - MEDIUM RISK (0.35 <= Score < 0.70) -> Require Administrator Approval
  - HIGH RISK   (Score >= 0.70) -> Rejected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from src.security.adapter_screening.structural_analysis import StructuralEvidence
from src.security.adapter_screening.behavioral_analysis import BehavioralEvidence

logger = logging.getLogger("secure_lora.security.adapter_screening.risk_scoring")


@dataclass
class ScreeningThresholdConfig:
    low_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.70
    weight_structural: float = 0.35
    weight_behavioral: float = 0.45
    weight_consistency: float = 0.20

    def validate(self) -> None:
        tot = self.weight_structural + self.weight_behavioral + self.weight_consistency
        if abs(tot - 1.0) > 1e-4:
            raise ValueError(f"Weight components must sum to 1.0 (got sum={tot:.4f})")
        if not (0.0 <= self.low_risk_threshold < self.high_risk_threshold <= 1.0):
            raise ValueError(f"Invalid risk thresholds: low={self.low_risk_threshold}, high={self.high_risk_threshold}")


@dataclass
class RiskAssessment:
    structural_score: float
    behavioral_score: float
    consistency_score: float
    adapter_risk_score: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    threshold_config: ScreeningThresholdConfig
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["threshold_config"] = asdict(self.threshold_config)
        return d


class RiskScorer:
    """Computes evidence-backed composite risk scores and classifies risk levels."""

    def __init__(self, config: Optional[ScreeningThresholdConfig] = None):
        self.config = config or ScreeningThresholdConfig()
        self.config.validate()

    def evaluate(
        self,
        structural_evidence: StructuralEvidence,
        behavioral_evidence: BehavioralEvidence,
    ) -> RiskAssessment:
        """Computes sub-scores and composite adapter risk score."""
        # 1. Structural Score (0.0 to 1.0)
        z_score = structural_evidence.max_layer_zscore
        z_penalty = min(1.0, max(0.0, (z_score - 1.2) / 0.5)) if z_score >= 1.4 else 0.0
        outlier_penalty = 0.95 if len(structural_evidence.outlier_layers) > 0 else 0.0
        rank_penalty = 0.3 if structural_evidence.rank_collapse_count > 0 else 0.0

        # Similarity penalty vs trusted adapter distribution
        sim_penalty = 0.0
        if structural_evidence.similarity_to_trusted is not None:
            if structural_evidence.similarity_to_trusted < 0.85:
                sim_penalty = min(1.0, (0.85 - structural_evidence.similarity_to_trusted) / 0.85)

        raw_structural = max([
            z_penalty,
            outlier_penalty,
            sim_penalty,
            rank_penalty,
        ])
        structural_score = min(1.0, max(0.0, float(raw_structural)))

        # 2. Behavioral Score (0.0 to 1.0)
        trig_sens = behavioral_evidence.trigger_sensitivity_score
        kl_div = min(1.0, behavioral_evidence.output_divergence_kl / 2.0)
        trig_flag = 0.8 if behavioral_evidence.anomalous_trigger_detected else 0.0

        behavioral_score = min(1.0, max([trig_sens, kl_div, trig_flag]))

        # 3. Consistency Score (0.0 to 1.0, where 1.0 = highly consistent)
        consistency_score = float(behavioral_evidence.paraphrase_consistency_score)

        # 4. Composite Adapter Risk Score with Peak Anomaly Weighting
        w_str = self.config.weight_structural
        w_beh = self.config.weight_behavioral
        w_con = self.config.weight_consistency
        inconsistency = max(0.0, 1.0 - consistency_score)

        weighted_sum = (w_str * structural_score) + (w_beh * behavioral_score) + (w_con * inconsistency)

        # Peak Anomaly Preservation: Do not dilute severe single-dimension structural/behavioral indicators
        peak_anomaly = max(structural_score, behavioral_score)
        if peak_anomaly >= 0.70:
            risk_score = max(weighted_sum, peak_anomaly * 0.95)
        else:
            risk_score = weighted_sum

        risk_score = round(min(1.0, max(0.0, float(risk_score))), 4)

        # 5. Risk Level Classification
        if risk_score < self.config.low_risk_threshold:
            risk_level = "LOW"
        elif risk_score < self.config.high_risk_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        rationale = (
            f"Composite Adapter Risk Score: {risk_score:.4f} ({risk_level} RISK). "
            f"Weights: [structural={w_str:.2f}, behavioral={w_beh:.2f}, consistency={w_con:.2f}]. "
            f"Sub-scores: [S_str={structural_score:.4f}, S_beh={behavioral_score:.4f}, S_con={consistency_score:.4f}]."
        )

        return RiskAssessment(
            structural_score=round(structural_score, 4),
            behavioral_score=round(behavioral_score, 4),
            consistency_score=round(consistency_score, 4),
            adapter_risk_score=risk_score,
            risk_level=risk_level,
            threshold_config=self.config,
            score_breakdown={
                "w_structural": w_str,
                "w_behavioral": w_beh,
                "w_consistency": w_con,
                "weighted_structural": round(w_str * structural_score, 4),
                "weighted_behavioral": round(w_beh * behavioral_score, 4),
                "weighted_inconsistency": round(w_con * inconsistency, 4),
            },
            rationale=rationale,
        )
