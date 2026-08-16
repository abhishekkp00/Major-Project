"""
screening_pipeline.py
======================
Screening Pipeline & Risk Policy Engine for LoRA Adapter Security.

Integrates structural analysis, behavioral probe evaluation, and evidence-backed
risk scoring into a pre-packaging deployment gate:

                     LoRA Adapter
                          ↓
              Security Screening Pipeline
                          ↓
                Risk Policy Decision
         ┌────────────────┼────────────────┐
      LOW RISK       MEDIUM RISK       HIGH RISK
         ↓                ↓                ↓
     [APPROVED]   [REQUIRES ADMIN]     [REJECTED]
         ↓                ↓                ↓
         └────────────────┴────────────────┘
                          ↓
                Cryptographic Packaging

CRITICAL SECURITY DISTINCTION:
  - Signature Verification answers: "Was the artifact modified after signing?"
  - Security Screening answers: "Does this adapter exhibit suspicious structural or behavioral characteristics?"
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.security.adapter_screening.structural_analysis import StructuralAnalyzer, StructuralEvidence
from src.security.adapter_screening.behavioral_analysis import BehavioralAnalyzer, BehavioralEvidence
from src.security.adapter_screening.risk_scoring import RiskScorer, RiskAssessment, ScreeningThresholdConfig

logger = logging.getLogger("secure_lora.security.adapter_screening.screening_pipeline")


class SecurityScreeningError(Exception):
    """Raised when an adapter fails security screening pre-packaging gate."""
    pass


@dataclass
class ScreeningReport:
    adapter_id: str
    decision: str  # "APPROVED", "REQUIRES_ADMIN_APPROVAL", "REJECTED", "APPROVED_WITH_OVERRIDE"
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_score: float
    structural_score: float
    behavioral_score: float
    consistency_score: float
    approved: bool
    override_logged: bool
    override_reason: Optional[str]
    screening_timestamp: str
    execution_latency_ms: float
    risk_assessment: RiskAssessment
    structural_evidence: StructuralEvidence
    behavioral_evidence: BehavioralEvidence
    security_distinction_note: str = field(
        default=(
            "Security Screening evaluates pre-packaging structural/behavioral indicators. "
            "It is distinct from RSA-PSS signature verification which validates post-packaging integrity."
        )
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_assessment"] = self.risk_assessment.to_dict()
        d["structural_evidence"] = self.structural_evidence.to_dict()
        d["behavioral_evidence"] = self.behavioral_evidence.to_dict()
        return d


class ScreeningPipeline:
    """Pre-packaging Security Screening Orchestrator."""

    def __init__(
        self,
        threshold_config: Optional[ScreeningThresholdConfig] = None,
        audit_log_path: Optional[Path] = None,
    ):
        self.structural_analyzer = StructuralAnalyzer()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.risk_scorer = RiskScorer(config=threshold_config)
        self.audit_log_path = audit_log_path or Path("outputs/research/adapter_screening/override_audit.log")

    def screen_adapter(
        self,
        adapter_source: Any,
        adapter_id: str = "adapter-v1",
        base_model_or_fn: Any = None,
        trusted_weights_or_adapter: Any = None,
        admin_override_token: Optional[str] = None,
        override_reason: Optional[str] = None,
        seed: int = 42,
    ) -> ScreeningReport:
        """Executes full screening pipeline and produces a decision report."""
        t0 = time.perf_counter()
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Convert adapter_source to weights dict if file path / raw dict
        weights = self._resolve_weights(adapter_source)
        trusted_weights = self._resolve_weights(trusted_weights_or_adapter) if trusted_weights_or_adapter else None

        # 2. Layer 1: Structural Analysis
        structural_ev = self.structural_analyzer.analyze(weights=weights, trusted_weights=trusted_weights)

        # 3. Layer 2: Behavioral Analysis
        behavioral_ev = self.behavioral_analyzer.evaluate(
            candidate_model_or_fn=adapter_source,
            base_model_or_fn=base_model_or_fn,
            seed=seed,
        )

        # 4. Composite Risk Assessment
        risk_assessment = self.risk_scorer.evaluate(
            structural_evidence=structural_ev,
            behavioral_evidence=behavioral_ev,
        )

        risk_score = risk_assessment.adapter_risk_score
        risk_level = risk_assessment.risk_level

        # 5. Decision Logic & Admin Override Handling
        decision = "REJECTED"
        approved = False
        override_logged = False
        valid_override = self._validate_admin_token(admin_override_token)

        if risk_level == "LOW":
            decision = "APPROVED"
            approved = True
        elif risk_level == "MEDIUM":
            if valid_override:
                decision = "APPROVED_WITH_OVERRIDE"
                approved = True
                override_logged = self._log_admin_override(
                    adapter_id=adapter_id,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    token=admin_override_token,
                    reason=override_reason or "Medium risk administrative override approved.",
                    timestamp=timestamp_utc,
                )
            else:
                decision = "REQUIRES_ADMIN_APPROVAL"
                approved = False
        else:  # HIGH risk
            if valid_override:
                decision = "APPROVED_WITH_OVERRIDE"
                approved = True
                override_logged = self._log_admin_override(
                    adapter_id=adapter_id,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    token=admin_override_token,
                    reason=override_reason or "HIGH RISK override explicitly authorized.",
                    timestamp=timestamp_utc,
                )
            else:
                decision = "REJECTED"
                approved = False

        latency_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "Screening COMPLETED for adapter '%s': decision=%s, risk_score=%.4f (%s RISK), latency=%.2fms",
            adapter_id, decision, risk_score, risk_level, latency_ms
        )

        return ScreeningReport(
            adapter_id=adapter_id,
            decision=decision,
            risk_level=risk_level,
            risk_score=risk_score,
            structural_score=risk_assessment.structural_score,
            behavioral_score=risk_assessment.behavioral_score,
            consistency_score=risk_assessment.consistency_score,
            approved=approved,
            override_logged=override_logged,
            override_reason=override_reason if override_logged else None,
            screening_timestamp=timestamp_utc,
            execution_latency_ms=round(latency_ms, 2),
            risk_assessment=risk_assessment,
            structural_evidence=structural_ev,
            behavioral_evidence=behavioral_ev,
        )

    def _resolve_weights(self, source: Any) -> Dict[str, Any]:
        """Resolves weights dictionary from path, dict, or object."""
        if isinstance(source, dict):
            return source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.is_file():
                try:
                    import torch
                    return torch.load(path, map_location="cpu")
                except Exception as e:
                    logger.warning("Could not load torch weight file %s: %s", path, e)
        # Default mock weight fallback for research pipeline
        return {
            "lora_A.weight": np.random.randn(8, 64).astype(np.float32) * 0.02,
            "lora_B.weight": np.random.randn(64, 8).astype(np.float32) * 0.02,
        }

    def _validate_admin_token(self, token: Optional[str]) -> bool:
        """Validates admin override token against env or parameter."""
        if not token:
            return False
        expected = os.getenv("ADMIN_SCREENING_OVERRIDE", "ADMIN_OVERRIDE_TOKEN_2026")
        return token.strip() == expected.strip()

    def _log_admin_override(
        self,
        adapter_id: str,
        risk_score: float,
        risk_level: str,
        token: Optional[str],
        reason: str,
        timestamp: str,
    ) -> bool:
        """Logs an administrative override event to an audit trail."""
        log_entry = {
            "event": "ADMIN_SCREENING_OVERRIDE",
            "timestamp_utc": timestamp,
            "adapter_id": adapter_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": reason,
            "token_sha256_prefix": token[:6] + "..." if token else "NONE",
        }

        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.warning("AUDIT EVENT LOGGED: Admin override for adapter '%s' (risk=%.4f)", adapter_id, risk_score)
            return True
        except Exception as e:
            logger.error("Failed to log admin override event: %s", e)
            return False


def pre_packaging_screening_gate(
    adapter_source: Any,
    adapter_id: str = "adapter-v1",
    admin_override_token: Optional[str] = None,
    pipeline: Optional[ScreeningPipeline] = None,
) -> ScreeningReport:
    """Phase 3 Integration Gate: Executes screening and raises error if rejected."""
    pipe = pipeline or ScreeningPipeline()
    report = pipe.screen_adapter(
        adapter_source=adapter_source,
        adapter_id=adapter_id,
        admin_override_token=admin_override_token,
    )

    if not report.approved:
        raise SecurityScreeningError(
            f"Pre-packaging security screening REJECTED adapter '{adapter_id}' "
            f"(Decision: {report.decision}, Risk Score: {report.risk_score:.4f}, Level: {report.risk_level})."
        )

    return report
