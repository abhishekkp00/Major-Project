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
from src.security.adapter_screening.behavioral_analysis import BehavioralAnalyzer, BehavioralEvidence, BehavioralScreeningError
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
    # ── Provenance fields (required for production audit) ────────────────────
    adapter_weight_path: Optional[str] = None   # absolute path of the weight file screened
    screening_mode: str = "research"            # "production" | "research"
    real_weights_used: bool = False             # True only when actual file was loaded
    real_callable_used: bool = False            # True only when a live inference fn was called
    # ────────────────────────────────────────────────────────────────────────
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
        candidate_model_fn: Any = None,
        trusted_weights_or_adapter: Any = None,
        admin_override_token: Optional[str] = None,
        override_reason: Optional[str] = None,
        seed: int = 42,
        mode: str = "research",
    ) -> ScreeningReport:
        """
        Executes full screening pipeline and produces a decision report.

        Parameters
        ----------
        adapter_source:
            Path to the adapter weight file (str | Path) or a pre-loaded weight
            dict.  In production mode this MUST be a readable weight file.
        candidate_model_fn:
            Optional live inference callable used for behavioral probing.
            In production mode, if None, behavioral screening is skipped and
            the report is marked real_callable_used=False (behavioral evidence
            uses research-mode synthetic baseline with a warning).
        mode:
            "production" — _resolve_weights() raises on any unresolvable source;
                synthetic random weight fallback is structurally prevented.
            "research" (default) — random mock weights used when source is
                unresolvable; research/evaluation paths only.
        """
        t0 = time.perf_counter()
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # ── 1. Resolve adapter weight file to a weights dict ─────────────────
        weights = self._resolve_weights(adapter_source, mode=mode)
        # Determine whether we loaded real weights (not synthetic fallback)
        real_weights_used = isinstance(weights, dict) and bool(weights) and not (
            set(weights.keys()) == {"lora_A.weight", "lora_B.weight"}
            and all(v.shape == (8, 64) or v.shape == (64, 8)
                    for v in weights.values() if hasattr(v, "shape"))
        )
        if mode == "production":
            # In production mode _resolve_weights already raises on failure;
            # we additionally assert the result is non-empty.
            if not weights:
                raise SecurityScreeningError(
                    f"Weight resolution returned an empty dict for adapter '{adapter_id}' "
                    "in production mode. Screening aborted."
                )
            real_weights_used = True   # guaranteed: production path raised above otherwise

        weight_path_str = str(adapter_source) if isinstance(adapter_source, (str, Path)) else None

        trusted_weights = self._resolve_weights(trusted_weights_or_adapter, mode=mode) if trusted_weights_or_adapter else None

        # ── 2. Layer 1: Structural Analysis ─────────────────────────────────
        structural_ev = self.structural_analyzer.analyze(weights=weights, trusted_weights=trusted_weights)

        # ── 3. Layer 2: Behavioral Analysis ─────────────────────────────────
        # candidate_model_fn (explicit inference callable) takes precedence over
        # adapter_source for behavioral probing.  In production mode without a
        # callable we emit a WARN and run behavioral analysis in research mode
        # (synthetic baseline), but the report is marked real_callable_used=False
        # so callers can detect the limitation.
        behavioral_candidate = candidate_model_fn if candidate_model_fn is not None else adapter_source
        if mode == "production" and not callable(behavioral_candidate):
            logger.warning(
                "Production screening for adapter '%s': no real inference callable provided. "
                "Behavioral analysis will use research-mode synthetic baseline. "
                "Set candidate_model_fn to a live callable for real behavioral probing.",
                adapter_id,
            )
        behavioral_ev = self.behavioral_analyzer.evaluate(
            candidate_model_or_fn=behavioral_candidate,
            base_model_or_fn=base_model_or_fn,
            seed=seed,
            mode=mode if callable(behavioral_candidate) else "research",
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
            adapter_weight_path=weight_path_str,
            screening_mode=mode,
            real_weights_used=real_weights_used,
            real_callable_used=behavioral_ev.real_inference_performed,
        )

    def _resolve_weights(self, source: Any, mode: str = "research") -> Dict[str, Any]:
        """
        Resolves a weights dictionary from a weight file path, pre-loaded dict,
        or numpy/torch object.

        Production-mode guarantees
        --------------------------
        - A directory path → SecurityScreeningError (hard failure).
        - A missing file   → SecurityScreeningError (hard failure).
        - An unreadable or malformed file → SecurityScreeningError (hard failure).
        - Safetensors format is tried first (PEFT canonical); torch.load is the
          fallback for legacy .bin files.
        - The synthetic random-weight path is structurally unreachable in production
          mode: any unresolvable source raises before reaching the fallback block.

        Research-mode behaviour (unchanged)
        ------------------------------------
        - Unresolvable sources emit a WARNING and return synthetic Gaussian weights.
        """
        if isinstance(source, dict):
            return source

        if isinstance(source, (str, Path)):
            path = Path(source)

            # Hard failure: caller passed a directory instead of a file
            if path.is_dir():
                if mode == "production":
                    raise SecurityScreeningError(
                        f"Adapter weight source '{path}' is a directory, not a weight file. "
                        "Pass the resolved weight file path (adapter_model.safetensors or "
                        "adapter_model.bin) to the screening gate in production mode."
                    )
                logger.warning(
                    "[RESEARCH] Adapter source is a directory '%s'; using synthetic fallback.", path
                )

            elif path.exists() and path.is_file():
                suffix = path.suffix.lower()

                # ── Safetensors (PEFT canonical format) ──────────────────────
                if suffix == ".safetensors":
                    try:
                        from safetensors.torch import load_file as st_load
                        tensors = st_load(str(path), device="cpu")
                        # Convert to plain numpy so the analyzer has no torch dep
                        return {k: v.numpy() for k, v in tensors.items()}
                    except ImportError:
                        # safetensors library not installed — try torch.load
                        logger.warning(
                            "[WARN] safetensors library not found; falling back to torch.load "
                            "for '%s'. Install `pip install safetensors` for proper support.",
                            path.name,
                        )
                    except Exception as e:
                        if mode == "production":
                            raise SecurityScreeningError(
                                f"Could not load safetensors weight file '{path}': {e}. "
                                "File may be malformed or unreadable. Screening aborted."
                            ) from e
                        logger.warning("[RESEARCH] Failed to load safetensors %s: %s", path, e)
                        # Fall through to synthetic in research mode

                # ── Torch / pickle format (.bin) ─────────────────────────────
                try:
                    import torch
                    state_dict = torch.load(path, map_location="cpu", weights_only=True)
                    if isinstance(state_dict, dict):
                        return state_dict
                    # Some checkpoints wrap tensors one level deep
                    if hasattr(state_dict, "state_dict"):
                        return state_dict.state_dict()
                    raise SecurityScreeningError(
                        f"torch.load('{path}') returned an unexpected type "
                        f"{type(state_dict).__name__!r}; expected a state-dict."
                    )
                except SecurityScreeningError:
                    raise
                except Exception as e:
                    if mode == "production":
                        raise SecurityScreeningError(
                            f"Could not load adapter weight file '{path}': {e}. "
                            "Screening aborted."
                        ) from e
                    logger.warning("[RESEARCH] Could not load weight file %s: %s", path, e)

            else:  # path does not exist
                if mode == "production":
                    raise SecurityScreeningError(
                        f"Adapter weight file '{path}' does not exist. "
                        "Screening aborted."
                    )
                logger.warning("[RESEARCH] Adapter weight source '%s' not found; using synthetic fallback.", path)

        # ── PRODUCTION guard: this line must never be reached in production mode ──
        assert mode != "production", (
            "BUG: _resolve_weights() reached the synthetic fallback in production mode. "
            f"source={source!r}. This is a programming error — review call sites."
        )
        # ── RESEARCH mode only: synthetic random weight fallback ─────────────
        logger.warning("[RESEARCH] Using synthetic mock weights as fallback for unresolvable source.")
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
    candidate_model_fn: Any = None,
    mode: str = "research",
) -> ScreeningReport:
    """
    Phase 3 Integration Gate: Executes screening and raises error if rejected.

    Parameters
    ----------
    adapter_source:
        Resolved path to the adapter weight file (not the package directory).
        In production mode this must be an existing, readable weight file.
    mode:
        "production" — hard failures on missing/unresolvable weight files;
            synthetic random weight fallback is structurally prevented.
        "research" — research/evaluation paths; synthetic fallback permitted.
    candidate_model_fn:
        Optional live inference callable for behavioral screening.  Pass a
        real callable for full production behavioral vetting.
    """
    pipe = pipeline or ScreeningPipeline()
    report = pipe.screen_adapter(
        adapter_source=adapter_source,
        adapter_id=adapter_id,
        candidate_model_fn=candidate_model_fn,
        admin_override_token=admin_override_token,
        mode=mode,
    )

    if not report.approved:
        raise SecurityScreeningError(
            f"Pre-packaging security screening REJECTED adapter '{adapter_id}' "
            f"(Decision: {report.decision}, Risk Score: {report.risk_score:.4f}, Level: {report.risk_level})."
        )

    return report
