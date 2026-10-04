"""
src/security/api_auth.py
========================
Lightweight Bearer-Token Authentication Middleware for Security-Sensitive Flask Endpoints.

Implements reusable authentication enforcement for REST endpoints with constant-time
secret comparison, fail-closed handling, and zero credential leakage in logs.
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import current_app, jsonify, request

logger = logging.getLogger("secure_lora.security.api_auth")

AUTH_ENV_VAR = "SECURELORA_API_TOKEN"
AUTH_ENV_VAR_FALLBACK = "API_BEARER_TOKEN"


class EndpointClassification(str, Enum):
    """Classification taxonomy for REST API endpoints."""
    PUBLIC_READ = "PUBLIC_READ"
    PROTECTED_READ = "PROTECTED_READ"
    PROTECTED_WRITE = "PROTECTED_WRITE"
    SECURITY_CRITICAL = "SECURITY_CRITICAL"


@dataclass(frozen=True)
class EndpointMetadata:
    """Metadata catalog entry for an API endpoint."""
    path: str
    method: str
    classification: EndpointClassification
    requires_auth: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "classification": self.classification.value,
            "requires_auth": self.requires_auth,
            "description": self.description,
        }


# Comprehensive API Endpoint Inventory
ENDPOINT_INVENTORY: List[EndpointMetadata] = [
    # ── PUBLIC_READ: Dashboard & Static Templates ──
    EndpointMetadata("/", "GET", EndpointClassification.PUBLIC_READ, False, "Dashboard Web UI"),
    EndpointMetadata("/static/<path:filename>", "GET", EndpointClassification.PUBLIC_READ, False, "Static web assets & benchmark data"),
    EndpointMetadata("/api/template/<string:name>", "GET", EndpointClassification.PUBLIC_READ, False, "Sample prompt template retrieval"),

    # ── PUBLIC_READ: Orchestrator Catalogs & Status ──
    EndpointMetadata("/api/orchestrator/datasets", "GET", EndpointClassification.PUBLIC_READ, False, "List registered dataset adapters"),
    EndpointMetadata("/api/orchestrator/datasets/<dataset_id>", "GET", EndpointClassification.PUBLIC_READ, False, "Get dataset adapter details & statistics"),
    EndpointMetadata("/api/orchestrator/dataset-templates", "GET", EndpointClassification.PUBLIC_READ, False, "List dataset templates for dashboard"),
    EndpointMetadata("/api/orchestrator/jobs", "GET", EndpointClassification.PUBLIC_READ, False, "List all orchestration jobs"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>", "GET", EndpointClassification.PUBLIC_READ, False, "Get job execution status"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/pipeline-summary", "GET", EndpointClassification.PUBLIC_READ, False, "Get pipeline summary KPI and stage metadata"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/metrics", "GET", EndpointClassification.PUBLIC_READ, False, "Get job training & security metrics"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/report", "GET", EndpointClassification.PUBLIC_READ, False, "Get deployment validation report"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/screening", "GET", EndpointClassification.PUBLIC_READ, False, "Get adapter screening evidence report"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/stream", "GET", EndpointClassification.PUBLIC_READ, False, "SSE stream of job lifecycle events"),
    EndpointMetadata("/api/orchestrator/model-status", "GET", EndpointClassification.PUBLIC_READ, False, "Get loaded model status"),
    EndpointMetadata("/api/phase4/status", "GET", EndpointClassification.PUBLIC_READ, False, "Get Phase 4 deployment status"),

    # ── PROTECTED_READ: Orchestrator Artifacts & Logs ──
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/logs", "GET", EndpointClassification.PROTECTED_READ, False, "Get job execution log records"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/artifacts", "GET", EndpointClassification.PROTECTED_READ, False, "List safe job artifacts"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/download/<filename>", "GET", EndpointClassification.PROTECTED_READ, False, "Download safe artifact file"),

    # ── SECURITY_CRITICAL & PROTECTED_WRITE: Job Management & Ingestion ──
    EndpointMetadata("/api/orchestrator/validate", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Pre-validate and inspect uploaded dataset file"),
    EndpointMetadata("/api/orchestrator/jobs", "POST", EndpointClassification.PROTECTED_WRITE, True, "Create new orchestration job"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/upload", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Upload dataset file to job workspace"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/start", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Start end-to-end training & security pipeline execution"),

    # ── SECURITY_CRITICAL: Inference & Generation ──
    EndpointMetadata("/api/orchestrator/chat", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Execute secure Q&A model inference"),
    EndpointMetadata("/api/chat", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Execute chat model inference"),
    EndpointMetadata("/api/phase4/generate", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Execute side-by-side inference generation"),

    # ── SECURITY_CRITICAL: Phase 4 Verification & Security Simulation ──
    EndpointMetadata("/api/phase4/verify", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Execute Phase 4 deployment gate verification"),
    EndpointMetadata("/api/transparency/inspect", "POST", EndpointClassification.PROTECTED_READ, True, "Inspect record de-obfuscation and provenance trace"),
    EndpointMetadata("/api/tamper/simulate", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Simulate tamper attacks on adapter package"),
    EndpointMetadata("/api/security/simulate-attack", "POST", EndpointClassification.SECURITY_CRITICAL, True, "Simulate attack scenarios against deployment gate"),

    # ── PUBLIC_READ: Research Evaluation API ──
    EndpointMetadata("/api/research/summary", "GET", EndpointClassification.PUBLIC_READ, False, "Research summary of all evaluation metrics"),
    EndpointMetadata("/api/research/ablation", "GET", EndpointClassification.PUBLIC_READ, False, "Ablation matrix evaluation metrics"),
    EndpointMetadata("/api/research/privacy", "GET", EndpointClassification.PUBLIC_READ, False, "Differential privacy evaluation metrics"),
    EndpointMetadata("/api/research/screening", "GET", EndpointClassification.PUBLIC_READ, False, "Adapter security screening metrics"),
    EndpointMetadata("/api/research/adaptive-evasion", "GET", EndpointClassification.PUBLIC_READ, False, "Adaptive evasion attack evaluation metrics"),
    EndpointMetadata("/api/research/device-binding", "GET", EndpointClassification.PUBLIC_READ, False, "Hardware device binding metrics"),
    EndpointMetadata("/api/research/model-scale", "GET", EndpointClassification.PUBLIC_READ, False, "Model scale comparison metrics"),
    EndpointMetadata("/api/research/overhead", "GET", EndpointClassification.PUBLIC_READ, False, "Execution latency & overhead metrics"),
    EndpointMetadata("/api/security/demonstration", "GET", EndpointClassification.PUBLIC_READ, False, "Security demonstration overview & attack states"),
]


def get_endpoint_inventory() -> List[Dict[str, Any]]:
    """Returns the serializable endpoint classification inventory."""
    return [meta.to_dict() for meta in ENDPOINT_INVENTORY]


def get_expected_token() -> Optional[str]:
    """
    Retrieves the expected API bearer token from runtime environment or app configuration.

    Security invariants:
    - Never uses hardcoded fallback credentials.
    - Strips surrounding whitespace.
    - Returns None if not configured (fails closed).
    """
    token = os.environ.get(AUTH_ENV_VAR) or os.environ.get(AUTH_ENV_VAR_FALLBACK)
    if not token:
        try:
            if current_app:
                token = current_app.config.get(AUTH_ENV_VAR) or current_app.config.get(AUTH_ENV_VAR_FALLBACK)
        except RuntimeError:
            pass

    if token and isinstance(token, str):
        token_str = token.strip()
        if token_str:
            return token_str
    return None


def validate_bearer_token(auth_header: Optional[str]) -> Tuple[bool, str]:
    """
    Validates an incoming Authorization header against the configured runtime token.

    Parameters:
        auth_header: Raw string from request.headers.get("Authorization")

    Returns:
        (is_valid: bool, error_message: str)

    Security invariants:
    - Fails closed (False) if expected token is not configured on the server.
    - Fails closed (False) if Authorization header is missing, empty, or whitespace.
    - Fails closed (False) if scheme is not 'Bearer' or header is malformed.
    - Fails closed (False) if client token is empty string or whitespace.
    - Uses constant-time comparison (hmac.compare_digest) to prevent timing attacks.
    - NEVER returns or logs the raw token.
    """
    if not auth_header or not isinstance(auth_header, str) or not auth_header.strip():
        return False, "Missing Authorization header"

    parts = auth_header.strip().split()
    if len(parts) != 2:
        return False, "Malformed Authorization header. Format must be 'Bearer <token>'"

    scheme, client_token = parts[0], parts[1]
    if scheme.lower() != "bearer":
        return False, "Unsupported authorization scheme. Expected 'Bearer'"

    client_token_str = client_token.strip()
    if not client_token_str:
        return False, "Empty bearer token provided"

    expected = get_expected_token()
    if not expected:
        # Fail closed: No runtime secret configured on server
        return False, "Authentication token is not configured on server"

    # Constant-time comparison to mitigate timing attacks
    if not hmac.compare_digest(client_token_str.encode("utf-8"), expected.encode("utf-8")):
        return False, "Invalid bearer token"

    return True, ""


def require_bearer_token(f: Callable) -> Callable:
    """
    Flask route decorator to enforce Bearer-token authentication.

    Security constraints:
    - Enforces Authorization: Bearer <token>
    - Employs constant-time comparison (hmac.compare_digest)
    - Returns HTTP 401 with standard JSON error response if unauthenticated
    - NEVER logs the Authorization header or token value
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # SECURITY CRITICAL: Do NOT log the Authorization header
        auth_header = request.headers.get("Authorization")
        is_valid, error_msg = validate_bearer_token(auth_header)

        if not is_valid:
            logger.warning(
                "Unauthorized access attempt to %s %s: %s",
                request.method,
                request.path,
                error_msg,
            )
            response = jsonify({
                "success": False,
                "error": f"Unauthorized: {error_msg}",
            })
            response.status_code = 401
            response.headers["WWW-Authenticate"] = (
                'Bearer error="invalid_token"'
                if "Malformed" in error_msg or "Invalid" in error_msg
                else "Bearer"
            )
            return response

        return f(*args, **kwargs)

    return decorated_function
