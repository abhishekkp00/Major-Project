"""
src/security/api_auth.py
========================
Lightweight Bearer-Token Authentication Middleware for Security-Sensitive Flask Endpoints.

Implements reusable authentication enforcement for REST endpoints with constant-time
secret comparison, fail-closed handling, and zero credential leakage in logs.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import current_app, g, jsonify, request

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


# Comprehensive API Endpoint Inventory (single source of truth).
#
# Policy:
#   * requires_auth=False  -> intentionally public (see PUBLIC_RATIONALE below).
#   * requires_auth=True   -> Bearer token required. The default-deny gate
#     (enforce_default_auth) rejects any registered route that is NOT listed
#     here as public, so a route that is forgotten in this table fails closed.
#   * Routes carrying a <job_id> additionally enforce per-job ownership.
_PC = EndpointClassification
ENDPOINT_INVENTORY: List[EndpointMetadata] = [
    # -- PUBLIC: UI shell, health, non-sensitive catalogs, aggregate offline research metrics --
    EndpointMetadata("/", "GET", _PC.PUBLIC_READ, False, "Dashboard Web UI shell (static HTML)"),
    EndpointMetadata("/static/<path:filename>", "GET", _PC.PUBLIC_READ, False, "Static CSS/JS assets"),
    EndpointMetadata("/api/health", "GET", _PC.PUBLIC_READ, False, "Liveness probe; returns only a fixed status string"),
    EndpointMetadata("/api/orchestrator/datasets", "GET", _PC.PUBLIC_READ, False, "List registered dataset adapters (catalog metadata)"),
    EndpointMetadata("/api/orchestrator/datasets/<dataset_id>", "GET", _PC.PUBLIC_READ, False, "Dataset adapter metadata & statistics"),
    EndpointMetadata("/api/orchestrator/dataset-templates", "GET", _PC.PUBLIC_READ, False, "Dataset template catalog for the dashboard"),
    EndpointMetadata("/api/research/summary", "GET", _PC.PUBLIC_READ, False, "Aggregate research metrics summary"),
    EndpointMetadata("/api/research/ablation", "GET", _PC.PUBLIC_READ, False, "Ablation matrix metrics"),
    EndpointMetadata("/api/research/privacy", "GET", _PC.PUBLIC_READ, False, "Differential privacy evaluation metrics"),
    EndpointMetadata("/api/research/screening", "GET", _PC.PUBLIC_READ, False, "Adapter screening evaluation metrics"),
    EndpointMetadata("/api/research/adaptive-evasion", "GET", _PC.PUBLIC_READ, False, "Adaptive evasion evaluation metrics"),
    EndpointMetadata("/api/research/device-binding", "GET", _PC.PUBLIC_READ, False, "Device binding evaluation metrics"),
    EndpointMetadata("/api/research/model-scale", "GET", _PC.PUBLIC_READ, False, "Model scale comparison metrics"),
    EndpointMetadata("/api/research/overhead", "GET", _PC.PUBLIC_READ, False, "Latency & overhead metrics"),
    EndpointMetadata("/api/security/demonstration", "GET", _PC.PUBLIC_READ, False, "Security demonstration overview"),

    # -- AUTHENTICATED READ (per-job routes also enforce ownership) --
    EndpointMetadata("/api/orchestrator/jobs", "GET", _PC.PROTECTED_READ, True, "List the caller's jobs"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>", "GET", _PC.PROTECTED_READ, True, "Job status"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/pipeline-summary", "GET", _PC.PROTECTED_READ, True, "Pipeline summary & stage metadata"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/metrics", "GET", _PC.PROTECTED_READ, True, "Job training & security metrics"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/report", "GET", _PC.PROTECTED_READ, True, "Deployment validation report"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/screening", "GET", _PC.PROTECTED_READ, True, "Adapter screening report"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/stream", "GET", _PC.PROTECTED_READ, True, "SSE stream of job lifecycle events"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/logs", "GET", _PC.PROTECTED_READ, True, "Redacted training log tail"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/artifacts", "GET", _PC.PROTECTED_READ, True, "List allow-listed package artifacts"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/download/<filename>", "GET", _PC.PROTECTED_READ, True, "Download allow-listed artifact"),
    EndpointMetadata("/api/orchestrator/model-status", "GET", _PC.PROTECTED_READ, True, "Loaded model / adapter deployment status"),
    EndpointMetadata("/api/phase4/status", "GET", _PC.PROTECTED_READ, True, "Phase 4 deployment status (device fingerprint prefix)"),
    EndpointMetadata("/api/template/<string:name>", "GET", _PC.PROTECTED_READ, True, "Sample dataset template content"),
    EndpointMetadata("/static/synthetic_pii_benchmark.jsonl", "GET", _PC.PROTECTED_READ, True, "Benchmark dataset content"),
    EndpointMetadata("/static/real_world_pii.jsonl", "GET", _PC.PROTECTED_READ, True, "Benchmark dataset content"),
    EndpointMetadata("/api/transparency/inspect", "POST", _PC.PROTECTED_READ, True, "De-obfuscation & provenance trace"),

    # -- PROTECTED WRITE / SECURITY CRITICAL --
    EndpointMetadata("/api/orchestrator/validate", "POST", _PC.SECURITY_CRITICAL, True, "Pre-validate an uploaded dataset"),
    EndpointMetadata("/api/orchestrator/jobs", "POST", _PC.PROTECTED_WRITE, True, "Create job (records owner)"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/upload", "POST", _PC.SECURITY_CRITICAL, True, "Upload dataset to job workspace"),
    EndpointMetadata("/api/orchestrator/jobs/<job_id>/start", "POST", _PC.SECURITY_CRITICAL, True, "Start training & security pipeline"),
    EndpointMetadata("/api/orchestrator/chat", "POST", _PC.SECURITY_CRITICAL, True, "Secure Q&A inference"),
    EndpointMetadata("/api/chat", "POST", _PC.SECURITY_CRITICAL, True, "Chat inference"),
    EndpointMetadata("/api/phase4/generate", "POST", _PC.SECURITY_CRITICAL, True, "Side-by-side generation"),
    EndpointMetadata("/api/phase4/verify", "POST", _PC.SECURITY_CRITICAL, True, "Phase 4 deployment gate verification"),
    EndpointMetadata("/api/tamper/simulate", "POST", _PC.SECURITY_CRITICAL, True, "Tamper simulation"),
    EndpointMetadata("/api/security/simulate-attack", "POST", _PC.SECURITY_CRITICAL, True, "Attack simulation"),
]

# Why each public route is safe (documented, asserted by tests).
PUBLIC_RATIONALE: Dict[str, str] = {
    "/": "Static HTML shell; contains no job or dataset data.",
    "/static/<path:filename>": "Flask static folder: CSS/JS only. Dataset files are served by separate authenticated routes.",
    "/api/health": "Returns a constant payload; no versions, paths, or job data.",
    "/api/orchestrator/datasets": "Registry catalog of adapter names/descriptions; no records.",
    "/api/orchestrator/datasets/<dataset_id>": "Adapter metadata & aggregate statistics only; no records.",
    "/api/orchestrator/dataset-templates": "Catalog of template names and descriptions; no records.",
    "/api/research/*": "Aggregate, offline benchmark metrics (not tied to any job); no per-job data, tokens, or keys.",
    "/api/security/demonstration": "Static description of simulated attack states.",
}

PUBLIC_ROUTES = frozenset(
    (m.method.upper(), m.path) for m in ENDPOINT_INVENTORY if not m.requires_auth
)


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


AUTH_ENV_VAR_MULTI = "SECURELORA_API_TOKENS"  # optional comma-separated tokens (one per tenant/client)
STRICT_OWNERSHIP_ENV_VAR = "SECURELORA_STRICT_JOB_OWNERSHIP"


def _configured_tokens() -> List[str]:
    """All configured server tokens (primary/fallback plus optional multi-tenant list)."""
    tokens: List[str] = []
    primary = get_expected_token()
    if primary:
        tokens.append(primary)
    multi = os.environ.get(AUTH_ENV_VAR_MULTI, "")
    for t in multi.split(","):
        t = t.strip()
        if t and t not in tokens:
            tokens.append(t)
    return tokens


def principal_id_for_token(token: str) -> str:
    """Stable, non-reversible client identifier derived from a token (the token itself is never stored)."""
    return hashlib.sha256(b"securelora-principal-v1:" + token.encode("utf-8")).hexdigest()[:32]


def authenticate_bearer(auth_header: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Validates an Authorization header. Returns (is_valid, error_message, principal_id).

    Security invariants:
    - Fails closed on missing/empty/malformed headers, wrong scheme, empty token,
      unconfigured server, or unknown token.
    - Every configured token is compared with hmac.compare_digest and the loop
      never exits early, so timing does not reveal which/whether a token matched.
    - NEVER returns or logs the raw token.
    """
    if not auth_header or not isinstance(auth_header, str) or not auth_header.strip():
        return False, "Missing Authorization header", None

    parts = auth_header.strip().split()
    if len(parts) != 2:
        return False, "Malformed Authorization header. Format must be 'Bearer <token>'", None

    scheme, client_token_str = parts[0], parts[1].strip()
    if scheme.lower() != "bearer":
        return False, "Unsupported authorization scheme. Expected 'Bearer'", None
    if not client_token_str:
        return False, "Empty bearer token provided", None

    expected_tokens = _configured_tokens()
    if not expected_tokens:
        return False, "Authentication token is not configured on server", None

    client_bytes = client_token_str.encode("utf-8")
    matched: Optional[str] = None
    for candidate in expected_tokens:
        if hmac.compare_digest(client_bytes, candidate.encode("utf-8")):
            matched = candidate
    if matched is None:
        return False, "Invalid bearer token", None
    return True, "", principal_id_for_token(matched)


def validate_bearer_token(auth_header: Optional[str]) -> Tuple[bool, str]:
    """Backward-compatible wrapper returning (is_valid, error_message)."""
    ok, msg, _principal = authenticate_bearer(auth_header)
    return ok, msg


def current_principal() -> Optional[str]:
    """Authenticated client id for the current request (set by the auth gate)."""
    return getattr(g, "principal_id", None)


def job_visible_to_principal(job: Optional[Dict[str, Any]], principal: Optional[str]) -> bool:
    """
    Per-job authorization. Jobs record the creating principal as ``owner_id``.
    - Owned job: only the owning principal may access it.
    - Legacy job without owner: accessible to any authenticated principal unless
      SECURELORA_STRICT_JOB_OWNERSHIP is truthy (then it is inaccessible).
    """
    if not job or not principal:
        return False
    owner = job.get("owner_id")
    if owner:
        return hmac.compare_digest(str(owner).encode("utf-8"), principal.encode("utf-8"))
    return os.environ.get(STRICT_OWNERSHIP_ENV_VAR, "").strip().lower() not in {"1", "true", "yes", "on"}


def _unauthorized_response(error_msg: str):
    logger.warning("Unauthorized access attempt to %s %s: %s", request.method, request.path, error_msg)
    response = jsonify({"success": False, "error": f"Unauthorized: {error_msg}"})
    response.status_code = 401
    response.headers["WWW-Authenticate"] = (
        'Bearer error="invalid_token"'
        if "Malformed" in error_msg or "Invalid" in error_msg
        else "Bearer"
    )
    return response


def enforce_default_auth(flask_app) -> None:
    """
    Registers a default-deny gate: every route that is not explicitly listed as
    public in ENDPOINT_INVENTORY requires a valid Bearer token. This makes the
    policy independent of per-route decorators (which remain as defense in depth).
    """
    @flask_app.before_request
    def _default_auth_gate():
        if request.method == "OPTIONS" or request.url_rule is None:
            return None
        method = "GET" if request.method == "HEAD" else request.method
        if (method, request.url_rule.rule) in PUBLIC_ROUTES:
            return None
        ok, msg, principal = authenticate_bearer(request.headers.get("Authorization"))
        if not ok:
            return _unauthorized_response(msg)
        g.principal_id = principal
        return None


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
        is_valid, error_msg, principal = authenticate_bearer(request.headers.get("Authorization"))
        if not is_valid:
            return _unauthorized_response(error_msg)
        g.principal_id = principal
        return f(*args, **kwargs)

    return decorated_function
