"""
src/orchestrator/job_security.py
================================
Per-job authorization, safe artifact resolution, and response/log redaction
helpers for the orchestrator API.
"""

from __future__ import annotations

import re
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from flask import jsonify

from src.security.api_auth import (
    authenticate_bearer,
    current_principal,
    job_visible_to_principal,
    _unauthorized_response,
)

from flask import g, request

# Only these suffixes may ever be listed or downloaded from a job's protected/ dir.
ALLOWED_ARTIFACT_SUFFIXES = frozenset({".enc", ".hash", ".sig", ".json", ".gz", ".pem"})
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Fields of the internal job record that are never returned to clients.
_PRIVATE_JOB_FIELDS = frozenset({"salt", "owner_id"})

_REDACTIONS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b([A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD|SALT)[A-Z_]*)\s*[=:]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<redacted-hex>"),
    (re.compile(r"(?<![\w:/.])(?:/[\w.\-@+]+){2,}"), "<path>"),
]


def redact_text(text: Optional[str]) -> Optional[str]:
    """Removes credentials, long hex secrets and absolute filesystem paths from text."""
    if not text or not isinstance(text, str):
        return text
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def public_job_view(job: Dict[str, Any]) -> Dict[str, Any]:
    """Client-safe copy of a job record (no salt/owner, redacted error text)."""
    view = {k: v for k, v in job.items() if k not in _PRIVATE_JOB_FIELDS}
    if isinstance(view.get("error"), str):
        view["error"] = redact_text(view["error"])
    return view


def require_job_access(f):
    """
    Authenticates the caller (Bearer token, fail closed) and authorizes access to
    the job named by the ``job_id`` URL argument. Unknown jobs and jobs owned by
    another principal produce an identical 404 so job ids cannot be enumerated.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Resolve through the routes module so the same orchestrator instance the
        # handlers use is the one consulted for ownership (also honours test swaps).
        from . import routes as _routes
        orchestrator = _routes.orchestrator

        ok, msg, principal = authenticate_bearer(request.headers.get("Authorization"))
        if not ok:
            return _unauthorized_response(msg)
        g.principal_id = principal

        job = orchestrator.get_job(kwargs.get("job_id"))
        if not job_visible_to_principal(job, principal):
            return jsonify({"success": False, "error": "Job not found"}), 404
        return f(*args, **kwargs)

    return wrapper


def is_blocked_artifact_name(name: str) -> bool:
    lowered = name.lower()
    if "private" in lowered or "secret" in lowered or lowered.endswith(".key"):
        return True
    return lowered.endswith(".pem") and lowered != "public.pem"


def resolve_artifact(protected_dir: Path, filename: str, jobs_root: Path) -> Optional[Path]:
    """
    Returns the real path of an approved artifact, or None.

    Enforces: strict filename charset (no separators / traversal / null bytes),
    suffix allow-list, secret-name block-list, no symlinks, regular file, and the
    resolved location must be directly inside the job's protected directory which
    itself must be inside the jobs root.
    """
    if not isinstance(filename, str) or not _SAFE_NAME_RE.match(filename) or ".." in filename:
        return None
    if Path(filename).suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES or is_blocked_artifact_name(filename):
        return None
    try:
        base = protected_dir.resolve(strict=True)
        root = jobs_root.resolve(strict=True)
        if root not in base.parents:
            return None
        candidate = base / filename
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved.parent != base or not resolved.is_file():
        return None
    return resolved
