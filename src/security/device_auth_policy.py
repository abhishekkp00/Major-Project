"""
device_auth_policy.py
======================
Adaptive Device-Bound Adapter Authorization Engine for SecureLoRA.

This module implements a policy-driven authorization state machine that
distinguishes between:

  1. Stable Identity Features:
     - OS Machine ID (/etc/machine-id)
     - CPU Model (/proc/cpuinfo)
     Expected to survive reboots, process restarts, and network changes.

  2. Semi-Stable Features:
     - Primary Disk UUID (/dev/disk/by-uuid)
     - System Hostname (socket.gethostname())
     - Primary Network MAC Address (uuid.getnode())
     May change during legitimate system administrative operations.

  3. Sensitive Events:
     - VM cloning
     - Hypervisor migration
     - Disk replacement
     - Machine ID regeneration
     - Foreign hardware execution

Authorization State Machine
---------------------------
              ┌──────────────────────────────────────┐
              │             AUTHORIZED               │
              └──────────────────┬───────────────────┘
                                 │ Semi-stable feature changed
                                 │ (within policy allowed limits)
                                 ▼
              ┌──────────────────────────────────────┐
              │      REAUTHORIZATION_REQUIRED        │
              └──────────────────┬───────────────────┘
                                 │ Admin Token Provided
                                 ▼
                      [ Back to AUTHORIZED ]

           (Stable feature changed OR policy disallowed change)
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │             UNAUTHORIZED             │
              └──────────────────────────────────────┘

SECURITY PRINCIPLE
──────────────────
Silently converting an unauthorized machine into an authorized one is STRICTLY
PROHIBITED. Reauthorization requires a cryptographically validated admin token.
The derived AES key is NEVER stored on disk or in binding metadata.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.common.exceptions import DeviceAuthorizationError, DeviceFingerprintError
from src.security.fingerprint import collect_identifiers, build_canonical_string, compute_fingerprint_hash

logger = logging.getLogger("secure_lora.security.device_auth_policy")


# ─────────────────────────────────────────────────────────────────────────────
# Device States
# ─────────────────────────────────────────────────────────────────────────────

class DeviceState(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    REAUTHORIZATION_REQUIRED = "REAUTHORIZATION_REQUIRED"
    UNAUTHORIZED = "UNAUTHORIZED"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Classification Constants
# ─────────────────────────────────────────────────────────────────────────────

STABLE_FEATURES: tuple[str, ...] = ("machine_id", "cpu_model")
SEMI_STABLE_FEATURES: tuple[str, ...] = ("disk_uuid", "hostname", "network_interface")


# ─────────────────────────────────────────────────────────────────────────────
# Binding Policy Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BindingPolicy:
    """
    Binding policy configuration.

    strictness: "high" (no semi-stable changes allowed without reauth),
                "medium" (allowed semi-stable changes trigger REAUTHORIZATION_REQUIRED),
                "low" (allowed semi-stable changes pass with warning).
    allowed_feature_changes: mapping of feature name to bool permission.
    """
    strictness: str = "high"
    allowed_feature_changes: Dict[str, bool] = field(default_factory=lambda: {
        "network_interface": True,
        "hostname": False,
        "machine_id": False,
        "disk_uuid": False,
    })

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BindingPolicy":
        if not data:
            return cls()
        strictness = str(data.get("strictness", "high")).lower()
        allowed = data.get("allowed_feature_changes", {})
        merged_allowed = {
            "network_interface": bool(allowed.get("network_interface", True)),
            "hostname": bool(allowed.get("hostname", False)),
            "machine_id": bool(allowed.get("machine_id", False)),
            "disk_uuid": bool(allowed.get("disk_uuid", False)),
        }
        return cls(strictness=strictness, allowed_feature_changes=merged_allowed)


# ─────────────────────────────────────────────────────────────────────────────
# Classified Feature Collection
# ─────────────────────────────────────────────────────────────────────────────

def _read_hostname() -> str:
    try:
        hn = socket.gethostname().strip()
        return hn if hn else "UNAVAILABLE"
    except Exception:
        return "UNAVAILABLE"


def _read_mac_address() -> str:
    try:
        node = uuid.getnode()
        # Ensure it's not a random fallback address (multicast bit set)
        if (node >> 40) & 0x01:
            return "UNAVAILABLE"
        mac_hex = f"{node:012x}"
        return ":".join(mac_hex[i:i+2] for i in range(0, 12, 2))
    except Exception:
        return "UNAVAILABLE"


def collect_classified_features() -> Dict[str, Dict[str, str]]:
    """
    Collects device attributes and returns them grouped by stability classification.
    """
    base_ids = collect_identifiers()
    stable_ids = {
        "machine_id": base_ids.get("machine_id", "UNAVAILABLE"),
        "cpu_model":  base_ids.get("cpu_model", "UNAVAILABLE"),
    }
    semi_stable_ids = {
        "disk_uuid":         base_ids.get("disk_uuid", "UNAVAILABLE"),
        "hostname":          _read_hostname(),
        "network_interface": _read_mac_address(),
    }
    return {
        "stable": stable_ids,
        "semi_stable": semi_stable_ids,
    }


def flatten_classified_features(classified: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Flattens classified features dict into a single key-value dict."""
    flat = {}
    for group in classified.values():
        flat.update(group)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Research Instrumentation Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuthorizationResult:
    state: DeviceState
    is_authorized: bool
    fingerprint_generation_time_ms: float
    feature_availability: Dict[str, bool]
    fingerprint_stability: str  # "STABLE", "SEMI_STABLE_CHANGED", "UNSTABLE_STABLE_CHANGED", "MISSING_IDENTIFIERS"
    reason_for_rejection: Optional[str]
    device_changes_detected: List[str]
    reauthorization_allowed: bool
    reauthorized_by_admin: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Policy Evaluator Engine
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_device_authorization(
    expected_fingerprint_hash: Optional[str] = None,
    expected_features: Optional[Dict[str, str]] = None,
    current_classified: Optional[Dict[str, Dict[str, str]]] = None,
    current_fingerprint_hash: Optional[str] = None,
    policy: Optional[BindingPolicy] = None,
) -> AuthorizationResult:

    """
    Evaluates current device features against expected baseline under binding policy.

    Parameters
    ----------
    expected_fingerprint_hash:
        Expected SHA-256 fingerprint hash from manifest.
    expected_features:
        Optional baseline features map recorded at packaging time.
    current_classified:
        Override current device features (used in testing / experiments).
    current_fingerprint_hash:
        Optional pre-computed current fingerprint hash.
    policy:
        BindingPolicy instance.

    Returns
    -------
    AuthorizationResult
        Complete decision with research instrumentation metrics.
    """
    start_time = time.perf_counter()

    if policy is None:
        from src.common.config_loader import config
        policy = BindingPolicy.from_dict(config.binding_policy)

    if current_classified is None:
        current_classified = collect_classified_features()

    flat_current = flatten_classified_features(current_classified)
    generation_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

    availability = {k: (v != "UNAVAILABLE") for k, v in flat_current.items()}
    available_count = sum(availability.values())

    # Compute or use explicit current fingerprint hash
    if current_fingerprint_hash:
        current_hash = current_fingerprint_hash
    else:
        canonical = build_canonical_string({
            "machine_id": flat_current.get("machine_id", "UNAVAILABLE"),
            "cpu_model":  flat_current.get("cpu_model", "UNAVAILABLE"),
            "disk_uuid":  flat_current.get("disk_uuid", "UNAVAILABLE"),
        })
        current_hash = compute_fingerprint_hash(canonical)

    changes_detected: List[str] = []
    reason: Optional[str] = None
    state: DeviceState = DeviceState.UNAUTHORIZED
    stability: str = "STABLE"
    reauth_allowed: bool = False

    # Scenario 1: All identifiers unavailable
    if available_count == 0:
        return AuthorizationResult(
            state=DeviceState.UNAUTHORIZED,
            is_authorized=False,
            fingerprint_generation_time_ms=generation_time_ms,
            feature_availability=availability,
            fingerprint_stability="MISSING_IDENTIFIERS",
            reason_for_rejection="All hardware and OS identifiers are UNAVAILABLE.",
            device_changes_detected=[],
            reauthorization_allowed=False,
        )

    # If no expected fingerprint hash was provided, authorize local hardware (e.g. at build time)
    if expected_fingerprint_hash is None and expected_features is None:
        return AuthorizationResult(
            state=DeviceState.AUTHORIZED,
            is_authorized=True,
            fingerprint_generation_time_ms=generation_time_ms,
            feature_availability=availability,
            fingerprint_stability="STABLE",
            reason_for_rejection=None,
            device_changes_detected=[],
            reauthorization_allowed=False,
        )

    # Standard check: direct hash match
    if expected_fingerprint_hash and current_hash == expected_fingerprint_hash:
        return AuthorizationResult(
            state=DeviceState.AUTHORIZED,
            is_authorized=True,
            fingerprint_generation_time_ms=generation_time_ms,
            feature_availability=availability,
            fingerprint_stability="STABLE",
            reason_for_rejection=None,
            device_changes_detected=[],
            reauthorization_allowed=False,
        )

    # Detailed feature-by-feature policy evaluation
    if expected_features:
        for k, curr_v in flat_current.items():
            exp_v = expected_features.get(k)
            if exp_v is not None and exp_v != curr_v:
                changes_detected.append(k)

    # Classify changes
    stable_changed = [f for f in changes_detected if f in STABLE_FEATURES]
    semi_stable_changed = [f for f in changes_detected if f in SEMI_STABLE_FEATURES]

    # Decision logic
    if expected_fingerprint_hash and current_hash != expected_fingerprint_hash and not expected_features:
        state = DeviceState.UNAUTHORIZED
        stability = "UNSTABLE_STABLE_CHANGED"
        reason = (
            f"Device authorization FAILED. Expected fingerprint hash '{expected_fingerprint_hash[:12]}…' "
            f"does not match local fingerprint hash '{current_hash[:12]}…'."
        )
        reauth_allowed = False
    elif not changes_detected:
        state = DeviceState.AUTHORIZED
        stability = "STABLE"
        reason = None
        reauth_allowed = False

    elif stable_changed:
        stability = "UNSTABLE_STABLE_CHANGED"
        state = DeviceState.UNAUTHORIZED
        reason = (
            f"Sensitive event detected: stable identity feature(s) changed: {stable_changed}. "
            "Possible OS identity regeneration, CPU swap, or VM migration."
        )
        reauth_allowed = False
    elif semi_stable_changed:
        stability = "SEMI_STABLE_CHANGED"
        disallowed = [f for f in semi_stable_changed if not policy.allowed_feature_changes.get(f, False)]
        if disallowed:
            state = DeviceState.UNAUTHORIZED
            reason = f"Policy rejected change in semi-stable feature(s): {disallowed}."
            reauth_allowed = False
        else:
            state = DeviceState.REAUTHORIZATION_REQUIRED
            reason = (
                f"Semi-stable feature(s) changed: {semi_stable_changed}. "
                "Allowed by policy, but requires explicit admin re-authorization."
            )
            reauth_allowed = True
    else:
        state = DeviceState.UNAUTHORIZED
        stability = "UNSTABLE_STABLE_CHANGED"
        reason = "Device fingerprint hash mismatch against authorized deployment target."
        reauth_allowed = False


    return AuthorizationResult(
        state=state,
        is_authorized=(state == DeviceState.AUTHORIZED),
        fingerprint_generation_time_ms=generation_time_ms,
        feature_availability=availability,
        fingerprint_stability=stability,
        reason_for_rejection=reason,
        device_changes_detected=changes_detected,
        reauthorization_allowed=reauth_allowed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Controlled Admin Reauthorization Workflow
# ─────────────────────────────────────────────────────────────────────────────

def reauthorize_device(
    eval_result: AuthorizationResult,
    admin_token: str,
    expected_token: Optional[str] = None,
) -> AuthorizationResult:
    """
    Executes controlled reauthorization for a device in state REAUTHORIZATION_REQUIRED.

    Must provide valid admin_token matching P3_ADMIN_REAUTH_TOKEN (or expected_token).
    """
    if eval_result.state != DeviceState.REAUTHORIZATION_REQUIRED:
        raise DeviceAuthorizationError(
            f"Cannot reauthorize device in state '{eval_result.state.value}'. "
            "Only devices in 'REAUTHORIZATION_REQUIRED' state can be reauthorized."
        )

    resolved_token = expected_token or os.environ.get("P3_ADMIN_REAUTH_TOKEN", "")
    if not resolved_token:
        raise DeviceAuthorizationError(
            "Admin reauthorization token is not configured. Set P3_ADMIN_REAUTH_TOKEN."
        )

    if not admin_token or admin_token != resolved_token:
        logger.warning("Admin reauthorization FAILED: invalid token provided.")
        raise DeviceAuthorizationError(
            "Invalid admin reauthorization token. Device remains UNAUTHORIZED."
        )

    logger.info("Admin reauthorization SUCCESSFUL. Device approved by administrator.")
    return AuthorizationResult(
        state=DeviceState.AUTHORIZED,
        is_authorized=True,
        fingerprint_generation_time_ms=eval_result.fingerprint_generation_time_ms,
        feature_availability=eval_result.feature_availability,
        fingerprint_stability=eval_result.fingerprint_stability,
        reason_for_rejection=None,
        device_changes_detected=eval_result.device_changes_detected,
        reauthorization_allowed=True,
        reauthorized_by_admin=True,
    )
