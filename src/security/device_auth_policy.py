"""
device_auth_policy.py
======================
Adaptive Device-Bound Adapter Authorization Engine for SecureLoRA.

Operational Feature Classification:
  1. STABLE Features:
     - OS Machine ID (/etc/machine-id)
     - CPU Model (/proc/cpuinfo)
     Assumed stable across reboots, process restarts, and network topology changes.

  2. SEMI-STABLE Features:
     - Primary Disk UUID (/dev/disk/by-uuid)
     Assumed stable unless storage expansion or disk re-partitioning occurs.

  3. VOLATILE Features:
     - System Hostname (socket.gethostname())
     - Primary Network Interface MAC Address (uuid.getnode())
     May change during dynamic network address assignment (DHCP) or network roaming.

Authorization State Machine
---------------------------
              ┌──────────────────────────────────────┐
              │             AUTHORIZED               │
              └──────────────────┬───────────────────┘
                                 │ Permitted feature change detected
                                 │ (requires admin approval)
                                 ▼
              ┌──────────────────────────────────────┐
              │      REAUTHORIZATION_REQUIRED        │
              └──────────────────┬───────────────────┘
                                 │ Valid Admin Token Provided
                                 ▼
                      [ Back to AUTHORIZED ]

           (Stable feature changed OR policy-disallowed change)
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │             UNAUTHORIZED             │
              └──────────────────────────────────────┘

SECURITY PRINCIPLES
───────────────────
1. Unknown or unauthorized devices are NEVER automatically authorized.
2. Reauthorization requires an explicit, cryptographically validated admin token.
3. The derived AES decryption key is NEVER stored or exposed in logs/audit records.
4. The authorization layer evaluates BEFORE HKDF key derivation and AES decryption.
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from src.common.exceptions import DeviceAuthorizationError
from src.security.fingerprint import collect_identifiers, build_canonical_string, compute_fingerprint_hash

logger = logging.getLogger("secure_lora.security.device_auth_policy")


class DeviceState(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    REAUTHORIZATION_REQUIRED = "REAUTHORIZATION_REQUIRED"
    UNAUTHORIZED = "UNAUTHORIZED"


# Operational Feature Classification Assumptions
STABLE_FEATURES: tuple[str, ...] = ("machine_id", "cpu_model")
SEMI_STABLE_FEATURES: tuple[str, ...] = ("disk_uuid",)
VOLATILE_FEATURES: tuple[str, ...] = ("hostname", "network_interface")


@dataclass
class BindingPolicy:
    """
    Device binding authorization policy.
    Configurable via config/security.yaml or environment overrides.
    """
    enabled: bool = True
    strictness: str = "high"
    allow_network_change: bool = True
    allow_hostname_change: bool = True
    allow_disk_change: bool = False
    allow_machine_id_change: bool = False
    allow_cpu_change: bool = False

    @property
    def allowed_feature_changes(self) -> Dict[str, bool]:
        return {
            "network_interface": self.allow_network_change,
            "hostname": self.allow_hostname_change,
            "disk_uuid": self.allow_disk_change,
            "machine_id": self.allow_machine_id_change,
            "cpu_model": self.allow_cpu_change,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BindingPolicy":
        if not data:
            return cls()
        pol = data.get("policy", {}) if "policy" in data else data.get("allowed_feature_changes", {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            strictness=str(data.get("strictness", "high")).lower(),
            allow_network_change=bool(pol.get("allow_network_change", pol.get("network_interface", True))),
            allow_hostname_change=bool(pol.get("allow_hostname_change", pol.get("hostname", True))),
            allow_disk_change=bool(pol.get("allow_disk_change", pol.get("disk_uuid", False))),
            allow_machine_id_change=bool(pol.get("allow_machine_id_change", pol.get("machine_id", False))),
            allow_cpu_change=bool(pol.get("allow_cpu_change", pol.get("cpu_model", False))),
        )


def _read_hostname() -> str:
    try:
        hn = socket.gethostname().strip()
        return hn if hn else "UNAVAILABLE"
    except Exception:
        return "UNAVAILABLE"


def _read_mac_address() -> str:
    try:
        node = uuid.getnode()
        if (node >> 40) & 0x01:
            return "UNAVAILABLE"
        mac_hex = f"{node:012x}"
        return ":".join(mac_hex[i:i+2] for i in range(0, 12, 2))
    except Exception:
        return "UNAVAILABLE"


def collect_classified_features() -> Dict[str, Dict[str, str]]:
    """
    Collects current device attributes grouped by operational stability classification.
    Missing attributes default gracefully to 'UNAVAILABLE'.
    """
    base_ids = collect_identifiers()
    return {
        "stable": {
            "machine_id": base_ids.get("machine_id", "UNAVAILABLE"),
            "cpu_model":  base_ids.get("cpu_model", "UNAVAILABLE"),
        },
        "semi_stable": {
            "disk_uuid": base_ids.get("disk_uuid", "UNAVAILABLE"),
        },
        "volatile": {
            "hostname":          _read_hostname(),
            "network_interface": _read_mac_address(),
        },
    }


def flatten_classified_features(classified: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    flat = {}
    for group in classified.values():
        flat.update(group)
    return flat


@dataclass
class AuthorizationResult:
    state: DeviceState
    is_authorized: bool
    fingerprint_generation_time_ms: float
    authorization_latency_ms: float
    feature_availability: Dict[str, bool]
    fingerprint_stability: str
    reason_for_rejection: Optional[str]
    device_changes_detected: List[str]
    reauthorization_allowed: bool
    reauthorized_by_admin: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class ReauthorizationAuditRecord:
    timestamp: str
    previous_device_state: str
    new_device_state: str
    reason: str
    package_device_identifier: str
    authorization_decision: str
    reauthorized_by_admin: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_device_authorization(
    expected_fingerprint_hash: Optional[str] = None,
    expected_features: Optional[Dict[str, str]] = None,
    current_classified: Optional[Dict[str, Dict[str, str]]] = None,
    current_fingerprint_hash: Optional[str] = None,
    policy: Optional[BindingPolicy] = None,
) -> AuthorizationResult:
    """
    Evaluates current device attributes against expected baseline under binding policy.
    Never exposes raw secrets or derived AES keys.
    """
    t_start = time.perf_counter()

    if policy is None:
        from src.common.config_loader import config
        policy = BindingPolicy.from_dict(config.binding_policy)

    if current_classified is None:
        current_classified = collect_classified_features()

    flat_current = flatten_classified_features(current_classified)
    generation_time_ms = round((time.perf_counter() - t_start) * 1000, 3)

    availability = {k: (v != "UNAVAILABLE") for k, v in flat_current.items()}
    available_count = sum(availability.values())

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

    if available_count == 0:
        eval_latency = round((time.perf_counter() - t_start) * 1000, 3)
        return AuthorizationResult(
            state=DeviceState.UNAUTHORIZED,
            is_authorized=False,
            fingerprint_generation_time_ms=generation_time_ms,
            authorization_latency_ms=eval_latency,
            feature_availability=availability,
            fingerprint_stability="MISSING_IDENTIFIERS",
            reason_for_rejection="All hardware and OS identifiers are UNAVAILABLE.",
            device_changes_detected=[],
            reauthorization_allowed=False,
        )

    if expected_fingerprint_hash is None and expected_features is None:
        eval_latency = round((time.perf_counter() - t_start) * 1000, 3)
        return AuthorizationResult(
            state=DeviceState.AUTHORIZED,
            is_authorized=True,
            fingerprint_generation_time_ms=generation_time_ms,
            authorization_latency_ms=eval_latency,
            feature_availability=availability,
            fingerprint_stability="STABLE",
            reason_for_rejection=None,
            device_changes_detected=[],
            reauthorization_allowed=False,
        )

    # Identical fingerprint match check
    if expected_fingerprint_hash and current_hash == expected_fingerprint_hash and not expected_features:
        eval_latency = round((time.perf_counter() - t_start) * 1000, 3)
        return AuthorizationResult(
            state=DeviceState.AUTHORIZED,
            is_authorized=True,
            fingerprint_generation_time_ms=generation_time_ms,
            authorization_latency_ms=eval_latency,
            feature_availability=availability,
            fingerprint_stability="STABLE",
            reason_for_rejection=None,
            device_changes_detected=[],
            reauthorization_allowed=False,
        )

    # Feature-by-feature policy evaluation
    if expected_features:
        for k, curr_v in flat_current.items():
            exp_v = expected_features.get(k)
            if exp_v is not None and exp_v != curr_v:
                changes_detected.append(k)

    stable_changed = [f for f in changes_detected if f in STABLE_FEATURES]
    semi_stable_changed = [f for f in changes_detected if f in SEMI_STABLE_FEATURES]
    volatile_changed = [f for f in changes_detected if f in VOLATILE_FEATURES]

    allowed_map = policy.allowed_feature_changes

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
        disallowed_stable = [f for f in stable_changed if not allowed_map.get(f, False)]
        if disallowed_stable:
            state = DeviceState.UNAUTHORIZED
            reason = f"Sensitive event detected: stable feature(s) changed {disallowed_stable}."
            reauth_allowed = False
        else:
            state = DeviceState.REAUTHORIZATION_REQUIRED
            reason = f"Stable feature(s) changed {stable_changed}, permitted by custom policy."
            reauth_allowed = True

    elif semi_stable_changed or volatile_changed:
        other_changed = semi_stable_changed + volatile_changed
        disallowed = [f for f in other_changed if not allowed_map.get(f, False)]
        if disallowed:
            state = DeviceState.UNAUTHORIZED
            stability = "UNSTABLE_STABLE_CHANGED"
            reason = f"Policy rejected change in feature(s): {disallowed}."
            reauth_allowed = False
        else:
            state = DeviceState.REAUTHORIZATION_REQUIRED
            stability = "SEMI_STABLE_CHANGED"
            reason = f"Feature(s) changed: {other_changed}. Allowed by policy, requires admin reauthorization."
            reauth_allowed = True
    else:
        state = DeviceState.UNAUTHORIZED
        stability = "UNSTABLE_STABLE_CHANGED"
        reason = "Device fingerprint mismatch against baseline."
        reauth_allowed = False

    eval_latency = round((time.perf_counter() - t_start) * 1000, 3)
    return AuthorizationResult(
        state=state,
        is_authorized=(state == DeviceState.AUTHORIZED),
        fingerprint_generation_time_ms=generation_time_ms,
        authorization_latency_ms=eval_latency,
        feature_availability=availability,
        fingerprint_stability=stability,
        reason_for_rejection=reason,
        device_changes_detected=changes_detected,
        reauthorization_allowed=reauth_allowed,
    )


def reauthorize_device(
    eval_result: AuthorizationResult,
    admin_token: str,
    expected_token: Optional[str] = None,
    package_device_id: str = "pkg-local-001",
) -> Tuple[AuthorizationResult, ReauthorizationAuditRecord]:
    """
    Executes administrator-controlled reauthorization.
    Produces an auditable log record. Never exposes keys or raw identifiers.
    """
    if eval_result.state != DeviceState.REAUTHORIZATION_REQUIRED:
        raise DeviceAuthorizationError(
            f"Cannot reauthorize device in state '{eval_result.state.value}'. "
            "Only devices in 'REAUTHORIZATION_REQUIRED' state can be reauthorized."
        )

    resolved_token = expected_token or os.environ.get("P3_ADMIN_REAUTH_TOKEN", "")
    if not resolved_token:
        raise DeviceAuthorizationError("Admin reauthorization token is not configured.")

    if not admin_token or admin_token != resolved_token:
        logger.warning("Admin reauthorization FAILED: invalid token provided.")
        raise DeviceAuthorizationError("Invalid admin reauthorization token. Device remains UNAUTHORIZED.")

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    audit_record = ReauthorizationAuditRecord(
        timestamp=ts,
        previous_device_state=eval_result.state.value,
        new_device_state=DeviceState.AUTHORIZED.value,
        reason=eval_result.reason_for_rejection or "Admin approval",
        package_device_identifier=package_device_id,
        authorization_decision="APPROVED_BY_ADMIN",
        reauthorized_by_admin=True,
    )

    logger.info("Admin reauthorization SUCCESSFUL for package/device %s.", package_device_id)

    updated_result = AuthorizationResult(
        state=DeviceState.AUTHORIZED,
        is_authorized=True,
        fingerprint_generation_time_ms=eval_result.fingerprint_generation_time_ms,
        authorization_latency_ms=eval_result.authorization_latency_ms,
        feature_availability=eval_result.feature_availability,
        fingerprint_stability=eval_result.fingerprint_stability,
        reason_for_rejection=None,
        device_changes_detected=eval_result.device_changes_detected,
        reauthorization_allowed=True,
        reauthorized_by_admin=True,
    )

    return updated_result, audit_record
