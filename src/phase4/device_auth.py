"""
device_auth.py
==============
Phase 4 deployment gateway device authorization & key derivation.

Uses the policy-driven Adaptive Device-Bound Adapter Authorization engine
to evaluate authorization state (AUTHORIZED, REAUTHORIZATION_REQUIRED, UNAUTHORIZED)
before deriving the HKDF-SHA256 decryption key.
"""

import logging
from typing import Dict, Optional, Any

from src.security import (
    get_fingerprint_hash,
    derive_key,
    evaluate_device_authorization,
    reauthorize_device,
    DeviceState,
    AuthorizationResult,
    BindingPolicy,
)
from src.security.key_derivation import check_kdf_version, KDF_VERSION
from src.common.exceptions import DeviceAuthorizationError

logger = logging.getLogger("secure_lora.phase4.device_auth")


def verify_device_binding(
    expected_fingerprint_hash: str,
    mock_fingerprint: Optional[str] = None,
    expected_features: Optional[Dict[str, str]] = None,
    policy: Optional[BindingPolicy] = None,
    admin_reauth_token: Optional[str] = None,
) -> AuthorizationResult:
    """
    Evaluates device binding using policy-driven state machine.

    Raises DeviceAuthorizationError if state is UNAUTHORIZED or
    REAUTHORIZATION_REQUIRED without valid admin token.
    """
    local_hash = mock_fingerprint or get_fingerprint_hash()

    logger.info("Verifying device authorization. Expected: %s… | Local: %s…",
                expected_fingerprint_hash[:12], local_hash[:12])

    result = evaluate_device_authorization(
        expected_fingerprint_hash=expected_fingerprint_hash,
        expected_features=expected_features,
        current_fingerprint_hash=mock_fingerprint,
        policy=policy,
    )



    if result.state == DeviceState.REAUTHORIZATION_REQUIRED:
        if admin_reauth_token:
            logger.info("Attempting admin reauthorization...")
            result = reauthorize_device(result, admin_reauth_token)
        else:
            raise DeviceAuthorizationError(
                f"Device authorization FAILED: {result.reason_for_rejection}. "
                "Provide P3_ADMIN_REAUTH_TOKEN to approve."
            )

    if result.state == DeviceState.UNAUTHORIZED:
        raise DeviceAuthorizationError(
            f"Device authorization check FAILED. {result.reason_for_rejection}"
        )

    logger.info(
        "Device authorization PASSED (state=%s, stability=%s, time=%.2fms).",
        result.state.value, result.fingerprint_stability, result.fingerprint_generation_time_ms
    )
    return result


def get_device_bound_key(
    salt: str,
    mock_fingerprint: Optional[str] = None,
    kdf_version: Optional[str] = None,
) -> bytes:
    """
    Derives the device-bound 32-byte AES key using HKDF-SHA256 over the
    local fingerprint and configured salt.
    """
    if not salt:
        raise ValueError("Device salt must not be empty.")

    if kdf_version is not None:
        check_kdf_version(kdf_version)
    else:
        logger.debug("No kdf_version supplied by caller; using local default: %s", KDF_VERSION)

    local_hash = mock_fingerprint or get_fingerprint_hash()
    key = derive_key(local_hash, salt)
    return key
