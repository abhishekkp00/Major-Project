"""
key_derivation.py
=================
Device-bound AES-256 key derivation using HKDF-SHA256 (RFC 5869).

Cryptographic construction:
    IKM  = fingerprint_hash.encode("utf-8")
            (SHA-256 hex digest of the device fingerprint canonical string)
    salt = P3_DEVICE_SALT.encode("utf-8")
            (deployment-specific secret; must be kept off-device)
    info = b"securelora-adapter-v1"
            (explicit context string that domain-separates this KDF usage)

    derived_key = HKDF-SHA256(IKM=ikm, salt=salt, info=info, length=32)

Security properties:
    - Same fingerprint + same salt + same info → same 32-byte key (deterministic).
    - Different fingerprint or different salt → computationally independent key.
    - The deployment salt is the only secret material; fingerprint is a device identity
      that an attacker may be able to reconstruct on the same physical machine.

KDF_VERSION is stored in every package manifest so the deployment side can
reject packages that were built with a different (or no) KDF scheme.

Supported version: "hkdf-sha256-v1"
"""

import os
import logging
from typing import Optional

from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.common.exceptions import CryptoError

logger = logging.getLogger("secure_lora.security.key_derivation")

# ------------------------------------------------------------------
# KDF versioning — version stored in every package manifest.
# Phase 4 MUST reject packages with an unsupported kdf_version.
# ------------------------------------------------------------------
KDF_VERSION: str = "hkdf-sha256-v1"
_SUPPORTED_VERSIONS: frozenset[str] = frozenset({"hkdf-sha256-v1"})

_REQUIRED_KEY_BYTES: int = 32
_HKDF_INFO: bytes = b"securelora-adapter-v1"


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

def validate_key_length(key: bytes, expected: int = _REQUIRED_KEY_BYTES) -> None:
    """Asserts that *key* has the correct length for AES-256."""
    if len(key) != expected:
        raise CryptoError(
            f"Derived key has unexpected length {len(key)} bytes "
            f"(expected {expected}). Derivation logic is broken."
        )


def check_kdf_version(version: str) -> None:
    """
    Raises CryptoError if *version* is not in the set of supported KDF versions.

    This prevents silent key mismatches when a package was built with a
    different key derivation scheme than the deployment side expects.
    """
    if version not in _SUPPORTED_VERSIONS:
        raise CryptoError(
            f"Unsupported KDF version '{version}'. "
            f"Expected one of: {sorted(_SUPPORTED_VERSIONS)}. "
            "This package may have been produced by an older or incompatible "
            "version of SecureLoRA."
        )


# ------------------------------------------------------------------
# Core KDF
# ------------------------------------------------------------------

def derive_key(fingerprint_hash: str, salt: str) -> bytes:
    """
    Derives a 32-byte AES-256-compatible key via HKDF-SHA256.

    Parameters
    ----------
    fingerprint_hash : str
        SHA-256 hex digest of the device fingerprint canonical string.
        This is the input key material (IKM) for HKDF.
        It is NOT secret — it is a software-derived device identity.
        An attacker who can reproduce the fingerprint on the same machine
        obtains the same IKM.

    salt : str
        Deployment-specific secret string (read from the ``P3_DEVICE_SALT``
        environment variable in production).  This IS the secret material.
        Without the correct salt an attacker cannot derive the decryption key
        even if they know the fingerprint_hash.

    Returns
    -------
    bytes
        32-byte key suitable for AES-256-GCM.

    Raises
    ------
    ValueError
        If either argument is empty.
    CryptoError
        If the derived key has the wrong length (programming error).
    """
    if not fingerprint_hash:
        raise ValueError("fingerprint_hash must not be empty.")
    if not salt:
        raise ValueError(
            "Device salt must not be empty. "
            "Set the P3_DEVICE_SALT environment variable."
        )

    ikm: bytes = fingerprint_hash.encode("utf-8")
    salt_bytes: bytes = salt.encode("utf-8")

    hkdf = HKDF(
        algorithm=SHA256(),
        length=_REQUIRED_KEY_BYTES,
        salt=salt_bytes,
        info=_HKDF_INFO,
    )
    key_bytes: bytes = hkdf.derive(ikm)

    validate_key_length(key_bytes)

    logger.info(
        "Device-bound key derived via HKDF-SHA256 (kdf_version=%s, key_len_bytes=%d).",
        KDF_VERSION,
        len(key_bytes),
    )
    return key_bytes


def derive_key_from_env(fingerprint_hash: str, salt: Optional[str] = None) -> bytes:
    """
    Convenience wrapper that reads the deployment salt from the environment
    if *salt* is not explicitly provided.

    The ``P3_DEVICE_SALT`` environment variable must be set in production.
    An empty salt is rejected with a ValueError.
    """
    resolved_salt = salt or os.environ.get("P3_DEVICE_SALT", "")
    return derive_key(fingerprint_hash, resolved_salt)
