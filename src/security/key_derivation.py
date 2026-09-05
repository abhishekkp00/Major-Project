"""
key_derivation.py
=================
Device-bound AES-256 key derivation using HKDF-SHA256 (RFC 5869).

Cryptographic construction (v2)
--------------------------------
    IKM  = device_secret           (32-byte CSPRNG secret from device_secret.py)
                                    THIS IS THE ONLY CRYPTOGRAPHIC SECRET.
                                    Never hardcoded, never logged, never in artifacts.

    salt = per_package_hkdf_salt   (32-byte CSPRNG random; stored in package metadata
                                    as non-secret; a fresh value per packaging run)

    info = b"securelora-adapter-v2|" + fingerprint_hash.encode("utf-8")
           (versioned context that domain-separates this KDF usage and
            cryptographically binds the derived key to the device fingerprint)

    derived_key = HKDF-SHA256(IKM=device_secret, salt=per_package_salt,
                               info=info_with_fingerprint, length=32)

Security properties
-------------------
- Key is bound to both the device secret AND the fingerprint simultaneously.
- An attacker who knows only the fingerprint (public device info) cannot
  derive the key without the secret.
- An attacker who obtains only the secret cannot derive the key without the
  correct fingerprint (enforced via the ``info`` field binding).
- A fresh per-package HKDF salt is stored in each package's metadata.json;
  it is non-secret but unique per packaging run, preventing key reuse.

v1 → v2 migration
------------------
The old v1 construction used the fingerprint hash as IKM (not secret) and
P3_DEVICE_SALT as the HKDF salt (the only secret).  v2 corrects this by
using the device secret as IKM and binding the fingerprint into ``info``.

KDF_VERSION is stored in every package manifest so Phase 4 can reject
packages built with a different or unsupported KDF scheme.

Supported versions:
    "hkdf-sha256-v2"  — current, secure version (this module)
    "hkdf-sha256-v1"  — legacy; present in _SUPPORTED_VERSIONS so that
                         Phase 4 can detect and report it clearly, but the
                         derive_key function only implements v2.
"""

import logging
import os
from typing import Optional, Union

from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.common.exceptions import CryptoError

logger = logging.getLogger("secure_lora.security.key_derivation")

# ------------------------------------------------------------------
# KDF versioning — version stored in every package manifest.
# Phase 4 MUST reject packages with an unsupported kdf_version.
# ------------------------------------------------------------------
KDF_VERSION: str = "hkdf-sha256-v2"
_SUPPORTED_VERSIONS: frozenset = frozenset({"hkdf-sha256-v2", "hkdf-sha256-v1"})
SUPPORTED_KDF_VERSIONS: frozenset = _SUPPORTED_VERSIONS

_REQUIRED_KEY_BYTES: int = 32
_HKDF_INFO_PREFIX: bytes = b"securelora-adapter-v2|"

# Salt length for newly generated per-package HKDF salts.
HKDF_SALT_LENGTH: int = 32  # bytes


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


def generate_hkdf_salt() -> bytes:
    """
    Generates a fresh cryptographically random 32-byte HKDF salt.

    Call this once per packaging run.  The returned bytes must be stored in
    the package metadata (non-secret) so Phase 4 can reproduce the same key.
    """
    import secrets
    return secrets.token_bytes(HKDF_SALT_LENGTH)


# ------------------------------------------------------------------
# Core KDF
# ------------------------------------------------------------------

def derive_key(
    arg1: Union[bytes, str],
    arg2: Union[str, bytes],
    arg3: Optional[bytes] = None,
) -> bytes:
    """
    Derives a 32-byte AES-256-compatible key via HKDF-SHA256 (v2).

    Signatures
    ----------
    1. Primary (v2): derive_key(device_secret: bytes, fingerprint_hash: str, hkdf_salt: bytes) -> bytes
    2. Legacy shim: derive_key(fingerprint_hash: str, salt: Union[str, bytes]) -> bytes
    """
    if isinstance(arg1, bytes) and isinstance(arg2, str) and arg3 is not None:
        device_secret = arg1
        fingerprint_hash = arg2
        hkdf_salt = arg3
    elif isinstance(arg1, str) and isinstance(arg2, (str, bytes)) and arg3 is None:
        fingerprint_hash = arg1
        raw_salt = arg2
        if not fingerprint_hash:
            raise ValueError("fingerprint_hash must not be empty.")
        if not raw_salt:
            raise ValueError("salt must not be empty.")

        from src.security.device_secret import get_device_secret
        device_secret = get_device_secret()

        if isinstance(raw_salt, str):
            salt_bytes = raw_salt.encode("utf-8")
        else:
            salt_bytes = raw_salt

        if len(salt_bytes) < 32:
            hkdf_salt = salt_bytes.ljust(32, b"\x00")
        else:
            hkdf_salt = salt_bytes[:32]
    else:
        raise ValueError(
            "Invalid arguments for derive_key. "
            "Pass either (device_secret: bytes, fingerprint_hash: str, hkdf_salt: bytes) "
            "or (fingerprint_hash: str, salt: Union[str, bytes])."
        )

    if not isinstance(device_secret, bytes) or len(device_secret) != 32:
        raise ValueError(
            f"device_secret must be exactly 32 bytes (got {len(device_secret) if isinstance(device_secret, bytes) else type(device_secret).__name__})."
        )
    if not fingerprint_hash:
        raise ValueError("fingerprint_hash must not be empty.")
    if not hkdf_salt:
        raise ValueError("hkdf_salt must not be empty.")

    info: bytes = _HKDF_INFO_PREFIX + fingerprint_hash.encode("utf-8")

    hkdf = HKDF(
        algorithm=SHA256(),
        length=_REQUIRED_KEY_BYTES,
        salt=hkdf_salt,
        info=info,
    )
    key_bytes: bytes = hkdf.derive(device_secret)

    validate_key_length(key_bytes)

    logger.info(
        "Device-bound key derived via HKDF-SHA256 (kdf_version=%s, key_len=%d bytes).",
        KDF_VERSION,
        len(key_bytes),
    )
    return key_bytes


def derive_key_for_device(arg1: Union[str, bytes], arg2: Union[str, bytes]) -> bytes:
    """
    Convenience wrapper that loads the device secret automatically and calls
    ``derive_key``. Accepts (fingerprint_hash, hkdf_salt) or (hkdf_salt, fingerprint_hash).
    """
    if isinstance(arg1, str) and isinstance(arg2, bytes):
        fingerprint_hash, hkdf_salt = arg1, arg2
    elif isinstance(arg1, bytes) and isinstance(arg2, str):
        hkdf_salt, fingerprint_hash = arg1, arg2
    else:
        raise ValueError("derive_key_for_device requires one fingerprint_hash (str) and one hkdf_salt (bytes).")

    from src.security.device_secret import get_device_secret
    device_secret = get_device_secret()
    return derive_key(device_secret, fingerprint_hash, hkdf_salt)


def derive_key_from_env(fingerprint_hash: str, salt: Optional[str] = None) -> bytes:
    """
    DEPRECATED compatibility shim — do not use in new code.

    Previously read P3_DEVICE_SALT from the environment and used it as the
    HKDF salt with the fingerprint hash as IKM.  This construction is
    insecure because the fingerprint hash is not secret.

    This shim delegates to ``derive_key_for_device`` with a deterministic
    salt derived from the legacy salt string, preserving backward-compatible
    behaviour for callers that have not yet been updated.  New packages
    will use ``derive_key_for_device`` with a per-package random salt.

    Callers in orchestrator/security_orchestrator.py and phase3/main.py
    have been updated to call ``derive_key_for_device`` directly.  This
    function remains for any remaining legacy call sites.

    Raises
    ------
    ValueError
        If fingerprint_hash is empty.
    DeviceSecretError
        If the device secret is missing (fail-closed).
    """
    logger.warning(
        "derive_key_from_env is deprecated; migrate to derive_key_for_device."
    )
    resolved_salt = salt or os.environ.get("P3_DEVICE_SALT", "")
    if not resolved_salt:
        raise ValueError(
            "Device salt must not be empty. "
            "Set the P3_DEVICE_SALT environment variable."
        )
    # Convert legacy string salt to bytes for HKDF compatibility.
    salt_bytes = resolved_salt.encode("utf-8")
    # Pad/truncate to 32 bytes so HKDF salt is well-formed.
    if len(salt_bytes) < 32:
        salt_bytes = salt_bytes.ljust(32, b"\x00")
    else:
        salt_bytes = salt_bytes[:32]
    return derive_key_for_device(fingerprint_hash, salt_bytes)
