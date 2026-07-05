import hashlib
import os
import logging
from typing import Optional

from src.common.exceptions import CryptoError

logger = logging.getLogger("secure_lora.security.key_derivation")

_REQUIRED_KEY_BYTES = 32


def derive_key(fingerprint_hash: str, salt: str) -> bytes:
    """Derives a 32-byte AES-256-compatible key from the fingerprint hash and salt."""
    if not fingerprint_hash:
        raise ValueError("fingerprint_hash must not be empty.")
    if not salt:
        raise ValueError(
            "Device salt must not be empty. "
            "Set the P3_DEVICE_SALT environment variable."
        )

    raw_material = f"{fingerprint_hash}:{salt}"
    key_bytes = hashlib.sha256(raw_material.encode("utf-8")).digest()

    validate_key_length(key_bytes)

    logger.info(
        "Device-bound key derived successfully. key_len_bytes=%d",
        len(key_bytes),
    )
    return key_bytes


def validate_key_length(key: bytes, expected: int = _REQUIRED_KEY_BYTES) -> None:
    """Asserts that key has the correct length for AES-256."""
    if len(key) != expected:
        raise ValueError(
            f"Derived key has unexpected length {len(key)} bytes "
            f"(expected {expected}). Derivation logic is broken."
        )


def derive_key_from_env(fingerprint_hash: str, salt: Optional[str] = None) -> bytes:
    """Convenience wrapper reading salt from env if not provided."""
    resolved_salt = salt or os.environ.get("P3_DEVICE_SALT", "")
    return derive_key(fingerprint_hash, resolved_salt)
