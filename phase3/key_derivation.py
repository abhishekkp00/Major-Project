"""
phase3/key_derivation.py  —  Step 3
--------------------------------------
Derives the 256-bit AES encryption key that is bound to this specific device.

Derivation formula
~~~~~~~~~~~~~~~~~~
::

    K = SHA-256( fingerprint_hash + ":" + salt )

where:
  * ``fingerprint_hash`` is the hex digest produced by ``device_fingerprint``
  * ``salt`` is a secret string sourced exclusively from the environment
    variable ``P3_DEVICE_SALT``

Design rationale
~~~~~~~~~~~~~~~~
* Using SHA-256 here is intentional and appropriate: the inputs already have
  high entropy (256-bit fingerprint + external secret salt), so the simpler
  SHA-256 construction avoids adding PBKDF2/scrypt complexity while still
  binding the key to device identity.
* The raw key bytes are never printed, logged, or written to disk.
* The module is stateless: given the same device and the same salt it will
  always produce the same key.  This means no key-wrapping file is needed on
  the authorised machine; the key is derived on demand and discarded after use.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# AES-256 requires exactly 32 bytes (256 bits).
_REQUIRED_KEY_BYTES = 32


def derive_key(fingerprint_hash: str, salt: str) -> bytes:
    """
    Derives a 32-byte AES-256-compatible key from the device fingerprint hash
    and a caller-supplied salt.

    Parameters
    ----------
    fingerprint_hash : str
        Hex digest returned by ``device_fingerprint.get_fingerprint_hash()``.
    salt : str
        Secret salt read from the ``P3_DEVICE_SALT`` environment variable.
        Must not be empty.

    Returns
    -------
    bytes
        32 raw key bytes suitable for use with AES-256-GCM.

    Raises
    ------
    ValueError
        If either argument is empty, or if the derived key has an unexpected
        length (should never happen with SHA-256, but validated defensively).
    """
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

    # Log derivation success — never log key material.
    logger.info(
        "Device-bound key derived successfully. key_len_bytes=%d",
        len(key_bytes),
    )
    return key_bytes


def validate_key_length(key: bytes, expected: int = _REQUIRED_KEY_BYTES) -> None:
    """
    Asserts that ``key`` has the correct length for AES-256.

    Raises
    ------
    ValueError
        If the key is not exactly ``expected`` bytes long.
    """
    if len(key) != expected:
        raise ValueError(
            f"Derived key has unexpected length {len(key)} bytes "
            f"(expected {expected}). Derivation logic is broken."
        )


def derive_key_from_env(fingerprint_hash: str, salt: Optional[str] = None) -> bytes:
    """
    Convenience wrapper: reads the salt from the ``P3_DEVICE_SALT`` env var
    when ``salt`` is not explicitly provided, then delegates to ``derive_key``.

    Parameters
    ----------
    fingerprint_hash : str
        Hex digest from ``device_fingerprint.get_fingerprint_hash()``.
    salt : str | None
        Optional override; if omitted the env var is used.

    Returns
    -------
    bytes
        32-byte AES key.
    """
    import os
    resolved_salt = salt or os.environ.get("P3_DEVICE_SALT", "")
    return derive_key(fingerprint_hash, resolved_salt)
