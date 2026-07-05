import logging
from typing import Optional

from src.security import get_fingerprint_hash, derive_key
from src.common.exceptions import DeviceAuthorizationError

logger = logging.getLogger("secure_lora.phase4.device_auth")


def verify_device_binding(expected_fingerprint_hash: str, mock_fingerprint: Optional[str] = None) -> None:
    """
    Compares the expected fingerprint hash from the package manifest with the local device fingerprint.
    """
    local_hash = mock_fingerprint or get_fingerprint_hash()

    logger.info("Verifying device binding. Expected: %s... | Local: %s...",
                expected_fingerprint_hash[:12], local_hash[:12])

    if expected_fingerprint_hash != local_hash:
        raise DeviceAuthorizationError(
            "Device authorization check FAILED. This machine is not authorized to deploy this adapter."
        )
    logger.info("Device authorization PASSED. This machine matches the package target fingerprint.")


def get_device_bound_key(salt: str, mock_fingerprint: Optional[str] = None) -> bytes:
    """
    Derives the device-bound 32-byte AES key using the local fingerprint and configured salt.
    """
    if not salt:
        raise ValueError("Device salt must not be empty.")

    local_hash = mock_fingerprint or get_fingerprint_hash()
    key = derive_key(local_hash, salt)
    return key
