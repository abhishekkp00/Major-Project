import logging
from typing import Optional

from src.security import get_fingerprint_hash, derive_key
from src.security.key_derivation import check_kdf_version, KDF_VERSION
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


def get_device_bound_key(
    salt: str,
    mock_fingerprint: Optional[str] = None,
    kdf_version: Optional[str] = None,
) -> bytes:
    """
    Derives the device-bound 32-byte AES key using HKDF-SHA256 over the
    local fingerprint and configured salt.

    Parameters
    ----------
    salt : str
        Deployment-specific secret.  Must not be empty.
    mock_fingerprint : str, optional
        Override the local fingerprint hash (for testing only).
    kdf_version : str, optional
        KDF version string read from the package manifest.  When provided,
        it is validated against the supported versions before key derivation.
        If the version is unsupported the function raises CryptoError.

    Raises
    ------
    ValueError
        If salt is empty.
    CryptoError
        If the kdf_version in the manifest is unsupported.
    """
    if not salt:
        raise ValueError("Device salt must not be empty.")

    # Validate KDF version from package manifest before deriving the key.
    # This prevents silent key mismatches against future KDF changes.
    if kdf_version is not None:
        check_kdf_version(kdf_version)
    else:
        # If caller did not supply a version we still log the version we will use.
        logger.debug("No kdf_version supplied by caller; using local default: %s", KDF_VERSION)

    local_hash = mock_fingerprint or get_fingerprint_hash()
    key = derive_key(local_hash, salt)
    return key
