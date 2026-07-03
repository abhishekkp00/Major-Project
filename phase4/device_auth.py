"""
phase4/device_auth.py
----------------------
Handles device authorization and device-bound key derivation.
Ensures hardware identifier safety by avoiding logging raw IDs.
"""

import logging
from typing import Optional

# Re-use Phase 3 device fingerprinting and key derivation algorithms to maintain consistency
from phase3.device_fingerprint import get_fingerprint_hash
from phase3.key_derivation import derive_key

logger = logging.getLogger(__name__)

class DeviceAuthorizationError(Exception):
    """Raised when the host device does not match the package target fingerprint."""
    pass

def verify_device_binding(expected_fingerprint_hash: str, mock_fingerprint: Optional[str] = None) -> None:
    """
    Compares the expected fingerprint hash from the package manifest with the local device fingerprint.
    
    Parameters
    ----------
    expected_fingerprint_hash : str
        The target fingerprint hash reference from the manifest.
    mock_fingerprint : str | None
        Optional mock fingerprint hash for testing.
    """
    local_hash = mock_fingerprint or get_fingerprint_hash()
    
    # Mask values in log output to prevent leakage of the exact hash
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
