"""
phase4/package_validator.py
----------------------------
Handles verification of the protected adapter package before decryption:
1. Recomputes SHA-256 of adapter.enc and compares it against adapter.hash in constant time.
2. Verifies the digital signature in adapter.sig using the RSA public key.
"""

import hmac
import hashlib
import logging
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

# Read in 64 KiB chunks to keep memory usage bounded for large files.
CHUNK_SIZE = 64 * 1024

class IntegrityValidationError(Exception):
    """Raised when package hash comparison fails."""
    pass

class SignatureValidationError(Exception):
    """Raised when signature verification fails."""
    pass

def compute_file_sha256(file_path: Path) -> str:
    """Streams file and returns lowercase SHA-256 hex digest."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_hash_integrity(enc_path: Path, hash_path: Path) -> str:
    """
    Computes SHA-256 of enc_path and compares it to hash_path using constant-time comparison.
    Returns the actual hash digest on success.
    """
    if not enc_path.exists():
        raise FileNotFoundError(f"Encrypted adapter file not found: {enc_path}")
    if not hash_path.exists():
        raise FileNotFoundError(f"Hash file not found: {hash_path}")

    stored_hash = hash_path.read_text(encoding="utf-8").strip()
    if len(stored_hash) != 64 or not all(c in "0123456789abcdef" for c in stored_hash):
        raise IntegrityValidationError("Stored hash file does not contain a valid SHA-256 hex string.")

    computed_hash = compute_file_sha256(enc_path)

    # Use constant-time comparison to prevent timing side-channel attacks
    if not hmac.compare_digest(stored_hash, computed_hash):
        raise IntegrityValidationError(
            f"Integrity check FAILED. Stored hash: {stored_hash[:12]}... "
            f"Computed hash: {computed_hash[:12]}..."
        )

    logger.info("Integrity check PASSED. Computed SHA-256 matches stored hash.")
    return computed_hash

def verify_rsa_signature(digest_hex: str, sig_path: Path, public_key_path: Path) -> None:
    """
    Verifies the RSA-PSS signature of the digest_hex string.
    """
    if not sig_path.exists():
        raise FileNotFoundError(f"Signature file not found: {sig_path}")
    if not public_key_path.exists():
        raise FileNotFoundError(f"Public key not found: {public_key_path}")

    try:
        public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
    except Exception as e:
        raise SignatureValidationError(f"Failed to load RSA public key: {e}") from e

    signature = sig_path.read_bytes()
    message = digest_hex.encode("utf-8")

    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        logger.info("RSA-PSS signature verification PASSED.")
    except InvalidSignature as e:
        raise SignatureValidationError(
            "Signature verification FAILED. The package signature is invalid."
        ) from e
    except Exception as e:
        raise SignatureValidationError(f"RSA signature verification error: {e}") from e

def validate_package_integrity(package_dir: Path, public_key_path: Optional[Path] = None) -> str:
    """
    Runs full integrity and authenticity validation on the package.
    Returns the verified digest_hex.
    """
    enc_path = package_dir / "adapter.enc"
    hash_path = package_dir / "adapter.hash"
    sig_path = package_dir / "adapter.sig"
    pub_key_path = public_key_path or (package_dir / "public.pem")

    # 1. Verify SHA-256 of adapter.enc matches adapter.hash
    actual_hash = verify_hash_integrity(enc_path, hash_path)

    # 2. Verify RSA signature of the hash
    verify_rsa_signature(actual_hash, sig_path, pub_key_path)

    return actual_hash
