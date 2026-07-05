import logging
from pathlib import Path
from typing import Optional

from src.security import verify_integrity, verify_signature, compute_sha256
from src.common.exceptions import IntegrityValidationError, SignatureValidationError

logger = logging.getLogger("secure_lora.phase4.package_validator")


def verify_hash_integrity(enc_path: Path, hash_path: Path) -> str:
    """
    Computes SHA-256 of enc_path and compares it to hash_path using constant-time comparison.
    Returns the actual hash digest on success.
    """
    try:
        verify_integrity(enc_path, hash_path)
        return compute_sha256(enc_path)
    except (FileNotFoundError, ValueError) as e:
        raise IntegrityValidationError(str(e)) from e


def verify_rsa_signature(digest_hex: str, sig_path: Path, public_key_path: Path) -> None:
    """
    Verifies the RSA-PSS signature of the digest_hex string.
    """
    try:
        verify_signature(digest_hex, sig_path, public_key_path)
    except (FileNotFoundError, ValueError) as e:
        raise SignatureValidationError(str(e)) from e


def validate_package_integrity(package_dir: Path, public_key_path: Optional[Path] = None) -> str:
    """
    Runs full integrity and authenticity validation on the package.
    Returns the verified digest_hex.
    """
    enc_path = package_dir / "adapter.enc"
    hash_path = package_dir / "adapter.hash"
    sig_path = package_dir / "adapter.sig"
    pub_key_path = public_key_path or (package_dir / "public.pem")

    actual_hash = verify_hash_integrity(enc_path, hash_path)
    verify_rsa_signature(actual_hash, sig_path, pub_key_path)
    return actual_hash
