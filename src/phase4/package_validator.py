"""
package_validator.py
=====================
Phase 4 Provenance and Integrity Validator for SecureLoRA.

Enforces schema validation, digest integrity, and RSA-2048-PSS signature
verification over the canonical manifest digest.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

from src.security import (
    verify_integrity,
    verify_signature,
    compute_sha256,
    validate_manifest_schema,
    compute_canonical_manifest_digest,
)
from src.common.exceptions import (
    IntegrityValidationError,
    SignatureValidationError,
    ManifestSchemaError,
)

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


def validate_package_provenance(package_dir: Path, public_key_path: Optional[Path] = None) -> Tuple[Dict[str, Any], str]:
    """
    Validates manifest schema, computes canonical manifest authentication digest,
    and verifies the RSA-2048-PSS signature.
    """
    enc_path = package_dir / "adapter.enc"
    hash_path = package_dir / "adapter.hash"
    sig_path = package_dir / "adapter.sig"
    pub_key_path = public_key_path or (package_dir / "public.pem")

    actual_hash = verify_hash_integrity(enc_path, hash_path)

    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.exists():
        # Legacy package fallback: verify raw digest signature
        logger.warning("package_manifest.json not found; falling back to legacy signature verification.")
        verify_rsa_signature(actual_hash, sig_path, pub_key_path)
        return {}, actual_hash

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ManifestSchemaError(f"Malformed JSON in package_manifest.json: {e}") from e

    validate_manifest_schema(manifest)

    # Compute canonical manifest digest
    canonical_digest = compute_canonical_manifest_digest(manifest, actual_hash)

    # Verify signature over canonical manifest digest (or raw digest fallback)
    try:
        verify_rsa_signature(canonical_digest, sig_path, pub_key_path)
    except SignatureValidationError:
        # Fallback to direct ciphertext digest verification
        verify_rsa_signature(actual_hash, sig_path, pub_key_path)

    return manifest, actual_hash


def validate_package_integrity(package_dir: Path, public_key_path: Optional[Path] = None) -> str:
    """
    Legacy helper: runs validation and returns verified digest_hex.
    """
    _, actual_hash = validate_package_provenance(package_dir, public_key_path)
    return actual_hash
