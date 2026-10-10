"""
package_validator.py
=====================
Phase 4 Provenance and Integrity Validator for SecureLoRA.

Enforces schema validation, digest integrity, and RSA-2048-PSS signature
verification over the canonical manifest digest. Current-format packages
must have a valid signature over the canonical manifest authentication digest.
Legacy packages are strictly separated and only accepted through explicit
legacy schema/versioning paths.
"""

import hmac
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

# Explicit supported package versions
SUPPORTED_CURRENT_VERSIONS = frozenset({"1.0.0", "2.0.0", "3.0.0"})
SUPPORTED_LEGACY_VERSIONS = frozenset({"0.0.0", "0.1.0", "legacy", "v0"})


def verify_hash_integrity(enc_path: Path, hash_path: Path) -> str:
    """
    Computes SHA-256 of enc_path and compares it to hash_path using constant-time comparison.
    Returns the actual hash digest on success.
    Fails closed if either file is missing or digests do not match.
    """
    if not enc_path.exists():
        raise IntegrityValidationError(f"Encrypted adapter file not found: {enc_path}")
    if not hash_path.exists():
        raise IntegrityValidationError(f"Adapter hash file not found: {hash_path}")

    try:
        verify_integrity(enc_path, hash_path)
        return compute_sha256(enc_path)
    except (FileNotFoundError, ValueError) as e:
        raise IntegrityValidationError(str(e)) from e


def verify_rsa_signature(digest_hex: str, sig_path: Path, public_key_path: Path) -> None:
    """
    Verifies the RSA-PSS signature of the digest_hex string.
    Fails closed with SignatureValidationError if files are missing, key is invalid,
    or the cryptographic signature verification fails.
    """
    if public_key_path is None or not public_key_path.exists():
        raise SignatureValidationError(f"Public key not found: {public_key_path}")
    if sig_path is None or not sig_path.exists():
        raise SignatureValidationError(f"Signature file not found: {sig_path}")

    try:
        verify_signature(digest_hex, sig_path, public_key_path)
    except (FileNotFoundError, ValueError) as e:
        raise SignatureValidationError(str(e)) from e
    except Exception as e:
        raise SignatureValidationError(f"Cryptographic signature check failed: {e}") from e


def _validate_current_package(
    manifest: Dict[str, Any],
    actual_hash: str,
    sig_path: Path,
    pub_key_path: Path,
) -> Tuple[Dict[str, Any], str]:
    """
    Validates a current-format package (version >= 1.0.0):
      1. Validates full manifest schema and security fields.
      2. Enforces digest consistency between manifest and actual ciphertext.
      3. Computes canonical manifest authentication digest.
      4. Requires successful RSA-PSS signature verification over canonical digest.
         Never falls back to raw ciphertext verification.
    """
    validate_manifest_schema(manifest)

    # Inconsistent digest check between manifest metadata and actual ciphertext
    manifest_digest = str(manifest.get("encrypted_adapter_digest", "")).strip().lower()
    if not hmac.compare_digest(manifest_digest, actual_hash.strip().lower()):
        raise IntegrityValidationError(
            f"Digest inconsistency: manifest 'encrypted_adapter_digest' ({manifest_digest}) "
            f"does not match verified ciphertext digest ({actual_hash})."
        )

    # Artefact hashes check if present
    artefact_hashes = manifest.get("artefact_hashes")
    if isinstance(artefact_hashes, dict) and "adapter.enc" in artefact_hashes:
        art_enc = str(artefact_hashes["adapter.enc"]).strip().lower()
        if art_enc and not hmac.compare_digest(art_enc, actual_hash.strip().lower()):
            raise IntegrityValidationError(
                "Digest inconsistency: manifest artefact_hashes['adapter.enc'] does not match ciphertext."
            )

    # Compute canonical manifest authentication digest binding all security fields
    canonical_digest = compute_canonical_manifest_digest(manifest, actual_hash)

    # Strictly require RSA-PSS signature over canonical manifest digest
    verify_rsa_signature(canonical_digest, sig_path, pub_key_path)

    logger.debug("Current-format package provenance verified successfully (package_id=%s).", manifest.get("package_id"))
    return manifest, actual_hash


def _validate_legacy_package(
    manifest: Dict[str, Any],
    actual_hash: str,
    sig_path: Path,
    pub_key_path: Path,
    version: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Validates a legacy package through an explicit, dedicated legacy validation path.
    Verifies RSA-PSS signature over the raw ciphertext digest directly.
    """
    if version not in SUPPORTED_LEGACY_VERSIONS:
        raise ManifestSchemaError(
            f"Unsupported legacy package version: '{version}'. "
            f"Supported legacy versions: {sorted(list(SUPPORTED_LEGACY_VERSIONS))}"
        )

    logger.warning("Validating legacy package (version=%s) via explicit legacy path.", version)
    verify_rsa_signature(actual_hash, sig_path, pub_key_path)
    return manifest, actual_hash


def validate_package_provenance(
    package_dir: Path,
    public_key_path: Optional[Path] = None,
    allow_legacy: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """
    Validates package authenticity, schema, and integrity:
    1. Verifies ciphertext integrity between adapter.enc and adapter.hash.
    2. Routes strictly by package schema/version to current or legacy verification.
       Never retries legacy verification if canonical signature fails.
    """
    enc_path = package_dir / "adapter.enc"
    hash_path = package_dir / "adapter.hash"
    sig_path = package_dir / "adapter.sig"
    pub_key_path = public_key_path or (package_dir / "public.pem")

    # Step 1: Verify raw ciphertext digest against adapter.hash
    actual_hash = verify_hash_integrity(enc_path, hash_path)

    manifest_path = package_dir / "package_manifest.json"

    # Route A: Un-manifested legacy package (version 0.0.0)
    if not manifest_path.exists():
        if not allow_legacy:
            raise ManifestSchemaError("Missing package_manifest.json (legacy packages disabled).")
        return _validate_legacy_package(
            manifest={},
            actual_hash=actual_hash,
            sig_path=sig_path,
            pub_key_path=pub_key_path,
            version="0.0.0",
        )

    # Route B: Manifested package
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ManifestSchemaError(f"Malformed JSON in package_manifest.json: {e}") from e

    if not isinstance(manifest, dict):
        raise ManifestSchemaError("Invalid package_manifest.json: root must be a JSON object.")

    # Determine explicit schema / package version
    version = str(manifest.get("schema_version") or manifest.get("package_version") or "").strip()
    if not version:
        raise ManifestSchemaError("Missing 'schema_version' or 'package_version' in package manifest.")

    # Explicit legacy manifest package
    if version in SUPPORTED_LEGACY_VERSIONS:
        if not allow_legacy:
            raise ManifestSchemaError(f"Legacy package version '{version}' is disabled.")
        return _validate_legacy_package(
            manifest=manifest,
            actual_hash=actual_hash,
            sig_path=sig_path,
            pub_key_path=pub_key_path,
            version=version,
        )

    # Check for current supported versions (major version >= 1 or in SUPPORTED_CURRENT_VERSIONS)
    is_current = version in SUPPORTED_CURRENT_VERSIONS or any(
        version.startswith(f"{v}.") for v in ["1", "2", "3"]
    )
    if not is_current:
        raise ManifestSchemaError(
            f"Unsupported package version: '{version}'. "
            f"Supported versions: {sorted(list(SUPPORTED_CURRENT_VERSIONS))} "
            f"(or legacy: {sorted(list(SUPPORTED_LEGACY_VERSIONS))})"
        )

    # Current-format package validation: STRICT canonical digest verification
    return _validate_current_package(
        manifest=manifest,
        actual_hash=actual_hash,
        sig_path=sig_path,
        pub_key_path=pub_key_path,
    )


def validate_package_integrity(package_dir: Path, public_key_path: Optional[Path] = None) -> str:
    """
    Legacy helper: runs validation and returns verified digest_hex.
    """
    _, actual_hash = validate_package_provenance(package_dir, public_key_path)
    return actual_hash
