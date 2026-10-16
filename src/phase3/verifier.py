"""
verifier.py
===========
Phase 3 Verification and Controlled Decryption pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

from src.security import (
    decrypt_adapter,
    get_fingerprint_hash,
    derive_key_from_env,
    compute_sha256,
)
from src.phase4.package_validator import validate_package_provenance
from src.phase3.package_builder import verify_package_completeness
from src.common.exceptions import VerificationError

logger = logging.getLogger("secure_lora.phase3.verifier")


def verify_and_decrypt(
    package_dir: Path,
    output_path: Path,
    salt: Optional[str] = None,
) -> Path:
    """
    Full verification + controlled decryption pipeline for Phase 3 CLI runner.
    """
    enc_path = package_dir / "adapter.enc"

    # ── Step 1: Package completeness ────────────────────────────────────────
    logger.info("[1/6] Checking package completeness…")
    try:
        verify_package_completeness(package_dir)
    except FileNotFoundError as exc:
        raise VerificationError(f"[Step 1] Package incomplete: {exc}") from exc
    logger.info("[1/6] PASS — all required files present.")

    # ── Step 2 & 3: Manifest Schema & RSA-PSS Signature Verification ────────
    logger.info("[2-3/6] Verifying SHA-256 integrity and RSA-PSS signature over canonical manifest digest…")
    try:
        manifest, ciphertext_digest = validate_package_provenance(package_dir)
    except Exception as exc:
        raise VerificationError(f"[Step 2-3] Provenance / signature verification failed: {exc}") from exc
    logger.info("[2-3/6] PASS — manifest schema & signature verified.")

    # ── Step 4: Device fingerprint & binding authorization ───────────────────
    logger.info("[4/6] Generating local device fingerprint and verifying binding…")
    try:
        local_fp_hash = get_fingerprint_hash()
    except Exception as exc:
        raise VerificationError(f"[Step 4] Fingerprint generation failed: {exc}") from exc

    expected_fp_ref = manifest.get("device_fingerprint_hash_ref")
    if expected_fp_ref and expected_fp_ref != local_fp_hash:
        raise VerificationError(
            f"[Step 4] Device authorization check FAILED: host fingerprint ({local_fp_hash[:12]}…) "
            f"does not match package device ({expected_fp_ref[:12]}…)."
        )
    logger.info("[4/6] PASS — device binding verified (hash_prefix=%s…).", local_fp_hash[:8])

    # ── Step 5: Key derivation ───────────────────────────────────────────────
    logger.info("[5/6] Deriving device-bound decryption key…")
    try:
        import json
        from src.security.key_derivation import (
            derive_key_for_device,
            validate_salt_length,
            check_kdf_version,
        )

        kdf_ver = manifest.get("kdf_version")
        if kdf_ver:
            check_kdf_version(kdf_ver)

        hkdf_salt_bytes = None
        manifest_path = package_dir / "package_manifest.json"
        meta_path = package_dir / "metadata.json"

        if manifest_path.exists():
            mdata = json.loads(manifest_path.read_text())
            salt_hex = mdata.get("hkdf_salt_hex")
            if salt_hex:
                try:
                    hkdf_salt_bytes = bytes.fromhex(salt_hex)
                except (ValueError, TypeError) as hex_err:
                    raise VerificationError(f"[Step 5] Malformed hex encoding in package salt: {hex_err}") from hex_err
                validate_salt_length(hkdf_salt_bytes)

        if not hkdf_salt_bytes and meta_path.exists():
            mdata = json.loads(meta_path.read_text())
            salt_hex = mdata.get("hkdf_salt_hex")
            if salt_hex:
                try:
                    hkdf_salt_bytes = bytes.fromhex(salt_hex)
                except (ValueError, TypeError) as hex_err:
                    raise VerificationError(f"[Step 5] Malformed hex encoding in package salt: {hex_err}") from hex_err
                validate_salt_length(hkdf_salt_bytes)

        if hkdf_salt_bytes:
            key = derive_key_for_device(local_fp_hash, hkdf_salt_bytes, kdf_version=kdf_ver)
        elif salt:
            key = derive_key_from_env(local_fp_hash, salt)
        else:
            raise VerificationError("[Step 5] Key derivation failed: HKDF salt missing from package manifest and no salt provided.")

    except (ValueError, EnvironmentError, VerificationError) as exc:
        raise VerificationError(f"[Step 5] Key derivation failed: {exc}") from exc
    except Exception as exc:
        raise VerificationError(f"[Step 5] Key derivation failed: {exc}") from exc
    logger.info("[5/6] PASS — key derived.")

    # ── Step 6: Controlled decryption ───────────────────────────────────────
    logger.info("[6/6] Attempting AES-256-GCM decryption…")
    try:
        decrypt_adapter(enc_path, output_path, key)
    except (FileNotFoundError, ValueError) as exc:
        raise VerificationError(
            f"[Step 6] Decryption failed — wrong device, wrong salt, or tampered ciphertext: {exc}"
        ) from exc
    logger.info("[6/6] PASS — adapter decrypted to %s.", output_path.name)

    logger.info(
        "=== All 6 verification steps PASSED. Adapter is authorised for use. ==="
    )
    return output_path
