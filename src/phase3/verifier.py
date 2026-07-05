import logging
from pathlib import Path
from typing import Optional

from src.security import (
    decrypt_adapter,
    get_fingerprint_hash,
    verify_integrity,
    derive_key_from_env,
    verify_signature,
)
from src.phase3.package_builder import verify_package_completeness
from src.common.exceptions import VerificationError

logger = logging.getLogger("secure_lora.phase3.verifier")


def verify_and_decrypt(
    package_dir: Path,
    output_path: Path,
    salt: Optional[str] = None,
) -> Path:
    """
    Full verification + controlled decryption pipeline.
    """
    enc_path = package_dir / "adapter.enc"
    hash_path = package_dir / "adapter.hash"
    sig_path  = package_dir / "adapter.sig"
    pub_path  = package_dir / "public.pem"

    # ── Step 1: package completeness ────────────────────────────────────────
    logger.info("[1/6] Checking package completeness…")
    try:
        verify_package_completeness(package_dir)
    except FileNotFoundError as exc:
        raise VerificationError(f"[Step 1] Package incomplete: {exc}") from exc
    logger.info("[1/6] PASS — all required files present.")

    # ── Step 2: SHA-256 integrity ────────────────────────────────────────────
    logger.info("[2/6] Verifying SHA-256 integrity…")
    try:
        verify_integrity(enc_path, hash_path)
    except (FileNotFoundError, ValueError) as exc:
        raise VerificationError(f"[Step 2] Integrity check failed: {exc}") from exc
    logger.info("[2/6] PASS — integrity verified.")

    # ── Step 3: RSA-PSS signature ────────────────────────────────────────────
    logger.info("[3/6] Verifying RSA-PSS signature…")
    try:
        stored_hash = hash_path.read_text(encoding="utf-8").strip()
        verify_signature(stored_hash, sig_path, pub_path)
    except (FileNotFoundError, ValueError) as exc:
        raise VerificationError(f"[Step 3] Signature verification failed: {exc}") from exc
    logger.info("[3/6] PASS — signature verified.")

    # ── Step 4: device fingerprint ───────────────────────────────────────────
    logger.info("[4/6] Generating local device fingerprint…")
    try:
        local_fp_hash = get_fingerprint_hash()
    except Exception as exc:
        raise VerificationError(f"[Step 4] Fingerprint generation failed: {exc}") from exc
    logger.info("[4/6] PASS — fingerprint computed (hash_prefix=%s…).", local_fp_hash[:8])

    # ── Step 5: key derivation ───────────────────────────────────────────────
    logger.info("[5/6] Deriving device-bound decryption key…")
    try:
        key = derive_key_from_env(local_fp_hash, salt)
    except (ValueError, EnvironmentError) as exc:
        raise VerificationError(f"[Step 5] Key derivation failed: {exc}") from exc
    logger.info("[5/6] PASS — key derived.")

    # ── Step 6: controlled decryption ───────────────────────────────────────
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
