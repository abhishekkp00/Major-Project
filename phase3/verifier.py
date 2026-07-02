"""
phase3/verifier.py  —  Step 8
--------------------------------
Authorised-device verification and controlled decryption.

Verification order (fail-closed at each step)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Package completeness — all required files must be present.
2. SHA-256 integrity   — recomputed hash of adapter.enc must match adapter.hash.
3. RSA-PSS signature   — adapter.hash content must pass signature check with
                          the bundled public key.
4. Device fingerprint  — regenerate on the local machine.
5. Key derivation      — derive AES key from local fingerprint + salt.
6. Decryption          — AES-256-GCM; authentication tag enforces device binding.

Each step raises an exception if it fails.  No step is skipped or reordered.
The decrypted adapter is written to a caller-supplied path and is only
returned if every verification step succeeds.

Design principles
~~~~~~~~~~~~~~~~~
* Fail closed — any failure raises immediately; no partial state is exposed.
* No plaintext leak — if decryption fails, no plaintext bytes are written.
* Atomic output — decrypted file is written via tmp-then-rename.
* Salt is never stored in the package — it must be present in the environment
  on the target device.
"""

import logging
from pathlib import Path
from typing import Optional

from .adapter_encryptor import decrypt_adapter
from .device_fingerprint import get_fingerprint_hash
from .integrity import verify_integrity
from .key_derivation import derive_key_from_env
from .package_builder import verify_package_completeness
from .signature_utils import verify_signature

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when any verification step fails.  Wraps the underlying cause."""


def verify_and_decrypt(
    package_dir: Path,
    output_path: Path,
    salt: Optional[str] = None,
) -> Path:
    """
    Full verification + controlled decryption pipeline.

    Parameters
    ----------
    package_dir : Path
        Directory containing the protected adapter package.
    output_path : Path
        Where to write the decrypted adapter (single file or tar.gz).
        This path is only created if ALL verification steps pass.
    salt : str | None
        Device salt override.  If omitted, ``P3_DEVICE_SALT`` env var is used.

    Returns
    -------
    Path
        ``output_path`` — only reached if authorised and verified.

    Raises
    ------
    VerificationError
        Wrapping the original exception from whichever step failed first.
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
        # Re-read the stored hash — the verifier computes it independently.
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
