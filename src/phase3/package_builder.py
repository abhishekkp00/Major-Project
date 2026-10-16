"""
package_builder.py
===================
Package Builder & Provenance Manifest Generator for SecureLoRA.

Extends package_manifest.json with cryptographically relevant fields:
  - package_id (UUIDv4)
  - adapter_id
  - base_model_id
  - model_revision
  - adapter_revision
  - package_version
  - creation_timestamp
  - expiration_timestamp
  - binding_policy_version
  - kdf_version
  - encryption_version
  - signature_algorithm
  - digest_algorithm
  - nonce_metadata
  - deployment_policy
  - sequence_number (monotonic anti-replay sequence)
  - device_fingerprint_hash_ref
  - encrypted_adapter_digest
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import shutil
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.security.key_derivation import KDF_VERSION
from src.security import (
    compute_sha256,
    compute_canonical_manifest_digest,
    sign_digest,
    save_signature,
    BindingPolicy,
    AntiReplayTracker,
)
from src.common.exceptions import (
    SigningKeyError,
    IncompletePackageError,
    PackageArtifactError,
    SignatureValidationError,
    IntegrityValidationError,
    ManifestSchemaError,
)

from src.common.config_loader import config

logger = logging.getLogger("secure_lora.phase3.package_builder")

# Ordered list of PEFT adapter weight file names, most-preferred first.
_PEFT_WEIGHT_CANDIDATES = [
    "adapter_model.safetensors",
    "adapter_model.bin",
]


def _resolve_adapter_weight_file(source: Union[Path, str]) -> Path:
    """
    Resolves the actual LoRA adapter weight file from within source (file or directory).

    Raises
    ------
    FileNotFoundError
        If neither adapter_model.safetensors nor adapter_model.bin exists.
    ValueError
        If the resolved path is not a regular file (e.g. a directory).
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Adapter weight source path does not exist: '{path}'")

    if path.is_file():
        return path

    if path.is_dir():
        for candidate in _PEFT_WEIGHT_CANDIDATES:
            weight_path = path / candidate
            if weight_path.exists():
                if not weight_path.is_file():
                    raise ValueError(
                        f"Expected a regular weight file but found a directory at '{weight_path}'. "
                        "Package directory may be malformed."
                    )
                logger.debug("Resolved adapter weight file: %s", weight_path.name)
                return weight_path

        raise FileNotFoundError(
            f"No supported adapter weight file found in '{path}'. "
            f"Expected one of: {_PEFT_WEIGHT_CANDIDATES}. "
            "Ensure the trained adapter has been saved into the package directory "
            "before calling build_package()."
        )

    raise ValueError(f"Invalid adapter weight source: '{path}'")

REQUIRED_ARTEFACTS = [
    "adapter.enc",
    "adapter.hash",
    "adapter.sig",
    "metadata.json",
    "public.pem",
]


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def verify_package_completeness(package_dir: Path, strict_format: bool = False) -> None:
    """
    Checks that all required artefacts are present, readable, non-empty, and regular files.
    Fails closed with IncompletePackageError if any required artifact is missing, or
    PackageArtifactError if an artifact is empty, unreadable, or invalid.
    """
    missing = [f for f in REQUIRED_ARTEFACTS if not (package_dir / f).exists()]
    if missing:
        raise IncompletePackageError(
            f"Incomplete package in '{package_dir}'. Missing: {missing}"
        )

    for fname in REQUIRED_ARTEFACTS:
        fpath = package_dir / fname
        if not fpath.is_file():
            raise PackageArtifactError(
                f"Required artifact '{fname}' in '{package_dir}' is not a regular file."
            )
        if fpath.stat().st_size == 0:
            raise PackageArtifactError(
                f"Required artifact '{fname}' in '{package_dir}' is empty (0 bytes)."
            )
        try:
            with open(fpath, "rb") as f:
                f.read(32)
        except Exception as exc:
            raise PackageArtifactError(
                f"Required artifact '{fname}' in '{package_dir}' is unreadable: {exc}"
            ) from exc

    if strict_format:
        verify_package_artifacts(package_dir)

    logger.debug("Package completeness check passed: %s", package_dir.name)


def verify_package_artifacts(
    package_dir: Path,
    manifest: Optional[Dict[str, Any]] = None,
    require_signature_match: bool = False,
) -> None:
    """
    Verifies that all package artifacts exist, are readable, have expected formats,
    and are consistent with the manifest and recorded hashes.
    """
    # 1. Validate metadata.json format
    meta_path = package_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta_data, dict):
                raise PackageArtifactError("metadata.json must contain a JSON object.")
        except Exception as exc:
            raise PackageArtifactError(f"metadata.json is malformed or invalid JSON: {exc}") from exc

    # 2. Validate package_manifest.json format if present
    manifest_path = package_dir / "package_manifest.json"
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_manifest, dict):
                raise PackageArtifactError("package_manifest.json must contain a JSON object.")
            if manifest is None:
                manifest = loaded_manifest
        except Exception as exc:
            raise PackageArtifactError(f"package_manifest.json is malformed: {exc}") from exc

    # 3. Validate adapter.hash format (must be 64-character hex string)
    hash_path = package_dir / "adapter.hash"
    if hash_path.exists():
        raw_hash = hash_path.read_text(encoding="utf-8").strip()
        if len(raw_hash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in raw_hash):
            raise PackageArtifactError(
                f"adapter.hash contains invalid digest format: '{raw_hash}' (expected 64 hex chars)."
            )

    # 4. Validate adapter.sig format (minimum length for RSA signature)
    sig_path = package_dir / "adapter.sig"
    if sig_path.exists():
        sig_bytes = sig_path.read_bytes()
        if len(sig_bytes) < 64:
            raise PackageArtifactError(
                f"adapter.sig is too short ({len(sig_bytes)} bytes; expected at least 64 bytes)."
            )

    # 5. Check manifest digest consistency if manifest is available
    if manifest is not None:
        enc_path = package_dir / "adapter.enc"
        if enc_path.exists():
            actual_ciphertext_hash = compute_sha256(enc_path)
            manifest_digest = str(manifest.get("encrypted_adapter_digest", "")).strip().lower()
            if manifest_digest and not hmac.compare_digest(manifest_digest, actual_ciphertext_hash):
                raise IntegrityValidationError(
                    f"Manifest encrypted_adapter_digest '{manifest_digest}' does not match "
                    f"actual ciphertext hash '{actual_ciphertext_hash}'."
                )

            # Check artefact_hashes consistency
            artefact_hashes = manifest.get("artefact_hashes", {})
            if isinstance(artefact_hashes, dict):
                for fname, expected_h in artefact_hashes.items():
                    if expected_h is not None and (package_dir / fname).exists():
                        actual_h = compute_sha256(package_dir / fname)
                        if not hmac.compare_digest(actual_h, str(expected_h).strip().lower()):
                            raise IntegrityValidationError(
                                f"Artefact hash mismatch for '{fname}': recorded '{expected_h}', "
                                f"actual '{actual_h}'."
                            )


def build_manifest(
    package_dir: Path,
    adapter_id: str = "medical-lora-adapter-v1",
    model_reference: str = "distilbert-base-uncased",
    fingerprint_hash: str = "",
    package_version: str = "1.0.0",
    enc_metadata: Optional[Dict[str, Any]] = None,
    sequence_number: int = 1,
    package_id: Optional[str] = None,
    expiration_timestamp: Optional[str] = None,
    model_revision: str = "main",
    adapter_revision: str = "v1.0.0",
    binding_policy_version: str = "1.0.0",
    screening_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds and writes the extended package_manifest.json with cryptographically
    relevant fields for provenance and anti-replay validation.
    """
    if sequence_number == 1:
        sequence_number = AntiReplayTracker().get_next_sequence_number(adapter_id)

    enc_path = package_dir / "adapter.enc"
    ciphertext_digest = compute_sha256(enc_path) if enc_path.exists() else ""

    pkg_id = package_id or str(uuid.uuid4())
    creation_time = datetime.now(timezone.utc).isoformat()

    policy_dict = config.binding_policy if hasattr(config, "binding_policy") else {
        "strictness": "high",
        "allowed_feature_changes": {
            "network_interface": True,
            "hostname": False,
            "machine_id": False,
            "disk_uuid": False,
        },
    }

    manifest = {
        "schema_version": package_version,
        "package_id": pkg_id,
        "adapter_id": adapter_id,
        "base_model_id": model_reference,
        "model_reference": model_reference,
        "model_revision": model_revision,
        "adapter_revision": adapter_revision,
        "package_version": package_version,
        "creation_timestamp": creation_time,
        "created_at_utc": creation_time,
        "expiration_timestamp": expiration_timestamp,
        "binding_policy_version": binding_policy_version,
        "kdf_version": (enc_metadata or {}).get("kdf_version", KDF_VERSION),
        "hkdf_salt_hex": (enc_metadata or {}).get("hkdf_salt_hex", ""),
        "encryption_version": "aes-256-gcm-v1",
        "signature_algorithm": "rsa-pss-2048-sha256",
        "digest_algorithm": "sha256",
        "nonce_metadata": {
            "iv_bytes": 12,
            "tag_bytes": 16,
            "salt_reference": "hkdf_csprng_salt",
        },
        "deployment_policy": policy_dict,
        "sequence_number": sequence_number,
        "device_fingerprint_hash_ref": fingerprint_hash,
        "encrypted_adapter_digest": ciphertext_digest,
        "verification_instructions": "Execute Phase 4 verification steps 1-9 in order before decryption or loading.",
        "artefact_hashes": {
            fname: (compute_sha256(package_dir / fname) if (package_dir / fname).exists() else None)
            for fname in REQUIRED_ARTEFACTS
        },

        "security_notes": {
            "plaintext_in_package": False,
            "private_key_in_package": False,
            "salt_in_package": False,
            "assurance": (
                "Provides cryptographic authenticity and provenance under the assumed private-key security model."
            ),
        },
    }

    if screening_report is not None:
        manifest["screening_report"] = {
            "adapter_weight_path": screening_report.get("adapter_weight_path"),
            "screening_mode": screening_report.get("screening_mode"),
            "real_weights_used": screening_report.get("real_weights_used"),
            "real_callable_used": screening_report.get("real_callable_used"),
            "risk_score": screening_report.get("risk_score"),
            "decision": screening_report.get("decision"),
            "risk_level": screening_report.get("risk_level"),
        }

    # Defense-in-depth: Ensure package manifest never contains secret or raw key material
    forbidden_secret_keys = {"secret", "device_secret", "raw_secret", "aes_key", "raw_key", "private_key"}
    for f_key in forbidden_secret_keys:
        if f_key in manifest:
            del manifest[f_key]

    _atomic_write_json(package_dir / "package_manifest.json", manifest)
    logger.info("Package manifest written → package_manifest.json (pkg_id=%s, seq=%d)", pkg_id, sequence_number)
    return manifest


def build_package(
    package_dir: Path,
    *,
    adapter_id: str = "medical-lora-adapter-v1",
    model_reference: str = "distilbert-base-uncased",
    fingerprint_hash: str = "",
    package_version: str = "1.0.0",
    enc_metadata: Optional[Dict[str, Any]] = None,
    public_key_src: Path,
    private_key_src: Optional[Path] = None,
    require_signing: bool = False,
    sequence_number: int = 1,
    expiration_timestamp: Optional[str] = None,
    enable_screening: bool = True,
    admin_override_token: Optional[str] = None,
    adapter_source: Optional[Union[Path, str]] = None,
    candidate_model_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    High-level package orchestrator:
      1. Pre-packaging Security Screening Gate (Structural + Behavioral vetting)
      2. Validates and copies public key to package
      3. Computes manifest with all security fields and embeds screening provenance
      4. Signs the canonical manifest authentication digest (manifest + ciphertext digest)
      5. Saves adapter.sig and updates artefact_hashes
      6. Executes immediate self-verification of the signature against public.pem
      7. Ensures RSA private signing keys, secrets, and plaintext files are removed from package directory
      8. Verifies package completeness and artifact formats
    """
    if enable_screening:
        from src.security.adapter_screening import (
            pre_packaging_screening_gate,
            SecurityScreeningError,
        )
        # ── Resolve the ACTUAL trained adapter weight file ──────────────────
        target_source = adapter_source if adapter_source is not None else package_dir
        adapter_weight_path = _resolve_adapter_weight_file(target_source)

        # ── Production-mode assertion guard ────────────────────────────────
        _SCREENING_MODE = "production"
        assert _SCREENING_MODE == "production", (
            "BUG: package_builder attempted to invoke screening in non-production mode. "
            "This is a programming error."
        )

        logger.info(
            "Executing Phase 3 pre-packaging security screening gate for '%s' "
            "(weight_file=%s, mode=%s)...",
            adapter_id, adapter_weight_path.name, _SCREENING_MODE,
        )
        screening_report = pre_packaging_screening_gate(
            adapter_source=adapter_weight_path,
            adapter_id=adapter_id,
            candidate_model_fn=candidate_model_fn,
            admin_override_token=admin_override_token,
            mode=_SCREENING_MODE,
        )
        logger.info(
            "Pre-packaging security screening passed "
            "(decision=%s, risk=%.4f, real_weights=%s, real_callable=%s)",
            screening_report.decision,
            screening_report.risk_score,
            screening_report.real_weights_used,
            screening_report.real_callable_used,
        )
        # Attach screening provenance to the manifest
        _screening_report_dict = screening_report.to_dict()
    else:
        _screening_report_dict = None

    if public_key_src is None or not Path(public_key_src).exists():
        raise FileNotFoundError(f"Public key source does not exist: '{public_key_src}'")

    dest_pub = package_dir / "public.pem"
    if Path(public_key_src).resolve() != dest_pub.resolve():
        shutil.copy2(public_key_src, dest_pub)
        logger.debug("Public key copied into package: %s", dest_pub.name)

    manifest = build_manifest(
        package_dir=package_dir,
        adapter_id=adapter_id,
        model_reference=model_reference,
        fingerprint_hash=fingerprint_hash,
        package_version=package_version,
        enc_metadata=enc_metadata,
        sequence_number=sequence_number,
        expiration_timestamp=expiration_timestamp,
        screening_report=_screening_report_dict,
    )

    # Sign canonical manifest digest if signing is required or private key provided
    should_sign = require_signing or (private_key_src is not None)
    if should_sign:
        if private_key_src is None:
            raise SigningKeyError(
                "Package signing is mandatory, but no private signing key (private_key_src) was provided."
            )
        priv_p = Path(private_key_src)
        if not priv_p.exists():
            raise SigningKeyError(
                f"Private signing key file does not exist: '{priv_p}'."
            )
        if not priv_p.is_file():
            raise SigningKeyError(
                f"Private signing key path is not a regular file: '{priv_p}'."
            )

        from src.security.signature import _load_private_key, verify_signature
        try:
            _load_private_key(priv_p)
        except Exception as exc:
            raise SigningKeyError(
                f"Private signing key at '{priv_p}' is invalid or unusable: {exc}"
            ) from exc

        enc_path = package_dir / "adapter.enc"
        ciphertext_digest = compute_sha256(enc_path) if enc_path.exists() else ""
        canonical_digest = compute_canonical_manifest_digest(manifest, ciphertext_digest)
        signature = sign_digest(canonical_digest, priv_p)
        save_signature(signature, package_dir / "adapter.sig")

        # Update non-canonical manifest artefact_hashes with adapter.sig hash
        manifest["artefact_hashes"]["adapter.sig"] = compute_sha256(package_dir / "adapter.sig")
        _atomic_write_json(package_dir / "package_manifest.json", manifest)

        # Immediate self-verification: confirm signature authenticates the final canonical manifest
        try:
            verify_signature(canonical_digest, package_dir / "adapter.sig", package_dir / "public.pem")
        except Exception as exc:
            raise SignatureValidationError(
                f"Self-verification of newly generated package signature failed: {exc}"
            ) from exc

    # Ensure RSA private signing key is NEVER retained inside the package directory
    for priv_pattern in ("*private*.pem", "*priv*.pem", "*.key", "*.secret", "device.secret", "dev_private.pem"):
        for priv_file in package_dir.glob(priv_pattern):
            try:
                priv_file.unlink()
                logger.info("Removed private secret file '%s' from package directory.", priv_file.name)
            except OSError:
                pass

    for pem_file in package_dir.glob("*.pem"):
        if pem_file.name != "public.pem":
            try:
                content = pem_file.read_text(encoding="utf-8", errors="ignore")
                if "PRIVATE KEY" in content:
                    pem_file.unlink()
                    logger.info("Removed private key PEM '%s' from package directory.", pem_file.name)
            except OSError:
                pass

    verify_package_completeness(package_dir)
    verify_package_artifacts(package_dir, manifest=manifest)
    return manifest


def _exclude_secrets_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    """Excludes private keys, secret keys, device secrets, and plaintext adapter files from exported tar archives."""
    name = Path(tarinfo.name).name.lower()

    # 1. Private keys, secret files, and device secrets
    if (
        "private" in name
        or "priv" in name
        or name.endswith(".key")
        or name.endswith(".secret")
        or "device_secret" in name
        or name == "device.secret"
        or name == "dev_private.pem"
    ):
        logger.warning("Excluding secret/private file '%s' from package archive.", tarinfo.name)
        return None

    # Exclude any other PEM files besides public.pem
    if name.endswith(".pem") and name != "public.pem":
        logger.warning("Excluding non-public PEM file '%s' from package archive.", tarinfo.name)
        return None

    # 2. Plaintext adapter weights and configuration files
    if (
        name in {"adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"}
        or name.endswith(".safetensors")
        or (name.endswith(".bin") and name != "adapter.enc")
        or name.endswith(".pt")
        or name.endswith(".pth")
        or name.endswith(".ckpt")
        or name.endswith(".tmp")
    ):
        logger.warning("Excluding plaintext adapter/weight file '%s' from package archive.", tarinfo.name)
        return None

    return tarinfo


def export_package_archive(package_dir: Path, archive_path: Optional[Path] = None) -> Path:
    """Compresses package_dir into a tar.gz for secure transport, excluding private keys/secrets."""
    if archive_path is None:
        archive_path = package_dir.with_suffix(".tar.gz")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name, filter=_exclude_secrets_filter)

    logger.info("Package archive created → %s (%d bytes)", archive_path.name, archive_path.stat().st_size)
    return archive_path

