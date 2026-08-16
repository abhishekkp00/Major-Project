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

import json
import logging
import os
import shutil
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.security import (
    compute_sha256,
    compute_canonical_manifest_digest,
    sign_digest,
    save_signature,
    BindingPolicy,
    AntiReplayTracker,
)

from src.common.config_loader import config

logger = logging.getLogger("secure_lora.phase3.package_builder")

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


def verify_package_completeness(package_dir: Path) -> None:
    """Checks that all required artefacts are present in package_dir."""
    missing = [f for f in REQUIRED_ARTEFACTS if not (package_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete package in '{package_dir}'. Missing: {missing}"
        )
    logger.debug("Package completeness check passed: %s", package_dir.name)


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
        "kdf_version": enc_metadata.get("kdf_version", "hkdf-sha256-v1"),
        "encryption_version": "aes-256-gcm-v1",
        "signature_algorithm": "rsa-pss-2048-sha256",
        "digest_algorithm": "sha256",
        "nonce_metadata": {
            "iv_bytes": 12,
            "tag_bytes": 16,
            "salt_reference": "P3_DEVICE_SALT",
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
    sequence_number: int = 1,
    expiration_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level package orchestrator:
      1. Copies public key to package
      2. Computes manifest with all 18 security fields
      3. Signs the canonical manifest authentication digest (manifest + ciphertext digest)
      4. Saves adapter.sig
      5. Verifies package completeness
    """
    dest_pub = package_dir / "public.pem"
    if public_key_src.resolve() != dest_pub.resolve():
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
    )

    # Sign canonical manifest digest if private key provided
    if private_key_src and private_key_src.exists():
        enc_path = package_dir / "adapter.enc"
        ciphertext_digest = compute_sha256(enc_path) if enc_path.exists() else ""
        canonical_digest = compute_canonical_manifest_digest(manifest, ciphertext_digest)
        signature = sign_digest(canonical_digest, private_key_src)
        save_signature(signature, package_dir / "adapter.sig")
        # Update manifest artefact_hashes with adapter.sig hash
        manifest["artefact_hashes"]["adapter.sig"] = compute_sha256(package_dir / "adapter.sig")
        _atomic_write_json(package_dir / "package_manifest.json", manifest)

    verify_package_completeness(package_dir)
    return manifest


def export_package_archive(package_dir: Path, archive_path: Optional[Path] = None) -> Path:
    """Compresses package_dir into a tar.gz for secure transport."""
    if archive_path is None:
        archive_path = package_dir.with_suffix(".tar.gz")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)

    logger.info("Package archive created → %s (%d bytes)", archive_path.name, archive_path.stat().st_size)
    return archive_path
