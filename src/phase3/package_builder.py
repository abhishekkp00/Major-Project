import json
import logging
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.security import compute_sha256

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
    adapter_id: str,
    model_reference: str,
    fingerprint_hash: str,
    package_version: str,
    enc_metadata: dict,
) -> dict:
    """Builds and writes package_manifest.json into package_dir."""
    # Hash every artefact that exists so Phase 4 can cross-check them.
    artefact_hashes: dict[str, Optional[str]] = {}
    for fname in REQUIRED_ARTEFACTS:
        fpath = package_dir / fname
        artefact_hashes[fname] = compute_sha256(fpath) if fpath.exists() else None

    manifest = {
        "schema_version": package_version,
        "adapter_id": adapter_id,
        "model_reference": model_reference,
        "device_fingerprint_hash_ref": fingerprint_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encryption": {
            "algorithm": enc_metadata.get("algorithm"),
            "adapter_format": enc_metadata.get("adapter_format"),
            "encrypted_at_utc": enc_metadata.get("timestamp_utc"),
        },
        "artefact_hashes": artefact_hashes,
        "verification_instructions": (
            "Phase 4 MUST execute these steps in order before loading the adapter:\n"
            "  1. Verify package completeness (all required files present).\n"
            "  2. Recompute SHA-256 of adapter.enc and compare with adapter.hash.\n"
            "  3. Verify RSA-PSS signature in adapter.sig against adapter.hash using public.pem.\n"
            "  4. Regenerate device fingerprint on the target machine.\n"
            "  5. Derive AES key from fingerprint + P3_DEVICE_SALT.\n"
            "  6. Attempt AES-256-GCM decryption; failure means wrong device or tampered file.\n"
            "  7. Only serve the adapter to inference if all six steps succeed."
        ),
        "security_notes": {
            "plaintext_in_package": False,
            "private_key_in_package": False,
            "salt_in_package": False,
        },
    }

    _atomic_write_json(package_dir / "package_manifest.json", manifest)
    logger.info("Package manifest written → package_manifest.json")
    return manifest


def build_package(
    package_dir: Path,
    *,
    adapter_id: str,
    model_reference: str,
    fingerprint_hash: str,
    package_version: str,
    enc_metadata: dict,
    public_key_src: Path,
) -> dict:
    """High-level orchestrator: copies public key, verifies completeness, builds manifest."""
    dest_pub = package_dir / "public.pem"
    if public_key_src.resolve() != dest_pub.resolve():
        shutil.copy2(public_key_src, dest_pub)
        logger.debug("Public key copied into package: %s", dest_pub.name)

    verify_package_completeness(package_dir)
    manifest = build_manifest(
        package_dir=package_dir,
        adapter_id=adapter_id,
        model_reference=model_reference,
        fingerprint_hash=fingerprint_hash,
        package_version=package_version,
        enc_metadata=enc_metadata,
    )
    return manifest


def export_package_archive(package_dir: Path, archive_path: Optional[Path] = None) -> Path:
    """Optionally compresses package_dir into a tar.gz for secure transport."""
    if archive_path is None:
        archive_path = package_dir.with_suffix(".tar.gz")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)

    logger.info("Package archive created → %s (%d bytes)", archive_path.name, archive_path.stat().st_size)
    return archive_path
