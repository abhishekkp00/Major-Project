"""
test_package_creation_audit.py
==============================
Architectural audit tests for SecureLoRA package creation, manifest consistency,
signature authentication, artifact completeness, and secrets exclusion.

Enforces:
1. Valid complete package builds and validates under current schema (1.0.0+).
2. Missing or unusable signing key fails closed when signing is mandatory.
3. Modified manifest, ciphertext, or artifact hashes are rejected.
4. Package metadata and signature correspond to the exact same final canonical manifest.
5. Private keys, device secrets, and plaintext adapter files are excluded from exported archives.
6. Incomplete or corrupted package creation does not report success.
"""

import json
import os
import tarfile
from pathlib import Path
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.phase3.package_builder import (
    build_package,
    build_manifest,
    export_package_archive,
    verify_package_completeness,
    verify_package_artifacts,
    REQUIRED_ARTEFACTS,
)
from src.phase4.package_validator import validate_package_provenance
from src.security import (
    compute_sha256,
    compute_canonical_manifest_digest,
    generate_dev_keypair,
)
from src.common.exceptions import (
    SigningKeyError,
    IncompletePackageError,
    PackageArtifactError,
    SignatureValidationError,
    IntegrityValidationError,
    ManifestSchemaError,
)


@pytest.fixture
def package_workspace(tmp_path: Path):
    """Sets up a clean workspace with valid encryption keypair and ciphertext."""
    pkg_dir = tmp_path / "protected_package"
    pkg_dir.mkdir()
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()

    priv_key_path = keys_dir / "dev_private.pem"
    pub_key_path = keys_dir / "public.pem"
    generate_dev_keypair(priv_key_path, pub_key_path, key_size=2048)

    # Valid ciphertext and metadata
    ciphertext = b"test-encrypted-adapter-weights-payload-bytes" * 16
    enc_path = pkg_dir / "adapter.enc"
    enc_path.write_bytes(ciphertext)

    ciphertext_hash = compute_sha256(enc_path)
    hash_path = pkg_dir / "adapter.hash"
    hash_path.write_text(ciphertext_hash)

    metadata = {
        "algorithm": "AES-256-GCM",
        "iv_hex": os.urandom(12).hex(),
        "tag_hex": os.urandom(16).hex(),
        "kdf_version": "hkdf-sha256-v2",
        "hkdf_salt_hex": os.urandom(32).hex(),
    }
    (pkg_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "pkg_dir": pkg_dir,
        "priv_key_path": priv_key_path,
        "pub_key_path": pub_key_path,
        "enc_path": enc_path,
        "ciphertext_hash": ciphertext_hash,
        "metadata": metadata,
    }


def test_valid_complete_package_builds_and_validates(package_workspace):
    """Requirement 1: Valid complete package builds, self-verifies, and validates with Phase 4 validator."""
    pkg_dir = package_workspace["pkg_dir"]
    priv_key_path = package_workspace["priv_key_path"]
    pub_key_path = package_workspace["pub_key_path"]

    manifest = build_package(
        package_dir=pkg_dir,
        adapter_id="medical-lora-v1",
        model_reference="distilbert-base-uncased",
        fingerprint_hash="a" * 64,
        package_version="1.0.0",
        enc_metadata=package_workspace["metadata"],
        public_key_src=pub_key_path,
        private_key_src=priv_key_path,
        require_signing=True,
        enable_screening=False,
    )

    # All required artifacts exist
    for artifact in REQUIRED_ARTEFACTS:
        assert (pkg_dir / artifact).exists(), f"Missing required artifact: {artifact}"
        assert (pkg_dir / artifact).stat().st_size > 0

    assert (pkg_dir / "package_manifest.json").exists()

    # Provenance validation passes strictly
    validated_manifest, verified_digest = validate_package_provenance(pkg_dir)
    assert validated_manifest["package_id"] == manifest["package_id"]
    assert verified_digest == package_workspace["ciphertext_hash"]
    assert verified_digest == manifest["encrypted_adapter_digest"]


def test_missing_signing_key_fails_when_signing_mandatory(package_workspace):
    """Requirement 2: Missing or unusable signing key fails closed when signing is mandatory."""
    pkg_dir = package_workspace["pkg_dir"]
    pub_key_path = package_workspace["pub_key_path"]

    # 1. require_signing=True but private_key_src=None
    with pytest.raises(SigningKeyError, match="mandatory, but no private signing key"):
        build_package(
            package_dir=pkg_dir,
            public_key_src=pub_key_path,
            private_key_src=None,
            require_signing=True,
            enable_screening=False,
        )

    # 2. require_signing=True with non-existent private key file
    with pytest.raises(SigningKeyError, match="does not exist"):
        build_package(
            package_dir=pkg_dir,
            public_key_src=pub_key_path,
            private_key_src=pkg_dir / "missing_priv.pem",
            require_signing=True,
            enable_screening=False,
        )

    # 3. Corrupt/unusable private key file
    corrupt_key = pkg_dir / "corrupt_key.pem"
    corrupt_key.write_text("---NOT A VALID RSA PRIVATE KEY---")
    with pytest.raises(SigningKeyError, match="invalid or unusable"):
        build_package(
            package_dir=pkg_dir,
            public_key_src=pub_key_path,
            private_key_src=corrupt_key,
            require_signing=True,
            enable_screening=False,
        )


def test_modified_manifest_ciphertext_or_artifact_hashes_rejected(package_workspace):
    """Requirement 3: Any tampering with manifest, ciphertext, or hashes is strictly rejected."""
    pkg_dir = package_workspace["pkg_dir"]
    priv_key_path = package_workspace["priv_key_path"]
    pub_key_path = package_workspace["pub_key_path"]

    # Build valid package
    build_package(
        package_dir=pkg_dir,
        adapter_id="medical-lora-v1",
        model_reference="distilbert-base-uncased",
        fingerprint_hash="a" * 64,
        package_version="1.0.0",
        enc_metadata=package_workspace["metadata"],
        public_key_src=pub_key_path,
        private_key_src=priv_key_path,
        require_signing=True,
        enable_screening=False,
    )

    # Test A: Modifying security field in manifest invalidates canonical signature
    manifest_path = pkg_dir / "package_manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["adapter_id"] = "malicious-tampered-id"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    with pytest.raises(SignatureValidationError):
        validate_package_provenance(pkg_dir)

    # Restore manifest
    manifest_data["adapter_id"] = "medical-lora-v1"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    # Test B: Modifying ciphertext causes hash mismatch
    enc_path = pkg_dir / "adapter.enc"
    enc_path.write_bytes(enc_path.read_bytes() + b"\x00tamper")
    with pytest.raises(IntegrityValidationError):
        validate_package_provenance(pkg_dir)


def test_package_metadata_and_signature_correspond_to_same_final_manifest(package_workspace):
    """Requirement 4: Package metadata and signature authenticate the exact same final manifest."""
    pkg_dir = package_workspace["pkg_dir"]
    priv_key_path = package_workspace["priv_key_path"]
    pub_key_path = package_workspace["pub_key_path"]

    manifest = build_package(
        package_dir=pkg_dir,
        adapter_id="audit-adapter",
        model_reference="audit-model",
        fingerprint_hash="b" * 64,
        package_version="1.0.0",
        enc_metadata=package_workspace["metadata"],
        public_key_src=pub_key_path,
        private_key_src=priv_key_path,
        require_signing=True,
        enable_screening=False,
    )

    # Verify on-disk manifest equals returned manifest
    disk_manifest = json.loads((pkg_dir / "package_manifest.json").read_text())
    assert disk_manifest["package_id"] == manifest["package_id"]
    assert disk_manifest["sequence_number"] == manifest["sequence_number"]

    # Compute canonical digest independently
    ciphertext_digest = compute_sha256(pkg_dir / "adapter.enc")
    canonical_digest = compute_canonical_manifest_digest(disk_manifest, ciphertext_digest)

    # Verify the signature on disk specifically authenticates this canonical digest
    from src.security.signature import verify_signature
    verify_signature(canonical_digest, pkg_dir / "adapter.sig", pkg_dir / "public.pem")


def test_secret_and_private_keys_and_plaintext_excluded_from_export(package_workspace, tmp_path):
    """Requirement 5: Private keys, secrets, and plaintext weights are excluded from exported archives."""
    pkg_dir = package_workspace["pkg_dir"]
    priv_key_path = package_workspace["priv_key_path"]
    pub_key_path = package_workspace["pub_key_path"]

    # Place sensitive secret and plaintext files in pkg_dir
    (pkg_dir / "dev_private.pem").write_text("DUMMY PRIVATE KEY CONTENT")
    (pkg_dir / "signing.key").write_text("SECRET KEY")
    (pkg_dir / "device.secret").write_bytes(os.urandom(32))
    (pkg_dir / "extra.secret").write_text("EXTRA SECRET")
    (pkg_dir / "adapter_model.safetensors").write_bytes(b"PLAINTEXT SAFETENSORS WEIGHTS")
    (pkg_dir / "adapter_model.bin").write_bytes(b"PLAINTEXT BIN WEIGHTS")
    (pkg_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')

    # Build package (cleans sensitive files from pkg_dir)
    build_package(
        package_dir=pkg_dir,
        public_key_src=pub_key_path,
        private_key_src=priv_key_path,
        require_signing=True,
        enable_screening=False,
    )

    # Export archive
    archive_path = export_package_archive(pkg_dir, tmp_path / "test_export.tar.gz")
    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as tar:
        member_names = [Path(m.name).name.lower() for m in tar.getmembers()]

    # Sensitive files MUST be excluded
    forbidden = [
        "dev_private.pem",
        "signing.key",
        "device.secret",
        "extra.secret",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_config.json",
    ]
    for forbidden_file in forbidden:
        assert forbidden_file not in member_names, f"Forbidden file found in archive: {forbidden_file}"

    # Required files MUST be included
    for req in REQUIRED_ARTEFACTS:
        assert req in member_names, f"Required artifact missing from archive: {req}"


def test_incomplete_package_creation_does_not_report_success(package_workspace):
    """Requirement 6: Incomplete or corrupted package creation fails closed and does not report success."""
    pkg_dir = package_workspace["pkg_dir"]
    pub_key_path = package_workspace["pub_key_path"]
    priv_key_path = package_workspace["priv_key_path"]

    # 1. Missing adapter.enc
    (pkg_dir / "adapter.enc").unlink()
    with pytest.raises(IncompletePackageError):
        build_package(
            package_dir=pkg_dir,
            public_key_src=pub_key_path,
            private_key_src=priv_key_path,
            enable_screening=False,
        )

    # Restore adapter.enc, make adapter.hash empty
    (pkg_dir / "adapter.enc").write_bytes(b"some ciphertext")
    (pkg_dir / "adapter.hash").write_text("")
    with pytest.raises(PackageArtifactError, match="empty"):
        build_package(
            package_dir=pkg_dir,
            public_key_src=pub_key_path,
            private_key_src=priv_key_path,
            enable_screening=False,
        )
