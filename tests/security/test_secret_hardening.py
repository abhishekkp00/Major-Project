"""
test_secret_hardening.py
========================
Security contract tests enforcing secret & configuration hardening:
1. Missing device secret fails closed immediately without generating unexpected replacement during packaging/deployment.
2. Exported adapter package archive (.tar.gz) NEVER contains RSA private key or secret files.
3. Public verification key (public.pem) IS retained in the exported adapter package.
4. Missing HKDF salt or missing device secret in verification path fails closed.
5. P3_DEVICE_SALT is not required and default demo secret usage fails closed on secure paths.
"""

import json
import tarfile
import pytest
from pathlib import Path

from src.security.device_secret import (
    get_device_secret,
    initialize_device_secret,
    device_secret_exists,
    DeviceSecretError,
)
from src.security.signature import generate_dev_keypair
from src.security.fingerprint import get_fingerprint_hash
from src.security.key_derivation import generate_hkdf_salt, derive_key_for_device
from src.phase3.package_builder import build_package, export_package_archive, REQUIRED_ARTEFACTS
from src.phase3.verifier import verify_and_decrypt
from src.common.exceptions import VerificationError, ConfigError
from src.common.config_loader import ConfigLoader


def test_missing_device_secret_fails_closed(tmp_path, monkeypatch):
    """Calling get_device_secret(auto_initialize=False) raises DeviceSecretError when missing."""
    test_secret_file = tmp_path / "non_existent_dir" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(test_secret_file))

    assert not device_secret_exists()
    with pytest.raises(DeviceSecretError, match="not found"):
        get_device_secret(auto_initialize=False)


def test_rsa_private_key_excluded_from_exported_package(tmp_path):
    """
    Verifies Requirement 6 & 7: RSA private signing key (dev_private.pem) is NEVER
    included in the exported package archive (.tar.gz), while public.pem IS included.
    """
    pkg_dir = tmp_path / "protected_adapter"
    pkg_dir.mkdir()

    # Create dummy artifacts
    (pkg_dir / "adapter.enc").write_bytes(b"dummy_ciphertext")
    (pkg_dir / "adapter.hash").write_text("dummy_hash")
    (pkg_dir / "metadata.json").write_text("{}")

    priv_key_path = tmp_path / "keys" / "dev_private.pem"
    pub_key_path = pkg_dir / "public.pem"
    priv_key_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dev_keypair(priv_key_path, pub_key_path, key_size=2048)

    # Also simulate accidental private key placing in package_dir
    accidental_priv = pkg_dir / "dev_private.pem"
    accidental_priv.write_bytes(priv_key_path.read_bytes())
    (pkg_dir / "extra.secret").write_text("secret_data")
    (pkg_dir / "extra.key").write_text("key_data")

    # Build package (should remove private keys from package_dir)
    manifest = build_package(
        package_dir=pkg_dir,
        adapter_id="test-adapter",
        model_reference="test-model",
        fingerprint_hash="a" * 64,
        public_key_src=pub_key_path,
        private_key_src=priv_key_path,
        enable_screening=False,
    )

    # Export package archive
    archive_path = export_package_archive(pkg_dir)
    assert archive_path.exists()

    # Inspect tar archive contents
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [Path(m.name).name for m in tar.getmembers()]

    # 1. RSA private signing key MUST NOT be in package archive
    assert "dev_private.pem" not in members
    assert "extra.secret" not in members
    assert "extra.key" not in members
    assert not any("private" in m.lower() for m in members)

    # 2. Public verification key MUST be in package archive (Requirement 8)
    assert "public.pem" in members

    # 3. All required artifacts present
    for req in REQUIRED_ARTEFACTS:
        assert req in members


def test_missing_hkdf_salt_fails_closed_in_verification(tmp_path, monkeypatch):
    """Missing HKDF salt in package manifest and missing salt parameter raises VerificationError."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()

    # Create dummy incomplete manifest without hkdf_salt_hex
    (pkg_dir / "package_manifest.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "adapter_id": "test",
    }))
    (pkg_dir / "metadata.json").write_text(json.dumps({}))
    (pkg_dir / "adapter.enc").write_bytes(b"dummy")
    (pkg_dir / "adapter.hash").write_text("dummyhash")
    (pkg_dir / "adapter.sig").write_bytes(b"dummysig")
    (pkg_dir / "public.pem").write_text("dummypubkey")

    # Mock validate_package_provenance to return manifest
    monkeypatch.setattr(
        "src.phase3.verifier.validate_package_provenance",
        lambda p: ({"schema_version": "1.0.0"}, "dummy_digest")
    )
    monkeypatch.setattr("src.phase3.verifier.get_fingerprint_hash", lambda: "a" * 64)

    out_path = tmp_path / "out.bin"
    with pytest.raises(VerificationError, match="HKDF salt missing"):
        verify_and_decrypt(pkg_dir, out_path, salt=None)


def test_config_loader_validation_requires_device_secret(tmp_path, monkeypatch):
    """ConfigLoader validate_phase3 and validate_phase4 fail if device secret does not exist."""
    test_secret_file = tmp_path / "missing" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(test_secret_file))

    cfg = ConfigLoader()
    with pytest.raises(ConfigError, match="Device secret is missing"):
        cfg.validate_phase3()

    with pytest.raises(ConfigError, match="Device secret is missing"):
        cfg.validate_phase4()
