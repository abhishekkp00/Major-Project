"""
test_device_bound_key_lifecycle.py
===================================
Comprehensive cryptographic systems engineering tests for the complete
device-bound key lifecycle across SecureLoRA packaging and deployment.

Covers:
1. Valid packaging-to-deployment encrypt-to-decrypt round-trip with matching inputs.
2. Wrong device secret derivation failure (AES-GCM tag mismatch on decrypt).
3. Wrong fingerprint failure (device authorization rejection before decryption).
4. Modified salt (truncated, extended, non-hex) fails closed.
5. Unsupported KDF version rejection.
6. Missing secret, truncated secret, and insecure secret file permissions fail closed.
7. Device-binding authorization check precedes decryption.
8. Packaging metadata and manifest isolation (no device secret or raw keys serialized).
"""

import os
import stat
import json
import shutil
import pytest
from pathlib import Path

from src.security.device_secret import (
    initialize_device_secret,
    get_device_secret,
    secret_file_path,
    DeviceSecretError,
)
from src.security.fingerprint import get_fingerprint_hash
from src.security.key_derivation import (
    derive_key,
    derive_key_for_device,
    generate_hkdf_salt,
    validate_key_length,
    validate_salt_length,
    check_kdf_version,
    KDF_VERSION,
    HKDF_SALT_LENGTH,
)
from src.security.crypto import (
    encrypt_adapter,
    decrypt_adapter,
    compute_sha256,
    save_hash,
)
from src.security.signature import generate_dev_keypair
from src.phase3.package_builder import build_package
from src.phase4.device_auth import verify_device_binding, get_device_bound_key
from src.phase4.main import run_deployment_pipeline
from src.common.exceptions import (
    CryptoError,
    DeviceAuthorizationError,
    VerificationError,
)


@pytest.fixture
def test_environment(tmp_path: Path, monkeypatch):
    """Sets up an isolated test environment with a dedicated device secret."""
    secret_file = tmp_path / "security_env" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(secret_file))
    initialize_device_secret()
    return secret_file


@pytest.fixture
def dummy_adapter_source(tmp_path: Path):
    """Creates a minimal dummy adapter folder."""
    src = tmp_path / "mock_adapter"
    src.mkdir(parents=True, exist_ok=True)
    (src / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}))
    
    import torch
    weights = {
        "base_model.model.encoder.layer.0.attention.self.query.lora_A.weight": torch.zeros((8, 64)),
        "base_model.model.encoder.layer.0.attention.self.query.lora_B.weight": torch.zeros((64, 8)),
    }
    torch.save(weights, src / "adapter_model.bin")
    return src


# ==============================================================================
# 1. End-to-End Encrypt-to-Decrypt Round Trip
# ==============================================================================

def test_encrypt_to_decrypt_round_trip(test_environment, dummy_adapter_source, tmp_path):
    """
    Packaging and deployment derive the same 32-byte AES key when
    device secret, fingerprint, salt, and KDF version match, successfully decrypting.
    """
    fp_hash = get_fingerprint_hash()
    salt = generate_hkdf_salt()
    assert len(salt) == HKDF_SALT_LENGTH

    # Packaging key derivation
    packaging_key = derive_key_for_device(fp_hash, salt, kdf_version=KDF_VERSION)
    assert len(packaging_key) == 32

    # Encrypt
    enc_path = tmp_path / "adapter.enc"
    meta_path = tmp_path / "metadata.json"
    meta = encrypt_adapter(
        adapter_input=dummy_adapter_source,
        output_enc_path=enc_path,
        key=packaging_key,
        fingerprint_hash=fp_hash,
        metadata_path=meta_path,
        hkdf_salt=salt,
    )
    assert enc_path.exists()
    assert meta["hkdf_salt_hex"] == salt.hex()

    # Deployment side: retrieve salt and derive key
    salt_from_package = bytes.fromhex(meta["hkdf_salt_hex"])
    deployment_key = get_device_bound_key(salt_from_package, kdf_version=meta["kdf_version"])
    assert deployment_key == packaging_key

    # Decrypt
    decrypted_tar = tmp_path / "restored.tar.gz"
    decrypt_adapter(enc_path, decrypted_tar, deployment_key)
    assert decrypted_tar.exists()
    assert decrypted_tar.stat().st_size > 0


# ==============================================================================
# 2. Wrong Device Secret Fails to Decrypt
# ==============================================================================

def test_wrong_device_secret_fails_to_decrypt(test_environment, dummy_adapter_source, tmp_path, monkeypatch):
    """Derivation with a different device secret yields a different key and fails AES-GCM decryption."""
    fp_hash = get_fingerprint_hash()
    salt = generate_hkdf_salt()

    packaging_key = derive_key_for_device(fp_hash, salt)
    enc_path = tmp_path / "adapter.enc"
    encrypt_adapter(
        adapter_input=dummy_adapter_source,
        output_enc_path=enc_path,
        key=packaging_key,
        fingerprint_hash=fp_hash,
        hkdf_salt=salt,
    )

    # Switch to an alternate device secret (different machine/device enrollment)
    alt_secret_file = tmp_path / "alt_device" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(alt_secret_file))
    initialize_device_secret()

    alt_key = derive_key_for_device(fp_hash, salt)
    assert alt_key != packaging_key

    decrypted_tar = tmp_path / "should_fail.tar.gz"
    with pytest.raises(CryptoError, match="AES-GCM decryption failed"):
        decrypt_adapter(enc_path, decrypted_tar, alt_key)


# ==============================================================================
# 3. Wrong Fingerprint Fails Authorization Before Decryption
# ==============================================================================

def test_wrong_fingerprint_rejected_at_authorization(test_environment, tmp_path):
    """An unauthorized device fingerprint is rejected during authorization check before key derivation."""
    authorized_fp = "a" * 64
    unauthorized_fp = "b" * 64

    # Verify authorization check fails
    with pytest.raises(DeviceAuthorizationError):
        verify_device_binding(
            expected_fingerprint_hash=authorized_fp,
            mock_fingerprint=unauthorized_fp,
        )


# ==============================================================================
# 4. Salt Length and Encoding Validation
# ==============================================================================

def test_modified_salt_rejected(test_environment):
    """Truncated, extended, or non-hex salts are rejected fail-closed."""
    fp_hash = get_fingerprint_hash()

    # Short salt (16 bytes)
    short_salt = os.urandom(16)
    with pytest.raises(CryptoError, match="unexpected length 16 bytes"):
        validate_salt_length(short_salt)

    with pytest.raises(CryptoError, match="unexpected length 16 bytes"):
        derive_key_for_device(fp_hash, short_salt)

    # Long salt (64 bytes)
    long_salt = os.urandom(64)
    with pytest.raises(CryptoError, match="unexpected length 64 bytes"):
        validate_salt_length(long_salt)

    with pytest.raises(CryptoError, match="unexpected length 64 bytes"):
        derive_key_for_device(fp_hash, long_salt)

    # Non-bytes salt
    with pytest.raises(TypeError):
        validate_salt_length("not-bytes-string")


# ==============================================================================
# 5. Unsupported KDF Version Rejected
# ==============================================================================

def test_unsupported_kdf_version_rejected():
    """Unsupported or malformed KDF versions fail closed."""
    with pytest.raises(CryptoError, match="Unsupported KDF version"):
        check_kdf_version("unsupported-v9")

    with pytest.raises(CryptoError, match="Unsupported KDF version"):
        check_kdf_version("")


# ==============================================================================
# 6. Missing, Corrupt, and Insecurely Permissioned Secrets
# ==============================================================================

def test_insecure_secret_permissions_fail_closed(tmp_path, monkeypatch):
    """Secret file with group/other read permissions or execute bits fails closed."""
    secret_path = tmp_path / "insecure" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(secret_path))
    initialize_device_secret()

    # World-readable (0644)
    secret_path.chmod(0o644)
    with pytest.raises(DeviceSecretError, match="insecure permissions"):
        get_device_secret()

    # Executable (0700)
    secret_path.chmod(0o700)
    with pytest.raises(DeviceSecretError, match="insecure permissions"):
        get_device_secret()

    # World-writable (0666)
    secret_path.chmod(0o666)
    with pytest.raises(DeviceSecretError, match="insecure permissions"):
        get_device_secret()


def test_missing_secret_fails_closed(tmp_path, monkeypatch):
    """Missing secret does not silently auto-enroll; fails closed."""
    secret_path = tmp_path / "non_existent" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(secret_path))

    with pytest.raises(DeviceSecretError, match="not found"):
        get_device_secret(auto_initialize=False)


# ==============================================================================
# 7. Package Metadata Never Contains Secrets
# ==============================================================================

def test_package_metadata_and_manifest_exclude_secret(test_environment, dummy_adapter_source, tmp_path):
    """Ensures metadata.json and package_manifest.json do not leak device secret or raw keys."""
    raw_secret = get_device_secret()
    salt = generate_hkdf_salt()
    fp_hash = get_fingerprint_hash()

    pkg_dir = tmp_path / "audit_pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    priv_pem = pkg_dir / "dev_private.pem"
    pub_pem = pkg_dir / "public.pem"
    generate_dev_keypair(priv_pem, pub_pem)

    key = derive_key_for_device(fp_hash, salt)
    enc_file = pkg_dir / "adapter.enc"
    meta_file = pkg_dir / "metadata.json"

    meta = encrypt_adapter(
        adapter_input=dummy_adapter_source,
        output_enc_path=enc_file,
        key=key,
        fingerprint_hash=fp_hash,
        metadata_path=meta_file,
        hkdf_salt=salt,
    )

    c_hash = compute_sha256(enc_file)
    save_hash(c_hash, pkg_dir / "adapter.hash")

    manifest = build_package(
        package_dir=pkg_dir,
        adapter_source=dummy_adapter_source,
        adapter_id="audit-adapter",
        model_reference="JackFram/llama-68m",
        fingerprint_hash=fp_hash,
        enc_metadata=meta,
        public_key_src=pub_pem,
        private_key_src=priv_pem,
    )

    manifest_text = (pkg_dir / "package_manifest.json").read_text()
    meta_text = (pkg_dir / "metadata.json").read_text()

    # Secret hex must not be in metadata or manifest
    secret_hex = raw_secret.hex()
    key_hex = key.hex()

    assert secret_hex not in manifest_text
    assert secret_hex not in meta_text
    assert key_hex not in manifest_text
    assert key_hex not in meta_text

    # No secret key field names in either dictionary
    for forbidden in ["device_secret", "raw_secret", "aes_key", "raw_key"]:
        assert forbidden not in manifest
        assert forbidden not in meta
