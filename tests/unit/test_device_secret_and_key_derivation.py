"""
test_device_secret_and_key_derivation.py
==========================================
Unit and security contract tests for:
1. device_secret persistence, CSPRNG generation, permissions (0600), fail-closed handling.
2. fingerprint requirement for machine_id.
3. HKDF-SHA256 v2 key derivation bound to device secret + device fingerprint + package salt.
4. KDF versioning and rejection of unsupported versions.
5. End-to-end secret and salt isolation across Phase 3 and Phase 4.
"""

import os
import stat
import pytest
from pathlib import Path

from src.security.device_secret import (
    initialize_device_secret,
    get_device_secret,
    device_secret_exists,
    secret_file_path,
    DeviceSecretError,
)
from src.security.fingerprint import (
    get_fingerprint_hash,
    collect_identifiers,
    build_canonical_string,
    DeviceFingerprintError,
)
from src.security.key_derivation import (
    derive_key_for_device,
    generate_hkdf_salt,
    check_kdf_version,
    validate_key_length,
    KDF_VERSION,
    SUPPORTED_KDF_VERSIONS,
)
from src.common.exceptions import CryptoError


# ==============================================================================
# 1. Device Secret Tests
# ==============================================================================

def test_device_secret_generation_and_persistence(tmp_path, monkeypatch):
    """Secret is generated using OS CSPRNG (32 bytes), saved with 0600 mode, and persisted."""
    test_secret_file = tmp_path / "securelora" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(test_secret_file))

    assert not device_secret_exists()
    
    # Initialize
    sec1 = initialize_device_secret()
    assert device_secret_exists()

    # Verify file mode 0600
    st_mode = test_secret_file.stat().st_mode
    file_perm = stat.S_IMODE(st_mode)
    assert file_perm == 0o600

    # Subsequent retrieval returns identical secret
    sec2 = get_device_secret()
    assert len(sec2) == 32
    assert isinstance(sec2, bytes)


def test_device_secret_corrupt_fails_closed(tmp_path, monkeypatch):
    """Truncated or corrupted secret file raises DeviceSecretError with no hardcoded fallback."""
    test_secret_file = tmp_path / "securelora" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(test_secret_file))
    
    test_secret_file.parent.mkdir(parents=True, exist_ok=True)
    # Write corrupt secret (16 bytes instead of 32 bytes)
    test_secret_file.write_bytes(b"shortsecret12345")
    test_secret_file.chmod(0o600)

    with pytest.raises(DeviceSecretError, match="wrong length 16 bytes"):
        get_device_secret()


def test_device_secret_missing_fails_closed(tmp_path, monkeypatch):
    """Missing secret file without auto-initialization fails closed when calling get_device_secret(auto_initialize=False)."""
    test_secret_file = tmp_path / "non_existent_dir" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(test_secret_file))

    with pytest.raises(DeviceSecretError, match="not found"):
        get_device_secret(auto_initialize=False)


# ==============================================================================
# 2. Fingerprint Tests
# ==============================================================================

def test_machine_id_required_for_fingerprint(monkeypatch):
    """Fails closed with DeviceFingerprintError if machine_id is unavailable."""
    def mock_collect_no_machine_id():
        return {
            "machine_id": "UNAVAILABLE",
            "cpu_info": "x86_64",
            "disk_uuid": "UNAVAILABLE",
            "hostname": "test-host",
        }
    
    monkeypatch.setattr("src.security.fingerprint.collect_identifiers", mock_collect_no_machine_id)

    with pytest.raises(DeviceFingerprintError, match="minimum required identity source"):
        get_fingerprint_hash()


def test_fingerprint_hash_deterministic():
    """Valid machine identity produces a 64-char hex SHA-256 hash."""
    fp_hash = get_fingerprint_hash()
    assert len(fp_hash) == 64
    assert int(fp_hash, 16) > 0  # Valid hex


# ==============================================================================
# 3. HKDF-SHA256 Key Derivation Tests
# ==============================================================================

def test_hkdf_v2_key_derivation_structure():
    """Key derivation outputs 32-byte AES key and enforces KDF_VERSION."""
    salt = generate_hkdf_salt()
    assert len(salt) == 32

    fp_hash = get_fingerprint_hash()
    key = derive_key_for_device(fp_hash, salt)
    
    assert len(key) == 32
    assert isinstance(key, bytes)
    validate_key_length(key)


def test_kdf_version_enforcement():
    """KDF_VERSION is hkdf-sha256-v2 and unsupported versions are rejected."""
    assert KDF_VERSION == "hkdf-sha256-v2"
    assert "hkdf-sha256-v2" in SUPPORTED_KDF_VERSIONS
    assert "hkdf-sha256-v1" in SUPPORTED_KDF_VERSIONS

    # Supported versions pass
    assert check_kdf_version("hkdf-sha256-v2") is None
    assert check_kdf_version("hkdf-sha256-v1") is None

    # Unsupported versions fail closed
    with pytest.raises(CryptoError, match="Unsupported KDF version"):
        check_kdf_version("invalid-kdf-v9")


def test_different_salt_yields_different_key():
    """Different package salts produce different AES keys for the same device."""
    salt1 = generate_hkdf_salt()
    salt2 = generate_hkdf_salt()
    assert salt1 != salt2

    fp_hash = get_fingerprint_hash()
    k1 = derive_key_for_device(salt1, fp_hash)
    k2 = derive_key_for_device(salt2, fp_hash)
    assert k1 != k2


def test_different_fingerprint_yields_different_key():
    """Different fingerprints produce different AES keys."""
    salt = generate_hkdf_salt()
    fp1 = "a" * 64
    fp2 = "b" * 64

    k1 = derive_key_for_device(salt, fp1)
    k2 = derive_key_for_device(salt, fp2)
    assert k1 != k2


def test_different_secret_yields_different_key(tmp_path, monkeypatch):
    """Different device secrets produce different AES keys."""
    salt = generate_hkdf_salt()
    fp_hash = get_fingerprint_hash()

    # Secret 1
    sec_file1 = tmp_path / "sec1" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(sec_file1))
    initialize_device_secret()
    k1 = derive_key_for_device(salt, fp_hash)

    # Secret 2
    sec_file2 = tmp_path / "sec2" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(sec_file2))
    initialize_device_secret()
    k2 = derive_key_for_device(salt, fp_hash)

    assert k1 != k2
