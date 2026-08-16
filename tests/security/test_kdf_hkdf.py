"""
test_kdf_hkdf.py
================
Tests for the HKDF-SHA256 key derivation scheme (requirements A–G from audit).

Coverage:
  1.  Same fingerprint + same salt → same key (determinism)
  2.  Different fingerprint → different key
  3.  Different salt → different key
  4.  Wrong KDF version → CryptoError rejection
  5.  Malformed / empty fingerprint → ValueError rejection
  6.  Tampered ciphertext → CryptoError via AES-GCM auth tag failure
  7.  Wrong device (different fingerprint) → different key → decryption failure
  8.  Temporary plaintext files cleaned up after successful decryption
  9.  Temporary plaintext files cleaned up after decryption exception
  10. Package metadata correctly records KDF version
  11. derive_key output is 32 bytes
  12. KDF_VERSION constant is the expected string
  13. check_kdf_version passes for the current version
  14. check_kdf_version rejects unknown versions
  15. derive_key_from_env reads salt from environment
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.security.key_derivation import (
    KDF_VERSION,
    check_kdf_version,
    derive_key,
    derive_key_from_env,
    validate_key_length,
)
from src.security.crypto import encrypt_adapter, decrypt_adapter
from src.phase4.decryptor import DecryptedAdapterContext
from src.common.exceptions import CryptoError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SALT_A = "test-salt-alpha-do-not-use-in-prod"
SALT_B = "test-salt-beta-different"
FP_A = "a" * 64   # SHA-256 hex digest is 64 chars
FP_B = "b" * 64


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def dummy_adapter_dir(tmp_dir: Path) -> Path:
    """Minimal PEFT adapter directory structure."""
    ad = tmp_dir / "final_adapter"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(os.urandom(512))
    (ad / "adapter_config.json").write_text(
        json.dumps({"base_model": "llama", "r": 8})
    )
    return ad


# ─────────────────────────────────────────────────────────────────────────────
# 1–3: Key derivation determinism and differentiation
# ─────────────────────────────────────────────────────────────────────────────

class TestHkdfKeyDerivationCoreProperties:

    def test_same_fingerprint_same_salt_same_key(self):
        """Requirement 1: same IKM + same salt → same derived key."""
        k1 = derive_key(FP_A, SALT_A)
        k2 = derive_key(FP_A, SALT_A)
        assert k1 == k2, "HKDF must be deterministic for identical inputs."

    def test_different_fingerprint_different_key(self):
        """Requirement 2: different fingerprint → different derived key."""
        k1 = derive_key(FP_A, SALT_A)
        k2 = derive_key(FP_B, SALT_A)
        assert k1 != k2, "Different fingerprints must produce different keys."

    def test_different_salt_different_key(self):
        """Requirement 3: same fingerprint, different salt → different key."""
        k1 = derive_key(FP_A, SALT_A)
        k2 = derive_key(FP_A, SALT_B)
        assert k1 != k2, "Different salts must produce different keys."

    def test_derived_key_is_32_bytes(self):
        """Requirement 11: output must be exactly 32 bytes for AES-256."""
        k = derive_key(FP_A, SALT_A)
        assert len(k) == 32

    def test_validate_key_length_passes(self):
        validate_key_length(b"x" * 32)

    def test_validate_key_length_rejects_short_key(self):
        with pytest.raises((ValueError, CryptoError)):
            validate_key_length(b"x" * 16)


# ─────────────────────────────────────────────────────────────────────────────
# 4: Wrong KDF version → rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestKdfVersionGating:

    def test_kdf_version_constant_value(self):
        """Requirement 12: KDF_VERSION must be the expected string."""
        assert KDF_VERSION == "hkdf-sha256-v1"

    def test_check_kdf_version_passes_for_current(self):
        """Requirement 13: current version passes without error."""
        check_kdf_version(KDF_VERSION)  # Must not raise.

    def test_check_kdf_version_rejects_unknown(self):
        """Requirement 4/14: unknown version → CryptoError."""
        with pytest.raises(CryptoError, match="Unsupported KDF version"):
            check_kdf_version("sha256-concat-v0")

    def test_check_kdf_version_rejects_empty_string(self):
        """Empty version string must also be rejected."""
        with pytest.raises(CryptoError, match="Unsupported KDF version"):
            check_kdf_version("")

    def test_check_kdf_version_rejects_none_like_string(self):
        """String 'None' is not a supported version."""
        with pytest.raises(CryptoError, match="Unsupported KDF version"):
            check_kdf_version("None")


# ─────────────────────────────────────────────────────────────────────────────
# 5: Malformed / empty fingerprint → rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:

    def test_empty_fingerprint_raises_value_error(self):
        """Requirement 5: empty fingerprint is rejected before HKDF runs."""
        with pytest.raises(ValueError, match="fingerprint_hash must not be empty"):
            derive_key("", SALT_A)

    def test_empty_salt_raises_value_error(self):
        """Empty salt is rejected (P3_DEVICE_SALT not set)."""
        with pytest.raises(ValueError, match="salt must not be empty"):
            derive_key(FP_A, "")

    def test_derive_key_from_env_reads_salt(self, monkeypatch):
        """Requirement 15: derive_key_from_env reads P3_DEVICE_SALT from env."""
        monkeypatch.setenv("P3_DEVICE_SALT", SALT_A)
        k_env = derive_key_from_env(FP_A)
        k_direct = derive_key(FP_A, SALT_A)
        assert k_env == k_direct

    def test_derive_key_from_env_empty_env_raises(self, monkeypatch):
        """derive_key_from_env raises if env var is not set."""
        monkeypatch.delenv("P3_DEVICE_SALT", raising=False)
        with pytest.raises(ValueError, match="salt must not be empty"):
            derive_key_from_env(FP_A, salt=None)


# ─────────────────────────────────────────────────────────────────────────────
# 6: Tampered ciphertext → CryptoError (AES-GCM auth tag failure)
# ─────────────────────────────────────────────────────────────────────────────

class TestTamperedCiphertextRejection:

    def test_tampered_ciphertext_raises(self, tmp_dir, dummy_adapter_dir):
        """Requirement 6: bit-flip in ciphertext triggers AES-GCM auth tag failure."""
        key = derive_key(FP_A, SALT_A)
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash=FP_A,
        )

        # Flip a byte in the middle of the ciphertext (after the 12-byte nonce).
        raw = bytearray(enc_path.read_bytes())
        raw[20] ^= 0xFF
        enc_path.write_bytes(bytes(raw))

        dec_path = tmp_dir / "restored.tar.gz"
        with pytest.raises((ValueError, CryptoError)):
            decrypt_adapter(enc_path, dec_path, key)


# ─────────────────────────────────────────────────────────────────────────────
# 7: Wrong device → different key → decryption failure
# ─────────────────────────────────────────────────────────────────────────────

class TestWrongDeviceRejection:

    def test_wrong_fingerprint_causes_decryption_failure(self, tmp_dir, dummy_adapter_dir):
        """Requirement 7: encrypting with FP_A and decrypting with FP_B must fail."""
        key_a = derive_key(FP_A, SALT_A)
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key_a,
            fingerprint_hash=FP_A,
        )

        key_b = derive_key(FP_B, SALT_A)
        assert key_a != key_b, "Keys must differ for different fingerprints."

        dec_path = tmp_dir / "restored.tar.gz"
        with pytest.raises((ValueError, CryptoError)):
            decrypt_adapter(enc_path, dec_path, key_b)


# ─────────────────────────────────────────────────────────────────────────────
# 8–9: Temporary plaintext cleanup (success and exception paths)
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporaryPlaintextCleanup:
    """
    Tests that DecryptedAdapterContext shreds all temporary plaintext files.
    Note: these tests verify file-system cleanup, not memory erasure.
    """

    def test_temp_files_cleaned_up_on_success(self, tmp_dir, dummy_adapter_dir):
        """Requirement 8: after successful context exit, temp dir is removed."""
        key = derive_key(FP_A, SALT_A)
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash=FP_A,
        )

        captured_temp_dir = None
        with DecryptedAdapterContext(enc_path, key) as adapter_dir:
            assert adapter_dir.exists(), "Adapter dir must exist inside context."
            captured_temp_dir = adapter_dir.parent  # The temp root

        # After exit, the temporary directory must be gone.
        assert not captured_temp_dir.exists(), (
            "Temporary plaintext directory must be removed after context exit."
        )

    def test_temp_files_cleaned_up_on_exception(self, tmp_dir, dummy_adapter_dir):
        """Requirement 9: temp dir is removed even if an exception is raised inside."""
        key = derive_key(FP_A, SALT_A)
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash=FP_A,
        )

        captured_temp_dir = None
        with pytest.raises(RuntimeError, match="deliberate test error"):
            with DecryptedAdapterContext(enc_path, key) as adapter_dir:
                captured_temp_dir = adapter_dir.parent
                raise RuntimeError("deliberate test error")

        assert not captured_temp_dir.exists(), (
            "Temporary plaintext directory must be removed even when an exception occurs."
        )

    def test_wrong_key_context_does_not_leave_temp_files(self, tmp_dir, dummy_adapter_dir):
        """Cleanup must run even when decryption itself fails (wrong key)."""
        key_good = derive_key(FP_A, SALT_A)
        key_bad = derive_key(FP_B, SALT_A)
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key_good,
            fingerprint_hash=FP_A,
        )

        with pytest.raises((ValueError, CryptoError)):
            with DecryptedAdapterContext(enc_path, key_bad):
                pass  # Should never reach here.

        # No lingering temp directories should be left.
        leftover = list(Path(tempfile.gettempdir()).glob("secure_lora_decrypted_*"))
        assert not leftover, f"Leftover temp dirs found: {leftover}"


# ─────────────────────────────────────────────────────────────────────────────
# 10: Package metadata records KDF version
# ─────────────────────────────────────────────────────────────────────────────

class TestPackageMetadataKdfVersion:

    def test_encrypt_adapter_metadata_includes_kdf_version(self, tmp_dir, dummy_adapter_dir):
        """Requirement 10: encrypt_adapter must record kdf_version in its metadata."""
        key = derive_key(FP_A, SALT_A)
        enc_path = tmp_dir / "adapter.enc"
        meta_path = tmp_dir / "metadata.json"

        meta = encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash=FP_A,
            metadata_path=meta_path,
        )

        assert "kdf_version" in meta, "metadata dict must contain 'kdf_version'."
        assert meta["kdf_version"] == KDF_VERSION

        # Also check the written JSON file.
        written = json.loads(meta_path.read_text())
        assert written.get("kdf_version") == KDF_VERSION
