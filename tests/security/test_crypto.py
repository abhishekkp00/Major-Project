import hashlib
import os
import json
from pathlib import Path
import pytest

from src.security import (
    encrypt_adapter,
    decrypt_adapter,
    derive_key,
    derive_key_from_env,
    validate_key_length,
    verify_integrity,
    save_hash,
    compute_sha256,
)

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path

@pytest.fixture()
def salt() -> str:
    return "phase3-test-salt-do-not-use-in-prod"

@pytest.fixture()
def dummy_adapter_file(tmp_dir: Path) -> Path:
    f = tmp_dir / "adapter.bin"
    f.write_bytes(os.urandom(1024))
    return f

@pytest.fixture()
def dummy_adapter_dir(tmp_dir: Path) -> Path:
    ad = tmp_dir / "final_adapter"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(os.urandom(512))
    (ad / "adapter_config.json").write_text(json.dumps({"base_model": "llama", "r": 8}))
    return ad


class TestKeyDerivation:
    def test_derive_key_returns_32_bytes(self, salt):
        k = derive_key("abc123" * 10, salt)
        assert len(k) == 32

    def test_derive_key_is_deterministic(self, salt):
        fp = "deadbeef" * 8
        k1 = derive_key(fp, salt)
        k2 = derive_key(fp, salt)
        assert k1 == k2

    def test_different_salt_produces_different_key(self):
        fp = "deadbeef" * 8
        k1 = derive_key(fp, "salt-one")
        k2 = derive_key(fp, "salt-two")
        assert k1 != k2

    def test_different_fingerprint_produces_different_key(self, salt):
        k1 = derive_key("aaaa" * 16, salt)
        k2 = derive_key("bbbb" * 16, salt)
        assert k1 != k2

    def test_empty_salt_raises(self):
        with pytest.raises(ValueError, match="salt must not be empty"):
            derive_key("deadbeef" * 8, "")

    def test_empty_fingerprint_raises(self, salt):
        with pytest.raises(ValueError, match="fingerprint_hash must not be empty"):
            derive_key("", salt)

    def test_validate_key_length_passes_for_32(self):
        validate_key_length(b"x" * 32)

    def test_validate_key_length_fails_for_wrong_size(self):
        with pytest.raises(ValueError):
            validate_key_length(b"x" * 16)


class TestEncryption:
    def _make_key(self) -> bytes:
        return hashlib.sha256(b"test-key-phase3").digest()

    def test_encrypt_single_file(self, tmp_dir, dummy_adapter_file):
        key = self._make_key()
        enc_path = tmp_dir / "adapter.enc"
        meta = encrypt_adapter(
            adapter_input=dummy_adapter_file,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash="fp" * 32,
            metadata_path=tmp_dir / "metadata.json",
        )
        assert enc_path.exists()
        assert enc_path.stat().st_size > 0
        assert meta["algorithm"] == "AES-256-GCM"
        assert "nonce_hex" in meta

    def test_encrypt_directory(self, tmp_dir, dummy_adapter_dir):
        key = self._make_key()
        enc_path = tmp_dir / "adapter.enc"
        meta = encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash="fp" * 32,
        )
        assert enc_path.exists()
        assert meta["algorithm"] == "AES-256-GCM"

    def test_decrypt_adapter_success(self, tmp_dir, dummy_adapter_dir):
        key = self._make_key()
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash="fp" * 32,
        )

        dec_tar_path = tmp_dir / "restored.tar.gz"
        decrypt_adapter(enc_path, dec_tar_path, key)
        assert dec_tar_path.exists()

    def test_decrypt_adapter_wrong_key_raises(self, tmp_dir, dummy_adapter_dir):
        key = self._make_key()
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash="fp" * 32,
        )

        dec_tar_path = tmp_dir / "restored.tar.gz"
        wrong_key = b"y" * 32
        with pytest.raises(ValueError, match="decryption failed"):
            decrypt_adapter(enc_path, dec_tar_path, wrong_key)
