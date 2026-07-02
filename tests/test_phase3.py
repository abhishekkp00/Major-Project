"""
tests/test_phase3.py
---------------------
Comprehensive test suite for Phase 3: Secure Adapter Protection and Device Binding.

Each test class corresponds to one implementation step.  Tests are designed to
be run independently with ``pytest`` and require no external services.

Run all Phase 3 tests:
    pytest tests/test_phase3.py -v

Run one step's tests only:
    pytest tests/test_phase3.py::TestStep2Fingerprint -v
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Returns an isolated temporary directory for each test."""
    return tmp_path


@pytest.fixture()
def salt() -> str:
    return "phase3-test-salt-do-not-use-in-prod"


@pytest.fixture()
def dummy_adapter_dir(tmp_dir: Path) -> Path:
    """Creates a fake PEFT adapter directory with two files."""
    ad = tmp_dir / "final_adapter"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(os.urandom(512))
    (ad / "adapter_config.json").write_text(json.dumps({"base_model": "llama", "r": 8}))
    return ad


@pytest.fixture()
def dummy_adapter_file(tmp_dir: Path) -> Path:
    """Creates a single fake adapter .bin file."""
    f = tmp_dir / "adapter.bin"
    f.write_bytes(os.urandom(1024))
    return f


# ---------------------------------------------------------------------------
# Step 2 — Device Fingerprinting
# ---------------------------------------------------------------------------

class TestStep2Fingerprint:
    """Validates device identity generation (phase3/device_fingerprint.py)."""

    def test_fingerprint_hash_is_64_hex_chars(self):
        from phase3.device_fingerprint import get_fingerprint_hash
        h = get_fingerprint_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_fingerprint_is_reproducible(self):
        """Same machine → same hash on repeated calls."""
        from phase3.device_fingerprint import get_fingerprint_hash
        h1 = get_fingerprint_hash()
        h2 = get_fingerprint_hash()
        assert h1 == h2, "Fingerprint must be deterministic on the same machine."

    def test_fingerprint_changes_with_different_identifiers(self):
        from phase3.device_fingerprint import build_canonical_string, compute_fingerprint_hash
        ids_a = {"machine_id": "aaa", "cpu_model": "Intel X", "disk_uuid": "uuid-1"}
        ids_b = {"machine_id": "bbb", "cpu_model": "Intel X", "disk_uuid": "uuid-1"}
        h_a = compute_fingerprint_hash(build_canonical_string(ids_a))
        h_b = compute_fingerprint_hash(build_canonical_string(ids_b))
        assert h_a != h_b

    def test_canonical_string_is_sorted(self):
        from phase3.device_fingerprint import build_canonical_string
        ids = {"z_key": "zzz", "a_key": "aaa", "m_key": "mmm"}
        canon = build_canonical_string(ids)
        # 'a_key' should appear before 'm_key' before 'z_key'
        assert canon.index("a_key") < canon.index("m_key") < canon.index("z_key")

    def test_unavailable_sources_produce_consistent_hash(self):
        """If a source is unavailable, 'UNAVAILABLE' must still produce a consistent hash."""
        from phase3.device_fingerprint import build_canonical_string, compute_fingerprint_hash
        ids = {"machine_id": "UNAVAILABLE", "cpu_model": "UNAVAILABLE", "disk_uuid": "UNAVAILABLE"}
        h1 = compute_fingerprint_hash(build_canonical_string(ids))
        h2 = compute_fingerprint_hash(build_canonical_string(ids))
        assert h1 == h2

    def test_raw_identifiers_not_in_hash_output(self):
        """The SHA-256 digest must not contain readable hardware strings."""
        from phase3.device_fingerprint import collect_identifiers, build_canonical_string, compute_fingerprint_hash
        ids = collect_identifiers()
        fp_hash = compute_fingerprint_hash(build_canonical_string(ids))
        for v in ids.values():
            if v != "UNAVAILABLE":
                assert v not in fp_hash, "Raw hardware value leaked into fingerprint hash!"


# ---------------------------------------------------------------------------
# Step 3 — Key Derivation
# ---------------------------------------------------------------------------

class TestStep3KeyDerivation:
    """Validates device-bound AES-256 key derivation (phase3/key_derivation.py)."""

    def test_derive_key_returns_32_bytes(self, salt):
        from phase3.key_derivation import derive_key
        k = derive_key("abc123" * 10, salt)
        assert len(k) == 32

    def test_derive_key_is_deterministic(self, salt):
        from phase3.key_derivation import derive_key
        fp = "deadbeef" * 8  # 64-char mock fingerprint hex
        k1 = derive_key(fp, salt)
        k2 = derive_key(fp, salt)
        assert k1 == k2

    def test_different_salt_produces_different_key(self):
        from phase3.key_derivation import derive_key
        fp = "deadbeef" * 8
        k1 = derive_key(fp, "salt-one")
        k2 = derive_key(fp, "salt-two")
        assert k1 != k2

    def test_different_fingerprint_produces_different_key(self, salt):
        from phase3.key_derivation import derive_key
        k1 = derive_key("aaaa" * 16, salt)
        k2 = derive_key("bbbb" * 16, salt)
        assert k1 != k2

    def test_empty_salt_raises(self):
        from phase3.key_derivation import derive_key
        with pytest.raises(ValueError, match="salt must not be empty"):
            derive_key("deadbeef" * 8, "")

    def test_empty_fingerprint_raises(self, salt):
        from phase3.key_derivation import derive_key
        with pytest.raises(ValueError, match="fingerprint_hash must not be empty"):
            derive_key("", salt)

    def test_validate_key_length_passes_for_32(self):
        from phase3.key_derivation import validate_key_length
        validate_key_length(b"x" * 32)  # should not raise

    def test_validate_key_length_fails_for_wrong_size(self):
        from phase3.key_derivation import validate_key_length
        with pytest.raises(ValueError):
            validate_key_length(b"x" * 16)


# ---------------------------------------------------------------------------
# Step 4 — Adapter Encryption
# ---------------------------------------------------------------------------

class TestStep4Encryption:
    """Validates AES-256-GCM adapter encryption/decryption (phase3/adapter_encryptor.py)."""

    def _make_key(self) -> bytes:
        return hashlib.sha256(b"test-key-phase3").digest()

    def test_encrypt_single_file(self, tmp_dir, dummy_adapter_file, salt):
        from phase3.adapter_encryptor import encrypt_adapter
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
        from phase3.adapter_encryptor import encrypt_adapter
        key = self._make_key()
        enc_path = tmp_dir / "adapter.enc"
        meta = encrypt_adapter(
            adapter_input=dummy_adapter_dir,
            output_enc_path=enc_path,
            key=key,
            fingerprint_hash="fp" * 32,
        )
        assert enc_path.exists()
        assert "tar.gz" in meta["adapter_format"]

    def test_decrypt_restores_original(self, tmp_dir, dummy_adapter_file):
        from phase3.adapter_encryptor import encrypt_adapter, decrypt_adapter
        key = self._make_key()
        enc_path = tmp_dir / "adapter.enc"
        out_path = tmp_dir / "restored.bin"
        original_bytes = dummy_adapter_file.read_bytes()

        encrypt_adapter(dummy_adapter_file, enc_path, key, "fp" * 32)
        decrypt_adapter(enc_path, out_path, key)

        assert out_path.read_bytes() == original_bytes

    def test_wrong_key_fails_decryption(self, tmp_dir, dummy_adapter_file):
        from phase3.adapter_encryptor import encrypt_adapter, decrypt_adapter
        key_correct = self._make_key()
        key_wrong   = hashlib.sha256(b"wrong-key").digest()
        enc_path    = tmp_dir / "adapter.enc"
        out_path    = tmp_dir / "restored.bin"

        encrypt_adapter(dummy_adapter_file, enc_path, key_correct, "fp" * 32)

        with pytest.raises(ValueError, match="AES-GCM decryption failed"):
            decrypt_adapter(enc_path, out_path, key_wrong)

        assert not out_path.exists(), "Plaintext output must not exist after failed decryption."

    def test_metadata_json_is_written(self, tmp_dir, dummy_adapter_file):
        from phase3.adapter_encryptor import encrypt_adapter
        meta_path = tmp_dir / "metadata.json"
        encrypt_adapter(dummy_adapter_file, tmp_dir / "adapter.enc", self._make_key(), "fp" * 32, meta_path)
        meta = json.loads(meta_path.read_text())
        for field in ["algorithm", "nonce_hex", "timestamp_utc", "version"]:
            assert field in meta, f"Missing metadata field: {field}"

    def test_short_key_raises(self, tmp_dir, dummy_adapter_file):
        from phase3.adapter_encryptor import encrypt_adapter
        with pytest.raises(ValueError, match="32 bytes"):
            encrypt_adapter(dummy_adapter_file, tmp_dir / "adapter.enc", b"short", "fp" * 32)

    def test_missing_adapter_raises(self, tmp_dir):
        from phase3.adapter_encryptor import encrypt_adapter
        with pytest.raises(FileNotFoundError):
            encrypt_adapter(tmp_dir / "does_not_exist.bin", tmp_dir / "adapter.enc", self._make_key(), "fp")

    def test_encrypted_file_differs_from_original(self, tmp_dir, dummy_adapter_file):
        from phase3.adapter_encryptor import encrypt_adapter
        enc_path = tmp_dir / "adapter.enc"
        encrypt_adapter(dummy_adapter_file, enc_path, self._make_key(), "fp" * 32)
        assert enc_path.read_bytes() != dummy_adapter_file.read_bytes()


# ---------------------------------------------------------------------------
# Step 5 — Integrity
# ---------------------------------------------------------------------------

class TestStep5Integrity:
    """Validates SHA-256 hash generation and constant-time verification (phase3/integrity.py)."""

    def _make_enc(self, tmp_dir: Path) -> Path:
        f = tmp_dir / "adapter.enc"
        f.write_bytes(os.urandom(256))
        return f

    def test_compute_hash_is_64_hex(self, tmp_dir):
        from phase3.integrity import compute_file_hash
        enc = self._make_enc(tmp_dir)
        h = compute_file_hash(enc)
        assert len(h) == 64

    def test_hash_is_reproducible(self, tmp_dir):
        from phase3.integrity import compute_file_hash
        enc = self._make_enc(tmp_dir)
        assert compute_file_hash(enc) == compute_file_hash(enc)

    def test_save_and_load_roundtrip(self, tmp_dir):
        from phase3.integrity import compute_file_hash, save_hash, load_hash
        enc = self._make_enc(tmp_dir)
        digest = compute_file_hash(enc)
        hash_path = tmp_dir / "adapter.hash"
        save_hash(digest, hash_path)
        assert load_hash(hash_path) == digest

    def test_verify_passes_on_unmodified_file(self, tmp_dir):
        from phase3.integrity import compute_file_hash, save_hash, verify_integrity
        enc = self._make_enc(tmp_dir)
        hash_path = tmp_dir / "adapter.hash"
        save_hash(compute_file_hash(enc), hash_path)
        verify_integrity(enc, hash_path)  # must not raise

    def test_verify_fails_on_modified_file(self, tmp_dir):
        from phase3.integrity import compute_file_hash, save_hash, verify_integrity
        enc = self._make_enc(tmp_dir)
        hash_path = tmp_dir / "adapter.hash"
        save_hash(compute_file_hash(enc), hash_path)

        # Flip one byte
        data = bytearray(enc.read_bytes())
        data[0] ^= 0xFF
        enc.write_bytes(bytes(data))

        with pytest.raises(ValueError, match="Integrity check FAILED"):
            verify_integrity(enc, hash_path)

    def test_load_hash_rejects_invalid_content(self, tmp_dir):
        from phase3.integrity import load_hash
        bad = tmp_dir / "adapter.hash"
        bad.write_text("not-a-valid-hash")
        with pytest.raises(ValueError, match="valid SHA-256"):
            load_hash(bad)


# ---------------------------------------------------------------------------
# Step 6 — RSA Signature
# ---------------------------------------------------------------------------

class TestStep6Signature:
    """Validates RSA-PSS signing and verification (phase3/signature_utils.py)."""

    @pytest.fixture(autouse=True)
    def keypair(self, tmp_dir):
        from phase3.signature_utils import generate_dev_keypair
        self.priv_path = tmp_dir / "dev_private.pem"
        self.pub_path  = tmp_dir / "public.pem"
        generate_dev_keypair(self.priv_path, self.pub_path, key_size=2048)

    def test_keypair_files_created(self):
        assert self.priv_path.exists()
        assert self.pub_path.exists()

    def test_private_key_permissions_are_600(self):
        mode = oct(self.priv_path.stat().st_mode)[-3:]
        assert mode == "600", f"Private key permissions should be 600, got {mode}"

    def test_sign_and_verify_succeeds(self):
        from phase3.signature_utils import sign_digest, save_signature, verify_signature
        digest = "a" * 64
        sig = sign_digest(digest, self.priv_path)
        sig_path = self.priv_path.parent / "adapter.sig"
        save_signature(sig, sig_path)
        verify_signature(digest, sig_path, self.pub_path)  # must not raise

    def test_verify_fails_on_modified_digest(self, tmp_dir):
        from phase3.signature_utils import sign_digest, save_signature, verify_signature
        digest_original = "a" * 64
        digest_modified = "b" * 64
        sig = sign_digest(digest_original, self.priv_path)
        sig_path = tmp_dir / "adapter.sig"
        save_signature(sig, sig_path)
        with pytest.raises(ValueError, match="Signature verification FAILED"):
            verify_signature(digest_modified, sig_path, self.pub_path)

    def test_verify_fails_with_wrong_public_key(self, tmp_dir):
        from phase3.signature_utils import generate_dev_keypair, sign_digest, save_signature, verify_signature
        # Generate a second keypair
        priv2 = tmp_dir / "other_private.pem"
        pub2  = tmp_dir / "other_public.pem"
        generate_dev_keypair(priv2, pub2, key_size=2048)

        digest = "a" * 64
        sig = sign_digest(digest, self.priv_path)  # sign with key 1
        sig_path = tmp_dir / "adapter.sig"
        save_signature(sig, sig_path)

        with pytest.raises(ValueError, match="Signature verification FAILED"):
            verify_signature(digest, sig_path, pub2)  # verify with key 2


# ---------------------------------------------------------------------------
# Step 7 — Package Builder
# ---------------------------------------------------------------------------

class TestStep7PackageBuilder:
    """Validates package assembly and manifest generation (phase3/package_builder.py)."""

    def _build_minimal_package(self, pkg_dir: Path):
        """Creates the minimum required artefacts for package tests."""
        (pkg_dir / "adapter.enc").write_bytes(os.urandom(128))
        (pkg_dir / "adapter.hash").write_text("a" * 64)
        (pkg_dir / "adapter.sig").write_bytes(os.urandom(256))
        (pkg_dir / "metadata.json").write_text("{}")
        (pkg_dir / "public.pem").write_text("---fake pem---")

    def test_completeness_check_passes(self, tmp_dir):
        from phase3.package_builder import verify_package_completeness
        self._build_minimal_package(tmp_dir)
        verify_package_completeness(tmp_dir)  # must not raise

    def test_completeness_check_fails_on_missing_file(self, tmp_dir):
        from phase3.package_builder import verify_package_completeness
        self._build_minimal_package(tmp_dir)
        (tmp_dir / "adapter.sig").unlink()
        with pytest.raises(FileNotFoundError, match="adapter.sig"):
            verify_package_completeness(tmp_dir)

    def test_manifest_contains_required_fields(self, tmp_dir):
        from phase3.package_builder import build_manifest
        self._build_minimal_package(tmp_dir)
        manifest = build_manifest(
            package_dir=tmp_dir,
            adapter_id="test-adapter",
            model_reference="test-model",
            fingerprint_hash="fp" * 32,
            package_version="3.0.0",
            enc_metadata={"algorithm": "AES-256-GCM", "adapter_format": "file:.bin"},
        )
        for field in [
            "schema_version", "adapter_id", "model_reference",
            "device_fingerprint_hash_ref", "created_at_utc",
            "artefact_hashes", "verification_instructions", "security_notes",
        ]:
            assert field in manifest, f"Manifest missing field: {field}"

    def test_manifest_reports_no_plaintext_in_package(self, tmp_dir):
        from phase3.package_builder import build_manifest
        self._build_minimal_package(tmp_dir)
        manifest = build_manifest(tmp_dir, "a", "b", "c" * 64, "3.0.0", {})
        assert manifest["security_notes"]["plaintext_in_package"] is False

    def test_artefact_hashes_match_actual_files(self, tmp_dir):
        from phase3.package_builder import build_manifest
        from phase3.integrity import compute_file_hash
        self._build_minimal_package(tmp_dir)
        manifest = build_manifest(tmp_dir, "a", "b", "c" * 64, "3.0.0", {})
        for fname, stored_hash in manifest["artefact_hashes"].items():
            if stored_hash is not None:
                actual = compute_file_hash(tmp_dir / fname)
                assert actual == stored_hash


# ---------------------------------------------------------------------------
# Step 8 — Verifier
# ---------------------------------------------------------------------------

class TestStep8Verifier:
    """Validates the 6-step authorised-device verification pipeline (phase3/verifier.py)."""

    @pytest.fixture()
    def full_package(self, tmp_dir, salt) -> Path:
        """
        Builds a complete, valid package in tmp_dir by running the protect
        pipeline using the current machine's fingerprint.
        """
        from phase3.adapter_encryptor import encrypt_adapter
        from phase3.device_fingerprint import get_fingerprint_hash
        from phase3.integrity import compute_file_hash, save_hash
        from phase3.key_derivation import derive_key
        from phase3.package_builder import build_package
        from phase3.signature_utils import generate_dev_keypair, save_signature, sign_digest

        pkg = tmp_dir / "pkg"
        pkg.mkdir()

        # Create a tiny fake adapter file
        adapter = tmp_dir / "adapter_model.safetensors"
        adapter.write_bytes(b"dummy-weights-" + os.urandom(64))

        fp_hash = get_fingerprint_hash()
        key     = derive_key(fp_hash, salt)

        enc_path  = pkg / "adapter.enc"
        priv_path = tmp_dir / "dev_private.pem"
        pub_path  = pkg / "public.pem"

        generate_dev_keypair(priv_path, pub_path, key_size=2048)

        enc_meta = encrypt_adapter(adapter, enc_path, key, fp_hash, pkg / "metadata.json")
        digest = compute_file_hash(enc_path)
        save_hash(digest, pkg / "adapter.hash")

        sig = sign_digest(digest, priv_path)
        save_signature(sig, pkg / "adapter.sig")

        build_package(
            package_dir=pkg,
            adapter_id="test-adapter",
            model_reference="test-model",
            fingerprint_hash=fp_hash,
            package_version="3.0.0",
            enc_metadata=enc_meta,
            public_key_src=pub_path,
        )
        return pkg, salt

    def test_authorised_device_succeeds(self, tmp_dir, full_package):
        from phase3.verifier import verify_and_decrypt
        pkg, salt = full_package
        out = tmp_dir / "restored.bin"
        result = verify_and_decrypt(pkg, out, salt=salt)
        assert result.exists()

    def test_tampered_adapter_fails_integrity(self, tmp_dir, full_package):
        from phase3.verifier import VerificationError, verify_and_decrypt
        pkg, salt = full_package
        # Flip one byte in the encrypted adapter
        data = bytearray((pkg / "adapter.enc").read_bytes())
        data[-1] ^= 0xFF
        (pkg / "adapter.enc").write_bytes(bytes(data))
        with pytest.raises(VerificationError, match=r"\[Step 2\]"):
            verify_and_decrypt(pkg, tmp_dir / "restored.bin", salt=salt)

    def test_missing_file_fails_completeness(self, tmp_dir, full_package):
        from phase3.verifier import VerificationError, verify_and_decrypt
        pkg, salt = full_package
        (pkg / "adapter.sig").unlink()
        with pytest.raises(VerificationError, match=r"\[Step 1\]"):
            verify_and_decrypt(pkg, tmp_dir / "restored.bin", salt=salt)

    def test_wrong_salt_fails_decryption(self, tmp_dir, full_package):
        from phase3.verifier import VerificationError, verify_and_decrypt
        pkg, _ = full_package
        with pytest.raises(VerificationError, match=r"\[Step 6\]"):
            verify_and_decrypt(pkg, tmp_dir / "restored.bin", salt="totally-wrong-salt")

    def test_wrong_public_key_fails_signature(self, tmp_dir, full_package):
        from phase3.signature_utils import generate_dev_keypair
        from phase3.verifier import VerificationError, verify_and_decrypt
        pkg, salt = full_package
        # Replace the bundled public key with a different one
        generate_dev_keypair(tmp_dir / "other_priv.pem", pkg / "public.pem", key_size=2048)
        with pytest.raises(VerificationError, match=r"\[Step 3\]"):
            verify_and_decrypt(pkg, tmp_dir / "restored.bin", salt=salt)


# ---------------------------------------------------------------------------
# Step 9 — Secure Cleanup & Crash Safety
# ---------------------------------------------------------------------------

class TestStep9CleanupSafety:
    """Validates safe temp-file handling and atomic write semantics."""

    def test_no_tmp_file_left_on_encryption_success(self, tmp_dir, dummy_adapter_file):
        from phase3.adapter_encryptor import encrypt_adapter
        key = hashlib.sha256(b"test").digest()
        encrypt_adapter(dummy_adapter_file, tmp_dir / "adapter.enc", key, "fp" * 32)
        tmp_files = list(tmp_dir.glob("*.tmp"))
        assert tmp_files == [], f"Stray .tmp files found: {tmp_files}"

    def test_no_enc_file_left_on_interrupted_encrypt(self, tmp_dir, dummy_adapter_file):
        """
        Simulate a mid-write failure: if write_bytes raises, the final .enc
        must not exist (because we write to .tmp first, then rename).
        """
        from unittest.mock import patch
        from phase3 import adapter_encryptor

        enc_path = tmp_dir / "adapter.enc"
        key = hashlib.sha256(b"test").digest()

        original_replace = os.replace

        def fail_replace(src, dst):
            raise OSError("Simulated write failure")

        with patch("os.replace", side_effect=fail_replace):
            with pytest.raises(OSError, match="Simulated write failure"):
                adapter_encryptor._encrypt_file(dummy_adapter_file, enc_path, key)

        assert not enc_path.exists(), "Partial .enc file must not survive a failed atomic write."

    def test_hash_file_atomic_write(self, tmp_dir):
        from phase3.integrity import save_hash
        hash_path = tmp_dir / "adapter.hash"
        save_hash("a" * 64, hash_path)
        assert hash_path.exists()
        # No .tmp sibling should remain
        assert not hash_path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Step 10 — Security Report
# ---------------------------------------------------------------------------

class TestStep10Report:
    """Validates that the security report contains required fields and no secrets."""

    def test_report_contains_required_fields(self, capsys):
        from phase3.main import _print_report
        _print_report(
            fp_hash="a" * 64,
            enc_meta={"algorithm": "AES-256-GCM", "adapter_format": "directory:tar.gz"},
            manifest={
                "schema_version": "3.0.0",
                "adapter_id": "test",
                "model_reference": "llama",
                "artefact_hashes": {"adapter.enc": "x" * 64},
                "security_notes": {"plaintext_in_package": False},
            },
            sig_status="SIGNED",
            integrity_status="HASHED",
            auth_result="ENCRYPTED_AND_PACKAGED",
        )
        # The report block is printed between the banner lines; extract the JSON object.
        captured = capsys.readouterr()
        output = captured.out
        json_start = output.index("{")
        json_end = output.rindex("}") + 1
        report = json.loads(output[json_start:json_end])
        assert report["signature_status"] == "SIGNED"
        assert report["integrity_status"] == "HASHED"
        assert report["authorization_result"] == "ENCRYPTED_AND_PACKAGED"
        assert report["encryption_algorithm"] == "AES-256-GCM"

    def test_report_does_not_contain_full_fingerprint(self, capsys):
        """The full 64-char fingerprint must not appear in the report — only the prefix."""
        from phase3.main import _print_report
        full_fp = "deadbeef" * 8  # 64 chars
        _print_report(
            fp_hash=full_fp,
            enc_meta={},
            manifest={"schema_version": "3.0.0", "adapter_id": "x",
                      "model_reference": "y", "artefact_hashes": {},
                      "security_notes": {}},
            sig_status="N/A",
            integrity_status="N/A",
            auth_result="N/A",
        )
        captured = capsys.readouterr()
        assert full_fp not in captured.out, "Full fingerprint must not appear in the report output."
