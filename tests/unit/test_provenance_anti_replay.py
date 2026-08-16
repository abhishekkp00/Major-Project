"""
test_provenance_anti_replay.py
==============================
Unit and Security Tests for Adapter Provenance and Anti-Replay System.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.security import (
    validate_manifest_schema,
    compute_canonical_manifest_digest,
    AntiReplayTracker,
    generate_dev_keypair,
    sign_digest,
    verify_signature,
    compute_sha256,
)
from src.phase3.package_builder import build_package
from src.phase4.package_validator import validate_package_provenance
from src.common.exceptions import (
    ManifestSchemaError,
    ReplayAttackError,
    PackageExpiredError,
    ModelMismatchError,
    AdapterMismatchError,
    SignatureValidationError,
    IntegrityValidationError,
)


@pytest.fixture
def valid_manifest_dict():
    return {
        "package_id": "11112222-3333-4444-5555-666677778888",
        "adapter_id": "medical-lora-v1",
        "base_model_id": "distilbert-base-uncased",
        "model_revision": "main",
        "adapter_revision": "v1.0.0",
        "package_version": "1.0.0",
        "creation_timestamp": "2026-08-16T12:00:00+00:00",
        "expiration_timestamp": None,
        "binding_policy_version": "1.0.0",
        "kdf_version": "hkdf-sha256-v1",
        "encryption_version": "aes-256-gcm-v1",
        "signature_algorithm": "rsa-pss-2048-sha256",
        "digest_algorithm": "sha256",
        "nonce_metadata": {"iv_bytes": 12, "tag_bytes": 16},
        "deployment_policy": {"strictness": "high"},
        "sequence_number": 10,
        "device_fingerprint_hash_ref": "3926c635fa8a12607cf843d884442ae151b5253f54529dc053cd6f0cebddfb93",
        "encrypted_adapter_digest": "4444555566667777888899990000aaaabbbbccccddddeeeeffff000011112222",
    }


class TestManifestSchemaValidation:
    def test_valid_manifest_schema_passes(self, valid_manifest_dict):
        validate_manifest_schema(valid_manifest_dict)

    def test_missing_required_field_raises_error(self, valid_manifest_dict):
        del valid_manifest_dict["sequence_number"]
        with pytest.raises(ManifestSchemaError) as exc_info:
            validate_manifest_schema(valid_manifest_dict)
        assert "sequence_number" in str(exc_info.value)

    def test_invalid_sequence_number_type_raises_error(self, valid_manifest_dict):
        valid_manifest_dict["sequence_number"] = -5
        with pytest.raises(ManifestSchemaError):
            validate_manifest_schema(valid_manifest_dict)

    def test_unsupported_kdf_version_raises_error(self, valid_manifest_dict):
        valid_manifest_dict["kdf_version"] = "pbkdf2-v1"
        with pytest.raises(ManifestSchemaError):
            validate_manifest_schema(valid_manifest_dict)


class TestCanonicalManifestDigest:
    def test_canonical_digest_deterministic(self, valid_manifest_dict):
        digest1 = compute_canonical_manifest_digest(valid_manifest_dict, "digest_abc")
        digest2 = compute_canonical_manifest_digest(valid_manifest_dict, "digest_abc")
        assert digest1 == digest2
        assert len(digest1) == 64

    def test_canonical_digest_changes_on_manifest_field_tamper(self, valid_manifest_dict):
        digest_orig = compute_canonical_manifest_digest(valid_manifest_dict, "digest_abc")
        valid_manifest_dict["base_model_id"] = "tampered-base-model"
        digest_tampered = compute_canonical_manifest_digest(valid_manifest_dict, "digest_abc")
        assert digest_orig != digest_tampered

    def test_canonical_digest_changes_on_ciphertext_digest_tamper(self, valid_manifest_dict):
        digest1 = compute_canonical_manifest_digest(valid_manifest_dict, "digest_abc")
        digest2 = compute_canonical_manifest_digest(valid_manifest_dict, "digest_xyz")
        assert digest1 != digest2


class TestAntiReplayTracker:
    def test_anti_replay_fresh_package_succeeds(self, valid_manifest_dict):
        with tempfile.TemporaryDirectory() as tmp:
            state_p = Path(tmp) / "state.json"
            tracker = AntiReplayTracker(state_file_path=state_p)
            tracker.check_and_update(valid_manifest_dict)

            # Check state was written
            state_data = json.loads(state_p.read_text())
            assert valid_manifest_dict["adapter_id"] in state_data["adapters"]
            rec = state_data["adapters"][valid_manifest_dict["adapter_id"]]
            assert rec["last_sequence"] == 10
            assert valid_manifest_dict["package_id"] in rec["processed_packages"]

    def test_anti_replay_duplicate_package_id_rejected(self, valid_manifest_dict):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AntiReplayTracker(state_file_path=Path(tmp) / "state.json")
            tracker.check_and_update(valid_manifest_dict)

            with pytest.raises(ReplayAttackError) as exc_info:
                tracker.check_and_update(valid_manifest_dict)
            assert "Package ID" in str(exc_info.value)

    def test_anti_replay_stale_sequence_number_rejected(self, valid_manifest_dict):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AntiReplayTracker(state_file_path=Path(tmp) / "state.json")
            tracker.check_and_update(valid_manifest_dict)

            stale_manifest = valid_manifest_dict.copy()
            stale_manifest["package_id"] = "22223333-4444-5555-6666-777788889999"
            stale_manifest["sequence_number"] = 8  # 8 <= 10
            with pytest.raises(ReplayAttackError) as exc_info:
                tracker.check_and_update(stale_manifest)
            assert "stale" in str(exc_info.value).lower()

    def test_anti_replay_expired_timestamp_rejected(self, valid_manifest_dict):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AntiReplayTracker(state_file_path=Path(tmp) / "state.json")
            valid_manifest_dict["expiration_timestamp"] = (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat()

            with pytest.raises(PackageExpiredError):
                tracker.check_and_update(valid_manifest_dict)

    def test_anti_replay_target_model_mismatch_rejected(self, valid_manifest_dict):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AntiReplayTracker(state_file_path=Path(tmp) / "state.json")
            with pytest.raises(ModelMismatchError):
                tracker.check_and_update(valid_manifest_dict, target_base_model_id="other-model-xl")

    def test_anti_replay_target_adapter_mismatch_rejected(self, valid_manifest_dict):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AntiReplayTracker(state_file_path=Path(tmp) / "state.json")
            with pytest.raises(AdapterMismatchError):
                tracker.check_and_update(valid_manifest_dict, target_adapter_id="foreign-adapter-id")


class TestPackageProvenanceVerificationOrder:
    def test_signature_validation_over_canonical_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp) / "pkg"
            pkg_dir.mkdir()

            (pkg_dir / "adapter.enc").write_bytes(b"CIPHERTEXT_BYTES_12345")
            (pkg_dir / "adapter.hash").write_text(compute_sha256(pkg_dir / "adapter.enc"))
            (pkg_dir / "metadata.json").write_text("{}")

            priv_key = Path(tmp) / "priv.pem"
            pub_key = Path(tmp) / "pub.pem"
            generate_dev_keypair(priv_key, pub_key)

            manifest = build_package(
                package_dir=pkg_dir,
                adapter_id="test-adapter",
                model_reference="distilbert-base-uncased",
                fingerprint_hash="3926c635fa8a12607cf843d884442ae151b5253f54529dc053cd6f0cebddfb93",
                package_version="1.0.0",
                enc_metadata={"kdf_version": "hkdf-sha256-v1"},
                public_key_src=pub_key,
                private_key_src=priv_key,
                sequence_number=1,
            )

            # Verification should pass
            m, digest = validate_package_provenance(pkg_dir)
            assert m["package_id"] == manifest["package_id"]
            assert digest == compute_sha256(pkg_dir / "adapter.enc")

    def test_tampered_manifest_fails_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp) / "pkg"
            pkg_dir.mkdir()

            (pkg_dir / "adapter.enc").write_bytes(b"CIPHERTEXT_BYTES_12345")
            (pkg_dir / "adapter.hash").write_text(compute_sha256(pkg_dir / "adapter.enc"))
            (pkg_dir / "metadata.json").write_text("{}")

            priv_key = Path(tmp) / "priv.pem"
            pub_key = Path(tmp) / "pub.pem"
            generate_dev_keypair(priv_key, pub_key)

            build_package(
                package_dir=pkg_dir,
                adapter_id="test-adapter",
                model_reference="distilbert-base-uncased",
                fingerprint_hash="3926c635fa8a12607cf843d884442ae151b5253f54529dc053cd6f0cebddfb93",
                package_version="1.0.0",
                enc_metadata={"kdf_version": "hkdf-sha256-v1"},
                public_key_src=pub_key,
                private_key_src=priv_key,
                sequence_number=1,
            )

            # Tamper with manifest
            man_p = pkg_dir / "package_manifest.json"
            m = json.loads(man_p.read_text())
            m["base_model_id"] = "malicious-model-injection"
            man_p.write_text(json.dumps(m, indent=2))

            with pytest.raises(SignatureValidationError):
                validate_package_provenance(pkg_dir)
