"""
test_phase4_trust_boundary.py
==============================
Unit tests for SecureLoRA Phase 4 inference & model-loading trust boundary enforcement.

Verifies that a SecureLoRA package MUST NOT enter the active model registry
or become available for inference until full Phase 4 verification succeeds.
"""

import json
import shutil
import pytest
from pathlib import Path

from src.orchestrator.model_registry import model_registry
from src.orchestrator.inference_service import generate_securelora, ensure_model_loaded
from src.phase4.main import run_deployment_pipeline
from src.phase3.package_builder import build_package
from src.security import (
    generate_dev_keypair,
    get_fingerprint_hash,
    encrypt_adapter,
    compute_sha256,
    save_hash,
    generate_hkdf_salt,
    derive_key_for_device,
)
from src.common.exceptions import (
    SignatureValidationError,
    IntegrityValidationError,
    DeviceAuthorizationError,
    ReplayAttackError,
    ModelMismatchError,
)


@pytest.fixture
def base_package_bundle(tmp_path: Path):
    """Generates a valid package bundle fixture."""
    pkg_out = tmp_path / "valid_pkg"
    pkg_out.mkdir()
    adapter_src = tmp_path / "dummy_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
    import torch
    torch.save({}, adapter_src / "adapter_model.bin")

    priv_pem = pkg_out / "dev_private.pem"
    pub_pem = pkg_out / "public.pem"
    generate_dev_keypair(priv_pem, pub_pem)

    salt = generate_hkdf_salt()
    fp_hash = get_fingerprint_hash()
    key = derive_key_for_device(fp_hash, salt)

    enc_file = pkg_out / "adapter.enc"
    hash_file = pkg_out / "adapter.hash"
    meta_file = pkg_out / "metadata.json"

    meta = encrypt_adapter(adapter_src, enc_file, key, fp_hash, hkdf_salt=salt)
    meta_file.write_text(json.dumps(meta, indent=2))

    c_hash = compute_sha256(enc_file)
    save_hash(c_hash, hash_file)

    build_package(
        package_dir=pkg_out,
        adapter_id="lora-adapter-v1",
        model_reference="JackFram/llama-68m",
        fingerprint_hash=fp_hash,
        enc_metadata=meta,
        public_key_src=pub_pem,
        private_key_src=priv_pem,
    )
    return pkg_out, salt


def test_valid_package_loads_and_runs_inference(base_package_bundle, tmp_path):
    """Valid package completes verification, enters model registry, and enables inference."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()
    assert not model_registry.is_verified()

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 0
    assert model_registry.is_verified()
    info = model_registry.get_info()
    assert info["deployment_status"] == "VERIFIED"
    assert info["adapter_loaded"] is True


def test_invalid_signature_rejected_before_loading(base_package_bundle, tmp_path):
    """Invalid RSA signature causes Phase 4 verification to fail before model loading."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()

    # Corrupt signature file
    (pkg_dir / "adapter.sig").write_bytes(b"invalid_signature_bytes_12345")

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    assert not model_registry.is_verified()
    info = model_registry.get_info()
    assert info["deployment_status"] == "UNAVAILABLE"
    assert info["peft_model"] is None


def test_modified_ciphertext_rejected(base_package_bundle, tmp_path):
    """Modified adapter.enc ciphertext causes SHA-256 integrity validation to fail."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()

    # Tamper ciphertext
    enc_path = pkg_dir / "adapter.enc"
    data = bytearray(enc_path.read_bytes())
    data[-1] ^= 0xFF
    enc_path.write_bytes(bytes(data))

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    assert not model_registry.is_verified()
    assert model_registry.get_info()["peft_model"] is None


def test_modified_manifest_rejected(base_package_bundle, tmp_path):
    """Modified package_manifest.json causes canonical digest signature check to fail."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()

    # Tamper manifest
    manifest_path = pkg_dir / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["adapter_id"] = "tampered-adapter-id"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    assert not model_registry.is_verified()


def test_wrong_device_rejected(base_package_bundle, tmp_path, monkeypatch):
    """Package with fingerprint for different device is rejected during device auth."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()

    # Mock device fingerprint to simulate running on unauthorized host machine
    monkeypatch.setattr("src.phase4.device_auth.get_fingerprint_hash", lambda: "1" * 64)

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    assert not model_registry.is_verified()


def test_wrong_model_binding_rejected(base_package_bundle, tmp_path):
    """Package built for model A fails verification when deployed for model B."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()

    # Run with mismatched target model name
    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="other-org/different-base-model",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    assert not model_registry.is_verified()


def test_decryption_failure_rejected(base_package_bundle, tmp_path):
    """Wrong decryption key or AES-GCM tag mismatch prevents adapter loading."""
    pkg_dir, _ = base_package_bundle
    output_dir = tmp_path / "out"

    model_registry.clear()

    # Tamper hkdf_salt_hex in manifest so key derivation generates a wrong key
    manifest_path = pkg_dir / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hkdf_salt_hex"] = "00" * 32
    manifest_path.write_text(json.dumps(manifest, indent=2))

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt="dummy",
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    assert not model_registry.is_verified()


def test_failed_verification_clears_existing_registry(base_package_bundle, tmp_path, monkeypatch):
    """If a previously verified adapter is loaded, a failing verification attempt MUST clear the registry."""
    pkg_dir, salt = base_package_bundle
    output_dir = tmp_path / "out"

    # 1. First successfully verify and register
    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )
    assert exit_code == 0
    assert model_registry.is_verified()

    # 2. Now attempt to verify a tampered package; it MUST clear the existing registry
    tamper_dir = tmp_path / "tampered"
    shutil.copytree(pkg_dir, tamper_dir)
    (tamper_dir / "adapter.sig").write_bytes(b"bad_sig")

    exit_code = run_deployment_pipeline(
        package_path=tamper_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Test prompt",
        output_dir=output_dir,
    )

    assert exit_code == 1
    # Check that previous adapter was cleared and is NO LONGER available
    assert not model_registry.is_verified()
    assert model_registry.get_info()["deployment_status"] == "UNAVAILABLE"

    with pytest.raises(RuntimeError, match="MODEL_UNAVAILABLE"):
        generate_securelora("Test prompt")
