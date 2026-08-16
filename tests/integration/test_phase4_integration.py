import os
import shutil
import pytest
from pathlib import Path
import json

from src.phase4.main import run_deployment_pipeline
from src.phase3.package_builder import build_package
from src.security import (
    generate_dev_keypair,
    derive_key,
    get_fingerprint_hash,
    encrypt_adapter,
    compute_sha256,
    save_hash,
)


@pytest.fixture
def valid_package_bundle(tmp_path: Path):
    pkg_out = tmp_path / "valid_pkg"
    pkg_out.mkdir()
    adapter_src = tmp_path / "dummy_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    import torch
    torch.save({}, adapter_src / "adapter_model.bin")

    priv_pem = pkg_out / "dev_private.pem"
    pub_pem = pkg_out / "public.pem"
    generate_dev_keypair(priv_pem, pub_pem)
    salt = "demo-integration-salt-abc123xyz"
    fp_hash = get_fingerprint_hash()
    key = derive_key(fp_hash, salt)

    enc_file = pkg_out / "adapter.enc"
    hash_file = pkg_out / "adapter.hash"
    meta_file = pkg_out / "metadata.json"

    meta = encrypt_adapter(adapter_src, enc_file, key, fp_hash)
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


@pytest.fixture
def temp_validation_dir(tmp_path: Path):
    return tmp_path / "deployment_validation"


def test_integration_success_path(valid_package_bundle, temp_validation_dir):
    pkg_dir, salt = valid_package_bundle

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 0
    assert (temp_validation_dir / "validation_report.json").exists()
    assert (temp_validation_dir / "validation_report.md").exists()

    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is True
    assert report["verification_pipeline"]["steps"]["Step 9: Adapter Load & Inference"] == "PASSED"


def test_integration_unauthorized_salt(valid_package_bundle, temp_validation_dir):
    pkg_dir, _ = valid_package_bundle

    exit_code = run_deployment_pipeline(
        package_path=pkg_dir,
        salt="wrong-unauthorized-salt-value",
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 1
    assert (temp_validation_dir / "validation_report.json").exists()

    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is False
    assert report["verification_pipeline"]["steps"]["Step 8: Decryption & Extraction"] == "FAILED"
    assert report["verification_pipeline"]["steps"]["Step 9: Adapter Load & Inference"] == "SKIPPED"


def test_integration_tampered_package(valid_package_bundle, tmp_path, temp_validation_dir):
    orig_dir, salt = valid_package_bundle

    tamper_dir = tmp_path / "tampered_pkg"
    shutil.copytree(orig_dir, tamper_dir)

    enc_file = tamper_dir / "adapter.enc"
    data = bytearray(enc_file.read_bytes())
    data[-1] ^= 0xFF
    enc_file.write_bytes(bytes(data))

    exit_code = run_deployment_pipeline(
        package_path=tamper_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 1
    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is False
    assert report["verification_pipeline"]["steps"]["Step 4: Digest Validation"] == "FAILED"


def test_integration_incomplete_manifest(valid_package_bundle, tmp_path, temp_validation_dir):
    orig_dir, salt = valid_package_bundle

    incomplete_dir = tmp_path / "incomplete_pkg"
    shutil.copytree(orig_dir, incomplete_dir)

    (incomplete_dir / "adapter.sig").unlink()

    exit_code = run_deployment_pipeline(
        package_path=incomplete_dir,
        salt=salt,
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 1
    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is False
    assert report["verification_pipeline"]["steps"]["Step 1: Package Completeness"] == "FAILED"
