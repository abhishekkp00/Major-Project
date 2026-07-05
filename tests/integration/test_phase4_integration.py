import os
import shutil
import pytest
from pathlib import Path
import json

from src.phase4.main import run_deployment_pipeline


@pytest.fixture
def original_package_dir():
    return Path("outputs/protected_adapter")


@pytest.fixture
def correct_salt():
    return "demo-integration-salt-abc123xyz"


@pytest.fixture
def temp_validation_dir(tmp_path: Path):
    return tmp_path / "deployment_validation"


def test_integration_success_path(original_package_dir, correct_salt, temp_validation_dir):
    if not original_package_dir.exists():
        pytest.skip("Original protected adapter package not found.")

    exit_code = run_deployment_pipeline(
        package_path=original_package_dir,
        salt=correct_salt,
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 0
    assert (temp_validation_dir / "validation_report.json").exists()
    assert (temp_validation_dir / "validation_report.md").exists()

    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is True
    assert report["verification_pipeline"]["steps"]["Step 8: Inference Validation"] == "PASSED"


def test_integration_unauthorized_salt(original_package_dir, temp_validation_dir):
    if not original_package_dir.exists():
        pytest.skip("Original protected adapter package not found.")

    exit_code = run_deployment_pipeline(
        package_path=original_package_dir,
        salt="wrong-unauthorized-salt-value",
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 1
    assert (temp_validation_dir / "validation_report.json").exists()

    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is False
    assert report["verification_pipeline"]["steps"]["Step 6: Decryption & Extraction"] == "FAILED"
    assert report["verification_pipeline"]["steps"]["Step 7: PEFT Model Loading"] == "SKIPPED"


def test_integration_tampered_package(original_package_dir, correct_salt, tmp_path, temp_validation_dir):
    if not original_package_dir.exists():
        pytest.skip("Original protected adapter package not found.")

    tamper_dir = tmp_path / "tampered_pkg"
    shutil.copytree(original_package_dir, tamper_dir)

    enc_file = tamper_dir / "adapter.enc"
    data = bytearray(enc_file.read_bytes())
    data[-1] ^= 0xFF
    enc_file.write_bytes(bytes(data))

    exit_code = run_deployment_pipeline(
        package_path=tamper_dir,
        salt=correct_salt,
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 1
    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is False
    assert report["verification_pipeline"]["steps"]["Step 2: Integrity Verification"] == "FAILED"
    assert report["verification_pipeline"]["steps"]["Step 3: Signature Verification"] == "SKIPPED"


def test_integration_incomplete_manifest(original_package_dir, correct_salt, tmp_path, temp_validation_dir):
    if not original_package_dir.exists():
        pytest.skip("Original protected adapter package not found.")

    incomplete_dir = tmp_path / "incomplete_pkg"
    shutil.copytree(original_package_dir, incomplete_dir)

    (incomplete_dir / "adapter.sig").unlink()

    exit_code = run_deployment_pipeline(
        package_path=incomplete_dir,
        salt=correct_salt,
        base_model_name="JackFram/llama-68m",
        prompt="Compare security models.",
        output_dir=temp_validation_dir
    )

    assert exit_code == 1
    report = json.loads((temp_validation_dir / "validation_report.json").read_text())
    assert report["verification_pipeline"]["success"] is False
    assert report["verification_pipeline"]["steps"]["Step 1: Package Completeness"] == "FAILED"
    assert report["verification_pipeline"]["steps"]["Step 2: Integrity Verification"] == "SKIPPED"
