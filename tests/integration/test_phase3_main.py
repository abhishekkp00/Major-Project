import os
import json
import pytest
from pathlib import Path

from src.phase3.config import Phase3Config
from src.phase3.main import main
from src.security import get_fingerprint_hash, derive_key


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_phase3_cli_pipeline(tmp_dir, monkeypatch):
    # Setup paths
    adapter_src = tmp_dir / "final_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text('{"r": 8, "peft_type": "LORA"}')
    (adapter_src / "adapter_model.safetensors").write_bytes(b"dummy-weights")

    output_dir = tmp_dir / "protected"
    output_dir.mkdir()

    # Monkeypatch the Phase3Config class attributes directly for absolute isolation
    monkeypatch.setattr(Phase3Config, "ADAPTER_INPUT_DIR", adapter_src)
    monkeypatch.setattr(Phase3Config, "PROTECTED_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(Phase3Config, "RSA_PRIVATE_KEY_PATH", output_dir / "dev_private.pem")
    monkeypatch.setattr(Phase3Config, "RSA_PUBLIC_KEY_PATH", output_dir / "public.pem")
    monkeypatch.setattr(Phase3Config, "DEVICE_SALT", "test-salt-12345")

    # Set env vars just in case other parts look there
    monkeypatch.setenv("P3_DEVICE_SALT", "test-salt-12345")
    monkeypatch.setenv("P3_ADAPTER_INPUT_DIR", str(adapter_src))
    monkeypatch.setenv("P3_PROTECTED_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("P3_RSA_PRIVATE_KEY_PATH", str(output_dir / "dev_private.pem"))
    monkeypatch.setenv("P3_RSA_PUBLIC_KEY_PATH", str(output_dir / "public.pem"))

    # 1. Protect CLI
    exit_code = main(["protect", "--archive"])
    assert exit_code == 0
    assert (output_dir / "adapter.enc").exists()
    assert (output_dir / "adapter.hash").exists()
    assert (output_dir / "adapter.sig").exists()
    assert (output_dir / "package_manifest.json").exists()
    assert (output_dir.with_suffix(".tar.gz")).exists()

    # 2. Report CLI
    exit_code = main(["report"])
    assert exit_code == 0

    # 3. Verify CLI
    restored_tar = tmp_dir / "restored.tar.gz"
    exit_code = main(["verify", "--output", str(restored_tar)])
    assert exit_code == 0
    assert restored_tar.exists()
