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
    test_secret_file = tmp_dir / "securelora" / "device.secret"
    monkeypatch.setenv("SECURELORA_SECRET_PATH", str(test_secret_file))
    from src.security.device_secret import initialize_device_secret
    initialize_device_secret()

    # Setup paths
    adapter_src = tmp_dir / "final_adapter"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text('{"r": 8, "peft_type": "LORA"}')
    import torch
    from src.evaluation.adapter_security import _generate_mock_lora_weights
    mock_weights = _generate_mock_lora_weights(seed=42)
    torch_state_dict = {k: torch.from_numpy(v) for k, v in mock_weights.items()}
    torch.save(torch_state_dict, adapter_src / "adapter_model.bin")

    output_dir = tmp_dir / "protected"
    output_dir.mkdir()

    def clean_model_fn(prompt, *args, **kwargs):
        return f"Clinical diagnosis response for: {prompt[:30]}"

    # Monkeypatch the Phase3Config class attributes directly for absolute isolation
    monkeypatch.setattr(Phase3Config, "ADAPTER_INPUT_DIR", adapter_src)
    monkeypatch.setattr(Phase3Config, "PROTECTED_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(Phase3Config, "RSA_PRIVATE_KEY_PATH", output_dir / "dev_private.pem")
    monkeypatch.setattr(Phase3Config, "RSA_PUBLIC_KEY_PATH", output_dir / "public.pem")
    monkeypatch.setattr(Phase3Config, "DEVICE_SALT", "test-salt-12345")
    monkeypatch.setattr(Phase3Config, "CANDIDATE_MODEL_FN", clean_model_fn, raising=False)

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
