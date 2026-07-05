import os
import json
import shutil
import pytest
from pathlib import Path

from src.orchestrator.security_orchestrator import run_security_orchestration
from src.common.exceptions import SecureLoraError


@pytest.fixture()
def mock_peft_and_base_model(monkeypatch):
    class MockTokenizer:
        eos_token = "</s>"
        def __call__(self, text, *args, **kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        def decode(self, *args, **kwargs):
            return "Decrypted and loaded adapter outputs secure text response."
            
    class MockModel:
        def generate(self, *args, **kwargs):
            import torch
            return torch.tensor([[1, 2, 3]])
            
    import sys
    import src.phase4.main
    phase4_main = sys.modules["src.phase4.main"]
    monkeypatch.setattr(phase4_main, "load_base_model_and_tokenizer", lambda name: (MockModel(), MockTokenizer()))
    monkeypatch.setattr(phase4_main, "load_peft_adapter", lambda base, path: MockModel())
    monkeypatch.setattr(phase4_main, "run_side_by_side_inference", lambda **kwargs: {"base_response": "base", "adapter_response": "adapter"})


@pytest.fixture()
def mock_job_workspace(tmp_path: Path):
    job_dir = tmp_path / "job_test_security"
    job_dir.mkdir()
    
    # 1. Create a dummy adapter directory
    adapter_dir = job_dir / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("dummy model weight tensors", encoding="utf-8")
    
    yield job_dir
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_security_orchestration_lifecycle_and_simulations(mock_job_workspace, mock_peft_and_base_model):
    job_id = "job_test_security_123"
    salt = "test-security-salt"
    base_model = "JackFram/llama-68m"
    
    updated_stages = []
    updated_statuses = []

    def mock_update_state(jid, **kwargs):
        assert jid == job_id
        if "stage" in kwargs:
            updated_stages.append(kwargs["stage"])
        if "status" in kwargs:
            updated_statuses.append(kwargs["status"])

    outcomes = run_security_orchestration(
        job_id=job_id,
        job_dir=mock_job_workspace,
        salt=salt,
        base_model_name=base_model,
        update_state_fn=mock_update_state
    )

    # Verify that status transitions occurred sequentially
    assert "preparing_adapter" in updated_stages
    assert "deriving_device_binding" in updated_stages
    assert "encrypting_adapter" in updated_stages
    assert "generating_hash" in updated_stages
    assert "generating_signature" in updated_stages
    assert "building_package" in updated_stages
    assert "running_integrity_check" in updated_stages
    assert "running_device_authorization_check" in updated_stages
    assert "running_secure_deployment_check" in updated_stages
    assert "secure_inference_validation" in updated_stages
    assert "security_validation_completed" in updated_stages

    # Verify metrics stored
    assert "adapter_size_before_encryption_bytes" in outcomes
    assert outcomes["adapter_size_before_encryption_bytes"] > 0
    assert "protected_package_size_bytes" in outcomes
    assert outcomes["protected_package_size_bytes"] > 0
    assert "encryption_time_seconds" in outcomes
    assert "verification_time_seconds" in outcomes
    
    # Verify simulation outcomes
    assert outcomes["authorized_deployment"] == "pass"
    assert outcomes["deployment_validation_result"] == "success"
    assert outcomes["tamper_simulation"] == "pass"
    assert outcomes["unauthorized_device_simulation"] == "pass"
