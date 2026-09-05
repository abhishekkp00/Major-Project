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
    import torch
    import numpy as np
    rng = np.random.RandomState(42)
    mock_weights = {}
    for i in range(4):
        a = rng.normal(0.0, 0.02, size=(8, 64)).astype(np.float32)
        b = rng.normal(0.0, 0.001, size=(64, 8)).astype(np.float32)
        mock_weights[f"base_model.model.encoder.layer.{i}.attention.self.query.lora_A.weight"] = torch.from_numpy(a)
        mock_weights[f"base_model.model.encoder.layer.{i}.attention.self.query.lora_B.weight"] = torch.from_numpy(b)
    torch.save(mock_weights, adapter_dir / "adapter_model.bin")
    
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


def test_missing_adapter_blocks_packaging_and_deployment(tmp_path: Path):
    """Verify missing adapter directory fails closed, sets status to SECURITY_SCREENING_FAILED, and prevents packaging/deployment."""
    from src.common.exceptions import SecurityScreeningFailedError

    job_dir = tmp_path / "job_missing_adapter"
    job_dir.mkdir()
    # Create empty adapter dir without weights file
    (job_dir / "adapter").mkdir()

    statuses = []
    stages = []

    def mock_update_state(jid, **kwargs):
        if "status" in kwargs:
            statuses.append(kwargs["status"])
        if "stage" in kwargs:
            stages.append(kwargs["stage"])

    with pytest.raises(SecurityScreeningFailedError):
        run_security_orchestration(
            job_id="job_missing_test",
            job_dir=job_dir,
            salt="test-salt",
            base_model_name="JackFram/llama-68m",
            update_state_fn=mock_update_state,
        )

    # Status must be SECURITY_SCREENING_FAILED
    assert "SECURITY_SCREENING_FAILED" in statuses
    # No package artifact or encrypted output should be written
    assert not (job_dir / "protected" / "adapter.enc").exists()
    assert not (job_dir / "protected" / "protected_package.tar.gz").exists()
    # Deployment step must NOT have been executed
    assert "running_secure_deployment_check" not in stages


def test_high_risk_adapter_blocks_packaging_and_deployment(tmp_path: Path):
    """Verify high-risk adapter raises SecurityPolicyRejectedError, sets status to SECURITY_POLICY_REJECTED, and blocks packaging/deployment."""
    import torch
    from src.common.exceptions import SecurityPolicyRejectedError
    from src.evaluation.adapter_security import _generate_mock_lora_weights

    job_dir = tmp_path / "job_high_risk_adapter"
    job_dir.mkdir()
    adapter_dir = job_dir / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 8}')

    # Create outlier weights that trigger HIGH risk level
    weights = _generate_mock_lora_weights(seed=42)
    first_key = list(weights.keys())[0]
    weights[first_key] = weights[first_key] * 200.0 + 100.0
    torch_state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}
    torch.save(torch_state_dict, adapter_dir / "adapter_model.bin")

    statuses = []
    stages = []

    def mock_update_state(jid, **kwargs):
        if "status" in kwargs:
            statuses.append(kwargs["status"])
        if "stage" in kwargs:
            stages.append(kwargs["stage"])

    with pytest.raises(SecurityPolicyRejectedError):
        run_security_orchestration(
            job_id="job_high_risk_test",
            job_dir=job_dir,
            salt="test-salt",
            base_model_name="JackFram/llama-68m",
            update_state_fn=mock_update_state,
        )

    assert "SECURITY_POLICY_REJECTED" in statuses
    assert not (job_dir / "protected" / "adapter.enc").exists()
    assert not (job_dir / "protected" / "protected_package.tar.gz").exists()
    assert "running_secure_deployment_check" not in stages


def test_screening_exception_blocks_packaging_and_deployment(tmp_path: Path, monkeypatch):
    """Verify arbitrary screening engine exception raises SecurityScreeningFailedError and blocks packaging/deployment."""
    from src.common.exceptions import SecurityScreeningFailedError
    import src.evaluation.adapter_security

    job_dir = tmp_path / "job_screening_exception"
    job_dir.mkdir()
    (job_dir / "adapter").mkdir()

    def mock_screen_error(*args, **kwargs):
        raise RuntimeError("Unexpected screening engine crash")

    monkeypatch.setattr(src.evaluation.adapter_security, "screen_adapter_and_enforce_policy", mock_screen_error)

    statuses = []
    stages = []

    def mock_update_state(jid, **kwargs):
        if "status" in kwargs:
            statuses.append(kwargs["status"])
        if "stage" in kwargs:
            stages.append(kwargs["stage"])

    with pytest.raises(SecurityScreeningFailedError) as exc_info:
        run_security_orchestration(
            job_id="job_exception_test",
            job_dir=job_dir,
            salt="test-salt",
            base_model_name="JackFram/llama-68m",
            update_state_fn=mock_update_state,
        )

    assert "SECURITY_SCREENING_FAILED" in statuses
    assert "Unexpected screening engine crash" in str(exc_info.value)
    assert not (job_dir / "protected" / "adapter.enc").exists()
    assert not (job_dir / "protected" / "protected_package.tar.gz").exists()
    assert "running_secure_deployment_check" not in stages


def test_invalid_incomplete_result_blocks_packaging(tmp_path: Path, monkeypatch):
    """Verify invalid/incomplete screening result raises SecurityScreeningFailedError and blocks packaging/deployment."""
    from src.common.exceptions import SecurityScreeningFailedError
    import src.evaluation.adapter_security

    job_dir = tmp_path / "job_invalid_result"
    job_dir.mkdir()
    (job_dir / "adapter").mkdir()

    monkeypatch.setattr(src.evaluation.adapter_security, "screen_adapter_and_enforce_policy", lambda *a, **kw: None)

    statuses = []

    def mock_update_state(jid, **kwargs):
        if "status" in kwargs:
            statuses.append(kwargs["status"])

    with pytest.raises(SecurityScreeningFailedError) as exc_info:
        run_security_orchestration(
            job_id="job_invalid_test",
            job_dir=job_dir,
            salt="test-salt",
            base_model_name="JackFram/llama-68m",
            update_state_fn=mock_update_state,
        )

    assert "SECURITY_SCREENING_FAILED" in statuses
    assert "invalid or incomplete result" in str(exc_info.value)
    assert not (job_dir / "protected" / "adapter.enc").exists()

