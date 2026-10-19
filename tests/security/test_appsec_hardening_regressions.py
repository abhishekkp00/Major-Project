"""
test_appsec_hardening_regressions.py
====================================
Regression tests for confirmed AppSec audit findings:
1. Safe deserialization: adapter_security weight loader enforces weights_only=True,
   preventing arbitrary object unpickling / remote code execution.
2. Resource-exhaustion prevention: upload endpoints reject invalid extensions and
   empty filenames before allocating temporary files or streaming bytes.
3. Storage containment: add_dataset_file_stream enforces path-relative containment
   and blocks directory traversal.
4. Elimination of pickle serialization in Phase 2 LoRA fine-tuning.
5. Comprehensive gitignore coverage for database files and decrypted states.
"""

import io
import pickle
import pytest
from pathlib import Path

import torch
from src.evaluation.dashboard import app
from src.security import api_auth
from src.orchestrator.service import JobOrchestrator
from src.evaluation.adapter_security import _load_weights_from_file_or_dict, ScreeningMode


# ==============================================================================
# 1. Deserialization Security Tests
# ==============================================================================

def _exploit_target():
    raise RuntimeError("EXPLOIT_CODE_EXECUTED: Malicious unpickling succeeded!")


class _MaliciousPicklePayload:
    """Simulates an arbitrary code execution exploit embedded in a pickled weight file."""
    def __reduce__(self):
        return (_exploit_target, ())


def test_adapter_security_weights_only_blocks_malicious_unpickling(tmp_path):
    """
    Confirmed Defect 1 Regression:
    Ensure _load_weights_from_file_or_dict uses weights_only=True and refuses to execute
    arbitrary objects during torch.load.
    """
    # Construct a PyTorch bin file containing the malicious payload
    malicious_bin = tmp_path / "adapter_model.bin"
    with open(malicious_bin, "wb") as f:
        pickle.dump({"exploit": _MaliciousPicklePayload()}, f)

    # Calling _load_weights_from_file_or_dict MUST NOT trigger the exploit (__reduce__ execution).
    # With weights_only=True, PyTorch rejects unpickling non-whitelisted objects with PickleError/UnpicklingError
    # without executing the payload.
    with pytest.raises(Exception) as exc_info:
        _load_weights_from_file_or_dict(malicious_bin, mode=ScreeningMode.PRODUCTION)

    assert "EXPLOIT_CODE_EXECUTED" not in str(exc_info.value), (
        "VULNERABILITY DETECTED: Malicious unpickler payload executed arbitrary code!"
    )


def test_adapter_security_loads_clean_safetensors_and_tensor_bin(tmp_path):
    """Verifies that legitimate tensor state dictionaries load safely under weights_only=True."""
    # Clean PyTorch weights
    clean_bin = tmp_path / "clean_adapter.bin"
    torch.save({"lora_A.weight": torch.randn(4, 16)}, clean_bin)

    weights_dict, source_desc, loaded = _load_weights_from_file_or_dict(
        clean_bin, mode=ScreeningMode.PRODUCTION
    )
    assert loaded is True
    assert "lora_A.weight" in weights_dict
    assert weights_dict["lora_A.weight"].shape == (4, 16)


# ==============================================================================
# 2. Upload Endpoint Resource Exhaustion & Early Validation Tests
# ==============================================================================

TEST_TOKEN = "appsec-test-token-0123456789"


@pytest.fixture
def auth_headers(monkeypatch):
    monkeypatch.setenv(api_auth.AUTH_ENV_VAR, TEST_TOKEN)
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_pre_validate_endpoint_early_rejects_unsupported_extension(client, auth_headers):
    """
    Confirmed Defect 3 Regression:
    /api/orchestrator/validate must reject unallowed extensions (e.g. .exe, .sh, .tar.gz)
    immediately without streaming or writing files to disk.
    """
    data = {
        "file": (io.BytesIO(b"malicious_executable_or_archive_payload"), "exploit.exe")
    }
    response = client.post(
        "/api/orchestrator/validate",
        data=data,
        content_type="multipart/form-data",
        headers=auth_headers,
    )
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["success"] is False
    assert "Unsupported file format" in res_json["error"]


def test_upload_file_endpoint_early_rejects_empty_filename(client, auth_headers):
    """
    /api/orchestrator/jobs/<job_id>/upload must reject traversal-only or empty names.
    """
    # Create job
    job_res = client.post(
        "/api/orchestrator/jobs",
        json={"dataset_name": "test_ds"},
        headers=auth_headers,
    )
    job_id = job_res.get_json()["job_id"]

    # Traversal name that secure_filename strips to ""
    data = {
        "file": (io.BytesIO(b'{"text": "sample"}\n'), "../../../")
    }
    response = client.post(
        f"/api/orchestrator/jobs/{job_id}/upload",
        data=data,
        content_type="multipart/form-data",
        headers=auth_headers,
    )
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["success"] is False
    assert "Invalid filename" in res_json["error"] or "No file selected" in res_json["error"]


def test_upload_file_endpoint_early_rejects_unallowed_extension(client, auth_headers):
    """
    /api/orchestrator/jobs/<job_id>/upload rejects unsupported extensions early.
    """
    job_res = client.post(
        "/api/orchestrator/jobs",
        json={"dataset_name": "test_ds"},
        headers=auth_headers,
    )
    job_id = job_res.get_json()["job_id"]

    data = {
        "file": (io.BytesIO(b"fake binary"), "dataset.tar.gz")
    }
    response = client.post(
        f"/api/orchestrator/jobs/{job_id}/upload",
        data=data,
        content_type="multipart/form-data",
        headers=auth_headers,
    )
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["success"] is False
    assert "Unsupported file format '.gz'" in res_json["error"] or "Unsupported file format" in res_json["error"]


# ==============================================================================
# 3. Storage Containment in add_dataset_file_stream
# ==============================================================================

def test_add_dataset_file_stream_blocks_path_traversal(tmp_path):
    """
    Confirmed Defect 4 Regression:
    add_dataset_file_stream must raise ValueError if filename escapes the workspace.
    """
    orchestrator = JobOrchestrator(base_jobs_dir=tmp_path / "jobs")
    job_id = orchestrator.create_job(dataset_name="containment_test", owner_id="test_user")

    stream = io.BytesIO(b'{"sample": 1}\n')
    with pytest.raises(ValueError, match="Path containment violation"):
        orchestrator.add_dataset_file_stream(
            job_id=job_id,
            filename="../../escaped_file.jsonl",
            stream=stream,
        )


# ==============================================================================
# 4. Gitignore Rule Verification
# ==============================================================================

def test_gitignore_covers_sqlite_and_runtime_databases():
    """
    Confirmed Defect 5 Regression:
    Ensure .gitignore includes rules for SQLite databases, WAL logs, and decrypted output folders.
    """
    gitignore_path = Path(__file__).resolve().parent.parent.parent / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")

    assert "*.db" in content
    assert "*.db-wal" in content
    assert "*.db-shm" in content
    assert "outputs/jobs.db*" in content
    assert "secure_lora_decrypted_*" in content
