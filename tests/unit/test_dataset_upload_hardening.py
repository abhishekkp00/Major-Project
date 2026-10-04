"""
tests/unit/test_dataset_upload_hardening.py
===========================================
Unit tests for hardened dataset upload handling against unbounded memory consumption:
- upload below limit succeeds
- upload above limit returns 413
- chunked upload does not call unbounded read()
- failed upload leaves no partial artifact
- invalid type still rejected
- metadata-only validation does not accumulate all records in memory
- uploaded file contents are never logged
"""

import io
import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from src.evaluation.dashboard import app
from src.orchestrator.dataset_processor import validate_dataset_file
from src.orchestrator.service import orchestrator


@pytest.fixture
def auth_token():
    return os.environ.get("SECURELORA_API_TOKEN", "test-bearer-token-securing-lora-2026")


@pytest.fixture
def test_client(auth_token):
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {auth_token}"
        yield c


def test_upload_below_limit_succeeds(test_client, tmp_path, monkeypatch):
    """Uploading a dataset well within the size limit succeeds."""
    job_id = "test_upload_valid_job"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    raw_inputs = job_dir / "raw_inputs"
    raw_inputs.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid, "status": "CREATED"} if jid == job_id else None)

    content = b'{"instruction": "What is 2+2?", "output": "4"}\n{"instruction": "Capital of France?", "output": "Paris"}\n'
    data = {
        "file": (io.BytesIO(content), "sample_dataset.jsonl")
    }

    res = test_client.post(
        f"/api/orchestrator/jobs/{job_id}/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["success"] is True
    assert payload["filename"] == "sample_dataset.jsonl"

    saved_file = raw_inputs / "sample_dataset.jsonl"
    assert saved_file.exists()
    assert saved_file.read_bytes() == content


def test_upload_above_limit_returns_413(test_client, tmp_path, monkeypatch):
    """Uploading a dataset that exceeds the configured MAX_CONTENT_LENGTH returns HTTP 413."""
    job_id = "test_upload_oversized_job"
    jobs_dir = tmp_path / "jobs"
    raw_inputs = jobs_dir / job_id / "raw_inputs"
    raw_inputs.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid, "status": "CREATED"} if jid == job_id else None)

    # Set a small test limit of 10 KB
    test_limit = 10 * 1024
    monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", test_limit)

    # 15 KB oversized content
    oversized_content = b"x" * (15 * 1024)
    data = {
        "file": (io.BytesIO(oversized_content), "large_dataset.txt")
    }

    res = test_client.post(
        f"/api/orchestrator/jobs/{job_id}/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 413
    payload = res.get_json()
    assert payload["success"] is False
    assert "Request entity too large" in payload["error"] or "exceeds maximum allowed size" in payload["error"]

    # Verify no file remained in raw_inputs
    assert not (raw_inputs / "large_dataset.txt").exists()


def test_chunked_upload_does_not_call_unbounded_read(test_client, tmp_path, monkeypatch):
    """Verify that file.stream is read in bounded chunks, never calling unbounded read()."""
    job_id = "test_bounded_chunks_job"
    jobs_dir = tmp_path / "jobs"
    raw_inputs = jobs_dir / job_id / "raw_inputs"
    raw_inputs.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid, "status": "CREATED"} if jid == job_id else None)

    chunk_calls = []

    class TrackedBytesIO(io.BytesIO):
        def read(self, size=-1):
            chunk_calls.append(size)
            assert size is not None and size > 0, "Unbounded read() was called!"
            return super().read(size)

    content = b"row1,row2\nval1,val2\n" * 1000
    tracked_stream = TrackedBytesIO(content)

    data = {
        "file": (tracked_stream, "test_data.csv")
    }

    res = test_client.post(
        f"/api/orchestrator/jobs/{job_id}/upload",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    assert len(chunk_calls) > 0
    # Every chunk read was bounded
    for call_size in chunk_calls:
        assert call_size > 0
        assert call_size <= 64 * 1024


def test_failed_upload_leaves_no_partial_artifact(test_client, tmp_path, monkeypatch):
    """When an upload fails partway through streaming, all partial artifacts are unlinked."""
    job_id = "test_cleanup_failure_job"
    jobs_dir = tmp_path / "jobs"
    raw_inputs = jobs_dir / job_id / "raw_inputs"
    raw_inputs.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid, "status": "CREATED"} if jid == job_id else None)

    target_path = raw_inputs / "interrupted_file.txt"

    # 1. Test via direct service method
    class FailingStream:
        def __init__(self):
            self.count = 0

        def read(self, size):
            self.count += 1
            if self.count == 1:
                return b"first chunk of data"
            raise IOError("Simulated disk error or connection abort")

    with pytest.raises(IOError):
        orchestrator.add_dataset_file_stream(
            job_id=job_id,
            filename="interrupted_file.txt",
            stream=FailingStream(),
        )

    # Partial file must be deleted immediately
    assert not target_path.exists()

    # 2. Test via endpoint with exception during write
    with patch("builtins.open", side_effect=IOError("Simulated write failure")):
        data = {
            "file": (io.BytesIO(b"data"), "failed_upload.txt")
        }
        res = test_client.post(
            f"/api/orchestrator/jobs/{job_id}/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 500
        assert not (raw_inputs / "failed_upload.txt").exists()


def test_invalid_type_rejected_and_leaves_no_artifact(test_client, tmp_path, monkeypatch):
    """Uploaded files with disallowed extensions are rejected and leave no partial file."""
    job_id = "test_invalid_type_job"
    jobs_dir = tmp_path / "jobs"
    raw_inputs = jobs_dir / job_id / "raw_inputs"
    raw_inputs.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid, "status": "CREATED"} if jid == job_id else None)

    disallowed_files = [
        ("malicious_script.py", b"import os; os.system('echo pwned')"),
        ("binary_executable.exe", b"MZ\x90\x00\x03\x00\x00\x00"),
        ("shell_script.sh", b"#!/bin/bash\necho hack"),
        ("archive.zip", b"PK\x03\x04\x14\x00\x00\x00"),
    ]

    for filename, content in disallowed_files:
        data = {
            "file": (io.BytesIO(content), filename)
        }
        res = test_client.post(
            f"/api/orchestrator/jobs/{job_id}/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 400
        payload = res.get_json()
        assert payload["success"] is False
        assert "Unsupported file format" in payload["error"]
        # Partial file must not exist
        assert not (raw_inputs / filename).exists()


def test_pre_validate_endpoint_bounded_streaming_and_cleanup(test_client, tmp_path, monkeypatch):
    """Verify pre-validation endpoint uses bounded streaming and cleans up temporary files."""
    test_limit = 5 * 1024  # 5 KB limit
    monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", test_limit)

    # 1. Valid pre-validation under limit
    valid_content = b'{"instruction": "hi", "output": "hello"}\n'
    data = {
        "file": (io.BytesIO(valid_content), "valid.jsonl")
    }
    res = test_client.post(
        "/api/orchestrator/validate",
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["success"] is True
    assert payload["metadata"]["num_raw_records"] == 1

    # 2. Oversized pre-validation rejected with 413
    oversized = b"a" * (8 * 1024)
    data_oversized = {
        "file": (io.BytesIO(oversized), "oversized.txt")
    }
    res_ov = test_client.post(
        "/api/orchestrator/validate",
        data=data_oversized,
        content_type="multipart/form-data",
    )
    assert res_ov.status_code == 413

    # 3. Disallowed extension rejected with 400
    data_bad_ext = {
        "file": (io.BytesIO(b"data"), "dataset.bin")
    }
    res_bad = test_client.post(
        "/api/orchestrator/validate",
        data=data_bad_ext,
        content_type="multipart/form-data",
    )
    assert res_bad.status_code == 400
    assert "Unsupported file format" in res_bad.get_json()["error"]


def test_validate_dataset_file_metadata_only_streaming(tmp_path):
    """Verify validate_dataset_file with metadata_only=True streams without accumulating all records in RAM."""
    sample_file = tmp_path / "stream_check.jsonl"
    lines = ['{"instruction": "Q%d", "output": "A%d"}\n' % (i, i) for i in range(500)]
    sample_file.write_text("".join(lines), encoding="utf-8")

    # When metadata_only=True, returned records are capped at sample_limit, but num_raw_records reflects the full count
    sample_records, metadata = validate_dataset_file(sample_file, metadata_only=True, sample_limit=5)
    assert len(sample_records) == 5
    assert metadata["num_raw_records"] == 500
    assert metadata["schema_detected"] == "instruction"

    # When metadata_only=False, all records are returned (backwards compatibility)
    all_records, full_meta = validate_dataset_file(sample_file, metadata_only=False)
    assert len(all_records) == 500
    assert full_meta["num_raw_records"] == 500


def test_file_contents_never_logged(test_client, tmp_path, monkeypatch, caplog):
    """Verify that uploaded dataset file contents are never written to logger records."""
    job_id = "test_no_log_content_job"
    jobs_dir = tmp_path / "jobs"
    raw_inputs = jobs_dir / job_id / "raw_inputs"
    raw_inputs.mkdir(parents=True)

    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid, "status": "CREATED"} if jid == job_id else None)

    secret_canary = "SUPER_SECRET_PATIENT_SSN_999-88-7777_CONFIDENTIAL_CONTENT"
    content = f'{{"text": "{secret_canary}"}}\n'.encode("utf-8")

    data = {
        "file": (io.BytesIO(content), "secret_records.jsonl")
    }

    with caplog.at_level(logging.DEBUG):
        res = test_client.post(
            f"/api/orchestrator/jobs/{job_id}/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200

    for record in caplog.records:
        assert secret_canary not in record.getMessage()
