"""
test_ephemeral_key_handling.py
==============================
Verifies:
1. In-memory symmetric key management in JobOrchestrator.
2. temporary_key_file context manager with restrictive 0600 permissions.
3. Guaranteed shredding and cleanup on normal exit and exception/failure paths.
4. Encryption succeeds and decryption succeeds without leaving permanent secrets.key.
5. Legacy secrets.key migration shreds the disk file upon memory loading.
6. Repository scan confirms no plaintext test keys remain.
"""

import os
import stat
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from src.security.crypto import (
    generate_key,
    temporary_key_file,
    encrypt_stream,
    decrypt_stream,
)
from src.orchestrator.service import JobOrchestrator


@pytest.fixture()
def custom_orchestrator(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    orch = JobOrchestrator(base_jobs_dir=str(jobs_dir))
    return orch


def test_temporary_key_file_lifecycle_and_permissions(tmp_path: Path):
    """Temporary key file is created with 0600 permissions and shredded on exit."""
    key = generate_key()
    captured_path = None

    with temporary_key_file(key, parent_dir=tmp_path) as key_path:
        captured_path = key_path
        assert captured_path.exists()
        assert captured_path.is_file()
        assert captured_path.read_bytes() == key

        if os.name == 'posix':
            mode = stat.S_IMODE(captured_path.stat().st_mode)
            assert mode == 0o600, f"Expected permissions 0600, got {oct(mode)}"

    # Upon context exit, key file must be shredded and unlinked
    assert not captured_path.exists(), "Temporary key file must not persist after exit."


def test_temporary_key_file_cleanup_on_exception(tmp_path: Path):
    """Exceptions within context block must also trigger shredding and deletion."""
    key = generate_key()
    captured_path = None

    with pytest.raises(RuntimeError, match="Simulated crash during training"):
        with temporary_key_file(key, parent_dir=tmp_path) as key_path:
            captured_path = key_path
            assert captured_path.exists()
            raise RuntimeError("Simulated crash during training")

    assert captured_path is not None
    assert not captured_path.exists(), "Failure path must securely shred and remove temporary key file."


def test_temporary_key_file_validation():
    """Validates key length and type constraints before file creation."""
    with pytest.raises(TypeError):
        with temporary_key_file("not_bytes"):
            pass

    with pytest.raises(ValueError, match="Key must be exactly 32 bytes"):
        with temporary_key_file(b"too_short"):
            pass


def test_orchestrator_job_encryption_and_decryption_without_secrets_key(custom_orchestrator, tmp_path):
    """
    Encryption and decryption succeed completely in-memory without creating
    or leaving permanent secrets.key artifacts on disk.
    """
    job_id = custom_orchestrator.create_job(
        dataset_name="clinical_records",
        version="1.0.0",
        epochs=1,
        salt="test-salt"
    )

    job_dir = custom_orchestrator.base_jobs_dir / job_id
    # Requirement: No secrets.key should be created on disk during job creation
    assert not (job_dir / "secrets.key").exists()

    # Verify key exists in memory
    in_memory_key = custom_orchestrator.get_job_key(job_id)
    assert in_memory_key is not None
    assert len(in_memory_key) == 32

    # Ingest test dataset
    record_content = b'{"instruction": "Treat headache", "input": "Patient has migraine", "output": "Administer analgesic"}\n'
    custom_orchestrator.add_dataset_file(job_id, "data.jsonl", record_content)

    # Run Phase 1 ingestion & encryption
    from src.orchestrator.dataset_processor import (
        validate_dataset_file,
        preprocess_and_standardize,
        encrypt_and_save_dataset
    )
    raw_file = job_dir / "raw_inputs" / "data.jsonl"
    raw_records, file_meta = validate_dataset_file(raw_file)
    processed_records = preprocess_and_standardize(raw_records)
    metadata = encrypt_and_save_dataset(
        processed_records=processed_records,
        key=in_memory_key,
        output_dir=job_dir / "encrypted",
        dataset_name="clinical_records",
        version="1.0.0",
        pii_summary=file_meta.get("pii_detected_summary", {})
    )

    assert metadata["encryption_status"] == "encrypted"
    encrypted_file = job_dir / "encrypted" / "encrypted_dataset.enc"
    assert encrypted_file.exists()

    # Still no secrets.key on disk
    assert not (job_dir / "secrets.key").exists()

    # Verify decryption succeeds with in-memory key
    decrypted_dest = tmp_path / "decrypted.jsonl"
    with open(encrypted_file, "rb") as fin, open(decrypted_dest, "wb") as fout:
        decrypt_stream(fin, fout, in_memory_key)

    decrypted_lines = decrypted_dest.read_text(encoding="utf-8").strip().splitlines()
    assert len(decrypted_lines) == 1
    decrypted_data = json.loads(decrypted_lines[0])
    assert "headache" in decrypted_data.get("instruction", "").lower()

    # Clean check: no secrets.key anywhere under job_dir
    key_files = list(job_dir.rglob("*.key")) + list(job_dir.rglob("secrets.key"))
    assert len(key_files) == 0, f"Found unexpected key files: {key_files}"


def test_orchestrator_pipeline_failure_cleans_temporary_key(custom_orchestrator, monkeypatch):
    """Simulated pipeline failure during Phase 2 training cleans up all key files."""
    job_id = custom_orchestrator.create_job(
        dataset_name="failure_test_dataset",
        version="1.0.0",
        epochs=1
    )
    job_dir = custom_orchestrator.base_jobs_dir / job_id
    raw_file = job_dir / "raw_inputs" / "data.jsonl"
    raw_file.write_bytes(b'{"instruction": "fail", "output": "test"}\n')

    # Mock subprocess to fail with returncode 1
    class FailingPopen:
        def __init__(self, *args, **kwargs):
            self.returncode = 1
            import io
            self.stdout = io.StringIO("Fatal GPU out of memory error\n")
            self.stderr = io.StringIO("")

        def wait(self):
            return 1

    monkeypatch.setattr("src.orchestrator.service.subprocess.Popen", FailingPopen)

    # Run pipeline and verify failure is handled
    custom_orchestrator._run_pipeline(job_id)

    job = custom_orchestrator.get_job(job_id)
    assert job["status"] == "FAILED"
    assert "Training failed with exit code 1" in str(job.get("error"))

    # Verify no persistent key files remain in the job directory
    key_files = list(job_dir.rglob("*.key")) + list(job_dir.rglob("secrets.key"))
    assert len(key_files) == 0, f"Found unexpected key files after failure: {key_files}"


def test_legacy_key_shred_on_recovery(custom_orchestrator):
    """
    If a legacy secrets.key file is encountered on disk, it is loaded into
    memory and immediately shredded from disk.
    """
    job_id = "job_legacy_123"
    job_dir = custom_orchestrator.base_jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    test_key = generate_key()
    legacy_key_path = job_dir / "secrets.key"
    legacy_key_path.write_bytes(test_key)
    assert legacy_key_path.exists()

    # Orchestrator does not yet have it in _job_keys
    assert job_id not in custom_orchestrator._job_keys

    # get_job_key recovers it, caches in memory, and shreds the file
    recovered_key = custom_orchestrator.get_job_key(job_id)
    assert recovered_key == test_key
    assert job_id in custom_orchestrator._job_keys

    # The file on disk must now be shredded and gone
    assert not legacy_key_path.exists(), "Legacy secrets.key should be shredded upon recovery"


def test_repository_scan_no_committed_or_lingering_keys():
    """Repository scan verifies no test secrets or *.key files exist under outputs/test_jobs/ or outputs/jobs/."""
    repo_root = Path(__file__).resolve().parents[2]
    
    # Check outputs/test_jobs and outputs/jobs specifically
    test_jobs_dir = repo_root / "outputs" / "test_jobs"
    if test_jobs_dir.exists():
        test_keys = list(test_jobs_dir.rglob("*.key")) + list(test_jobs_dir.rglob("secrets.key"))
        assert len(test_keys) == 0, f"Found lingering test secrets in {test_jobs_dir}: {test_keys}"

    jobs_dir = repo_root / "outputs" / "jobs"
    if jobs_dir.exists():
        job_keys = list(jobs_dir.rglob("*.key")) + list(jobs_dir.rglob("secrets.key"))
        assert len(job_keys) == 0, f"Found lingering job secrets in {jobs_dir}: {job_keys}"
