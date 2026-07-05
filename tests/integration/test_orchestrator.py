import os
import json
import shutil
import pytest
from pathlib import Path
from src.orchestrator.service import JobOrchestrator


@pytest.fixture()
def custom_orchestrator(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    orch = JobOrchestrator(base_jobs_dir=str(jobs_dir))
    yield orch
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_orchestrator_job_lifecycle(custom_orchestrator):
    # 1. Create job
    job_id = custom_orchestrator.create_job(
        dataset_name="health_records",
        version="1.0.0",
        epochs=1,
        salt="test-salt-xyz"
    )

    assert job_id.startswith("job_")
    
    # Verify directory structure
    job_dir = custom_orchestrator.base_jobs_dir / job_id
    assert (job_dir / "raw_inputs").exists()
    assert (job_dir / "encrypted").exists()
    assert (job_dir / "secrets.key").exists()
    
    # 2. Add dataset file
    dataset_content = b'{"instruction": "test", "output": "response"}\n'
    custom_orchestrator.add_dataset_file(job_id, "data.jsonl", dataset_content)
    
    saved_file = job_dir / "raw_inputs" / "data.jsonl"
    assert saved_file.exists()
    assert saved_file.read_bytes() == dataset_content

    # 3. Retrieve status
    job = custom_orchestrator.get_job(job_id)
    assert job is not None
    assert job["dataset_name"] == "health_records"
    assert job["status"] == "CREATED"
    assert job["stage"] == "dataset_intake"
    assert job["epochs"] == 1
    assert job["salt"] == "test-salt-xyz"

    # 4. Check list jobs
    jobs = custom_orchestrator.get_all_jobs()
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job_id

    # 5. Update state
    custom_orchestrator.update_job_state(job_id, status="TRAINING", stage="fine_tuning", progress=40)
    job = custom_orchestrator.get_job(job_id)
    assert job["status"] == "TRAINING"
    assert job["progress"] == 40
