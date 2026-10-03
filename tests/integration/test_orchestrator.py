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


def test_orchestrator_training_subprocess_uses_sys_executable(custom_orchestrator, monkeypatch):
    """Regression test: verify training subprocess uses sys.executable and preserves all invocation options."""
    import io
    import sys
    import subprocess
    import src.orchestrator.security_orchestrator

    job_id = custom_orchestrator.create_job(
        dataset_name="health_records",
        version="1.0.0",
        epochs=1,
        salt="test-salt-xyz"
    )
    dataset_content = b'{"instruction": "test instruction", "input": "test input", "output": "test output"}\n'
    custom_orchestrator.add_dataset_file(job_id, "data.jsonl", dataset_content)

    captured_popen_calls = []
    real_popen = subprocess.Popen

    class MockPopen:
        def __init__(self, cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and "src.phase2.train_lora" in cmd:
                self.cmd = cmd
                self.kwargs = kwargs
                self.returncode = 0
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                captured_popen_calls.append((cmd, kwargs))
                self._is_mock = True
            else:
                self._real = real_popen(cmd, *args, **kwargs)
                self._is_mock = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self._is_mock:
                return self._real.__exit__(exc_type, exc_val, exc_tb)

        def wait(self, *args, **kwargs):
            return 0 if self._is_mock else self._real.wait(*args, **kwargs)

        def communicate(self, *args, **kwargs):
            return ("", "") if self._is_mock else self._real.communicate(*args, **kwargs)

        def __getattr__(self, name):
            if not self._is_mock:
                return getattr(self._real, name)
            raise AttributeError(name)

    monkeypatch.setattr("src.orchestrator.service.subprocess.Popen", MockPopen)
    monkeypatch.setattr(
        "src.orchestrator.security_orchestrator.run_security_orchestration",
        lambda **kwargs: {"status": "mocked"}
    )

    custom_orchestrator._run_pipeline(job_id)

    assert len(captured_popen_calls) == 1
    cmd, kwargs = captured_popen_calls[0]

    # Verify requirement 2 & 6: executable argument equals sys.executable
    assert cmd[0] == sys.executable
    # Verify requirement 3: preserve module invocation, working directory, env vars, stdout/stderr
    assert cmd[1:] == ["-m", "src.phase2.train_lora"]
    assert kwargs.get("cwd") == str(Path.cwd())
    assert kwargs.get("stdout") == subprocess.PIPE
    assert kwargs.get("stderr") == subprocess.STDOUT
    assert kwargs.get("text") is True
    assert "SECURE_LORA_KEY_HEX" in kwargs.get("env", {})
    assert kwargs.get("env", {}).get("SECURE_LORA_EPOCHS") == "1"


def test_orchestrator_training_subprocess_dynamic_sys_executable(custom_orchestrator, monkeypatch):
    """Regression test: verify arbitrary sys.executable is respected (venv, .venv, system Python, IDE terminal)."""
    import io
    import sys
    import subprocess
    import src.orchestrator.security_orchestrator

    custom_python = "/custom/virtualenv/bin/python3"
    monkeypatch.setattr(sys, "executable", custom_python)

    job_id = custom_orchestrator.create_job(
        dataset_name="health_records",
        version="1.0.0",
        epochs=2,
        salt="test-salt-custom"
    )
    dataset_content = b'{"instruction": "test", "input": "", "output": "result"}\n'
    custom_orchestrator.add_dataset_file(job_id, "data.jsonl", dataset_content)

    captured_popen_calls = []
    real_popen = subprocess.Popen

    class MockPopen:
        def __init__(self, cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and "src.phase2.train_lora" in cmd:
                self.cmd = cmd
                self.kwargs = kwargs
                self.returncode = 0
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                captured_popen_calls.append((cmd, kwargs))
                self._is_mock = True
            else:
                self._real = real_popen(cmd, *args, **kwargs)
                self._is_mock = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self._is_mock:
                return self._real.__exit__(exc_type, exc_val, exc_tb)

        def wait(self, *args, **kwargs):
            return 0 if self._is_mock else self._real.wait(*args, **kwargs)

        def communicate(self, *args, **kwargs):
            return ("", "") if self._is_mock else self._real.communicate(*args, **kwargs)

        def __getattr__(self, name):
            if not self._is_mock:
                return getattr(self._real, name)
            raise AttributeError(name)

    monkeypatch.setattr("src.orchestrator.service.subprocess.Popen", MockPopen)
    monkeypatch.setattr(
        "src.orchestrator.security_orchestrator.run_security_orchestration",
        lambda **kwargs: {"status": "mocked"}
    )

    custom_orchestrator._run_pipeline(job_id)

    assert len(captured_popen_calls) == 1
    cmd, kwargs = captured_popen_calls[0]
    assert cmd[0] == custom_python
    assert cmd[0] == sys.executable
    assert cmd[1:] == ["-m", "src.phase2.train_lora"]
    assert kwargs.get("env", {}).get("SECURE_LORA_EPOCHS") == "2"

