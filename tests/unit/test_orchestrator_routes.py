import json
import pytest
from pathlib import Path
from flask import Flask

from src.orchestrator.routes import orchestrator_bp
from src.orchestrator.service import orchestrator


@pytest.fixture()
def app():
    flask_app = Flask(__name__)
    flask_app.register_blueprint(orchestrator_bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_get_jobs(client, monkeypatch):
    mock_jobs = [{"job_id": "job_1", "status": "COMPLETED"}]
    monkeypatch.setattr(orchestrator, "get_all_jobs", lambda: mock_jobs)

    response = client.get("/api/orchestrator/jobs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["jobs"] == mock_jobs


def test_get_job_status(client, monkeypatch):
    mock_job = {"job_id": "job_1", "status": "RUNNING", "stage": "training"}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: mock_job if jid == "job_1" else None)

    response = client.get("/api/orchestrator/jobs/job_1")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["job"] == mock_job

    response = client.get("/api/orchestrator/jobs/job_unknown")
    assert response.status_code == 404


def test_get_job_metrics(client, monkeypatch):
    mock_job = {
        "job_id": "job_1",
        "loss_history": [{"loss": 0.5}],
        "current_epoch": 1,
        "pii_summary": {"email": 2},
        "num_records": 100,
        "security_metrics": {"encryption_time_seconds": 1.2}
    }
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: mock_job if jid == "job_1" else None)

    response = client.get("/api/orchestrator/jobs/job_1/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    metrics = data["metrics"]
    assert metrics["current_epoch"] == 1
    assert metrics["pii_detected_summary"] == {"email": 2}


def test_list_job_artifacts(client, tmp_path, monkeypatch):
    job_id = "job_1"
    # Setup temporary directory for jobs
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    protected_dir = job_dir / "protected"
    protected_dir.mkdir(parents=True)
    
    # Create safe and unsafe files
    (protected_dir / "adapter.enc").write_bytes(b"encrypted weights")
    (protected_dir / "public.pem").write_bytes(b"public key")
    (protected_dir / "private.pem").write_bytes(b"private key") # Should be excluded
    (protected_dir / "dev_private.pem").write_bytes(b"another private key") # Should be excluded
    (protected_dir / "secrets.key").write_bytes(b"secret symmetric key") # Should be excluded
    (protected_dir / "package_manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid} if jid == job_id else None)
    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)

    response = client.get(f"/api/orchestrator/jobs/{job_id}/artifacts")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    
    artifacts = data["artifacts"]
    names = [art["name"] for art in artifacts]
    assert "adapter.enc" in names
    assert "public.pem" in names
    assert "package_manifest.json" in names
    assert "private.pem" not in names
    assert "dev_private.pem" not in names
    assert "secrets.key" not in names


def test_download_job_artifact(client, tmp_path, monkeypatch):
    job_id = "job_1"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    protected_dir = job_dir / "protected"
    protected_dir.mkdir(parents=True)

    (protected_dir / "adapter.enc").write_bytes(b"encrypted data")
    (protected_dir / "dev_private.pem").write_bytes(b"private key data")

    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid} if jid == job_id else None)
    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)

    # Test downloading safe file
    response = client.get(f"/api/orchestrator/jobs/{job_id}/download/adapter.enc")
    assert response.status_code == 200
    assert response.data == b"encrypted data"

    # Test downloading unsafe file (private key)
    response = client.get(f"/api/orchestrator/jobs/{job_id}/download/dev_private.pem")
    assert response.status_code == 403

    # Test non-existent file
    response = client.get(f"/api/orchestrator/jobs/{job_id}/download/nonexistent.enc")
    assert response.status_code == 403


def test_get_job_report(client, tmp_path, monkeypatch):
    job_id = "job_1"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    deployment_dir = job_dir / "deployment"
    deployment_dir.mkdir(parents=True)

    report_content = {"verification_pipeline": {"success": True}}
    (deployment_dir / "validation_report.json").write_text(json.dumps(report_content), encoding="utf-8")

    monkeypatch.setattr(orchestrator, "get_job", lambda jid: {"job_id": jid} if jid == job_id else None)
    monkeypatch.setattr(orchestrator, "base_jobs_dir", jobs_dir)

    response = client.get(f"/api/orchestrator/jobs/{job_id}/report")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["report"]["verification_pipeline"] == report_content["verification_pipeline"]
    assert "security_validation_outcomes" in data["report"]


def test_stream_job_events(client, monkeypatch):
    job_id = "job_1"
    mock_job = {"job_id": job_id, "status": "COMPLETED", "stage": "completed"}
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: mock_job if jid == job_id else None)

    response = client.get(f"/api/orchestrator/jobs/{job_id}/stream")
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    
    # Read first event from response stream
    stream_data = response.data.decode("utf-8")
    assert "data: " in stream_data
    payload = json.loads(stream_data.replace("data: ", "").strip())
    assert payload["job_id"] == job_id
    assert payload["status"] == "COMPLETED"
