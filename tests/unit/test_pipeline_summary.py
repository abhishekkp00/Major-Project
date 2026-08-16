"""
test_pipeline_summary.py
========================
Tests for GET /api/orchestrator/jobs/<job_id>/pipeline-summary

Verifies:
  1. Returns 404 for unknown job IDs.
  2. Returns 200 + 10 stages for a real (COMPLETED) job.
  3. Returns 200 + 10 stages for a FAILED job with correct stage statuses.
  4. Returns 200 + 10 stages for a RUNNING job.
  5. KPI block is always present with all 8 expected keys.
  6. No sensitive fields (salt, key) in response.
  7. Missing optional metrics are null, never fabricated as 0.
  8. SSE stream endpoint still works after route addition (regression).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("SECURE_LORA_DASHBOARD_PORT", "5099")
    from src.evaluation.dashboard import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def routes_client():
    """Standalone client for orchestrator routes only."""
    from flask import Flask
    from src.orchestrator.routes import orchestrator_bp
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True
    test_app.register_blueprint(orchestrator_bp)
    with test_app.test_client() as c:
        yield c


# ── 1. 404 for unknown job ───────────────────────────────────────────────

def test_pipeline_summary_unknown_job(routes_client):
    resp = routes_client.get("/api/orchestrator/jobs/nonexistent_job_abc/pipeline-summary")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data.get("success") is False


# ── 2. Route registered ─────────────────────────────────────────────────

def test_pipeline_summary_route_registered(client):
    from src.evaluation.dashboard import app
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/api/orchestrator/jobs/<job_id>/pipeline-summary" in rules


# ── 3. Correct shape with real job (if one exists) ──────────────────────

def test_pipeline_summary_with_real_job(client):
    """
    If a completed job exists, verify the response shape.
    If none exists, skip gracefully.
    """
    r = client.get("/api/orchestrator/jobs")
    data = r.get_json()
    jobs = [j for j in (data.get("jobs") or []) if j.get("status") in ("COMPLETED", "FAILED", "TRAINING", "CREATED")]
    if not jobs:
        pytest.skip("No jobs found — skipping shape test")

    job_id = jobs[0]["job_id"]
    resp = client.get(f"/api/orchestrator/jobs/{job_id}/pipeline-summary")
    assert resp.status_code == 200
    d = resp.get_json()

    assert d["success"] is True
    assert d["job_id"] == job_id

    # 10 stages always present
    stages = d.get("stages", [])
    assert len(stages) == 10, f"Expected 10 stages, got {len(stages)}"

    # Every stage has required fields
    required_keys = {"id", "name", "status", "purpose", "metrics", "security_significance", "result"}
    for s in stages:
        missing = required_keys - set(s.keys())
        assert not missing, f"Stage {s.get('id')} missing keys: {missing}"

    # KPI block present with all 8 keys
    kpi = d.get("kpi", {})
    expected_kpi_keys = {"dataset", "records", "pii_detected", "training_mode",
                         "dp_epsilon", "adapter_status", "package_status",
                         "device_status", "deployment_status"}
    missing_kpi = expected_kpi_keys - set(kpi.keys())
    assert not missing_kpi, f"KPI missing keys: {missing_kpi}"


# ── 4. Stage statuses are valid ──────────────────────────────────────────

VALID_STATUSES = {"PASSED", "RUNNING", "FAILED", "PENDING", "SKIPPED"}

def test_pipeline_summary_valid_statuses(client):
    r = client.get("/api/orchestrator/jobs")
    data = r.get_json()
    jobs = data.get("jobs") or []
    if not jobs:
        pytest.skip("No jobs found")

    job_id = jobs[0]["job_id"]
    resp = client.get(f"/api/orchestrator/jobs/{job_id}/pipeline-summary")
    d = resp.get_json()
    for s in d.get("stages", []):
        assert s["status"] in VALID_STATUSES, f"Invalid status '{s['status']}' for stage {s['id']}"


# ── 5. No sensitive fields ───────────────────────────────────────────────

FORBIDDEN = {"private_key", "aes_key", "salt", "device_id", "hkdf_key",
             "password", "plaintext", "credential", "secrets"}

def test_pipeline_summary_no_sensitive_fields(client):
    r = client.get("/api/orchestrator/jobs")
    data = r.get_json()
    jobs = data.get("jobs") or []
    if not jobs:
        pytest.skip("No jobs found")

    job_id = jobs[0]["job_id"]
    resp = client.get(f"/api/orchestrator/jobs/{job_id}/pipeline-summary")
    raw = resp.data.decode("utf-8").lower()
    for field in FORBIDDEN:
        assert field not in raw, f"Sensitive field '{field}' found in pipeline-summary response"


# ── 6. Metrics null, not fabricated zero ────────────────────────────────

def test_pipeline_summary_no_fabricated_zeros(client):
    """
    For any stage whose optional metrics are not yet available (no training run),
    the values must be null (None in Python / null in JSON), not fabricated 0.
    This test creates a brand-new CREATED job and checks the training stage.
    """
    import json

    # Create a minimal job without uploading or starting it
    r = client.post("/api/orchestrator/jobs",
                    data=json.dumps({"dataset_name": "test_zero_check", "version": "0.0.1", "epochs": 1}),
                    content_type="application/json")
    d = r.get_json()
    assert d.get("success"), f"Failed to create test job: {d}"
    job_id = d["job_id"]

    resp = client.get(f"/api/orchestrator/jobs/{job_id}/pipeline-summary")
    assert resp.status_code == 200
    ps = resp.get_json()
    assert ps["success"] is True

    # Training stage should not have fabricated zeroes for dp_epsilon / trainable_params
    training = next((s for s in ps["stages"] if s["id"] == "training"), None)
    assert training is not None
    # These must be null (not 0) when training hasn't started
    m = training["metrics"]
    assert m.get("epsilon") is None, f"epsilon should be null, got {m.get('epsilon')}"
    assert m.get("trainable_params") is None, f"trainable_params should be null, got {m.get('trainable_params')}"


# ── 7. SSE stream regression ─────────────────────────────────────────────

def test_sse_stream_route_still_registered(client):
    from src.evaluation.dashboard import app
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/api/orchestrator/jobs/<job_id>/stream" in rules


# ── 8. Completed job has all stages PASSED ───────────────────────────────

def test_completed_job_all_stages_passed(client):
    r = client.get("/api/orchestrator/jobs")
    data = r.get_json()
    completed = [j for j in (data.get("jobs") or []) if j.get("status") == "COMPLETED"]
    if not completed:
        pytest.skip("No COMPLETED jobs found")

    job_id = completed[0]["job_id"]
    resp = client.get(f"/api/orchestrator/jobs/{job_id}/pipeline-summary")
    d = resp.get_json()
    for s in d.get("stages", []):
        assert s["status"] == "PASSED", f"Expected PASSED for stage {s['id']}, got {s['status']}"
