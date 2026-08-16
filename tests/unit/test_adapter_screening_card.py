"""
test_adapter_screening_card.py
==============================
Unit tests for Step 3 Adapter Security Screening API endpoint and data contracts:
  - GET /api/orchestrator/jobs/<job_id>/screening
  - Validates decision badges: SCREENED, REVIEW, REJECTED (never "safe")
  - Validates structural analysis & behavioral analysis data shapes
  - Validates decision explanation rationale & research disclaimer
  - Validates missing screening result handling without metric fabrication
"""

import os
import json
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("SECURE_LORA_DASHBOARD_PORT", "5099")
    from src.evaluation.dashboard import app
    app.testing = True
    with app.test_client() as c:
        yield c


def test_adapter_screening_unknown_job(client):
    """Unknown job returns 404."""
    res = client.get("/api/orchestrator/jobs/job_nonexistent_xyz999/screening")
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert "Job not found" in data["error"]


def test_adapter_screening_missing_result(client, tmp_path):
    """Job without screening result returns available=False without fabricating metrics."""
    from src.orchestrator.service import JobOrchestrator
    orchestrator = JobOrchestrator(base_jobs_dir=str(tmp_path))
    job_id = orchestrator.create_job(dataset_name="test_dataset_dummy")

    # Temporarily point routes orchestrator to our test orchestrator
    import src.orchestrator.routes as routes_mod
    orig_orch = routes_mod.orchestrator
    routes_mod.orchestrator = orchestrator

    try:
        res = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["available"] is False
        assert "not evaluated" in data["reason"].lower() or "report not" in data["reason"].lower()
    finally:
        routes_mod.orchestrator = orig_orch


def test_adapter_screening_clean_adapter(client, tmp_path):
    """Clean adapter screening returns SCREENED decision, low risk score, structural & behavioral metrics."""
    from src.orchestrator.service import JobOrchestrator
    orchestrator = JobOrchestrator(base_jobs_dir=str(tmp_path))
    job_id = orchestrator.create_job(dataset_name="clean_medical_dataset")

    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Mock clean screening report
    report_data = {
        "adapter_id": job_id,
        "timestamp_utc": "2026-08-16T12:00:00Z",
        "risk_level": "LOW",
        "adapter_risk_score": 0.12,
        "decision": "APPROVED",
        "bypassed_via_force": False,
        "screening_latency_ms": 42.5,
        "structural_report": {
            "total_parameters": 294912,
            "layer_count": 8,
            "global_frobenius_norm": 4.12,
            "global_l2_norm": 3.85,
            "global_linf_norm": 0.45,
            "outlier_layer_count": 0,
            "outlier_layers": [],
            "max_layer_zscore": 1.15,
            "sparsity_ratio": 0.02,
            "rank_utilization_mean": 0.88,
            "cosine_similarity_ref": 0.98,
            "parameter_drift_score": 0.04,
            "structural_risk_score": 0.10
        },
        "behavioral_report": {
            "normal_output_divergence": 0.08,
            "trigger_sensitivity": 0.05,
            "paraphrase_consistency": 0.92,
            "abnormal_response_rate": 0.0,
            "classification_flip_rate": 0.0,
            "behavioral_risk_score": 0.14,
            "consistency_risk_score": 0.05,
            "probe_results": [{"category": "normal"}, {"category": "trigger"}]
        },
        "risk_breakdown": {
            "structural_risk_score": 0.10,
            "behavioral_risk_score": 0.14,
            "consistency_risk_score": 0.05
        }
    }

    (job_dir / "screening_report.json").write_text(json.dumps(report_data), encoding="utf-8")

    import src.orchestrator.routes as routes_mod
    orig_orch = routes_mod.orchestrator
    routes_mod.orchestrator = orchestrator

    try:
        res = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["available"] is True
        assert data["job_id"] == job_id
        assert data["decision"] == "SCREENED"
        assert data["decision_code"] == "APPROVED"
        assert data["risk_level"] == "LOW"
        assert data["adapter_risk_score"] == 0.12
        assert data["structural_score"] == 0.10
        assert data["behavioral_score"] == 0.14

        # Verify research disclaimer
        rc = data["research_context"]
        assert rc["label"] == "PRE-DEPLOYMENT ADAPTER SCREENING"
        assert "risk assessment mechanism" in rc["disclaimer"]
        assert "not a formal proof" in rc["disclaimer"]
    finally:
        routes_mod.orchestrator = orig_orch


def test_adapter_screening_suspicious_adapter(client, tmp_path):
    """High risk suspicious adapter returns REJECTED decision badge."""
    from src.orchestrator.service import JobOrchestrator
    orchestrator = JobOrchestrator(base_jobs_dir=str(tmp_path))
    job_id = orchestrator.create_job(dataset_name="suspicious_dataset")

    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "adapter_id": job_id,
        "timestamp_utc": "2026-08-16T12:00:00Z",
        "risk_level": "HIGH",
        "adapter_risk_score": 0.85,
        "decision": "REJECTED",
        "bypassed_via_force": False,
        "screening_latency_ms": 50.0,
        "structural_report": {
            "total_parameters": 294912,
            "layer_count": 8,
            "global_frobenius_norm": 28.5,
            "global_l2_norm": 22.1,
            "outlier_layer_count": 2,
            "outlier_layers": ["layer.0.lora_A", "layer.3.lora_B"],
            "max_layer_zscore": 4.2,
            "sparsity_ratio": 0.01,
            "cosine_similarity_ref": 0.35,
            "parameter_drift_score": 3.1,
            "structural_risk_score": 0.82
        },
        "behavioral_report": {
            "normal_output_divergence": 0.65,
            "trigger_sensitivity": 0.88,
            "paraphrase_consistency": 0.40,
            "abnormal_response_rate": 0.50,
            "classification_flip_rate": 0.60,
            "behavioral_risk_score": 0.88,
            "consistency_risk_score": 0.60,
            "probe_results": [{"category": "trigger"}]
        },
        "risk_breakdown": {
            "structural_risk_score": 0.82,
            "behavioral_risk_score": 0.88,
            "consistency_risk_score": 0.60
        }
    }

    (job_dir / "screening_report.json").write_text(json.dumps(report_data), encoding="utf-8")

    import src.orchestrator.routes as routes_mod
    orig_orch = routes_mod.orchestrator
    routes_mod.orchestrator = orchestrator

    try:
        res = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["available"] is True
        assert data["decision"] == "REJECTED"
        assert data["risk_level"] == "HIGH"
        assert data["adapter_risk_score"] == 0.85
    finally:
        routes_mod.orchestrator = orig_orch


def test_adapter_screening_never_calls_adapter_safe(client, tmp_path):
    """Verifies that decision is never string 'SAFE' or 'PROVEN_SAFE'."""
    from src.orchestrator.service import JobOrchestrator
    orchestrator = JobOrchestrator(base_jobs_dir=str(tmp_path))
    job_id = orchestrator.create_job(dataset_name="clean_dataset_2")

    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "adapter_id": job_id,
        "risk_level": "LOW",
        "adapter_risk_score": 0.05,
        "decision": "APPROVED"
    }

    (job_dir / "screening_report.json").write_text(json.dumps(report_data), encoding="utf-8")

    import src.orchestrator.routes as routes_mod
    orig_orch = routes_mod.orchestrator
    routes_mod.orchestrator = orchestrator

    try:
        res = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        data = res.get_json()
        assert data["decision"] in ["SCREENED", "REVIEW", "REJECTED"]
        assert data["decision"] != "SAFE"
    finally:
        routes_mod.orchestrator = orig_orch
