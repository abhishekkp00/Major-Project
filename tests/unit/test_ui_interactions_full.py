"""
test_ui_interactions_full.py
=============================
Comprehensive unit tests for SecureLoRA Dashboard UI interactions & workflows:
- Research Workbench API endpoints & ablation modal data binding.
- Dataset template endpoints & preview modal data.
- Pipeline job creation, status stream, and screening gate details.
- Deployment gate status & secure aggregate Q&A interactions.
"""

import json
import pytest
from pathlib import Path
from src.evaluation.dashboard import app
from src.orchestrator.service import orchestrator


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_pipeline_summary_and_stages_endpoint(client, monkeypatch):
    """Test job pipeline summary endpoint driving Step 2 KPI row and Step 3 stages."""
    job_id = "test_job_pipeline_999"
    mock_job = {
        "job_id": job_id,
        "status": "COMPLETED",
        "stage": "completed",
        "progress": 100.0,
        "created_at": "2026-08-16T12:00:00Z",
        "updated_at": "2026-08-16T12:05:00Z",
        "dataset_name": "real_world_pii.jsonl",
        "num_records": 100,
        "pii_summary": {"total_entities_detected": 142},
        "eval_metrics": {
            "training_config": {"dp_enabled": True, "dp_epsilon": 2.44}
        },
        "security_metrics": {
            "adapter_screening_outcome": "SCREENED",
            "tamper_rejection_rate": 1.0,
            "cross_device_rejection_rate": 1.0
        },
        "verification_steps": {
            "Step 4: Device Authorization": "PASSED",
            "Step 8: Inference Validation": "PASSED"
        }
    }

    monkeypatch.setattr(orchestrator, "get_job", lambda jid: mock_job if jid == job_id else None)

    rv = client.get(f"/api/orchestrator/jobs/{job_id}/pipeline-summary")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert data["pipeline_status"] == "COMPLETED"
    assert "kpi" in data
    assert data["kpi"]["training_mode"] == "DP-LoRA"
    assert data["kpi"]["dp_epsilon"] == 2.44
    assert len(data["stages"]) == 10


def test_research_workbench_full_flow(client):
    """Test full Research Workbench data fetching flow for Step 4."""
    # 1. Summary Endpoint
    r_sum = client.get("/api/research/summary")
    assert r_sum.status_code == 200
    d_sum = r_sum.get_json()
    assert d_sum["available"] is True
    assert d_sum["classification"] == "HISTORICAL"
    assert "utility" in d_sum
    assert "privacy" in d_sum
    assert "security" in d_sum
    assert "overhead" in d_sum

    # 2. Ablation Endpoint
    r_abl = client.get("/api/research/ablation")
    assert r_abl.status_code == 200
    d_abl = r_abl.get_json()
    assert d_abl["available"] is True
    assert len(d_abl["ablation_rows"]) > 0
    assert "E0" in d_abl["experiment_summaries"]
    assert "E9" in d_abl["experiment_summaries"]

    # 3. Privacy Endpoint
    r_priv = client.get("/api/research/privacy")
    assert r_priv.status_code == 200
    d_priv = r_priv.get_json()
    assert d_priv["available"] is True
    assert "full_pipeline_privacy" in d_priv

    # 4. Screening Endpoint
    r_scr = client.get("/api/research/screening")
    assert r_scr.status_code == 200
    d_scr = r_scr.get_json()
    assert d_scr["available"] is True
    assert d_scr["detection_metrics"]["f1_score"] == 1.0

    # 5. Adaptive Evasion Endpoint
    r_evas = client.get("/api/research/adaptive-evasion")
    assert r_evas.status_code == 200
    d_evas = r_evas.get_json()
    assert d_evas["available"] is True
    assert "level_summary" in d_evas
    assert "seed_stats" in d_evas

    # 6. Overhead Endpoint
    r_over = client.get("/api/research/overhead")
    assert r_over.status_code == 200
    d_over = r_over.get_json()
    assert d_over["available"] is True
    assert "full_pipeline_overhead" in d_over


def test_phase4_status_endpoint(client):
    """Test Phase 4 status endpoint driving deployment authorization check."""
    r_stat = client.get("/api/phase4/status")
    assert r_stat.status_code == 200
    d_stat = r_stat.get_json()
    assert "loaded" in d_stat
    assert "fingerprint_prefix" in d_stat


def test_security_demonstration_endpoints(client):
    """Test Step 5 Security Demonstration endpoints."""
    # 1. Demonstration endpoint
    r_demo = client.get("/api/security/demonstration")
    assert r_demo.status_code == 200
    d_demo = r_demo.get_json()
    assert d_demo["success"] is True
    assert "device_info" in d_demo
    assert d_demo["device_info"]["authorization_state"] in ["AUTHORIZED", "REAUTHORIZATION_REQUIRED", "UNAUTHORIZED"]
    assert "provenance" in d_demo
    assert len(d_demo["attacks"]) == 6
    assert len(d_demo["history"]) > 0

    # 2. Simulate Attack endpoint
    for attack_id in ["tampering", "replay", "unauthorized_device", "signature_forgery", "suspicious_adapter", "adaptive_suspicious_adapter"]:
        r_sim = client.post(
            "/api/security/simulate-attack",
            data=json.dumps({"attack_id": attack_id, "payload": "Test payload"}),
            content_type="application/json"
        )
        assert r_sim.status_code == 200
        d_sim = r_sim.get_json()
        assert d_sim["success"] is True
        assert d_sim["attack"]["result"] in ["BLOCKED", "DETECTED", "ALLOWED", "NOT TESTED"]
        assert "evidence" in d_sim["attack"]


def test_deployment_gate_scenarios(client):
    """Test Step 6 Deployment Gate scenarios: successful, tampered, wrong device, invalid signature, replay."""
    scenarios = [
        ("successful", True, None),
        ("tampered_package", False, "Step 2: Integrity Verification"),
        ("invalid_signature", False, "Step 3: Signature Verification"),
        ("wrong_device", False, "Step 4: Device Authorization"),
        ("replay", False, "Step 5: Key Derivation"),
    ]

    for scenario, expected_success, failed_step in scenarios:
        res = client.post(
            "/api/phase4/verify",
            data=json.dumps({"scenario": scenario}),
            content_type="application/json"
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is expected_success
        assert "steps" in data
        assert len(data["steps"]) == 8
        if expected_success:
            assert all(status == "PASSED" for status in data["steps"].values())
        else:
            assert data["steps"][failed_step] == "FAILED"
            assert len(data["error"]) > 0


def test_workbench_html_structure_and_templates(client):
    """Test SecureLoRA Workbench HTML structure & dataset templates API endpoint."""
    # 1. Dataset Templates API Endpoint
    res_tmpl = client.get("/api/orchestrator/dataset-templates")
    assert res_tmpl.status_code == 200
    d_tmpl = res_tmpl.get_json()
    assert d_tmpl["success"] is True
    assert len(d_tmpl["templates"]) == 3
    ids = [t["id"] for t in d_tmpl["templates"]]
    assert "pii_corporate" in ids
    assert "clinical_notes" in ids
    assert "real_world_pii" in ids

    # 2. Main Index Route Rendering
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert 'id="tabRun"' in html
    assert 'id="tabMetrics"' in html
    assert 'id="tabSecurity"' in html
    assert 'id="tabModel"' in html
    assert 'id="dataset-cards-grid"' in html
    assert 'id="btn-start-pipeline"' in html
    assert 'id="tstep-1"' in html
    assert 'id="tstep-8"' in html




