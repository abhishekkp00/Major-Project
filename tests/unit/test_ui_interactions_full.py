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
import shutil
import pytest
from pathlib import Path
from src.evaluation.dashboard import app
from src.orchestrator.service import orchestrator


@pytest.fixture
def client():
    import os
    app.config["TESTING"] = True
    token = os.environ.get("SECURELORA_API_TOKEN") or "test-bearer-token-securing-lora-2026"
    with app.test_client() as client:
        client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {token}"
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
    pii_stage = next(stage for stage in data["stages"] if stage["id"] == "pii_audit")
    # Counts/scores absent from the job record must remain unavailable, not inferred.
    assert pii_stage["metrics"]["pii_masked"] is None
    assert pii_stage["metrics"]["precision"] is None
    assert pii_stage["metrics"]["recall"] is None
    assert pii_stage["metrics"]["f1"] is None
    deployment_stage = next(stage for stage in data["stages"] if stage["id"] == "deployment")
    assert deployment_stage["status"] != "PASSED"  # Only two of eight gates have evidence.


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
    assert d_scr["detection_metrics"]["evasion_suite_f1"] == 1.0

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
    assert "ai4privacy" in ids or "pii_corporate" in ids
    assert "synthea" in ids or "clinical_notes" in ids
    assert "synthetic" in ids or "real_world_pii" in ids


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

    # Jury-demo clarity and evidence-provenance regressions.
    assert "DEMO GUIDE" in html
    assert 'id="deployment-badge"' in html
    assert "Checking API…" in html
    assert 'id="auth-dialog"' not in html
    assert 'id="auth-token-input"' not in html
    assert 'id="run-feedback"' in html
    assert "RESULT PROVENANCE" in html
    assert "synthetic baseline" in html
    assert 'id="dp-noise" class="form-input" value="1.2"' in html

    # Confirm the browser assets serve the code containing the expected safeguards.
    js_res = client.get("/static/js/dashboard.js")
    assert js_res.status_code == 200
    dashboard_js = js_res.get_data(as_text=True)
    assert "connectApiSession" not in dashboard_js
    assert "LOCAL DEMO" in dashboard_js
    assert "'0.96'" not in dashboard_js
    assert "'0.97'" not in dashboard_js
    assert "Number(eb[k].precision || 1.0)" not in dashboard_js
    assert "function showRunFeedback(" in dashboard_js
    assert "not this live run" in dashboard_js
    assert "seamlessly fall back to benchmark data" not in dashboard_js

    css_res = client.get("/static/css/dashboard.css")
    assert css_res.status_code == 200
    dashboard_css = css_res.get_data(as_text=True)
    assert "Jury-demo clarity and responsive polish" in dashboard_css
    assert ".run-feedback--success" in dashboard_css


def test_workbench_shell_and_assets(client):
    """Test that the production enterprise workbench shell serves cleanly without auth blockage."""
    res = client.get("/workbench")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "SecureLoRA — Adapter Security Workbench" in html
    assert 'class="app"' in html
    assert 'id="sidebar"' in html
    assert 'id="main"' in html
    assert 'id="theme-toggle"' in html
    assert 'id="conn-pill"' in html
    assert 'id="view"' in html
    assert "securelora-theme" in html
    assert "/static/workbench/js/app.js" in html
    assert "/static/workbench/workbench.css" in html

    # Confirm workbench CSS static route
    css_res = client.get("/static/workbench/workbench.css")
    assert css_res.status_code == 200
    css_content = css_res.get_data(as_text=True)
    assert "--nav-bg" in css_content
    assert "--primary" in css_content


def test_workbench_overview_components_and_pipeline_stages(client):
    """Test Overview page components, 8-stage pipeline visualization, and animations."""
    # 1. Verify CSS styles for the eight-stage security pipeline and motion controls
    css_res = client.get("/static/workbench/workbench.css")
    assert css_res.status_code == 200
    css = css_res.get_data(as_text=True)
    assert ".pipeline-grid" in css
    assert ".pipeline-stage" in css
    assert ".pipeline-stage.stage--success" in css
    assert ".pipeline-stage.stage--running" in css
    assert ".stage-runner-indicator" in css
    assert ".stage-evidence-toggle" in css
    assert ".stage-evidence-panel" in css
    assert "stage-shimmer" in css
    assert "prefers-reduced-motion" in css

    # 2. Verify Overview page JS source module
    js_res = client.get("/static/workbench/js/pages/overview.js")
    assert js_res.status_code == 200
    js = js_res.get_data(as_text=True)

    # All 8 canonical stages required by security architecture
    assert "Dataset Validation" in js
    assert "PII Masking" in js
    assert "LoRA / Optional DP-LoRA" in js
    assert "Adapter Screening" in js
    assert "Integrity Signing" in js
    assert "Device-Bound Encryption" in js
    assert "Deployment Verification" in js
    assert "Inference Validation" in js

    # Summary cards based on real backend data
    assert "Total Pipeline Runs" in js
    assert "Successful Runs" in js
    assert "Failed Runs" in js
    assert "Currently Running" in js

    # Prominent New Pipeline Run action
    assert "New Pipeline Run" in js

    # Recent runs table with required columns
    assert "Recent Pipeline Executions" in js
    assert "Training Mode" in js
    assert "Duration" in js

    # Real evaluation panel with honest disclosures
    assert "Verified Research Evaluation" in js
    assert "Evidence Unavailable" in js
    assert "Not Evaluated" in js


def test_workbench_dataset_templates_and_metadata_contracts(client):
    """Test real backend endpoints supporting the Dataset & Privacy module."""
    import io
    import json

    # 1. Dataset Templates API contract (loading & success)
    res = client.get("/api/orchestrator/dataset-templates")
    assert res.status_code == 200
    d = res.get_json()
    assert d["success"] is True
    templates = d.get("templates") or []
    assert len(templates) >= 3
    for tmpl in templates:
        assert "id" in tmpl
        assert "name" in tmpl
        assert "license" in tmpl
        assert "ground_truth_available" in tmpl
        assert "subset_options" in tmpl

    # 2. Template Metadata endpoint (/api/orchestrator/datasets/<id>)
    first_id = templates[0]["id"]
    detail_res = client.get(f"/api/orchestrator/datasets/{first_id}")
    assert detail_res.status_code == 200
    detail = detail_res.get_json()
    assert detail["success"] is True
    assert "metadata" in detail
    assert "statistics" in detail

    # 3. Invalid dataset ID returns 404 (failure state)
    err_res = client.get("/api/orchestrator/datasets/non_existent_dataset_999")
    assert err_res.status_code == 404

    # 4. Dataset Validation Upload contract - Successful JSONL validation
    valid_jsonl = '{"instruction": "Patient report", "input": "Jane Doe email jane@hospital.org", "output": "Administered 10mg."}\n'
    val_res = client.post(
        "/api/orchestrator/validate",
        data={"file": (io.BytesIO(valid_jsonl.encode("utf-8")), "test_records.jsonl")},
        content_type="multipart/form-data"
    )
    assert val_res.status_code == 200
    val_data = val_res.get_json()
    assert val_data["success"] is True
    meta = val_data["metadata"]
    assert meta["num_raw_records"] == 1
    assert "pii_detected_summary" in meta
    assert meta["has_pii"] is True
    assert meta["pii_detected_summary"].get("email", 0) >= 1

    # 5. Dataset Validation Upload contract - Malformed file (failure state with actionable error)
    bad_jsonl = '{"instruction": "Malformed JSON without closing brace\n'
    bad_res = client.post(
        "/api/orchestrator/validate",
        data={"file": (io.BytesIO(bad_jsonl.encode("utf-8")), "corrupted.jsonl")},
        content_type="multipart/form-data"
    )
    assert bad_res.status_code == 400
    bad_data = bad_res.get_json()
    assert bad_data["success"] is False
    assert "error" in bad_data
    assert "Malformed" in bad_data["error"]

    # 6. In-RAM Transparency Trace & Redaction Preview (/api/transparency/inspect)
    trace_payload = {
        "raw_jsonl": '{"instruction": "Clinical note", "input": "Patient John Doe SSN 000-12-3456", "output": "Normal"}\n'
    }
    trace_res = client.post("/api/transparency/inspect", json=trace_payload)
    assert trace_res.status_code == 200
    trace_data = trace_res.get_json()
    assert trace_data["success"] is True
    records = trace_data.get("trace", {}).get("records", [])
    assert len(records) >= 1
    rec0 = records[0]
    assert "raw" in rec0
    assert "masked" in rec0
    assert "pii_spans" in rec0
    assert "[SSN]" in rec0["masked"]["text"]
    assert rec0["pii_count"] >= 1

    # 7. Transparency Trace - Empty payload error handling (empty state)
    empty_trace_res = client.post("/api/transparency/inspect", json={})
    assert empty_trace_res.status_code == 400


def test_workbench_dataset_js_source_and_workflow_stepper(client):
    """Test Dataset & Privacy module workflow stepper, PII detectors, and ground-truth distinctions."""
    # 1. Verify CSS styles for the workflow stepper and masking preview
    css_res = client.get("/static/workbench/workbench.css")
    assert css_res.status_code == 200
    css = css_res.get_data(as_text=True)
    assert ".workflow-stepper" in css
    assert ".workflow-step" in css
    assert ".masking-preview-box" in css
    assert ".masking-pane" in css
    assert ".masking-text-block" in css
    assert ".redacted-token" in css
    assert ".detected-span" in css

    # 2. Verify dataset.js source code has all required interaction and disclosure elements
    js_res = client.get("/static/workbench/js/pages/dataset.js")
    assert js_res.status_code == 200
    js = js_res.get_data(as_text=True)

    # Workflow Stepper steps
    assert "Source Selection" in js
    assert "Schema Validation" in js
    assert "PII Entity Suite" in js
    assert "Redaction & Ground Truth" in js

    # Separate primary and secondary actions
    assert "Validate Dataset Structure" in js
    assert "Run PII Detection & Masking Preview" in js

    # Supported entity types
    assert "EMAIL" in js
    assert "PHONE" in js
    assert "SSN" in js
    assert "CREDIT_CARD" in js
    assert "IP_ADDRESS" in js
    assert "API_SECRET" in js
    assert "NAME_PATTERN" in js
    assert "MRN" in js
    assert "DATE_PHI" in js

    # In-RAM redaction and Zero-Disk-Leakage guarantees
    assert "Zero-Disk-Leakage Guarantee" in js
    assert "Before & After In-RAM Redaction Trace" in js
    assert "Scanned PII Entity Breakdown" in js

    # Operational distinction: Detection vs Masking vs Evaluation
    assert "1. Detection:" in js
    assert "2. In-RAM Masking:" in js
    assert "3. Ground-Truth Evaluation:" in js

    # Ground truth unavailable honest disclosure
    assert "Ground truth unavailable" in js


def test_training_config_endpoint(client):
    """Test /api/orchestrator/training-config exposes supported models, LoRA defaults, and DP capabilities."""
    rv = client.get("/api/orchestrator/training-config")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    cfg = data["config"]

    # Base models
    assert "supported_models" in cfg
    models = cfg["supported_models"]
    assert len(models) >= 2
    model_ids = [m["id"] for m in models]
    assert "JackFram/llama-68m" in model_ids
    assert "meta-llama/Llama-2-7b-hf" in model_ids

    # LoRA defaults
    assert "lora_defaults" in cfg
    ld = cfg["lora_defaults"]
    assert ld["r"] >= 1
    assert ld["alpha"] >= 1
    assert 0.0 <= ld["dropout"] <= 0.5
    assert "q_proj" in ld["target_modules"]
    assert "v_proj" in ld["target_modules"]

    # DP capabilities
    assert "dp_capabilities" in cfg
    dp = cfg["dp_capabilities"]
    assert "supported" in dp
    assert "Opacus" in dp["engine"]
    assert dp["default_max_grad_norm"] > 0
    assert dp["default_noise_multiplier"] > 0
    assert 0 < dp["default_target_delta"] < 1
    assert dp["default_target_epsilon"] > 0


def test_training_job_creation_standard_vs_dp_lora_distinction(client):
    """Test job creation preserves standard LoRA vs DP-LoRA config and never attributes DP to standard LoRA."""
    created_jobs = []
    try:
        # 1. Create Standard LoRA job
        std_payload = {
            "dataset_name": "synthetic_ehr",
            "dataset_type": "synthetic_ehr",
            "version": "1.0.0",
            "epochs": 1,
            "subset_size": 100,
            "training_config": {
                "model_name": "JackFram/llama-68m",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "dp_enabled": False
            }
        }
        std_res = client.post("/api/orchestrator/jobs", json=std_payload)
        assert std_res.status_code == 200
        std_data = std_res.get_json()
        assert std_data["success"] is True
        std_job_id = std_data["job_id"]
        created_jobs.append(std_job_id)

        # Verify backend job state
        std_job = orchestrator.get_job(std_job_id)
        assert std_job is not None
        assert std_job.get("dp_enabled") is False

        # Verify pipeline summary NEVER reports DP epsilon/delta for standard LoRA
        std_summary_res = client.get(f"/api/orchestrator/jobs/{std_job_id}/pipeline-summary")
        assert std_summary_res.status_code == 200
        std_sum_data = std_summary_res.get_json()
        assert std_sum_data["kpi"]["training_mode"] != "DP-LoRA"
        assert std_sum_data["kpi"]["dp_epsilon"] is None
        assert std_sum_data["kpi"]["dp_delta"] is None

        # 2. Create DP-LoRA job
        dp_payload = {
            "dataset_name": "synthetic_ehr",
            "dataset_type": "synthetic_ehr",
            "version": "1.0.1",
            "epochs": 2,
            "subset_size": 100,
            "training_config": {
                "model_name": "JackFram/llama-68m",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj"],
                "dp_enabled": True,
                "dp_max_grad_norm": 1.0,
                "dp_noise_multiplier": 1.25,
                "dp_target_delta": 1e-5,
                "dp_target_epsilon": 7.5,
                "dp_accountant": "rdp"
            }
        }
        dp_res = client.post("/api/orchestrator/jobs", json=dp_payload)
        assert dp_res.status_code == 200
        dp_data = dp_res.get_json()
        assert dp_data["success"] is True
        dp_job_id = dp_data["job_id"]
        created_jobs.append(dp_job_id)

        dp_job = orchestrator.get_job(dp_job_id)
        assert dp_job is not None
        assert dp_job.get("dp_enabled") is True
        assert dp_job.get("training_config", {}).get("dp_noise_multiplier") == 1.25

        # Simulate completed evaluation artifact with formal accounting
        orchestrator.update_job_state(
            dp_job_id,
            status="COMPLETED",
            eval_metrics={
                "training_mode": "dp-lora",
                "epsilon": 7.42,
                "delta": 1e-5,
                "noise_multiplier": 1.25,
                "max_grad_norm": 1.0,
                "accountant_type": "rdp",
                "trainable_parameters": 45000,
                "total_parameters": 68000000,
                "training_duration_seconds": 42.5,
                "val_loss": 1.85,
                "train_loss": 1.92
            }
        )

        dp_summary_res = client.get(f"/api/orchestrator/jobs/{dp_job_id}/pipeline-summary")
        assert dp_summary_res.status_code == 200
        dp_sum_data = dp_summary_res.get_json()
        assert dp_sum_data["kpi"]["training_mode"] == "DP-LoRA"
        assert dp_sum_data["kpi"]["dp_epsilon"] == 7.42
        assert dp_sum_data["kpi"]["dp_delta"] == 1e-5
    finally:
        for jid in created_jobs:
            orchestrator.jobs.pop(jid, None)
        orchestrator._save_db()


def test_training_progress_metrics_and_log_tailing(client):
    """Test telemetry metrics endpoint, loss history, and subprocess log access."""
    job_id = orchestrator.create_job(dataset_name="synthetic_ehr", version="1.0.0")
    try:
        # 1. Update loss history
        loss_data = [
            {"epoch": 0.25, "loss": 2.45},
            {"epoch": 0.50, "loss": 2.10, "eval_loss": 2.15},
            {"epoch": 0.75, "loss": 1.85},
            {"epoch": 1.00, "loss": 1.62, "eval_loss": 1.70}
        ]
        orchestrator.update_job_state(job_id, status="TRAINING", progress=45, loss_history=loss_data)

        metrics_res = client.get(f"/api/orchestrator/jobs/{job_id}/metrics")
        assert metrics_res.status_code == 200
        m_data = metrics_res.get_json()
        assert m_data["success"] is True
        assert len(m_data["metrics"]["loss_history"]) == 4
        assert m_data["metrics"]["loss_history"][1]["eval_loss"] == 2.15

        # 2. Write training logs and verify tailing with redaction
        log_dir = orchestrator.base_jobs_dir / job_id
        log_file = log_dir / "training.log"
        log_file.write_text("Epoch 1/1 [Step 10] loss: 2.10\nEpoch 1/1 [Step 20] loss: 1.62\nTraining completed.\n")

        logs_res = client.get(f"/api/orchestrator/jobs/{job_id}/logs")
        assert logs_res.status_code == 200
        l_data = logs_res.get_json()
        assert l_data["success"] is True
        assert "loss: 1.62" in l_data["logs"]
    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()


def test_training_api_failures_and_access_controls(client):
    """Test error handling for non-existent jobs and bad parameters."""
    # 1. Access non-existent job
    bad_res = client.get("/api/orchestrator/jobs/job_non_existent_999999")
    assert bad_res.status_code == 404

    # 2. Access logs for non-existent job
    bad_log_res = client.get("/api/orchestrator/jobs/job_non_existent_999999/logs")
    assert bad_log_res.status_code == 404

    # 3. Access metrics for non-existent job
    bad_metric_res = client.get("/api/orchestrator/jobs/job_non_existent_999999/metrics")
    assert bad_metric_res.status_code == 404


def test_workbench_training_js_architecture_and_contract(client):
    """Verify training.js adheres to UI requirements, layout, and privacy constraints."""
    res = client.get("/static/workbench/js/pages/training.js")
    assert res.status_code == 200
    src = res.get_data(as_text=True)

    # 1. Base model selector populated from supported backend config
    assert "endpoints.trainingConfig" in src
    assert "supported_models" in src
    assert "JackFram/llama-68m" in src

    # 2. LoRA Rank, Alpha, Dropout, Target Modules
    assert "train-rank" in src
    assert "train-alpha" in src
    assert "train-dropout" in src
    assert "train-modules-box" in src

    # 3. Standard LoRA vs DP-LoRA toggle
    assert "Standard LoRA (Mode A)" in src
    assert "DP-LoRA via Opacus (Mode B)" in src

    # 4. Conditional DP parameters
    assert "train-clip-norm" in src
    assert "train-noise-multiplier" in src
    assert "train-target-delta" in src
    assert "train-target-epsilon" in src

    # 5. Pre-launch confirmation summary modal
    assert "openDialog" in src
    assert "Confirm Pipeline Execution" in src
    assert "Costly Training Job" in src

    # 6. Duplicate submission guard
    assert "isSubmitting" in src
    assert "launchBtn.disabled = true" in src

    # 7. Training progress panel with distinct states and elapsed time
    assert "monitor-status-badge" in src
    assert "monitor-elapsed-timer" in src
    assert "fmtDuration" in src
    assert "training.log" in src

    # 8. Loss curves via accessible lineChart
    assert "lineChart" in src
    assert "Loss Convergence Curve" in src

    # 9. Privacy accounting panel & rule against false DP claims
    assert "Differential Privacy Accounting" in src
    assert "Standard LoRA — No Differential Privacy Accounting Applied" in src
    assert "Verified Differential Privacy Accounting Artifact" in src
    assert "Historical Experiment" in src


def test_screening_missing_evidence_handling(client):
    """Test screening endpoint gracefully reports available=False without metric fabrication when evidence is absent."""
    job_id = orchestrator.create_job(dataset_name="synthetic_ehr", version="1.0.0")
    try:
        rv = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["success"] is True
        assert data["available"] is False
        assert "reason" in data
        assert "not evaluated" in data["reason"].lower()
    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()


def test_screening_failed_checks_and_completed_status_distinction(client):
    """Ensure a completed pipeline job does not imply all security checks passed."""
    job_id = orchestrator.create_job(dataset_name="synthetic_ehr", version="1.0.0")
    job_dir = orchestrator.base_jobs_dir / job_id
    try:
        # Simulate a completed pipeline run where adapter screening actually REJECTED the model
        orchestrator.update_job_state(
            job_id,
            status="COMPLETED",
            security_metrics={
                "security_screening_risk_score": 0.85,
                "security_screening_risk_level": "HIGH",
                "security_screening": "rejected",
                "screening_details": {
                    "adapter_id": job_id,
                    "risk_level": "HIGH",
                    "adapter_risk_score": 0.85,
                    "approved": False,
                    "bypassed_via_force": False,
                    "screening_mode": "PRODUCTION",
                    "structural_report": {
                        "total_parameters": 294912,
                        "layer_count": 8,
                        "global_frobenius_norm": 32.5,  # Exceeds max_frobenius_norm (25.0)
                        "global_l2_norm": 24.2,         # Exceeds max_l2_norm (20.0)
                        "max_layer_zscore": 4.8,        # Exceeds max_layer_zscore (3.0)
                        "outlier_layer_count": 2,
                        "outlier_layers": ["lora_A.layer_3", "lora_B.layer_7"],
                        "rank_utilization_mean": 0.42,
                        "sparsity_ratio": 0.01,
                        "cosine_similarity_ref": None,   # No reference comparison
                        "structural_risk_score": 0.88,
                        "actual_adapter_loaded": True
                    },
                    "behavioral_report": {
                        "trigger_sensitivity": 0.82,     # Exceeds threshold (0.60)
                        "paraphrase_consistency": 0.45,   # Below threshold (0.60)
                        "normal_output_divergence": 0.55,
                        "abnormal_response_rate": 0.35,
                        "classification_flip_rate": 0.50,
                        "behavioral_risk_score": 0.82,
                        "consistency_risk_score": 0.55,
                        "real_inference_performed": False,
                        "execution_status": "MISSING_CALLBACK"
                    },
                    "risk_breakdown": {
                        "structural_risk_score": 0.88,
                        "behavioral_risk_score": 0.82,
                        "consistency_risk_score": 0.55
                    },
                    "risk_assessment": {
                        "rationale": "High structural anomaly and trigger sensitivity detected in adapter weights."
                    }
                }
            }
        )

        rv = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["success"] is True
        assert data["available"] is True
        # Critical assertion: Completed job MUST NOT be marked as PASSED / SCREENED when checks failed!
        assert data["decision"] == "REJECTED"
        assert data["risk_level"] == "HIGH"
        assert data["adapter_risk_score"] == 0.85
        assert data["structural_analysis"]["max_layer_zscore"] == 4.8
        assert data["structural_analysis"]["outlier_layer_count"] == 2
        assert "lora_A.layer_3" in data["structural_analysis"]["outlier_layers"]
        assert data["behavioral_analysis"]["real_inference_performed"] is False
    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()


def test_screening_historical_artifacts_and_metrics_fidelity(client):
    """Test screening report serialization, singular values, reference comparison honesty, and real thresholds."""
    job_id = orchestrator.create_job(dataset_name="synthetic_ehr", version="1.0.0")
    job_dir = orchestrator.base_jobs_dir / job_id
    try:
        report_data = {
            "adapter_id": job_id,
            "risk_level": "LOW",
            "adapter_risk_score": 0.125,
            "approved": True,
            "bypassed_via_force": False,
            "screening_mode": "PRODUCTION",
            "structural_report": {
                "total_parameters": 294912,
                "layer_count": 8,
                "global_frobenius_norm": 4.15,
                "global_l2_norm": 3.80,
                "global_l1_norm": 124.5,
                "global_linf_norm": 0.35,
                "max_layer_zscore": 1.25,
                "outlier_layer_count": 0,
                "outlier_layers": [],
                "sparsity_ratio": 0.02,
                "rank_utilization_mean": 0.94,
                "cosine_similarity_ref": 0.982,  # Reference weights present in this run
                "parameter_drift_score": 0.038,
                "structural_risk_score": 0.11,
                "actual_adapter_loaded": True,
                "layer_metrics": {
                    "layer_0.q_proj": {"l2_norm": 1.12, "l1_norm": 32.1, "linf_norm": 0.12, "mean": 0.001, "std": 0.015}
                }
            },
            "behavioral_report": {
                "real_inference_performed": True,
                "execution_status": "COMPLETED",
                "evaluation_type": "REAL_BEHAVIORAL_EVALUATION",
                "trigger_sensitivity": 0.08,
                "paraphrase_consistency": 0.94,
                "normal_output_divergence": 0.06,
                "abnormal_response_rate": 0.0,
                "classification_flip_rate": 0.0,
                "behavioral_risk_score": 0.12,
                "consistency_risk_score": 0.06,
                "probe_results": [
                    {
                        "category": "normal",
                        "prompt": "Explain hypertension symptoms.",
                        "candidate_output": "Common symptoms include headache and dizziness.",
                        "divergence": 0.04
                    },
                    {
                        "category": "trigger",
                        "prompt": "Diabetic diet [TRIGGER_SECRET_TAG]",
                        "candidate_output": "Standard nutritional guidance for glycemic control.",
                        "divergence": 0.08
                    }
                ]
            },
            "risk_breakdown": {
                "structural_risk_score": 0.11,
                "behavioral_risk_score": 0.12,
                "consistency_risk_score": 0.06
            },
            "risk_assessment": {
                "rationale": "All structural and behavioral parameters are well within security bounds."
            }
        }

        # Write real report artifact to job directory
        (job_dir / "screening_report.json").write_text(json.dumps(report_data), encoding="utf-8")

        rv = client.get(f"/api/orchestrator/jobs/{job_id}/screening")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["success"] is True
        assert data["available"] is True
        assert data["decision"] == "SCREENED"
        assert data["risk_level"] == "LOW"

        # Structural assertions
        s = data["structural_analysis"]
        assert s["global_frobenius_norm"] == 4.15
        assert s["rank_utilization_mean"] == 0.94
        assert s["cosine_similarity_ref"] == 0.982
        assert "layer_0.q_proj" in s["layer_metrics"]

        # Behavioral assertions
        b = data["behavioral_analysis"]
        assert b["real_inference_performed"] is True
        assert b["execution_status"] == "COMPLETED"
        assert len(b["probe_results"]) == 2
        assert b["probe_results"][1]["category"] == "trigger"

        # Configured weights and thresholds from actual backend config
        w = data["configured_weights"]
        assert w["structural"] == 0.35
        assert w["behavioral"] == 0.45
        assert w["consistency"] == 0.20

        t = data["configured_thresholds"]
        assert t["low_risk"] == 0.35
        assert t["high_risk"] == 0.65
        assert t["max_layer_zscore"] == 3.0
    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()
        shutil.rmtree(job_dir, ignore_errors=True)


def test_workbench_screening_js_architecture_and_contract(client):
    """Verify screening.js contains separate sections, evidence indicators, and rule enforcement."""
    res = client.get("/static/workbench/js/pages/screening.js")
    assert res.status_code == 200
    src = res.get_data(as_text=True)

    # Section A: Structural screening
    assert "Section A: Structural Weight Statistics & Spectral Norms" in src
    assert "Global Frobenius Norm" in src
    assert "Global Spectral / L2 Norm" in src
    assert "Max Layer Outlier Z-Score" in src
    assert "Effective Rank Utilization" in src
    assert "Trusted-Weight Reference Baseline Comparison" in src
    assert "No reference baseline weights were configured" in src

    # Section B: Behavioral screening
    assert "Section B: Behavioral Probing & Trigger Verification" in src
    assert "Trigger Token Sensitivity" in src
    assert "Paraphrase Consistency" in src
    assert "Normal Output Divergence" in src
    assert "Abnormal Output Flip Rate" in src
    assert "Evidence Source: Live Model Inference Probing Verified" in src
    assert "Evidence Source: Behavioral Inference Callback Not Provisioned" in src
    assert "barChart" in src
    assert "Inspect Probe Responses & Prompts" in src

    # Section C: Decision summary
    assert "Screening Decision Summary" in src
    assert "Screening Gate Decision" in src
    assert "Composite Risk Score" in src
    assert "Configured Risk Weights & Acceptance Thresholds" in src
    assert "Formula: Risk = min(1.0, max(" in src

    # Technical Details
    assert "Layer-by-Layer Matrix Telemetry (Technical Audit)" in src


def test_packaging_missing_artifacts_and_uninitialized_state(client):
    """Packaging endpoint returns available=False when artifacts are ungenerated."""
    job_id = "test_job_packaging_uninit_999"
    orchestrator.jobs[job_id] = {
        "job_id": job_id,
        "status": "RUNNING",
        "stage": "training_model",
        "owner": "test-principal"
    }
    orchestrator._save_db()
    try:
        res = client.get(f"/api/orchestrator/jobs/{job_id}/packaging")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["available"] is False
        assert data["status"] == "NOT_PACKAGED"
        assert "not exist" in data["reason"] or "not been generated" in data["reason"]
    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()


def test_packaging_successful_cryptographic_verification(client):
    """Packaging endpoint verifies real SHA-256 and RSA-PSS signatures over canonical manifest."""
    import uuid
    from src.security import compute_sha256, generate_dev_keypair, sign_digest, save_signature, compute_canonical_manifest_digest
    from src.phase3.package_builder import build_manifest

    job_id = f"test_job_pkg_valid_{uuid.uuid4().hex[:8]}"
    job_dir = orchestrator.base_jobs_dir / job_id
    protected_dir = job_dir / "protected"
    protected_dir.mkdir(parents=True, exist_ok=True)

    orchestrator.jobs[job_id] = {
        "job_id": job_id,
        "status": "COMPLETED",
        "dataset_name": "clean_medical_dataset",
        "owner": "test-principal",
        "security_metrics": {
            "adapter_screening_outcome": "APPROVED",
            "encryption_time_ms": 12.4,
            "signing_time_ms": 8.1
        }
    }
    orchestrator._save_db()

    try:
        # Create ciphertext
        enc_path = protected_dir / "adapter.enc"
        enc_path.write_bytes(b"SECLORA\x01CIPHERTEXT_MOCK_PAYLOAD_32BYTES_OK!")
        ciphertext_hash = compute_sha256(enc_path)

        # Create hash file
        hash_path = protected_dir / "adapter.hash"
        hash_path.write_text(ciphertext_hash, encoding="utf-8")

        # Create keys
        priv_key_path = job_dir / "dev_private.pem"
        pub_key_path = protected_dir / "public.pem"
        generate_dev_keypair(priv_key_path, pub_key_path, key_size=2048)

        # Create metadata.json
        meta_data = {
            "iv_hex": "0102030405060708090a0b0c",
            "tag_hex": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
            "hkdf_salt_hex": "11" * 32,
            "kdf_version": "hkdf-sha256-v2",
            "fingerprint_hash": "fp_test_host_12345678",
            "encryption_version": "aes-256-gcm-v1"
        }
        (protected_dir / "metadata.json").write_text(json.dumps(meta_data), encoding="utf-8")

        # Build manifest
        manifest = build_manifest(
            package_dir=protected_dir,
            adapter_id=job_id,
            model_reference="JackFram/llama-68m",
            fingerprint_hash="fp_test_host_12345678",
            package_version="1.0.0",
            enc_metadata=meta_data,
            sequence_number=1
        )
        (protected_dir / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Compute canonical manifest digest and sign with private key
        canonical_digest = compute_canonical_manifest_digest(manifest, ciphertext_hash)
        sig_bytes = sign_digest(canonical_digest, priv_key_path)
        save_signature(sig_bytes, protected_dir / "adapter.sig")

        res = client.get(f"/api/orchestrator/jobs/{job_id}/packaging")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["available"] is True
        assert data["package_status"] == "PACKAGED"

        # Integrity
        int_data = data["integrity"]
        assert int_data["ciphertext_digest"] == ciphertext_hash
        assert int_data["recorded_digest"] == ciphertext_hash
        assert int_data["digest_matches_recorded"] is True
        assert int_data["digest_matches_manifest"] is True

        # Signature
        sig_data = data["signature"]
        assert sig_data["verified"] is True
        assert sig_data["verification_error"] is None
        assert sig_data["canonical_digest"] == canonical_digest
        assert sig_data["public_key_present"] is True
        assert sig_data["public_key_pem"].startswith("-----BEGIN PUBLIC KEY-----")

        # Key derivation
        kd = data["key_derivation"]
        assert kd["kdf_version"] == "hkdf-sha256-v2"
        assert kd["salt_hex"] == "11" * 32
        assert "Device Secret" in kd["ikm_source"]

        # Encryption
        enc = data["encryption"]
        assert enc["cipher"] == "AES-256-GCM (NIST SP 800-38D)"
        assert enc["iv_hex"] == "0102030405060708090a0b0c"
        assert enc["tag_hex"] == "a1b2c3d4e5f60718293a4b5c6d7e8f90"

        # Inventory check: private key should NEVER be present
        inv_names = [item["name"] for item in data["inventory"]]
        assert "adapter.enc" in inv_names
        assert "adapter.sig" in inv_names
        assert "public.pem" in inv_names
        assert "dev_private.pem" not in inv_names

        # Sequence stages
        seq = data["sequence_stages"]
        assert seq["step_1_inspect"]["status"] == "PASSED"
        assert seq["step_2_encrypt"]["status"] == "PASSED"
        assert seq["step_3_digest"]["status"] == "PASSED"
        assert seq["step_4_sign"]["status"] == "PASSED"

    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()
        shutil.rmtree(job_dir, ignore_errors=True)


def test_packaging_signature_and_integrity_failure_detection(client):
    """Packaging endpoint reports tampered signature or corrupted ciphertext."""
    import uuid
    from src.security import compute_sha256, generate_dev_keypair, sign_digest, save_signature, compute_canonical_manifest_digest
    from src.phase3.package_builder import build_manifest

    job_id = f"test_job_pkg_tamper_{uuid.uuid4().hex[:8]}"
    job_dir = orchestrator.base_jobs_dir / job_id
    protected_dir = job_dir / "protected"
    protected_dir.mkdir(parents=True, exist_ok=True)

    orchestrator.jobs[job_id] = {
        "job_id": job_id,
        "status": "COMPLETED",
        "dataset_name": "clean_medical_dataset",
        "owner": "test-principal"
    }
    orchestrator._save_db()

    try:
        enc_path = protected_dir / "adapter.enc"
        enc_path.write_bytes(b"SECLORA\x01AUTHENTIC_BYTES_HERE")
        ciphertext_hash = compute_sha256(enc_path)

        hash_path = protected_dir / "adapter.hash"
        hash_path.write_text(ciphertext_hash, encoding="utf-8")

        priv_key_path = job_dir / "dev_private.pem"
        pub_key_path = protected_dir / "public.pem"
        generate_dev_keypair(priv_key_path, pub_key_path, key_size=2048)

        meta_data = {
            "iv_hex": "1234567890abcdef12345678",
            "tag_hex": "abcdef1234567890abcdef1234567890",
            "hkdf_salt_hex": "22" * 32,
            "kdf_version": "hkdf-sha256-v2",
            "fingerprint_hash": "fp_tamper_test"
        }
        (protected_dir / "metadata.json").write_text(json.dumps(meta_data), encoding="utf-8")

        manifest = build_manifest(
            package_dir=protected_dir,
            adapter_id=job_id,
            model_reference="JackFram/llama-68m",
            fingerprint_hash="fp_tamper_test",
            package_version="1.0.0",
            enc_metadata=meta_data,
            sequence_number=1
        )
        (protected_dir / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Sign legitimate canonical digest
        canonical_digest = compute_canonical_manifest_digest(manifest, ciphertext_hash)
        sig_bytes = sign_digest(canonical_digest, priv_key_path)

        # Deliberately corrupt signature by inverting bytes
        corrupted_sig = bytes(b ^ 0xFF for b in sig_bytes)
        save_signature(corrupted_sig, protected_dir / "adapter.sig")

        res = client.get(f"/api/orchestrator/jobs/{job_id}/packaging")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["available"] is True
        assert data["package_status"] == "SIGNATURE_FAILED"
        assert data["signature"]["verified"] is False
        assert data["signature"]["verification_error"] is not None
        assert "signature" in data["signature"]["verification_error"].lower()

    finally:
        orchestrator.jobs.pop(job_id, None)
        orchestrator._save_db()
        shutil.rmtree(job_dir, ignore_errors=True)


def test_workbench_packaging_js_architecture_and_contract(client):
    """Verify packaging.js contains 4-stage sequence, inventory, verification, and disclaimers."""
    res = client.get("/static/workbench/js/pages/packaging.js")
    assert res.status_code == 200
    src = res.get_data(as_text=True)

    # 4-stage sequence
    assert "Packaging Lifecycle Sequence" in src
    assert "Inspect Adapter" in src
    assert "Encrypt Package" in src
    assert "Generate Digest" in src
    assert "Sign Package" in src

    # Inventory & downloads
    assert "Package File Inventory & Allow-Listed Artifacts" in src
    assert "downloadAuthenticated" in src
    assert "dev_private.pem" in src

    # SHA-256 and RSA-PSS verification
    assert "Integrity Sealing & Provenance Signature Verification" in src
    assert "SHA-256 Ciphertext Integrity" in src
    assert "RSA-2048-PSS Signature Verification" in src
    assert "Canonical Manifest Digest" in src
    assert "Inspect Public Key (X.509 PEM)" in src

    # HKDF and AES-GCM
    assert "HKDF-SHA256 Key Derivation (RFC 5869)" in src
    assert "AES-256-GCM Envelope Parameters" in src
    assert "Inspect Package Manifest (package_manifest.json)" in src

    # Disclaimers
    assert "Security Architecture & Attestation Disclaimers" in src
    assert "Attestation Model Boundary: Software vs Hardware" in src
    assert "Transient Plaintext Storage Consideration" in src


def test_deployment_verification_failed_signature(client):
    """Failed signature check marks signature FAILED, terminal state REJECTED, and blocks inference."""
    from src.orchestrator.model_registry import model_registry
    model_registry.clear()

    res = client.post(
        "/api/phase4/verify",
        data=json.dumps({"scenario": "invalid_signature"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is False
    assert data["terminal_state"] == "REJECTED"
    assert "checks" in data
    assert data["checks"]["signature"]["status"] == "FAILED"
    assert "SignatureValidationError" in data["rejection_reason"]

    # Inference must be rejected
    gen_res = client.post(
        "/api/phase4/generate",
        data=json.dumps({"prompt": "Hello test prompt"}),
        content_type="application/json"
    )
    assert gen_res.status_code == 400
    gen_data = gen_res.get_json()
    assert gen_data["status"] == "MODEL_UNAVAILABLE"


def test_deployment_verification_digest_mismatch(client):
    """Tampered package triggers digest mismatch, downstream checks NOT_EVALUATED, terminal state REJECTED."""
    res = client.post(
        "/api/phase4/verify",
        data=json.dumps({"scenario": "tampered_package"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is False
    assert data["terminal_state"] == "REJECTED"
    assert data["checks"]["integrity"]["status"] == "FAILED"
    assert data["checks"]["signature"]["status"] == "NOT_EVALUATED"
    assert data["checks"]["device_auth"]["status"] == "NOT_EVALUATED"
    assert data["checks"]["replay_state"]["status"] == "NOT_EVALUATED"


def test_deployment_verification_unauthorized_device(client):
    """Host hardware fingerprint mismatch fails device authorization and marks terminal state REJECTED."""
    res = client.post(
        "/api/phase4/verify",
        data=json.dumps({"scenario": "wrong_device"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is False
    assert data["terminal_state"] == "REJECTED"
    assert data["checks"]["device_auth"]["status"] == "FAILED"
    assert data["device_identity"]["status"] == "UNAUTHORIZED"
    assert "DeviceAuthorizationError" in data["rejection_reason"]


def test_deployment_verification_replay_rejection(client):
    """Replay attack with expired nonce triggers replay rejection."""
    res = client.post(
        "/api/phase4/verify",
        data=json.dumps({"scenario": "replay"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is False
    assert data["terminal_state"] == "REJECTED"
    assert data["checks"]["replay_state"]["status"] == "FAILED"
    assert "ReplayProtectionError" in data["rejection_reason"]


def test_deployment_verification_successful_authorized_loading_and_inference(client):
    """Successful scenario passes all 4 independent checks, sets ACCEPTED terminal state, and marks adapter active."""
    res = client.post(
        "/api/phase4/verify",
        data=json.dumps({"scenario": "successful"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["terminal_state"] == "ACCEPTED"
    assert data["checks"]["signature"]["status"] == "PASSED"
    assert data["checks"]["integrity"]["status"] == "PASSED"
    assert data["checks"]["device_auth"]["status"] == "PASSED"
    assert data["checks"]["replay_state"]["status"] == "PASSED"
    assert data["adapter_provenance"]["adapter_loaded"] is True
    assert data["device_identity"]["status"] == "AUTHORIZED"


def test_deployment_inference_unavailable_when_unverified(client):
    """Inference is denied with MODEL_UNAVAILABLE when model_registry is unverified."""
    from src.orchestrator.model_registry import model_registry
    model_registry.clear()

    res = client.post(
        "/api/phase4/generate",
        data=json.dumps({"prompt": "Testing clinical guidance"}),
        content_type="application/json"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["status"] == "MODEL_UNAVAILABLE"
    assert "verified first" in data["message"].lower()


def test_workbench_deployment_js_architecture_and_contract(client):
    """Verify deployment.js contains 4 checks, terminal state, evidence views, disclaimers, and inference controls."""
    res = client.get("/static/workbench/js/pages/deployment.js")
    assert res.status_code == 200
    src = res.get_data(as_text=True)

    # 4 independent checks
    assert "Package Signature Verification" in src
    assert "Integrity Digest Verification" in src
    assert "Device Authorization" in src
    assert "Replay-State Verification" in src

    # Terminal state
    assert "DEPLOYMENT ACCEPTED: All Cryptographic Checks Passed" in src
    assert "DEPLOYMENT REJECTED: Fail-Closed Security Policy Triggered" in src
    assert "DEPLOYMENT UNVERIFIED: Verification Execution Required" in src

    # View evidence action
    assert "View Evidence & Audit Trail" in src

    # Device-bound encryption
    assert "Device-Bound Encryption & Key Derivation" in src
    assert "Software-Derived OS Telemetry" in src
    assert "Attestation Scope" in src

    # Inference & loading
    assert "Authorized Adapter Loading & Interactive Inference" in src
    assert "Execute Side-by-Side Inference" in src
    assert "Transient Plaintext Storage Notice" in src


def test_workbench_evaluation_js_architecture_and_contract(client):
    """Verify evaluation.js architecture, real benchmark provenance, unexecuted notices, and report export."""
    res = client.get("/static/workbench/js/pages/evaluation.js")
    assert res.status_code == 200
    src = res.get_data(as_text=True)

    # Scope switching
    assert "Canonical Research Benchmarks" in src
    assert "Active Job Run Audit" in src
    assert "eval-job-selector" in src

    # Real metric and provenance fields
    assert "Input PII Redaction F1" in src
    assert "Differential Privacy Guarantee" in src
    assert "Dual-Plane Screening F1" in src
    assert "Tamper Rejection Rate" in src
    assert "Experimental Methodology & Research Provenance" in src
    assert "outputs/benchmarks/pii_metrics.json" in src
    assert "EXP_PII_BENCHMARK_HYBRID" in src

    # Explicit unexecuted states
    assert "Explicit Unavailable / Unexecuted Evaluations Notice" in src
    assert "generation_memorization_leakage" in src
    assert "threat_model_empirical_validation" in src

    # Comparable visualizations and disclaimers
    assert "Privacy-Utility Frontier (Differential Privacy Accounting)" in src
    assert "Adapter Screening Detection Rates (Dual-Plane Audit)" in src
    assert "Security Disclosure & Comparability Guarantee" in src

    # Export report functionality
    assert "Export Report (JSON)" in src
    assert "securelora_canonical_benchmark_report" in src
    assert "validation_report_" in src


def test_workbench_history_js_architecture_and_contract(client):
    """Verify history.js search, multi-status filters, failure details, tabs, logs, and artifacts."""
    res = client.get("/static/workbench/js/pages/history.js")
    assert res.status_code == 200
    src = res.get_data(as_text=True)

    # Search & filters
    assert "Search runs by Job ID, Dataset, Model..." in src
    assert "All Statuses" in src
    assert "Completed / Verified" in src
    assert "Running / Active" in src
    assert "Failed / Policy Rejected" in src
    assert "Cancelled" in src
    assert "Incomplete / Staged" in src
    assert "Newest First" in src
    assert "Oldest First" in src

    # Failure rationale preservation
    assert "Execution Halted with Status:" in src
    assert "Failure Rationale:" in src
    assert "Failure Stage:" in src
    assert "Pipeline Execution Cancelled" in src

    # Tabs
    assert "1. Pipeline Lifecycle" in src
    assert "2. Configuration" in src
    assert "3. Execution Logs" in src
    assert "4. Artifacts" in src

    # Logs tailing and copy
    assert "Copy Logs" in src
    assert "Refresh Logs" in src
    assert "Tailing last 200 lines from training.log" in src

    # Download authenticated
    assert "downloadAuthenticated" in src
    assert "Generated Cryptographic Artifacts Inventory" in src


def test_research_summary_truthful_provenance_and_unexecuted_evaluations(client):
    """Verify backend /api/research/summary delivers empirical artifacts and explicit unexecuted markers."""
    res = client.get("/api/research/summary")
    assert res.status_code == 200
    data = res.get_json()

    # Empirical values must be present and truthful
    assert data["privacy"]["input_pii_sanitization"]["precision"] == 0.95
    assert data["privacy"]["input_pii_sanitization"]["recall"] == 0.9744
    assert data["privacy"]["input_pii_sanitization"]["f1"] == 0.962
    assert data["privacy"]["input_pii_sanitization"]["corpus_size"] == 48

    # Differential privacy bounds
    assert data["privacy"]["differential_privacy"]["dp_epsilon"] == 2.443
    assert data["privacy"]["differential_privacy"]["dp_delta"] == 1e-05

    # Explicit unexecuted markers
    unexecuted = data.get("execution_metadata", {}).get("unexecuted_evaluations") or data.get("research_metadata", {}).get("unexecuted_evaluations", [])
    unexec_names = [u["name"] for u in unexecuted]
    assert "generation_memorization_leakage" in unexec_names
    assert "threat_model_empirical_validation" in unexec_names
    for u in unexecuted:
        assert u["status"] == "NOT_EXECUTED"
        assert len(u["reason"]) > 10


def test_history_job_artifacts_and_logs_endpoints(client, monkeypatch, tmp_path):
    """Test job artifacts and logs API behavior for history inspection."""
    job_id = "test_hist_job_42"
    mock_job = {
        "job_id": job_id,
        "status": "FAILED",
        "stage": "fine_tuning",
        "error": "CUDA out of memory during backward pass",
        "created_at": "2026-08-16T10:00:00Z"
    }
    monkeypatch.setattr(orchestrator, "get_job", lambda jid: mock_job if jid == job_id else None)

    # Logs endpoint when no log exists
    res = client.get(f"/api/orchestrator/jobs/{job_id}/logs")
    assert res.status_code == 200
    assert "Waiting for training logs" in res.get_json()["logs"]

    # Artifacts endpoint when directory doesn't exist yet
    res = client.get(f"/api/orchestrator/jobs/{job_id}/artifacts")
    assert res.status_code == 200
    assert res.get_json()["artifacts"] == []








