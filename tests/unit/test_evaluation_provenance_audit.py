"""
test_evaluation_provenance_audit.py
====================================
Comprehensive ML evaluation provenance, execution status, and audit tests.

Verifies:
  1. Provenance traceability: Every metric links to its canonical artifact and experiment.
  2. Truthful execution status: VERIFIED, EXECUTED_UNVERIFIED, SIMULATED, NOT_EXECUTED.
  3. No fabricated/hardcoded success scores (1.0) or fallbacks.
  4. Strict semantic categorization into the 6 distinct domains.
  5. Unexecuted evaluations (generation memorization leakage) report None and NOT_EXECUTED.
  6. Simulations are visibly and explicitly labeled with is_simulated=True.
  7. Missing and malformed artifacts fail safely without inventing default values.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def research_app():
    """Standalone Flask test client for research_api blueprint."""
    from flask import Flask
    from src.evaluation.research_api import research_api_bp
    app = Flask("audit_research_app")
    app.config["TESTING"] = True
    app.register_blueprint(research_api_bp)
    return app.test_client()


@pytest.fixture
def dashboard_app():
    """Flask test client for dashboard simulation endpoints."""
    from src.evaluation.dashboard import app
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# Test 1: Provenance and Semantic Categories in /api/research/summary
# ---------------------------------------------------------------------------

def test_summary_provenance_and_semantic_categories(research_app):
    resp = research_app.get("/api/research/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    prov = data["provenance"]

    expected_categories = {
        "utility.train_loss": "utility",
        "utility.val_loss": "utility",
        "utility.perplexity": "utility",
        "utility.accuracy": "utility",
        "utility.f1": "utility",
        "privacy.pii_precision": "input_pii_sanitization",
        "privacy.pii_recall": "input_pii_sanitization",
        "privacy.pii_f1": "input_pii_sanitization",
        "privacy.pii_corpus_size": "input_pii_sanitization",
        "privacy.dp_epsilon": "differential_privacy",
        "privacy.dp_delta": "differential_privacy",
        "privacy.dp_noise_multiplier": "differential_privacy",
        "privacy.dp_clipping_norm": "differential_privacy",
        "privacy.generation_memorization_leakage": "generation_memorization_leakage",
        "security.tamper_rejection_rate": "deployment_security",
        "security.signature_rejection_rate": "deployment_security",
        "security.device_rejection_rate": "deployment_security",
        "security.replay_rejection_rate": "deployment_security",
        "overhead.encryption_ms": "overhead",
        "overhead.decryption_ms": "overhead",
        "overhead.verification_ms": "overhead",
        "overhead.deployment_gate_ms": "overhead",
        "overhead.screening_ms": "overhead",
    }

    for key, expected_cat in expected_categories.items():
        assert key in prov, f"Metric '{key}' missing from summary provenance"
        m = prov[key]
        assert m["semantic_category"] == expected_cat, \
            f"Metric '{key}' category '{m.get('semantic_category')}' != expected '{expected_cat}'"
        assert "is_simulated" in m
        assert "metric_status" in m
        assert "execution_status" in m

    # Verify execution metadata describes executed vs unexecuted
    assert "execution_metadata" in data
    meta = data["execution_metadata"]
    assert "experiments_executed" in meta
    assert "unexecuted_evaluations" in meta
    unexec_names = [u["name"] for u in meta["unexecuted_evaluations"]]
    assert "generation_memorization_leakage" in unexec_names


# ---------------------------------------------------------------------------
# Test 2: Generation-level Memorization Leakage Truthfulness
# ---------------------------------------------------------------------------

def test_generation_memorization_leakage_is_unexecuted(research_app):
    """Probing LLM output for verbatim training extraction is unexecuted; must be None / NOT_EXECUTED."""
    # Check in summary
    resp = research_app.get("/api/research/summary")
    data = resp.get_json()
    mem_prov = data["provenance"]["privacy.generation_memorization_leakage"]
    assert mem_prov["value"] is None
    assert mem_prov["metric_status"] == "NOT_EXECUTED"
    assert mem_prov["execution_status"] == "NOT_EXECUTED"
    assert mem_prov["semantic_category"] == "generation_memorization_leakage"

    # Check in privacy endpoint
    resp_priv = research_app.get("/api/research/privacy")
    data_priv = resp_priv.get_json()
    assert data_priv["full_pipeline_privacy"]["generation_memorization_leakage"] is None
    priv_mem = data_priv["provenance"]["generation_memorization_leakage"]
    assert priv_mem["value"] is None
    assert priv_mem["metric_status"] == "NOT_EXECUTED"
    assert priv_mem["execution_status"] == "NOT_EXECUTED"


# ---------------------------------------------------------------------------
# Test 3: No Fabricated 1.0 Fallbacks in Summary (Device Data Fallback)
# ---------------------------------------------------------------------------

def test_no_fabricated_tamper_signature_fallbacks_when_device_data_used(tmp_path):
    """When falling back to device_data (which only evaluates device binding and replay),
    tamper and signature rejection rates must be None and NOT_EXECUTED, never fabricated 1.0.
    """
    import src.evaluation.research_api as ra
    orig_paths = dict(ra._PATHS)
    try:
        # Point summary_metrics and e9_run to non-existent files so device_data is reached
        ra._PATHS["summary_metrics"] = tmp_path / "nonexistent_summary.json"
        ra._PATHS["e9_run"] = tmp_path / "nonexistent_e9.json"

        from flask import Flask
        from src.evaluation.research_api import research_api_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(research_api_bp)
        with app.test_client() as c:
            resp = c.get("/api/research/summary")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["available"] is True

            sec = data["security"]
            prov = data["provenance"]

            # tamper and signature must NOT be fabricated as 1.0
            assert sec["tamper_rejection_rate"] is None, "tamper_rejection_rate was fabricated instead of None"
            assert sec["signature_rejection_rate"] is None, "signature_rejection_rate was fabricated instead of None"
            assert prov["security.tamper_rejection_rate"]["metric_status"] == "NOT_EXECUTED"
            assert prov["security.signature_rejection_rate"]["metric_status"] == "NOT_EXECUTED"

            # device and replay rejection should still be loaded from device_data
            assert sec["device_rejection_rate"] == 1.0
            assert sec["replay_rejection_rate"] == 1.0
    finally:
        ra._PATHS.update(orig_paths)


# ---------------------------------------------------------------------------
# Test 4: No Fabricated Behavioral Detection Fallback in Adaptive Evasion
# ---------------------------------------------------------------------------

def test_adaptive_evasion_no_fabricated_behavioral_inversion(research_app):
    """Ensure behavioral_detection is never calculated via (1.0 - structural_detection)."""
    resp = research_app.get("/api/research/adaptive-evasion")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True

    level_summary = data.get("level_summary", {})
    # In canonical adaptive_evasion_metrics.json, level_2 has struct_only_detection_rate = 1.0.
    # The old buggy code did: 1.0 - 1.0 = 0.0 for behavioral_detection.
    # Truthful behavior: behav_detection_rate is None per-level (or actual empirical number).
    if "level_2" in level_summary:
        l2 = level_summary["level_2"]
        assert l2.get("behavioral_detection") is None or l2.get("behavioral_detection") == 1.0, \
            "Behavioral detection in level_2 must not be fabricated as 0.0 via inversion"

    # Category summary must contain real empirical rates
    if "category_summary" in data:
        cat = data["category_summary"]
        if "BASIC_SUSPICIOUS" in cat:
            assert cat["BASIC_SUSPICIOUS"]["behav_detection_rate"] == 1.0
        if "CLEAN" in cat:
            assert cat["CLEAN"]["behav_detection_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test 5: Simulated Endpoints are Explicitly Tagged as SIMULATED
# ---------------------------------------------------------------------------

def test_security_demonstration_is_labeled_simulated(research_app):
    """The /api/security/demonstration endpoint must be explicitly labeled SIMULATED."""
    resp = research_app.get("/api/security/demonstration")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "SIMULATED"
    assert data["classification"] == "SIMULATED_DEMONSTRATION"
    assert data["is_simulated"] is True
    assert "demonstration_notice" in data
    assert data["provenance"]["is_simulated"] is True
    assert data["provenance"]["execution_status"] == "SIMULATED"

    for attack in data["attacks"]:
        assert attack.get("is_simulated") is True
        assert attack.get("execution_status") == "SIMULATED"


def test_dashboard_tamper_simulate_is_labeled_simulated(dashboard_app):
    """POST /api/tamper/simulate must include is_simulated=True."""
    token = os.environ.get("SECURELORA_API_TOKEN") or "test-bearer-token-securing-lora-2026"
    resp = dashboard_app.post(
        "/api/tamper/simulate",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "Hello world", "stage": "3", "payload": "CORRUPT"}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("is_simulated") is True
    assert data.get("simulation_type") == "INTERACTIVE_DEMONSTRATION"
    assert data.get("execution_status") == "SIMULATED"


def test_dashboard_security_simulate_attack_is_labeled_simulated(dashboard_app):
    """POST /api/security/simulate-attack must include is_simulated=True."""
    token = os.environ.get("SECURELORA_API_TOKEN") or "test-bearer-token-securing-lora-2026"
    resp = dashboard_app.post(
        "/api/security/simulate-attack",
        headers={"Authorization": f"Bearer {token}"},
        json={"attack_id": "tampering", "payload": "Tamper test payload"}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("is_simulated") is True
    assert data.get("simulation_type") == "INTERACTIVE_SECURITY_SIMULATION"
    assert data.get("attack", {}).get("is_simulated") is True


# ---------------------------------------------------------------------------
# Test 6: Defensible Metric Status Criteria
# ---------------------------------------------------------------------------

def test_defensible_status_criteria():
    """Verify make_metric distinguishes VERIFIED, EXECUTED_UNVERIFIED, SIMULATED, NOT_EXECUTED."""
    from src.evaluation.research_api import make_metric

    # Case A: null value or NOT_EXECUTED
    m1 = make_metric(None, "path/to/file.json", "COMPLETED")
    assert m1["value"] is None
    assert m1["metric_status"] == "NOT_EXECUTED"
    assert m1["execution_status"] == "NOT_EXECUTED"

    m2 = make_metric(0.85, "path/to/file.json", "NOT_EXECUTED")
    assert m2["value"] is None
    assert m2["metric_status"] == "NOT_EXECUTED"

    # Case B: simulated
    m3 = make_metric(1.0, "path/to/demo.json", is_simulated=True)
    assert m3["is_simulated"] is True
    assert m3["metric_status"] == "SIMULATED"
    assert m3["execution_status"] == "SIMULATED"

    # Case C: single-run execution without multi-seed verification
    m4 = make_metric(0.75, "outputs/single_run.json", "COMPLETED")
    assert m4["metric_status"] == "EXECUTED_UNVERIFIED"

    # Case D: multi-seed verified run
    m5 = make_metric(0.8301, "outputs/research/metrics/summary_metrics.json", "COMPLETED", num_seeds=3)
    assert m5["metric_status"] == "VERIFIED"

    # Case E: canonical variance-proven artifact
    m6 = make_metric(0.962, "outputs/benchmarks/pii_metrics.json", "COMPLETED", has_variance_proof=True)
    assert m6["metric_status"] == "VERIFIED"


# ---------------------------------------------------------------------------
# Test 7: Metric Traceability and Required Metadata in All Endpoints
# ---------------------------------------------------------------------------

def test_all_endpoints_metadata_traceability(research_app):
    """Ensure every research endpoint returns valid source_artifact and metadata."""
    routes = [
        "/api/research/summary",
        "/api/research/privacy",
        "/api/research/screening",
        "/api/research/adaptive-evasion",
        "/api/research/device-binding",
        "/api/research/model-scale",
        "/api/research/overhead",
    ]
    for route in routes:
        resp = research_app.get(route)
        assert resp.status_code == 200, f"Route {route} returned HTTP {resp.status_code}"
        data = resp.get_json()
        assert data.get("available") is True
        prov = data.get("provenance", {})
        assert len(prov) > 0, f"No provenance returned in {route}"
        for metric_name, m_info in prov.items():
            assert "source_artifact" in m_info
            assert "metric_status" in m_info
            assert "execution_status" in m_info
            assert "semantic_category" in m_info
            assert "is_simulated" in m_info
