"""
test_research_api.py
====================
Unit tests for the read-only research API blueprint.

Tests:
  1. Each endpoint returns HTTP 200 when result files exist.
  2. Each endpoint returns {"available": false} gracefully when files are missing.
  3. Malformed JSON files are handled without a 500 crash.
  4. No private/sensitive fields leak through any endpoint.
  5. Existing /api/phase4/status still works after blueprint registration.
  6. All 6 new endpoints are present in the Flask app's URL map.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import pytest

# Ensure project root on sys.path before any src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Returns a Flask test client with the full dashboard app."""
    os.environ.setdefault("SECURE_LORA_DASHBOARD_PORT", "5099")
    from src.evaluation.dashboard import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def research_api_app():
    """Standalone test client for research_api blueprint only (no ML imports)."""
    from flask import Flask
    from src.evaluation.research_api import research_api_bp
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True
    test_app.register_blueprint(research_api_bp)
    with test_app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1: All 6 new research endpoints present in full app URL map
# ---------------------------------------------------------------------------

def test_research_endpoints_registered(client):
    from src.evaluation.dashboard import app
    rules = {r.rule for r in app.url_map.iter_rules()}
    expected = [
        "/api/research/summary",
        "/api/research/ablation",
        "/api/research/privacy",
        "/api/research/screening",
        "/api/research/adaptive-evasion",
        "/api/research/overhead",
    ]
    for route in expected:
        assert route in rules, f"Route {route} not registered in Flask app."


# ---------------------------------------------------------------------------
# Test 2: Existing /api/phase4/status still works (regression guard)
# ---------------------------------------------------------------------------

def test_phase4_status_still_works(client):
    resp = client.get("/api/phase4/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "loaded" in data
    assert "fingerprint_prefix" in data
    assert "base_model_name" in data


# ---------------------------------------------------------------------------
# Test 3: Research endpoints return 200 with real result files
# ---------------------------------------------------------------------------

def test_summary_endpoint_200(research_api_app):
    resp = research_api_app.get("/api/research/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    # If the file exists it must have available=True and utility block
    if data.get("available"):
        assert "utility" in data
        assert "privacy" in data
        assert "security" in data
        assert "overhead" in data
    else:
        assert "reason" in data


def test_ablation_endpoint_200(research_api_app):
    resp = research_api_app.get("/api/research/ablation")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "available" in data


def test_privacy_endpoint_200(research_api_app):
    resp = research_api_app.get("/api/research/privacy")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "available" in data


def test_screening_endpoint_200(research_api_app):
    resp = research_api_app.get("/api/research/screening")
    assert resp.status_code == 200
    data = resp.get_json()
    if data.get("available"):
        assert "confusion_matrix" in data
        assert "detection_metrics" in data
    else:
        assert "reason" in data


def test_adaptive_evasion_endpoint_200(research_api_app):
    resp = research_api_app.get("/api/research/adaptive-evasion")
    assert resp.status_code == 200
    data = resp.get_json()
    if data.get("available"):
        assert "level_summary" in data
        assert "hypotheses" in data
        assert "seed_stats" in data
    else:
        assert "reason" in data


def test_overhead_endpoint_200(research_api_app):
    resp = research_api_app.get("/api/research/overhead")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "available" in data


# ---------------------------------------------------------------------------
# Test 4: Graceful handling when result files are missing (no 500 crash)
# ---------------------------------------------------------------------------

def test_graceful_missing_files(tmp_path, monkeypatch):
    """Patch _PATHS to point to a non-existent directory; verify no crash."""
    import src.evaluation.research_api as ra
    original_paths = dict(ra._PATHS)
    try:
        # Point all paths to a non-existent directory
        for k in ra._PATHS:
            ra._PATHS[k] = tmp_path / "does_not_exist" / f"{k}.json"

        from flask import Flask
        from src.evaluation.research_api import research_api_bp
        test_app = Flask(__name__)
        test_app.config["TESTING"] = True
        test_app.register_blueprint(research_api_bp)

        with test_app.test_client() as c:
            for route in ["/api/research/summary", "/api/research/ablation",
                           "/api/research/privacy", "/api/research/screening",
                           "/api/research/adaptive-evasion", "/api/research/overhead"]:
                resp = c.get(route)
                assert resp.status_code == 200, f"Expected 200, got {resp.status_code} for {route}"
                data = resp.get_json()
                assert data.get("available") is False, f"Expected available=false for {route} with missing file"
                assert "reason" in data, f"Missing 'reason' key for {route}"
    finally:
        ra._PATHS.update(original_paths)


# ---------------------------------------------------------------------------
# Test 5: Malformed JSON files handled gracefully
# ---------------------------------------------------------------------------

def test_malformed_json_handled(tmp_path, monkeypatch):
    """Write a malformed JSON file and verify the endpoint returns available=false."""
    import src.evaluation.research_api as ra
    original_paths = dict(ra._PATHS)
    try:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ NOT VALID JSON !!!", encoding="utf-8")
        ra._PATHS["b8_summary"] = bad_file

        from flask import Flask
        from src.evaluation.research_api import research_api_bp
        test_app = Flask(__name__)
        test_app.config["TESTING"] = True
        test_app.register_blueprint(research_api_bp)

        with test_app.test_client() as c:
            resp = c.get("/api/research/summary")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data.get("available") is False
            assert "reason" in data
    finally:
        ra._PATHS.update(original_paths)


# ---------------------------------------------------------------------------
# Test 6: No sensitive fields leak from any endpoint
# ---------------------------------------------------------------------------

FORBIDDEN_FIELDS = {
    "private_key", "aes_key", "salt", "device_id", "machine_id",
    "hkdf_key", "secret", "password", "plaintext", "credential",
    "processed_packages",  # raw UUIDs in deployment state
}


def test_no_sensitive_fields_in_summary(research_api_app):
    resp = research_api_app.get("/api/research/summary")
    raw = resp.data.decode("utf-8").lower()
    for field in FORBIDDEN_FIELDS:
        assert field not in raw, f"Sensitive field '{field}' found in /api/research/summary response"


def test_no_sensitive_fields_in_screening(research_api_app):
    resp = research_api_app.get("/api/research/screening")
    raw = resp.data.decode("utf-8").lower()
    for field in FORBIDDEN_FIELDS:
        assert field not in raw, f"Sensitive field '{field}' found in /api/research/screening response"


def test_no_sensitive_fields_in_evasion(research_api_app):
    resp = research_api_app.get("/api/research/adaptive-evasion")
    raw = resp.data.decode("utf-8").lower()
    for field in FORBIDDEN_FIELDS:
        assert field not in raw, f"Sensitive field '{field}' found in /api/research/adaptive-evasion response"


# ---------------------------------------------------------------------------
# Test 7: Classification field present and correct in all endpoints
# ---------------------------------------------------------------------------

def test_classification_field_present(research_api_app):
    for route in ["/api/research/summary", "/api/research/screening",
                  "/api/research/adaptive-evasion", "/api/research/overhead",
                  "/api/research/privacy"]:
        resp = research_api_app.get(route)
        data = resp.get_json()
        if data.get("available"):
            assert data.get("classification") == "HISTORICAL", \
                f"{route} should have classification=HISTORICAL"


# ---------------------------------------------------------------------------
# Test 8: Executed metric artifact returns actual verified value and provenance
# ---------------------------------------------------------------------------

def test_executed_metric_artifact_returns_actual_value(research_api_app):
    resp = research_api_app.get("/api/research/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("available") is True

    # Utility metrics loaded from real summary_metrics.json (E9)
    assert data["utility"]["train_loss"] == 0.8909
    assert data["utility"]["val_loss"] == 0.8301
    assert data["utility"]["perplexity"] == 2.2938
    assert data["utility"]["accuracy"] == 0.8780
    assert data["utility"]["f1"] == 0.8672

    # Separate provenance metadata
    assert "provenance" in data
    assert "utility.train_loss" in data["provenance"]
    prov_loss = data["provenance"]["utility.train_loss"]
    assert prov_loss["value"] == 0.8909
    assert prov_loss["metric_status"] == "VERIFIED"
    assert prov_loss["execution_status"] == "COMPLETED"
    assert "summary_metrics.json" in prov_loss["source_artifact"]

    # Input PII sanitization vs model DP separation
    assert "input_pii_sanitization" in data["privacy"]
    assert data["privacy"]["input_pii_sanitization"]["precision"] == 0.95
    assert data["privacy"]["input_pii_sanitization"]["f1"] == 0.962
    # Verify PII F1 is NOT conflated with model task utility F1
    assert data["utility"]["f1"] != data["privacy"]["pii_f1"]


# ---------------------------------------------------------------------------
# Test 9: Missing / unexecuted artifact returns null and NOT_EXECUTED status
# ---------------------------------------------------------------------------

def test_missing_unexecuted_artifact_returns_null_and_status(tmp_path):
    import src.evaluation.research_api as ra
    original_paths = dict(ra._PATHS)
    try:
        # Point summary_metrics, e9_run, pii_metrics to non-existent files
        # Keep b8_summary valid so endpoint executes
        ra._PATHS["summary_metrics"] = tmp_path / "missing_summary.json"
        ra._PATHS["e9_run"] = tmp_path / "missing_e9.json"
        ra._PATHS["pii_metrics"] = tmp_path / "missing_pii.json"
        ra._PATHS["model_scale"] = tmp_path / "missing_scale.json"

        from flask import Flask
        from src.evaluation.research_api import research_api_bp
        test_app = Flask(__name__)
        test_app.config["TESTING"] = True
        test_app.register_blueprint(research_api_bp)

        with test_app.test_client() as c:
            resp = c.get("/api/research/summary")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["available"] is True

            # Values must be None (JSON null), NOT fabricated numbers
            assert data["utility"]["train_loss"] is None
            assert data["utility"]["val_loss"] is None
            assert data["utility"]["perplexity"] is None
            assert data["utility"]["accuracy"] is None
            assert data["utility"]["f1"] is None

            # Provenance metadata must report NOT_EXECUTED
            prov_loss = data["provenance"]["utility.train_loss"]
            assert prov_loss["value"] is None
            assert prov_loss["metric_status"] == "NOT_EXECUTED"
            assert prov_loss["execution_status"] == "NOT_EXECUTED"

            # Privacy metrics must be null
            assert data["privacy"]["dp_epsilon"] is None
            assert data["privacy"]["pii_precision"] is None
    finally:
        ra._PATHS.update(original_paths)


# ---------------------------------------------------------------------------
# Test 10: Verify no fabricated loss/perplexity/KPI defaults in research API
# ---------------------------------------------------------------------------

def test_no_fabricated_defaults_in_research_api(research_api_app):
    FORBIDDEN_FABRICATED_VALUES = [
        "0.4200",  # old fabricated train_loss
        "0.4500",  # old fabricated val_loss
        "1.5700",  # old fabricated perplexity
        "0.9400",  # old fabricated accuracy
        "2.1795",  # old frontend live fallback
        "1.7289",  # old frontend live fallback
        "5.6346",  # old frontend live fallback
        "0.0500",  # old structural fallback
        "0.0300",  # old behavioral fallback
        "0.0800",  # old risk fallback
    ]

    for route in ["/api/research/summary", "/api/research/ablation",
                  "/api/research/privacy", "/api/research/screening",
                  "/api/research/adaptive-evasion", "/api/research/overhead"]:
        resp = research_api_app.get(route)
        raw = resp.data.decode("utf-8")
        for bad_val in FORBIDDEN_FABRICATED_VALUES:
            assert f'"{bad_val}"' not in raw, f"Fabricated literal {bad_val} found in {route}"

    # Verify model memorization leakage is marked NOT_EXECUTED and null
    resp_priv = research_api_app.get("/api/research/privacy")
    data_priv = resp_priv.get_json()
    assert data_priv["full_pipeline_privacy"]["generation_memorization_leakage"] is None
    assert data_priv["provenance"]["generation_memorization_leakage"]["metric_status"] == "NOT_EXECUTED"


# ---------------------------------------------------------------------------
# Test 11: Frontend dashboard.js contains no hardcoded empirical literals
# ---------------------------------------------------------------------------

def test_dashboard_js_no_hardcoded_empirical_fallbacks():
    js_path = PROJECT_ROOT / "src" / "evaluation" / "static" / "js" / "dashboard.js"
    assert js_path.exists()
    content = js_path.read_text(encoding="utf-8")

    # Assert old fabricated live run fallback metrics are gone
    assert "2.1795" not in content, "Hardcoded train_loss fallback 2.1795 found in dashboard.js"
    assert "1.7289" not in content, "Hardcoded val_loss fallback 1.7289 found in dashboard.js"
    assert "5.6346" not in content, "Hardcoded perplexity fallback 5.6346 found in dashboard.js"
    assert "0.0420" not in content, "Hardcoded structural score fallback 0.0420 found in dashboard.js"
    assert "0.0310" not in content, "Hardcoded behavioral score fallback 0.0310 found in dashboard.js"
    assert "0.1546" not in content, "Hardcoded risk score fallback 0.1546 found in dashboard.js"
    assert "0.0500" not in content, "Hardcoded score fallback 0.0500 found in dashboard.js"
    assert "0.0300" not in content, "Hardcoded score fallback 0.0300 found in dashboard.js"
    assert "0.0800" not in content, "Hardcoded score fallback 0.0800 found in dashboard.js"
    assert "PASS (100.0%)" not in content, "Hardcoded fallback PASS (100.0%) found in dashboard.js"

    # Assert old fabricated chart dataset arrays are gone
    assert "2.10, 1.72, 1.57" not in content, "Hardcoded perplexity curve array found in dashboard.js"
    assert "0.7419, 0.5423" not in content, "Hardcoded val loss curve array found in dashboard.js"
    assert "100, 100, 75, 100" not in content, "Hardcoded PII breakdown array found in dashboard.js"

