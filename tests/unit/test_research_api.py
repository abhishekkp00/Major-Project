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
