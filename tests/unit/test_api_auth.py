"""
tests/unit/test_api_auth.py
===========================
Unit test suite for Bearer-token authentication middleware and endpoint protection.

Tests:
- missing Authorization -> 401
- malformed Authorization -> 401
- wrong token -> 401
- valid token -> normal endpoint behavior
- security-sensitive endpoints are covered
- Authorization header never appears in logs
- constant-time comparison path is verified
- server fail-closed on missing/empty environment token
"""

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask, jsonify

from src.evaluation.dashboard import app
from src.security.api_auth import (
    AUTH_ENV_VAR,
    AUTH_ENV_VAR_FALLBACK,
    EndpointClassification,
    ENDPOINT_INVENTORY,
    get_endpoint_inventory,
    get_expected_token,
    require_bearer_token,
    validate_bearer_token,
)


@pytest.fixture
def auth_test_app():
    """Isolated Flask application with both public and protected test endpoints."""
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True

    @test_app.route("/api/test/public", methods=["GET"])
    def public_route():
        return jsonify({"success": True, "message": "public data"})

    @test_app.route("/api/test/protected", methods=["POST"])
    @require_bearer_token
    def protected_route():
        return jsonify({"success": True, "message": "protected action completed"})

    return test_app


@pytest.fixture
def auth_client(auth_test_app):
    return auth_test_app.test_client()


def test_endpoint_inventory_classification():
    """Verify endpoint inventory classifications and structure."""
    inventory = get_endpoint_inventory()
    assert len(inventory) > 0

    required_keys = {"path", "method", "classification", "requires_auth", "description"}
    for item in inventory:
        assert required_keys.issubset(item.keys())
        assert item["classification"] in [
            EndpointClassification.PUBLIC_READ.value,
            EndpointClassification.PROTECTED_READ.value,
            EndpointClassification.PROTECTED_WRITE.value,
            EndpointClassification.SECURITY_CRITICAL.value,
        ]

    # Verify key security-sensitive endpoints are explicitly classified as requiring auth
    paths_requiring_auth = {item["path"]: item for item in inventory if item["requires_auth"]}
    assert "/api/orchestrator/jobs" in paths_requiring_auth
    assert "/api/orchestrator/jobs/<job_id>/upload" in paths_requiring_auth
    assert "/api/orchestrator/jobs/<job_id>/start" in paths_requiring_auth
    assert "/api/orchestrator/chat" in paths_requiring_auth
    assert "/api/phase4/verify" in paths_requiring_auth
    assert "/api/phase4/generate" in paths_requiring_auth
    assert "/api/security/simulate-attack" in paths_requiring_auth


def test_missing_authorization_returns_401(auth_client, monkeypatch):
    """Missing Authorization header must return HTTP 401 with standard JSON error."""
    monkeypatch.setenv(AUTH_ENV_VAR, "test-secret-token-xyz")
    resp = auth_client.post("/api/test/protected")
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["success"] is False
    assert "Unauthorized" in data["error"]
    assert "Missing Authorization header" in data["error"]
    assert "WWW-Authenticate" in resp.headers


def test_malformed_authorization_headers_return_401(auth_client, monkeypatch):
    """Malformed Authorization headers must be rejected with HTTP 401."""
    monkeypatch.setenv(AUTH_ENV_VAR, "test-secret-token-xyz")

    malformed_headers = [
        "Basic dXNlcjpwYXNz",
        "Bearer",
        "Bearer ",
        "Bearer token1 token2",
        "Token xyz123",
        "bearer_without_space",
        "CustomAuth 999",
    ]

    for header in malformed_headers:
        resp = auth_client.post("/api/test/protected", headers={"Authorization": header})
        assert resp.status_code == 401, f"Expected 401 for header: {header}"
        data = resp.get_json()
        assert data["success"] is False
        assert "Unauthorized" in data["error"]


def test_wrong_token_returns_401(auth_client, monkeypatch):
    """Supplying an incorrect token must return HTTP 401."""
    monkeypatch.setenv(AUTH_ENV_VAR, "correct-runtime-secret-token")
    resp = auth_client.post(
        "/api/test/protected",
        headers={"Authorization": "Bearer wrong-token-value-1234"},
    )
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["success"] is False
    assert "Unauthorized: Invalid bearer token" in data["error"]


def test_valid_token_allows_request(auth_client, monkeypatch):
    """Supplying a matching token must allow the request to proceed normally."""
    valid_token = "valid-runtime-token-987654"
    monkeypatch.setenv(AUTH_ENV_VAR, valid_token)

    resp = auth_client.post(
        "/api/test/protected",
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["message"] == "protected action completed"


def test_fallback_env_var_supported(auth_client, monkeypatch):
    """API_BEARER_TOKEN fallback environment variable is supported when primary is not set."""
    monkeypatch.delenv(AUTH_ENV_VAR, raising=False)
    monkeypatch.setenv(AUTH_ENV_VAR_FALLBACK, "fallback-token-456")

    resp = auth_client.post(
        "/api/test/protected",
        headers={"Authorization": "Bearer fallback-token-456"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True


def test_missing_server_token_fails_closed(auth_client, monkeypatch):
    """When the server token environment variable is absent, requests fail closed (401)."""
    monkeypatch.delenv(AUTH_ENV_VAR, raising=False)
    monkeypatch.delenv(AUTH_ENV_VAR_FALLBACK, raising=False)

    resp = auth_client.post(
        "/api/test/protected",
        headers={"Authorization": "Bearer any-token-attempt"},
    )
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["success"] is False
    assert "not configured on server" in data["error"]


def test_empty_or_whitespace_server_token_fails_closed(auth_client, monkeypatch):
    """Empty or whitespace-only server token must fail closed."""
    for empty_val in ["", "   ", "\t\n"]:
        monkeypatch.setenv(AUTH_ENV_VAR, empty_val)
        resp = auth_client.post(
            "/api/test/protected",
            headers={"Authorization": "Bearer any-token-attempt"},
        )
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["success"] is False


def test_constant_time_comparison(auth_client, monkeypatch):
    """Token validation must use constant-time comparison path (hmac.compare_digest)."""
    import hmac

    token = "constant-time-test-token"
    monkeypatch.setenv(AUTH_ENV_VAR, token)

    with patch("hmac.compare_digest", wraps=hmac.compare_digest) as mock_compare:
        resp = auth_client.post(
            "/api/test/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert mock_compare.called


def test_authorization_header_never_logged(auth_client, monkeypatch, caplog):
    """Raw Authorization header and secret tokens must never appear in log records."""
    canary_token = "canary-ultra-secret-token-do-not-log-12345"
    monkeypatch.setenv(AUTH_ENV_VAR, "different-token")

    with caplog.at_level(logging.DEBUG):
        # 1. Test unauthorized attempt with canary
        auth_client.post(
            "/api/test/protected",
            headers={"Authorization": f"Bearer {canary_token}"},
        )

        # 2. Test malformed attempt with canary
        auth_client.post(
            "/api/test/protected",
            headers={"Authorization": f"Token {canary_token}"},
        )

    for record in caplog.records:
        assert canary_token not in record.getMessage()
        assert "Authorization:" not in record.getMessage()


def test_public_read_endpoints_remain_accessible(auth_client):
    """Public read endpoints must not require an Authorization header."""
    resp = auth_client.get("/api/test/public")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True


def test_all_security_sensitive_endpoints_reject_unauthenticated():
    """Verify that every security-critical endpoint in the application rejects requests without valid auth."""
    app.config["TESTING"] = True
    test_client = app.test_client()

    # Clear authorization header from test client
    test_client.environ_base.pop("HTTP_AUTHORIZATION", None)

    # 1. Job Creation
    res = test_client.post("/api/orchestrator/jobs", json={"dataset_name": "synthetic"})
    assert res.status_code == 401

    # 2. File Upload
    res = test_client.post("/api/orchestrator/jobs/job_123/upload")
    assert res.status_code == 401

    # 3. Job Start
    res = test_client.post("/api/orchestrator/jobs/job_123/start")
    assert res.status_code == 401

    # 4. Orchestrator Chat / Inference
    res = test_client.post("/api/orchestrator/chat", json={"question": "hello"})
    assert res.status_code == 401

    # 5. Phase 4 Verification
    res = test_client.post("/api/phase4/verify", json={"scenario": "successful"})
    assert res.status_code == 401

    # 6. Phase 4 Generation
    res = test_client.post("/api/phase4/generate", json={"prompt": "test"})
    assert res.status_code == 401

    # 7. Transparency Inspection
    res = test_client.post("/api/transparency/inspect", json={"raw_jsonl": "{}"})
    assert res.status_code == 401

    # 8. Tamper Simulation
    res = test_client.post("/api/tamper/simulate", json={"text": "hello"})
    assert res.status_code == 401

    # 9. Security Attack Simulation
    res = test_client.post("/api/security/simulate-attack", json={"attack_id": "tampering"})
    assert res.status_code == 401

    # 10. Dashboard Chat
    res = test_client.post("/api/chat", json={"question": "hello"})
    assert res.status_code == 401


def test_security_sensitive_endpoints_accept_authenticated(monkeypatch):
    """Verify that protected endpoints accept requests with valid Authorization Bearer token."""
    valid_token = "valid-endpoint-token-2026"
    monkeypatch.setenv(AUTH_ENV_VAR, valid_token)

    app.config["TESTING"] = True
    test_client = app.test_client()
    headers = {"Authorization": f"Bearer {valid_token}"}

    # Job Creation accepts valid token (returns 200 or 400 validation error, not 401)
    res = test_client.post(
        "/api/orchestrator/jobs",
        json={"dataset_name": "synthetic", "epochs": 1},
        headers=headers,
    )
    assert res.status_code != 401

    # Security attack simulation accepts valid token (returns 200)
    res = test_client.post(
        "/api/security/simulate-attack",
        json={"attack_id": "tampering", "payload": "Test payload"},
        headers=headers,
    )
    assert res.status_code == 200
    assert res.get_json()["success"] is True
