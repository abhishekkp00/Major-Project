"""
Route-level authentication, per-job authorization, artifact-download hardening,
and credential-hygiene tests for the SecureLoRA Flask API.
"""
import logging
import os

import pytest

from src.evaluation.dashboard import app
from src.orchestrator.service import orchestrator
from src.security import api_auth
from src.security.api_auth import ENDPOINT_INVENTORY, PUBLIC_ROUTES, principal_id_for_token

TOKEN_A = "tenant-a-token-0123456789abcdef"
TOKEN_B = "tenant-b-token-fedcba9876543210"


@pytest.fixture(autouse=True)
def _tokens(monkeypatch):
    monkeypatch.setenv(api_auth.AUTH_ENV_VAR, TOKEN_A)
    monkeypatch.setenv(api_auth.AUTH_ENV_VAR_MULTI, TOKEN_B)
    monkeypatch.delenv(api_auth.STRICT_OWNERSHIP_ENV_VAR, raising=False)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def H(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    """Two jobs owned by different principals with real artifact directories."""
    root = tmp_path / "jobs"
    for jid, tok in (("job_a", TOKEN_A), ("job_b", TOKEN_B)):
        prot = root / jid / "protected"
        prot.mkdir(parents=True)
        (prot / "adapter.enc").write_bytes(b"cipher-" + jid.encode())
        (prot / "public.pem").write_bytes(b"pub")
        (prot / "dev_private.pem").write_bytes(b"PRIVATE")
        (root / jid / "training.log").write_text(
            "step 1 loss 0.5\nSECURE_LORA_KEY_HEX=" + "ab" * 32 + "\nloading /home/user/secret/model\n"
            "Authorization: Bearer " + TOKEN_A + "\n", encoding="utf-8")
    (root / "outside.json").write_text('{"leak": true}')
    records = {
        "job_a": {"job_id": "job_a", "owner_id": principal_id_for_token(TOKEN_A), "status": "COMPLETED",
                  "salt": "SUPERSECRETSALT", "created_at": "2026-01-02", "error": "failed at /srv/app/x.py"},
        "job_b": {"job_id": "job_b", "owner_id": principal_id_for_token(TOKEN_B), "status": "COMPLETED",
                  "salt": "OTHERSALT", "created_at": "2026-01-01"},
    }
    monkeypatch.setattr(orchestrator, "base_jobs_dir", root)
    monkeypatch.setattr(orchestrator, "jobs", records)
    monkeypatch.setattr(orchestrator, "_save_db", lambda: None)
    return root


# ---- route inventory vs. actual Flask registration --------------------------
def _concrete(rule):
    url = rule.rule
    for conv in rule.arguments:
        url = url.replace(f"<{conv}>", "x").replace(f"<string:{conv}>", "x").replace(f"<path:{conv}>", "x")
    return url


def test_every_registered_route_is_in_inventory():
    inventory = {(m.method, m.path) for m in ENDPOINT_INVENTORY}
    for rule in app.url_map.iter_rules():
        for method in rule.methods - {"HEAD", "OPTIONS"}:
            assert (method, rule.rule) in inventory, f"Unclassified route {method} {rule.rule}"


def test_every_non_public_route_rejects_unauthenticated(client):
    checked = 0
    for rule in app.url_map.iter_rules():
        for method in rule.methods - {"HEAD", "OPTIONS"}:
            if (method, rule.rule) in PUBLIC_ROUTES:
                continue
            resp = client.open(_concrete(rule), method=method)
            assert resp.status_code == 401, f"{method} {rule.rule} -> {resp.status_code}"
            checked += 1
    assert checked >= 25


def test_public_routes_stay_available(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}
    assert client.get("/").status_code == 200
    assert client.get("/api/orchestrator/datasets").status_code != 401
    assert client.get("/api/research/summary").status_code != 401


def test_public_routes_have_documented_rationale():
    for _method, path in PUBLIC_ROUTES:
        key = "/api/research/*" if path.startswith("/api/research/") else path
        assert key in api_auth.PUBLIC_RATIONALE, f"missing rationale for {path}"


# ---- token validation -------------------------------------------------------
@pytest.mark.parametrize("hdr", [None, "", "   ", "Bearer", "Bearer ", "Basic abc", "Bearer a b",
                                 "Token " + TOKEN_A, "Bearer wrong-token", "bearer"])
def test_bad_credentials_fail_closed_on_get_and_post(client, jobs, hdr):
    headers = {"Authorization": hdr} if hdr is not None else {}
    assert client.get("/api/orchestrator/jobs/job_a/logs", headers=headers).status_code == 401
    assert client.get("/api/orchestrator/jobs", headers=headers).status_code == 401
    assert client.post("/api/orchestrator/jobs/job_a/start", headers=headers).status_code == 401


def test_unconfigured_server_fails_closed(client, monkeypatch):
    monkeypatch.delenv(api_auth.AUTH_ENV_VAR, raising=False)
    monkeypatch.delenv(api_auth.AUTH_ENV_VAR_FALLBACK, raising=False)
    monkeypatch.delenv(api_auth.AUTH_ENV_VAR_MULTI, raising=False)
    assert client.get("/api/orchestrator/jobs", headers=H("anything")).status_code == 401


def test_constant_time_comparison_used(monkeypatch):
    calls = []
    real = api_auth.hmac.compare_digest
    monkeypatch.setattr(api_auth.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    ok, _, _ = api_auth.authenticate_bearer(f"Bearer {TOKEN_B}")
    assert ok and len(calls) == 2  # compared against every configured token, no early exit


# ---- authenticated access & per-job isolation -------------------------------
def test_valid_token_allows_own_job(client, jobs):
    for suffix in ("", "/logs", "/metrics", "/artifacts", "/pipeline-summary"):
        r = client.get(f"/api/orchestrator/jobs/job_a{suffix}", headers=H(TOKEN_A))
        assert r.status_code == 200, suffix
    assert client.get("/api/orchestrator/jobs/job_a/download/adapter.enc", headers=H(TOKEN_A)).data == b"cipher-job_a"


def test_cross_job_access_is_rejected_uniformly(client, jobs):
    for suffix in ("", "/logs", "/metrics", "/artifacts", "/report", "/screening",
                   "/pipeline-summary", "/stream", "/download/adapter.enc"):
        other = client.get(f"/api/orchestrator/jobs/job_b{suffix}", headers=H(TOKEN_A))
        missing = client.get(f"/api/orchestrator/jobs/nope{suffix}", headers=H(TOKEN_A))
        assert other.status_code == 404 == missing.status_code, suffix
        assert other.get_json() == missing.get_json()  # no job-id enumeration oracle
    assert client.post("/api/orchestrator/jobs/job_b/start", headers=H(TOKEN_A)).status_code == 404
    assert client.post("/api/orchestrator/jobs/job_b/upload", headers=H(TOKEN_A)).status_code == 404


def test_job_list_is_filtered_per_principal(client, jobs):
    ids = [j["job_id"] for j in client.get("/api/orchestrator/jobs", headers=H(TOKEN_A)).get_json()["jobs"]]
    assert ids == ["job_a"]


def test_transparency_and_chat_job_ids_are_isolated(client, jobs):
    r = client.post("/api/transparency/inspect", json={"job_id": "job_b"}, headers=H(TOKEN_A))
    assert r.status_code == 404
    r = client.post("/api/chat", json={"question": "hi", "job_id": "job_b"}, headers=H(TOKEN_A))
    assert r.status_code == 404


def test_create_job_records_owner_and_hides_it(client, tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "base_jobs_dir", tmp_path)
    monkeypatch.setattr(orchestrator, "jobs", {})
    monkeypatch.setattr(orchestrator, "_save_db", lambda: None)
    jid = client.post("/api/orchestrator/jobs", json={"dataset_name": "synthetic"}, headers=H(TOKEN_B)).get_json()["job_id"]
    assert orchestrator.jobs[jid]["owner_id"] == principal_id_for_token(TOKEN_B)
    assert client.get(f"/api/orchestrator/jobs/{jid}", headers=H(TOKEN_A)).status_code == 404
    body = client.get(f"/api/orchestrator/jobs/{jid}", headers=H(TOKEN_B)).get_json()["job"]
    assert "owner_id" not in body and "salt" not in body


def test_legacy_ownerless_job_policy(client, jobs, monkeypatch):
    orchestrator.jobs["legacy"] = {"job_id": "legacy", "status": "COMPLETED", "created_at": "2025"}
    assert client.get("/api/orchestrator/jobs/legacy", headers=H(TOKEN_A)).status_code == 200
    monkeypatch.setenv(api_auth.STRICT_OWNERSHIP_ENV_VAR, "1")
    assert client.get("/api/orchestrator/jobs/legacy", headers=H(TOKEN_A)).status_code == 404


# ---- response hygiene --------------------------------------------------------
def test_status_response_strips_salt_owner_and_paths(client, jobs):
    body = client.get("/api/orchestrator/jobs/job_a", headers=H(TOKEN_A)).get_data(as_text=True)
    assert "SUPERSECRETSALT" not in body and "owner_id" not in body and "/srv/app" not in body


def test_logs_are_redacted(client, jobs):
    text = client.get("/api/orchestrator/jobs/job_a/logs", headers=H(TOKEN_A)).get_json()["logs"]
    assert "ab" * 32 not in text and TOKEN_A not in text and "/home/user" not in text
    assert "loss 0.5" in text


def test_internal_errors_do_not_leak(client, monkeypatch):
    def boom():
        raise RuntimeError("secret internal path /etc/shadow")
    monkeypatch.setattr(orchestrator, "get_all_jobs", boom)
    r = client.get("/api/orchestrator/jobs", headers=H(TOKEN_A))
    assert r.status_code == 500 and "shadow" not in r.get_data(as_text=True)


# ---- download path safety ----------------------------------------------------
@pytest.mark.parametrize("name", [
    "../outside.json", "..%2foutside.json", "%2e%2e/outside.json", "..\\outside.json",
    "dev_private.pem", "adapter.enc%00.json", "nonexistent.enc", "evil.sh", ".hidden.json",
])
def test_download_cannot_escape_or_fetch_unapproved(client, jobs, name):
    r = client.get(f"/api/orchestrator/jobs/job_a/download/{name}", headers=H(TOKEN_A))
    assert r.status_code in (403, 404)
    assert b"leak" not in r.data and b"PRIVATE" not in r.data


def test_download_blocks_symlink_escape(client, jobs):
    link = jobs / "job_a" / "protected" / "link.json"
    link.symlink_to(jobs / "outside.json")
    r = client.get("/api/orchestrator/jobs/job_a/download/link.json", headers=H(TOKEN_A))
    assert r.status_code == 403 and b"leak" not in r.data
    names = [a["name"] for a in client.get("/api/orchestrator/jobs/job_a/artifacts",
                                           headers=H(TOKEN_A)).get_json()["artifacts"]]
    assert "link.json" not in names and "dev_private.pem" not in names


def test_cannot_download_other_jobs_artifact_via_traversal(client, jobs):
    r = client.get("/api/orchestrator/jobs/job_a/download/..%2f..%2fjob_b%2fprotected%2fadapter.enc", headers=H(TOKEN_A))
    assert r.status_code in (403, 404) and b"cipher-job_b" not in r.data


# ---- credential hygiene in logs ---------------------------------------------
def test_no_credentials_in_logs(client, jobs, caplog):
    with caplog.at_level(logging.DEBUG):
        client.get("/api/orchestrator/jobs", headers=H("wrong-canary-token-XYZ"))
        client.get("/api/orchestrator/jobs/job_a/logs", headers=H(TOKEN_A))
        client.get("/api/orchestrator/jobs/job_b/logs", headers=H(TOKEN_A))
    for secret in ("wrong-canary-token-XYZ", TOKEN_A, TOKEN_B):
        assert secret not in caplog.text
