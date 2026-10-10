"""Regression tests: PII masking must fail closed, never pass raw text through."""
import json
import logging

import pytest

from src.common.exceptions import PIIMaskingError
from src.orchestrator import dataset_processor as dp
from src.phase1 import preprocessing as pp

SECRET_EMAIL = "alice.secret@example.com"
SECRET_PHONE = "555-867-5309"


def _boom(_text):
    raise RuntimeError(f"engine crashed on {SECRET_EMAIL}")


@pytest.fixture
def broken_phase1(monkeypatch):
    monkeypatch.setattr(pp, "_pii_masker", _boom)


@pytest.fixture
def broken_orchestrator(monkeypatch):
    monkeypatch.setattr(dp, "_get_pii_masker", lambda: _boom)


# --- successful masking preserved -------------------------------------------
def test_phase1_masks_email_and_phone():
    r = pp.preprocess_record({"text": f"Mail {SECRET_EMAIL} or call {SECRET_PHONE}."})
    assert SECRET_EMAIL not in r["text"]
    assert SECRET_PHONE not in r["text"]


def test_orchestrator_masks_email_and_phone_all_fields():
    out = dp.preprocess_and_standardize([
        {"instruction": f"Write to {SECRET_EMAIL}", "input": f"Call {SECRET_PHONE}", "output": "ok"}
    ])
    blob = json.dumps(out)
    assert SECRET_EMAIL not in blob and SECRET_PHONE not in blob


# --- fail closed: phase1 ----------------------------------------------------
def test_phase1_mask_raises_and_does_not_return_original(broken_phase1):
    with pytest.raises(PIIMaskingError) as ei:
        pp._mask(f"contact {SECRET_EMAIL}", "text")
    assert SECRET_EMAIL not in str(ei.value)


def test_phase1_preprocess_record_propagates(broken_phase1):
    with pytest.raises(PIIMaskingError):
        pp.preprocess_record({"instruction": "a", "input": "", "output": SECRET_EMAIL})


def test_phase1_dataset_aborts_with_record_index_and_no_pii(broken_phase1, caplog):
    records = [{"text": "clean"}, {"text": f"leak {SECRET_EMAIL}"}]
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PIIMaskingError) as ei:
            pp.preprocess_dataset(records)
    assert ei.value.record_index == 1 or ei.value.record_index == 2
    assert SECRET_EMAIL not in str(ei.value)
    assert SECRET_EMAIL not in caplog.text


def test_phase1_masker_load_failure_is_fail_closed(monkeypatch):
    monkeypatch.setattr(pp, "_pii_masker", None)
    monkeypatch.setattr(pp, "_get_pii_masker", lambda: (_ for _ in ()).throw(ImportError("no model")))
    with pytest.raises(PIIMaskingError):
        pp._mask("anything")


def test_phase1_non_string_masker_output_rejected(monkeypatch):
    monkeypatch.setattr(pp, "_pii_masker", lambda t: (None, {}))
    with pytest.raises(PIIMaskingError):
        pp._mask("something")


# --- fail closed: orchestrator dataset processor ----------------------------
def test_orchestrator_preprocess_raises_and_no_output(broken_orchestrator, caplog):
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PIIMaskingError) as ei:
            dp.preprocess_and_standardize([{"text": "ok"}, {"text": f"x {SECRET_EMAIL}"}])
    assert ei.value.record_index == 1
    assert ei.value.field == "text"
    assert SECRET_EMAIL not in str(ei.value)
    assert SECRET_EMAIL not in caplog.text


def test_orchestrator_masker_unavailable_fails_closed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "src.security.pii_engine":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(PIIMaskingError):
        dp._get_pii_masker()


def test_mask_pii_false_is_explicit_opt_out_only():
    out = dp.preprocess_and_standardize([{"text": SECRET_EMAIL}], mask_pii=False)
    assert out[0]["text"] == SECRET_EMAIL  # explicit caller opt-out, not a failure path


# --- orchestrator job: failure surfaced, nothing encrypted, no training ------
def test_job_fails_closed_without_encrypting_or_training(tmp_path, monkeypatch, broken_orchestrator):
    from src.orchestrator.service import JobOrchestrator
    import subprocess

    popen_calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popen_calls.append(a) or (_ for _ in ()).throw(AssertionError("training started")))

    orch = JobOrchestrator(base_jobs_dir=str(tmp_path))
    job_id = "job-pii"
    job_dir = tmp_path / job_id
    (job_dir / "raw_inputs").mkdir(parents=True)
    (job_dir / "raw_inputs" / "d.jsonl").write_text(json.dumps({"text": f"mail {SECRET_EMAIL}"}) + "\n")
    orch.jobs[job_id] = {
        "job_id": job_id, "status": "CREATED", "salt": "00" * 16, "epochs": 1,
        "dataset_name": "d.jsonl", "dataset_type": "d.jsonl", "version": "1", "subset_size": 1,
    }
    try:
        orch._run_pipeline(job_id)
    except Exception:
        pass
    job = orch.jobs[job_id]
    assert job["status"] == "FAILED"
    assert job["failed_stage"] == "preprocessing"
    assert job["failure_reason"] == "pii_masking_failed"
    assert SECRET_EMAIL not in json.dumps(job)
    assert not popen_calls
    assert not list(job_dir.rglob("encrypted_dataset.enc"))
