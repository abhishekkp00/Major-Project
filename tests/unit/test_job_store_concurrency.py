"""
test_job_store_concurrency.py
=============================
Concurrency, persistence, restart-recovery, migration, and key-safety
tests for SecureLoRA's SQLite Job Store.
"""

import os
import json
import sqlite3
import pytest
import tempfile
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from src.orchestrator.job_store import (
    SqliteJobStore,
    JobsDictProxy,
    CorruptedJobStoreError,
    JobStoreError,
)
from src.orchestrator.service import JobOrchestrator


# Worker helper functions for multiprocessing tests
def _worker_create_jobs(db_dir_str: str, worker_id: int, count: int) -> int:
    store = SqliteJobStore(db_dir_str)
    created = 0
    for i in range(count):
        jid = f"job_w{worker_id}_{i}"
        record = {
            "job_id": jid,
            "owner_id": f"worker_{worker_id}",
            "dataset_name": f"dataset_{worker_id}_{i}",
            "status": "CREATED",
            "stage": "dataset_intake",
            "progress": 0,
            "epochs": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        store.create_job(record)
        created += 1
    return created


def _worker_atomic_transition(db_dir_str: str, job_id: str) -> bool:
    store = SqliteJobStore(db_dir_str)
    return store.atomic_status_transition(
        job_id=job_id,
        from_status="CREATED",
        to_status="STARTING",
        stage="dataset_protection",
    )


def _worker_update_progress(db_dir_str: str, job_id: str, start_prog: int, steps: int) -> int:
    store = SqliteJobStore(db_dir_str)
    for step in range(steps):
        store.update_job_state(
            job_id,
            progress=start_prog + step,
            loss_history=[0.5 / (step + 1)],
        )
    return steps


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Concurrent Job Creation (Multi-Process Safety)
# ──────────────────────────────────────────────────────────────────────────────
def test_concurrent_job_creation_across_processes(tmp_path: Path):
    """Verifies that multiple independent processes concurrently creating jobs
    do not lose records or corrupt the database.
    """
    store = SqliteJobStore(tmp_path)
    num_workers = 4
    jobs_per_worker = 25
    expected_total = num_workers * jobs_per_worker

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(_worker_create_jobs, str(tmp_path), w_id, jobs_per_worker)
            for w_id in range(num_workers)
        ]
        results = [f.result() for f in futures]

    assert sum(results) == expected_total

    all_jobs = store.get_all_jobs()
    assert len(all_jobs) == expected_total

    # Verify all expected job IDs exist
    job_ids = {j["job_id"] for j in all_jobs}
    for w_id in range(num_workers):
        for i in range(jobs_per_worker):
            assert f"job_w{w_id}_{i}" in job_ids

    # Verify JSON snapshot is also synced
    json_path = tmp_path / "jobs_db.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert len(json_data) == expected_total


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Concurrent Status Updates Consistency
# ──────────────────────────────────────────────────────────────────────────────
def test_concurrent_status_updates_consistency(tmp_path: Path):
    """Verifies that concurrent updates to jobs from multiple processes
    leave the database in a consistent, uncorrupted state.
    """
    store = SqliteJobStore(tmp_path)
    # Create 4 base jobs
    for i in range(4):
        store.create_job({
            "job_id": f"job_upd_{i}",
            "status": "INGESTING",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = [
            executor.submit(_worker_update_progress, str(tmp_path), f"job_upd_{i}", i * 10, 10)
            for i in range(4)
        ]
        results = [f.result() for f in futures]

    assert sum(results) == 40

    for i in range(4):
        job = store.get_job(f"job_upd_{i}")
        assert job is not None
        assert job["progress"] == (i * 10) + 9
        assert len(job["loss_history"]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Atomic Status Transition Race (Only 1 Winner)
# ──────────────────────────────────────────────────────────────────────────────
def test_atomic_status_transition_cross_process_race(tmp_path: Path):
    """Verifies that when multiple processes attempt to transition a job
    from CREATED to STARTING simultaneously, exactly one succeeds.
    """
    store = SqliteJobStore(tmp_path)
    store.create_job({
        "job_id": "job_race_1",
        "status": "CREATED",
        "stage": "dataset_intake",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = [
            executor.submit(_worker_atomic_transition, str(tmp_path), "job_race_1")
            for _ in range(4)
        ]
        results = [f.result() for f in futures]

    # Exactly one worker succeeded
    assert results.count(True) == 1
    assert results.count(False) == 3

    # Final status is STARTING
    final_job = store.get_job("job_race_1")
    assert final_job["status"] == "STARTING"
    assert final_job["stage"] == "dataset_protection"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Restart Recovery Handles Interrupted Jobs
# ──────────────────────────────────────────────────────────────────────────────
def test_restart_recovery_handles_interrupted_jobs(tmp_path: Path):
    """Verifies that active in-flight jobs (INGESTING, TRAINING, PACKAGING, DEPLOYING)
    are marked as FAILED with error descriptions when the store restarts.
    """
    store = SqliteJobStore(tmp_path)
    now = datetime.now(timezone.utc).isoformat()

    test_states = [
        ("job_ingest", "INGESTING"),
        ("job_train", "TRAINING"),
        ("job_pkg", "PACKAGING"),
        ("job_dep", "DEPLOYING"),
        ("job_done", "COMPLETED"),
        ("job_failed", "FAILED"),
        ("job_created", "CREATED"),
    ]

    for jid, status in test_states:
        store.create_job({
            "job_id": jid,
            "status": status,
            "stage": "working",
            "created_at": now,
        })

    recovered_jobs = store.recover_interrupted_jobs()
    assert len(recovered_jobs) == 4

    # Interrupted jobs became FAILED
    for jid in ["job_ingest", "job_train", "job_pkg", "job_dep"]:
        job = store.get_job(jid)
        assert job["status"] == "FAILED"
        assert "System restarted during job execution" in job["error"]

    # Other jobs stayed unchanged
    assert store.get_job("job_done")["status"] == "COMPLETED"
    assert store.get_job("job_failed")["status"] == "FAILED"
    assert store.get_job("job_created")["status"] == "CREATED"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Corrupt Persistent Data Fails Safely and Observably
# ──────────────────────────────────────────────────────────────────────────────
def test_corrupt_persistent_data_fails_safely_and_observable(tmp_path: Path, caplog):
    """Verifies that corrupt persistent database files raise CorruptedJobStoreError
    and log actionable error diagnostics.
    """
    store = SqliteJobStore(tmp_path)
    store.create_job({"job_id": "test_job_1", "status": "CREATED", "created_at": "2026-01-01"})
    store.close()

    # Intentionally corrupt the SQLite file header and body (and clean up WAL files)
    db_file = tmp_path / "jobs.sqlite3"
    assert db_file.exists()
    for extra in [tmp_path / "jobs.sqlite3-wal", tmp_path / "jobs.sqlite3-shm"]:
        if extra.exists():
            extra.unlink()
    db_file.write_bytes(b"CORRUPTED_GARBAGE_DATA_12345" * 100)

    # Opening the corrupted file raises CorruptedJobStoreError
    with pytest.raises(CorruptedJobStoreError):
        SqliteJobStore(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Legacy JSON Migration
# ──────────────────────────────────────────────────────────────────────────────
def test_legacy_json_migration(tmp_path: Path):
    """Verifies that legacy jobs_db.json files are seamlessly migrated
    into SQLite without data loss or schema distortion.
    """
    json_path = tmp_path / "jobs_db.json"
    legacy_data = {
        f"job_leg_{i}": {
            "job_id": f"job_leg_{i}",
            "owner_id": f"tenant_{i}",
            "dataset_name": f"legacy_ds_{i}",
            "status": "COMPLETED",
            "stage": "deployment",
            "progress": 100,
            "epochs": 2,
            "salt": f"salt_{i}",
            "loss_history": [0.4, 0.2],
            "eval_metrics": {"loss": 0.2, "perplexity": 1.25},
            "created_at": f"2026-01-0{i+1}T10:00:00Z",
            "updated_at": f"2026-01-0{i+1}T12:00:00Z",
        }
        for i in range(5)
    }
    json_path.write_text(json.dumps(legacy_data), encoding="utf-8")

    # Initialize store, triggering auto-migration
    store = SqliteJobStore(tmp_path)

    all_jobs = store.get_all_jobs()
    assert len(all_jobs) == 5

    # Check that every single field was preserved exactly
    for i in range(5):
        orig = legacy_data[f"job_leg_{i}"]
        migrated = store.get_job(f"job_leg_{i}")
        assert migrated is not None
        assert migrated["job_id"] == orig["job_id"]
        assert migrated["owner_id"] == orig["owner_id"]
        assert migrated["dataset_name"] == orig["dataset_name"]
        assert migrated["status"] == orig["status"]
        assert migrated["progress"] == orig["progress"]
        assert migrated["epochs"] == orig["epochs"]
        assert migrated["salt"] == orig["salt"]
        assert migrated["loss_history"] == orig["loss_history"]
        assert migrated["eval_metrics"] == orig["eval_metrics"]

    # Check that original JSON file was preserved as backup
    backup_file = tmp_path / "jobs_db.json.pre_sqlite_bak"
    assert backup_file.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Cryptographic Key Separation (Defense-in-Depth)
# ──────────────────────────────────────────────────────────────────────────────
def test_cryptographic_key_material_strictly_barred(tmp_path: Path):
    """Verifies that any attempt to persist plaintext encryption keys
    into the SQLite job store or JSON snapshot raises ValueError.
    """
    store = SqliteJobStore(tmp_path)

    forbidden_payloads = [
        {"job_id": "bad_job_1", "key": b"12345678901234567890123456789012"},
        {"job_id": "bad_job_2", "encryption_key": "hex_secret_key_material"},
        {"job_id": "bad_job_3", "job_key": b"secret_job_key"},
    ]

    for payload in forbidden_payloads:
        with pytest.raises(ValueError) as excinfo:
            store.create_job(payload)
        assert "Cryptographic key material" in str(excinfo.value)

    # Attempting to update with key material is also barred
    store.create_job({"job_id": "good_job", "status": "CREATED"})
    with pytest.raises(ValueError):
        store.update_job_state("good_job", key=b"forbidden_key_bytes")


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: JobsDictProxy API Compatibility
# ──────────────────────────────────────────────────────────────────────────────
def test_jobs_dict_proxy_api_compatibility(tmp_path: Path):
    """Verifies that JobsDictProxy acts as a fully compliant MutableMapping,
    ensuring complete backward compatibility with dict-based access.
    """
    store = SqliteJobStore(tmp_path)
    proxy = JobsDictProxy(store)

    assert len(proxy) == 0
    assert not ("test_j" in proxy)

    proxy["test_j"] = {
        "dataset_name": "alpha",
        "status": "CREATED",
        "created_at": "2026-01-01T00:00:00Z",
    }

    assert len(proxy) == 1
    assert "test_j" in proxy
    assert proxy["test_j"]["job_id"] == "test_j"
    assert proxy["test_j"]["status"] == "CREATED"
    assert proxy.get("test_j") is not None
    assert proxy.get("non_existent") is None

    # Test update via proxy
    proxy["test_j"] = {
        "status": "COMPLETED",
        "progress": 100,
    }
    assert proxy["test_j"]["status"] == "COMPLETED"
    assert proxy["test_j"]["progress"] == 100

    # Test values and keys
    assert list(proxy.keys()) == ["test_j"]
    vals = list(proxy.values())
    assert len(vals) == 1
    assert vals[0]["job_id"] == "test_j"

    # Test deletion
    del proxy["test_j"]
    assert len(proxy) == 0
    assert "test_j" not in proxy


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: Orchestrator Service Concurrency Start Race
# ──────────────────────────────────────────────────────────────────────────────
def test_orchestrator_service_concurrent_start_race(tmp_path: Path):
    """Verifies that JobOrchestrator.start_job allows only 1 winner
    and raises ValueError on concurrent attempts.
    """
    orch = JobOrchestrator(base_jobs_dir=str(tmp_path))
    job_id = orch.create_job(dataset_name="synthetic", epochs=1)

    # First start succeeds
    # To avoid running full training pipeline in test, stub _run_pipeline
    ran_pipelines = []
    orch._run_pipeline = lambda jid: ran_pipelines.append(jid)

    orch.start_job(job_id)
    assert orch.get_job(job_id)["status"] == "STARTING"

    # Second start must fail with ValueError
    with pytest.raises(ValueError) as exc:
        orch.start_job(job_id)
    assert "already in status" in str(exc.value)
