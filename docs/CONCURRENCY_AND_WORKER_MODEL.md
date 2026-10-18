# SecureLoRA Concurrency and Worker Model

## 1. Executive Summary & Architectural Evaluation

### The Problem in Prior Architecture
Prior to this migration, SecureLoRA stored orchestration job records in a single JSON document (`outputs/jobs/jobs_db.json`). In-memory updates were guarded by a Python `threading.Lock()` inside `JobOrchestrator`.

While a `threading.Lock` protects concurrent threads **within a single Python OS process**, it provides **zero synchronization across multiple processes**. In modern production deployments utilizing WSGI/ASGI servers (such as Gunicorn, uWSGI, or multi-container horizontal scaling):
- Multiple worker processes simultaneously modifying `jobs_db.json` would overwrite each other's updates (lost updates / read-modify-write race conditions).
- A process reading while another process wrote could read partial or corrupted JSON.
- Two worker processes receiving concurrent requests to start the same job could both read `status == "CREATED"` simultaneously, leading to duplicate execution races.

### The Solution: Transactional SQLite Job Store
SecureLoRA now utilizes an ACID-compliant transactional store implemented via `SqliteJobStore` (`src/orchestrator/job_store.py`):
1. **Write-Ahead Logging (WAL)**: `PRAGMA journal_mode = WAL` enables concurrent readers and writers without read/write blocking.
2. **Atomic Immediate Transactions**: `BEGIN IMMEDIATE` ensures cross-process exclusive write locks are acquired upfront, eliminating deadlocks.
3. **Cross-Process Busy Timeout**: `PRAGMA busy_timeout = 30000` (30 seconds) handles lock contention across workers gracefully.
4. **Parameterized Queries & Schema Constraints**: Prevent SQL injection and enforce schema integrity.
5. **Zero-Loss Auto-Migration**: Automatically migrates existing legacy JSON records into SQLite on first startup, while keeping a pre-migration backup (`jobs_db.json.pre_sqlite_bak`).
6. **Continuous JSON Snapshot Sync**: Continuously exports an atomic JSON snapshot (`jobs_db.json`) via tempfile rename, ensuring 100% backward compatibility for legacy offline scripts (e.g. `scripts/generate_paper_figures.py`).
7. **Transparent Mapping Proxy**: `JobsDictProxy(MutableMapping)` provides backward-compatible dict-like semantics (`orchestrator.jobs[...]`) across the codebase and existing test suites.

---

## 2. Supported Worker & Deployment Model

SecureLoRA defines a clear separation of concerns across tiers:

```
┌─────────────────────────────────────────────────────────────┐
│                       HTTP Ingress                          │
│               (Nginx / Reverse Proxy / LB)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌───────────────────────────────┐     ┌───────────────────────────────┐
│       API Worker (PID 101)    │     │       API Worker (PID 102)    │
│    Gunicorn / Flask Worker    │     │    Gunicorn / Flask Worker    │
└──────────────┬────────────────┘     └───────────────┬───────────────┘
               │                                      │
               │  Multi-Process Safe Transactions     │
               │  (WAL Mode + Parameterized SQL)      │
               ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Persistent SQLite Database                  │
│                   (outputs/jobs/jobs.sqlite3)               │
└─────────────────────────────────────────────────────────────┘
```

### A. API Tier (Multi-Worker Safe)
- **Supported Configurations**:
  - Gunicorn with multiple worker processes (e.g., `gunicorn -w 4 -b 0.0.0.0:5000 src.evaluation.dashboard:app`)
  - Flask threaded development servers
  - Standalone CLI commands running concurrently against the workspace
- **Safety Guarantees**:
  - Concurrent `create_job()` calls across workers will not lose records or produce duplicate IDs.
  - Concurrent `get_job()` and `get_all_jobs()` read consistent snapshot states.
  - Concurrent `update_job_state()` commits execute transactionally.
  - Concurrent `start_job()` calls race safely: only one worker successfully executes the atomic status transition (`CREATED` -> `STARTING`); the other worker receives a `ValueError` indicating the job is already running.

### B. Training Pipeline Execution Tier
- **Worker Execution Model**:
  - Fine-tuning and security screening pipelines require dedicated GPU/CPU resources.
  - When `start_job(job_id)` is invoked, `SqliteJobStore.atomic_status_transition` guarantees that only one process can take ownership of transitioning `CREATED` -> `STARTING`.
  - The pipeline executes phases sequentially:
    1. Phase 1: Ingestion & PII Masking
    2. Phase 2: Secure LoRA Fine-Tuning
    3. Phase 3: Adapter Screening & Security Verification
    4. Phase 4: Encryption, Signature, and Deployment Packaging
- **Multi-Node Caveat**:
  - Training pipelines that fork local background threads assume execution occurs on the node where `outputs/jobs/<job_id>` filesystem directories reside.
  - In a multi-machine cluster, SQLite should be deployed with a distributed transactional store (such as PostgreSQL) and a shared artifact volume (or object storage such as S3/GCS).

---

## 3. Cryptographic Key Management & Storage Boundaries

### Strict Invariant: Plaintext Keys Never Persisted
SecureLoRA adheres to strict cryptographic separation:
1. **Volatile Ephemeral Keyring**:
   - The 256-bit AES encryption key for each job is generated in volatile memory (`JobOrchestrator._job_keys[job_id]`).
   - Plaintext keys are passed to training subprocesses only via ephemeral, securely scoped environment variables or memory channels, and cleared immediately via `clear_job_key()`.
2. **Persistent Store Enforcement (Defense-in-Depth)**:
   - `SqliteJobStore._serialize_row_values` strictly inspects all incoming record dictionaries.
   - If any field matches `key`, `encryption_key`, `secret_key`, `job_key`, or `symmetric_key`, a `ValueError` is raised immediately, aborting the write before any data touches SQLite or JSON.
3. **Legacy Migration Cleansing**:
   - During migration from legacy JSON or disk artifacts, legacy `.key` files on disk are loaded into volatile memory and immediately wiped using DOD-standard multipass overwrite (`src.security.shred.shred_file`).

---

## 4. Failure Modes & Crash Recovery

### Interrupted Job Recovery
In the event of an unplanned process crash, power failure, or node reboot during pipeline execution:
- Jobs in non-terminal active states (`INGESTING`, `TRAINING`, `PACKAGING`, `DEPLOYING`) are left in an indeterminate state.
- Upon restart, `SqliteJobStore.recover_interrupted_jobs()` automatically identifies all interrupted jobs and transitions them to `FAILED` with the audit message:
  `"System restarted during job execution."`
- Terminal states (`COMPLETED`, `FAILED`, `SECURITY_POLICY_REJECTED`, `SECURITY_SCREENING_FAILED`) and non-started jobs (`CREATED`) remain untouched.

### Database Corruption Safety
- SQLite integrity is verified via `PRAGMA integrity_check`.
- If persistent database bytes are corrupted or altered out-of-band:
  - `SqliteJobStore` detects corruption on initialization or read and raises `CorruptedJobStoreError`.
  - Errors are logged with full contextual diagnostic output, preventing silent data distortion.

---

## 5. Migration Strategy & Compatibility

| Component | Legacy Model | Migrated Architecture |
|---|---|---|
| **Primary Storage** | `outputs/jobs/jobs_db.json` | `outputs/jobs/jobs.sqlite3` |
| **Concurrency Control** | Single-process `threading.Lock` | OS-level SQLite WAL transactions |
| **Query Mechanism** | Full-dict memory load & rewrite | Parameterized SQL queries with indexed lookups |
| **API Contract** | `orch.jobs[job_id]`, `orch.get_job()` | `JobsDictProxy` wrapping `SqliteJobStore` (100% compatible) |
| **Legacy Backup** | None | `outputs/jobs/jobs_db.json.pre_sqlite_bak` |
| **External Scripts** | Read `outputs/jobs/jobs_db.json` | Continues to read auto-synced `outputs/jobs/jobs_db.json` |

### Running the Verification Suite
All concurrency, migration, and persistence properties are verified by automated tests:
```bash
./venv/bin/pytest tests/unit/test_job_store_concurrency.py -v
```
To verify the full orchestrator and authorization suite:
```bash
./venv/bin/pytest tests/unit/test_job_store_concurrency.py \
                  tests/integration/test_orchestrator.py \
                  tests/unit/test_api_authorization.py \
                  tests/unit/test_pii_fail_closed.py \
                  tests/unit/test_orchestrator_routes.py \
                  tests/security/test_ephemeral_key_handling.py -v
```
