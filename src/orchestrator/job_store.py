"""
job_store.py
============
Transactional, Cross-Process SQLite Job Store for SecureLoRA.

Replaces the legacy process-local JSON-based store with an ACID-compliant,
multi-process safe SQLite transactional store:
  - WAL (Write-Ahead Logging) mode enabled for high-concurrency read/write.
  - Parameterized queries and schema constraints.
  - Cross-process transactional safety using OS file-level advisory locking and busy timeouts.
  - Safe, automatic migration from legacy jobs_db.json without data loss.
  - Restart recovery: interrupted jobs (INGESTING, TRAINING, PACKAGING, DEPLOYING)
    atomically transition to FAILED.
  - Zero cryptographic key persistence: encryption keys are strictly barred from
    persistent records.
  - Backward-compatible dictionary proxy interface (JobsDictProxy) for seamless
    integration with existing code and tests.
  - Continuous atomic JSON snapshot sync for backward compatibility with external scripts.
"""

import os
import json
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Iterator, Mapping, MutableMapping

logger = logging.getLogger("secure_lora.orchestrator.job_store")


class JobStoreError(Exception):
    """Base exception for job store operations."""
    pass


class CorruptedJobStoreError(JobStoreError):
    """Raised when persistent job store data is corrupted or unreadable."""
    pass


class JobStoreMigrationError(JobStoreError):
    """Raised when migration from legacy records fails."""
    pass


# Known columns in the SQLite schema
SQLITE_COLUMNS = [
    "job_id",
    "owner_id",
    "dataset_name",
    "dataset_type",
    "subset_size",
    "version",
    "status",
    "stage",
    "progress",
    "epochs",
    "salt",
    "loss_history",
    "eval_metrics",
    "verification_steps",
    "security_metrics",
    "pii_summary",
    "schema_detected",
    "num_records",
    "current_epoch",
    "error",
    "failed_stage",
    "failure_reason",
    "failed_record_index",
    "failed_field",
    "extra_metadata",
    "created_at",
    "updated_at",
]

JSON_COLUMNS = {
    "loss_history",
    "eval_metrics",
    "verification_steps",
    "security_metrics",
    "pii_summary",
}


def _sanitize_for_storage(val: Any) -> Any:
    """Recursively replaces NaN, Infinity, -Infinity with None for clean JSON serialization."""
    if isinstance(val, float):
        import math
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(val, dict):
        return {k: _sanitize_for_storage(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize_for_storage(v) for v in val]
    return val


class SqliteJobStore:
    """
    Transactional SQLite-backed job store supporting multi-process concurrency,
    safe migration, atomic status transitions, and restart recovery.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        owner_id TEXT,
        dataset_name TEXT,
        dataset_type TEXT,
        subset_size INTEGER,
        version TEXT,
        status TEXT NOT NULL,
        stage TEXT,
        progress INTEGER DEFAULT 0,
        epochs INTEGER,
        salt TEXT,
        loss_history TEXT,
        eval_metrics TEXT,
        verification_steps TEXT,
        security_metrics TEXT,
        pii_summary TEXT,
        schema_detected TEXT,
        num_records INTEGER,
        current_epoch REAL,
        error TEXT,
        failed_stage TEXT,
        failure_reason TEXT,
        failed_record_index INTEGER,
        failed_field TEXT,
        extra_metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
    CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_id);
    """

    def __init__(
        self,
        base_jobs_dir: Path,
        db_filename: str = "jobs.sqlite3",
        legacy_json_filename: str = "jobs_db.json",
        auto_sync_json: bool = True
    ):
        self.base_jobs_dir = Path(base_jobs_dir)
        self.base_jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_jobs_dir / db_filename
        self.json_path = self.base_jobs_dir / legacy_json_filename
        self.auto_sync_json = auto_sync_json

        self._local = threading.local()
        self._init_lock = threading.Lock()

        # Initialize schema and verify integrity
        self._init_db()

        # Migrate from legacy jobs_db.json if SQLite is empty
        self._check_and_migrate_legacy()

        # Recover interrupted jobs
        self.recover_interrupted_jobs()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection configured with WAL and busy timeout."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    check_same_thread=False,
                    isolation_level=None  # Managed transactions
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA busy_timeout = 30000;")
                self._local.conn = conn
            except sqlite3.DatabaseError as exc:
                logger.error("Failed to connect to SQLite job store at %s: %s", self.db_path, exc)
                raise CorruptedJobStoreError(f"Cannot open job database: {exc}") from exc
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Provides an atomic transaction block with immediate lock acquisition."""
        conn = self.get_connection()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise

    def verify_integrity(self) -> bool:
        """Runs PRAGMA integrity_check to detect database corruption."""
        try:
            conn = self.get_connection()
            cur = conn.execute("PRAGMA integrity_check;")
            rows = cur.fetchall()
            if not rows or rows[0][0] != "ok":
                err_msg = "; ".join(r[0] for r in rows)
                logger.error("SQLite integrity check failed for %s: %s", self.db_path, err_msg)
                raise CorruptedJobStoreError(f"Database corruption detected: {err_msg}")
            return True
        except sqlite3.DatabaseError as exc:
            logger.error("Database error during integrity check: %s", exc)
            raise CorruptedJobStoreError(f"Database error during integrity check: {exc}") from exc

    def close(self):
        """Closes the current thread's connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _init_db(self):
        """Initializes database schema and indexes."""
        with self._init_lock:
            try:
                conn = self.get_connection()
                cur = conn.execute("PRAGMA integrity_check;")
                rows = cur.fetchall()
                if not rows or rows[0][0] != "ok":
                    err_msg = "; ".join(r[0] for r in rows)
                    raise CorruptedJobStoreError(f"Database corruption detected: {err_msg}")
                conn.executescript(self.SCHEMA)
            except sqlite3.DatabaseError as exc:
                logger.error("Failed to initialize SQLite schema at %s: %s", self.db_path, exc)
                raise CorruptedJobStoreError(f"Failed to initialize schema: {exc}") from exc

    def _check_and_migrate_legacy(self):
        """Automatically migrates existing jobs_db.json into SQLite if table is empty."""
        with self._init_lock:
            if not self.json_path.exists():
                return

            try:
                conn = self.get_connection()
                cur = conn.execute("SELECT count(*) FROM jobs;")
                count = cur.fetchone()[0]
                if count == 0:
                    logger.info("Empty SQLite jobs table detected. Migrating from %s...", self.json_path)
                    migrated = self.migrate_from_json(self.json_path)
                    logger.info("Successfully migrated %d legacy jobs into SQLite.", migrated)
            except CorruptedJobStoreError:
                raise
            except Exception as exc:
                logger.warning("Legacy migration check encountered an error: %s", exc)

    def migrate_from_json(self, json_path: Path) -> int:
        """
        Parses and migrates legacy jobs_db.json into the SQLite database.
        Runs within a single atomic transaction. Does not discard existing records.
        """
        path = Path(json_path)
        if not path.exists():
            return 0

        # Create safe backup copy of legacy json if not already present
        bak_path = path.with_suffix(".json.pre_sqlite_bak")
        if not bak_path.exists():
            try:
                import shutil
                shutil.copy2(path, bak_path)
            except Exception as e:
                logger.warning("Could not create legacy backup: %s", e)

        try:
            raw_text = path.read_text(encoding="utf-8")
            if not raw_text.strip():
                return 0
            records_map = json.loads(raw_text)
            if not isinstance(records_map, dict):
                raise JobStoreMigrationError("Legacy jobs_db.json root must be a JSON object mapping job_id to record.")
        except json.JSONDecodeError as exc:
            logger.error("Corrupted legacy JSON file at %s: %s", path, exc)
            raise CorruptedJobStoreError(f"Malformed legacy JSON records at {path}: {exc}") from exc

        count = 0
        with self.transaction() as conn:
            for job_id, job in records_map.items():
                if not isinstance(job, dict):
                    continue
                # Ensure job_id is present
                job_record = dict(job)
                job_record["job_id"] = job_id
                self._insert_row(conn, job_record)
                count += 1

        logger.info("Migrated %d records from %s into SQLite.", count, path)
        return count

    def _serialize_row_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validates security constraints and formats record fields for SQLite columns."""
        # Task 5 Requirement: Strictly prohibit cryptographic key persistence
        for forbidden in ("key", "encryption_key", "secret_key", "job_key", "symmetric_key"):
            if forbidden in record:
                raise ValueError(
                    f"Security violation: Cryptographic key material ('{forbidden}') "
                    "must NEVER be persisted to the job store."
                )

        sanitized = _sanitize_for_storage(record)
        known_keys = set(SQLITE_COLUMNS)
        extra = {}

        for k, v in sanitized.items():
            if k not in known_keys and k != "extra_metadata":
                extra[k] = v

        row = {}
        for col in SQLITE_COLUMNS:
            if col == "extra_metadata":
                row[col] = json.dumps(extra) if extra else None
            elif col in JSON_COLUMNS:
                val = sanitized.get(col)
                if val is not None:
                    row[col] = json.dumps(val) if not isinstance(val, str) else val
                else:
                    row[col] = None
            else:
                row[col] = sanitized.get(col)

        # Enforce required defaults
        if not row["created_at"]:
            row["created_at"] = datetime.now(timezone.utc).isoformat()
        if not row["updated_at"]:
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
        if not row["status"]:
            row["status"] = "CREATED"

        return row

    def _deserialize_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Converts SQLite Row to dict, deserializing JSON fields and unrolling extra_metadata."""
        data = dict(row)

        for col in JSON_COLUMNS:
            raw_json = data.get(col)
            if raw_json and isinstance(raw_json, str):
                try:
                    data[col] = json.loads(raw_json)
                except Exception:
                    pass
            elif raw_json is None:
                if col in ("loss_history",):
                    data[col] = []
                elif col in ("eval_metrics", "verification_steps", "security_metrics", "pii_summary"):
                    data[col] = {}

        extra_json = data.pop("extra_metadata", None)
        if extra_json and isinstance(extra_json, str):
            try:
                extra = json.loads(extra_json)
                if isinstance(extra, dict):
                    data.update(extra)
            except Exception:
                pass

        return data

    def _insert_row(self, conn: sqlite3.Connection, record: Dict[str, Any]):
        """Inserts or replaces a row inside an active connection."""
        row_data = self._serialize_row_values(record)
        cols = list(row_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [row_data[c] for c in cols]

        sql = f"INSERT OR REPLACE INTO jobs ({col_names}) VALUES ({placeholders});"
        conn.execute(sql, values)

    def create_job(self, job_record: Dict[str, Any]) -> str:
        """
        Atomically creates a new job record.
        Raises ValueError if job_id already exists or contains key material.
        """
        job_id = job_record.get("job_id")
        if not job_id:
            raise ValueError("job_record must contain 'job_id'")

        with self.transaction() as conn:
            cur = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?;", (job_id,))
            if cur.fetchone():
                raise ValueError(f"Job '{job_id}' already exists in store.")

            self._insert_row(conn, job_record)

        if self.auto_sync_json:
            self._sync_json_snapshot()

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single job by ID. Returns None if not found."""
        try:
            conn = self.get_connection()
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._deserialize_row(row)
        except sqlite3.DatabaseError as exc:
            logger.error("Error reading job '%s' from SQLite: %s", job_id, exc)
            raise CorruptedJobStoreError(f"Database error reading job '{job_id}': {exc}") from exc

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Retrieves all jobs ordered by created_at DESC."""
        try:
            conn = self.get_connection()
            cur = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC;")
            return [self._deserialize_row(r) for r in cur.fetchall()]
        except sqlite3.DatabaseError as exc:
            logger.error("Error reading all jobs from SQLite: %s", exc)
            raise CorruptedJobStoreError(f"Database error reading jobs: {exc}") from exc

    def get_all_job_ids(self) -> List[str]:
        """Retrieves list of all job_ids."""
        conn = self.get_connection()
        cur = conn.execute("SELECT job_id FROM jobs ORDER BY created_at DESC;")
        return [r[0] for r in cur.fetchall()]

    def count(self) -> int:
        """Returns total count of jobs."""
        conn = self.get_connection()
        cur = conn.execute("SELECT count(*) FROM jobs;")
        return cur.fetchone()[0]

    def has_job(self, job_id: str) -> bool:
        """Checks if a job exists."""
        conn = self.get_connection()
        cur = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?;", (job_id,))
        return cur.fetchone() is not None

    def update_job_state(self, job_id: str, **kwargs) -> bool:
        """
        Updates fields of an existing job within an atomic transaction.
        Automatically updates 'updated_at' timestamp.
        """
        # Security check: Prohibit storing keys
        for forbidden in ("key", "encryption_key", "secret_key", "job_key", "symmetric_key"):
            if forbidden in kwargs:
                raise ValueError(
                    f"Security violation: Cryptographic key material ('{forbidden}') "
                    "must NEVER be persisted to the job store."
                )

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
            row = cur.fetchone()
            if row is None:
                return False

            existing = self._deserialize_row(row)
            existing.update(kwargs)
            existing["updated_at"] = now_iso
            self._insert_row(conn, existing)

        if self.auto_sync_json:
            self._sync_json_snapshot()

        return True

    def atomic_status_transition(
        self,
        job_id: str,
        expected_status: Optional[str] = None,
        new_status: Optional[str] = None,
        from_status: Optional[str] = None,
        to_status: Optional[str] = None,
        **additional_updates
    ) -> bool:
        """
        Atomically transitions a job's status if and only if its current status
        matches expected_status. Prevents concurrent execution races across workers.
        Returns True if transition succeeded, False if current status did not match.
        """
        expected = expected_status or from_status
        target = new_status or to_status
        if not expected or not target:
            raise ValueError("Both expected_status (or from_status) and new_status (or to_status) are required.")

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND status = ?;",
                (job_id, expected)
            )
            row = cur.fetchone()
            if row is None:
                return False

            existing = self._deserialize_row(row)
            existing["status"] = target
            existing["updated_at"] = now_iso
            existing.update(additional_updates)
            self._insert_row(conn, existing)

        if self.auto_sync_json:
            self._sync_json_snapshot()

        return True

    def delete_job(self, job_id: str) -> bool:
        """Deletes a job record by ID."""
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?;", (job_id,))
            deleted = cur.rowcount > 0

        if deleted and self.auto_sync_json:
            self._sync_json_snapshot()

        return deleted

    def recover_interrupted_jobs(self) -> List[str]:
        """
        On restart/startup recovery: transitions any non-terminal active jobs
        (INGESTING, TRAINING, PACKAGING, DEPLOYING) to FAILED status.
        Runs as a single atomic transaction.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        recovered_ids = []

        try:
            with self.transaction() as conn:
                cur = conn.execute(
                    """
                    SELECT job_id FROM jobs
                    WHERE status IN ('INGESTING', 'TRAINING', 'PACKAGING', 'DEPLOYING');
                    """
                )
                recovered_ids = [r[0] for r in cur.fetchall()]

                if recovered_ids:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'FAILED',
                            error = 'System restarted during job execution.',
                            updated_at = ?
                        WHERE status IN ('INGESTING', 'TRAINING', 'PACKAGING', 'DEPLOYING');
                        """,
                        (now_iso,)
                    )
                    logger.warning(
                        "Restart recovery: transitioned %d interrupted jobs to FAILED: %s",
                        len(recovered_ids), recovered_ids
                    )
        except Exception as exc:
            logger.error("Failed during restart recovery: %s", exc)

        if recovered_ids and self.auto_sync_json:
            self._sync_json_snapshot()

        return recovered_ids

    def export_json(self, target_path: Optional[Path] = None) -> Path:
        """
        Exports all jobs as an atomic JSON snapshot to support legacy file readers.
        Uses atomic file rename (.tmp -> replace).
        """
        out_path = Path(target_path) if target_path else self.json_path
        all_jobs = self.get_all_jobs()
        export_map = {j["job_id"]: _sanitize_for_storage(j) for j in all_jobs}

        temp_path = out_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(export_map, indent=4), encoding="utf-8")
        temp_path.replace(out_path)
        return out_path

    def _sync_json_snapshot(self):
        """Internal helper to refresh JSON snapshot for backward compatibility."""
        try:
            self.export_json(self.json_path)
        except Exception as exc:
            logger.debug("Failed syncing jobs_db.json snapshot: %s", exc)


class JobsDictProxy(MutableMapping[str, Dict[str, Any]]):
    """
    Dictionary proxy backed directly by SqliteJobStore.
    Maintains 100% backward compatibility for all code and tests accessing
    orchestrator.jobs as a mapping (getitem, setitem, delitem, iter, len, get, etc.).
    """

    def __init__(self, store: SqliteJobStore):
        self._store = store

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def __setitem__(self, job_id: str, record: Dict[str, Any]):
        rec = dict(record)
        rec["job_id"] = job_id
        if not self._store.has_job(job_id):
            self._store.create_job(rec)
        else:
            rec.pop("job_id", None)
            self._store.update_job_state(job_id, **rec)

    def __delitem__(self, job_id: str):
        if not self._store.delete_job(job_id):
            raise KeyError(job_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.get_all_job_ids())

    def __len__(self) -> int:
        return self._store.count()

    def __contains__(self, job_id: object) -> bool:
        if not isinstance(job_id, str):
            return False
        return self._store.has_job(job_id)

    def get(self, job_id: str, default: Any = None) -> Any:
        job = self._store.get_job(job_id)
        return job if job is not None else default

    def values(self):
        return self._store.get_all_jobs()

    def items(self):
        return [(j["job_id"], j) for j in self._store.get_all_jobs()]

    def keys(self):
        return self._store.get_all_job_ids()
