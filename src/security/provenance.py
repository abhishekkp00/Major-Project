"""
provenance.py
=============
Adapter Provenance and Anti-Replay System for SecureLoRA.

This module provides cryptographic authenticity and provenance validation
under the assumed private-key security model. It enforces:

  1. Manifest Schema & Security Field Validation
  2. Canonical Manifest Authentication (combining metadata + payload digest)
  3. Offline Anti-Replay State Tracking (monotonic sequence numbers, UUID checking, expiration)
  4. Target Model & Adapter Binding Verification

Security Scope & Terminology
----------------------------
This mechanism provides cryptographic authenticity and provenance under the
assumed private-key security model. It does NOT claim non-repudiation in the
absolute sense (since private key protection depends on environment security).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

from src.common.exceptions import (
    AdapterMismatchError,
    DeploymentStateIntegrityError,
    ManifestSchemaError,
    ModelMismatchError,
    PackageExpiredError,
    ReplayAttackError,
    SecurityError,
)
from src.security.crypto import compute_sha256

logger = logging.getLogger("secure_lora.security.provenance")

DEFAULT_STATE_FILE = Path("outputs/.deployment_state.json")

REQUIRED_MANIFEST_FIELDS = [
    "package_id",
    "adapter_id",
    "base_model_id",
    "model_revision",
    "adapter_revision",
    "package_version",
    "creation_timestamp",
    "expiration_timestamp",
    "binding_policy_version",
    "kdf_version",
    "encryption_version",
    "signature_algorithm",
    "digest_algorithm",
    "nonce_metadata",
    "deployment_policy",
    "sequence_number",
    "device_fingerprint_hash_ref",
    "encrypted_adapter_digest",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Manifest Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_manifest_schema(manifest: Dict[str, Any]) -> None:
    """
    Validates that package_manifest.json contains all required cryptographic provenance fields
    and that data types strictly conform to the spec.
    """
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise ManifestSchemaError(f"Package manifest is missing required security fields: {missing}")

    if not isinstance(manifest["package_id"], str) or len(manifest["package_id"]) < 8:
        raise ManifestSchemaError("Invalid 'package_id': must be a valid non-empty string UUID.")

    if not isinstance(manifest["sequence_number"], int) or manifest["sequence_number"] < 1:
        raise ManifestSchemaError("Invalid 'sequence_number': must be a positive integer >= 1.")

    from src.security.key_derivation import SUPPORTED_KDF_VERSIONS
    if manifest["kdf_version"] not in SUPPORTED_KDF_VERSIONS:
        raise ManifestSchemaError(f"Unsupported KDF version: {manifest['kdf_version']}")

    if manifest["encryption_version"] != "aes-256-gcm-v1":
        raise ManifestSchemaError(f"Unsupported encryption version: {manifest['encryption_version']}")

    if manifest["signature_algorithm"] != "rsa-pss-2048-sha256":
        raise ManifestSchemaError(f"Unsupported signature algorithm: {manifest['signature_algorithm']}")

    if manifest["digest_algorithm"] != "sha256":
        raise ManifestSchemaError(f"Unsupported digest algorithm: {manifest['digest_algorithm']}")

    if not isinstance(manifest.get("adapter_id"), str) or not manifest["adapter_id"].strip():
        raise ManifestSchemaError("Invalid 'adapter_id': must be a non-empty string.")

    if not isinstance(manifest.get("base_model_id"), str) or not manifest["base_model_id"].strip():
        raise ManifestSchemaError("Invalid 'base_model_id': must be a non-empty string.")

    if not isinstance(manifest.get("package_version"), str) or not manifest["package_version"].strip():
        raise ManifestSchemaError("Invalid 'package_version': must be a non-empty string.")

    if not isinstance(manifest.get("deployment_policy"), dict):
        raise ManifestSchemaError("Invalid 'deployment_policy': must be a dictionary.")

    if not isinstance(manifest.get("encrypted_adapter_digest"), str) or not manifest["encrypted_adapter_digest"].strip():
        raise ManifestSchemaError("Invalid 'encrypted_adapter_digest': must be a non-empty string.")

    logger.debug("Package manifest schema validation PASSED (package_id=%s).", manifest["package_id"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Canonical Authenticated Digest Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_canonical_manifest_digest(manifest: Dict[str, Any], ciphertext_digest: str) -> str:
    """
    Constructs a deterministic canonical string combining all security-critical manifest metadata
    and the adapter ciphertext SHA-256 digest, then computes its SHA-256 hash.

    This authenticated structure ensures the RSA-PSS signature validates the full security scope:
      canonical_manifest + ciphertext_digest + version + binding_metadata -> RSA-PSS signature
    """
    canonical_dict = {
        "package_id": str(manifest.get("package_id")),
        "adapter_id": str(manifest.get("adapter_id")),
        "base_model_id": str(manifest.get("base_model_id")),
        "model_revision": str(manifest.get("model_revision")),
        "adapter_revision": str(manifest.get("adapter_revision")),
        "package_version": str(manifest.get("package_version")),
        "creation_timestamp": str(manifest.get("creation_timestamp")),
        "expiration_timestamp": manifest.get("expiration_timestamp"),
        "sequence_number": int(manifest.get("sequence_number", 1)),
        "binding_policy_version": str(manifest.get("binding_policy_version", "1.0.0")),
        "kdf_version": str(manifest.get("kdf_version")),
        "encryption_version": str(manifest.get("encryption_version")),
        "signature_algorithm": str(manifest.get("signature_algorithm")),
        "digest_algorithm": str(manifest.get("digest_algorithm")),
        "deployment_policy": manifest.get("deployment_policy", {}),
        "device_fingerprint_hash_ref": str(manifest.get("device_fingerprint_hash_ref")),
        "encrypted_adapter_digest": ciphertext_digest.strip().lower(),
    }

    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    digest_hash = compute_sha256_text(canonical_json)
    return digest_hash


def compute_sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Offline Anti-Replay & Provenance Validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_deployment_state_schema(state: Any) -> None:
    """
    Validates deployment state structure and types before trusting state.

    Schema:
    {
        "schema_version": int,
        "adapters": {
            "<adapter_id>": {
                "last_sequence": int,
                "processed_packages": List[str],
                "last_deployed_utc": Optional[str]
            }
        }
    }

    Raises DeploymentStateIntegrityError on any structural or type violation.
    """
    if not isinstance(state, dict):
        raise DeploymentStateIntegrityError(
            f"Deployment state root must be a JSON object, got {type(state).__name__}."
        )

    if "adapters" not in state:
        raise DeploymentStateIntegrityError(
            "Deployment state schema violation: missing required 'adapters' mapping."
        )

    adapters = state["adapters"]
    if not isinstance(adapters, dict):
        raise DeploymentStateIntegrityError(
            f"Deployment state 'adapters' must be a dictionary, got {type(adapters).__name__}."
        )

    schema_version = state.get("schema_version")
    if schema_version is not None and (
        isinstance(schema_version, bool) or not isinstance(schema_version, (int, str))
    ):
        raise DeploymentStateIntegrityError(
            f"Deployment state 'schema_version' must be an integer or string, got {type(schema_version).__name__}."
        )

    for adapter_id, record in adapters.items():
        if not isinstance(adapter_id, str) or not adapter_id.strip():
            raise DeploymentStateIntegrityError(
                f"Deployment state contains invalid or empty adapter_id: {adapter_id!r}."
            )
        if not isinstance(record, dict):
            raise DeploymentStateIntegrityError(
                f"Deployment state record for adapter '{adapter_id}' must be a dictionary, got {type(record).__name__}."
            )

        last_seq = record.get("last_sequence")
        if (
            last_seq is None
            or isinstance(last_seq, bool)
            or not isinstance(last_seq, int)
            or last_seq < 0
        ):
            raise DeploymentStateIntegrityError(
                f"Deployment state record for adapter '{adapter_id}' has invalid 'last_sequence': {last_seq!r}."
            )

        processed = record.get("processed_packages")
        if processed is None or not isinstance(processed, list):
            raise DeploymentStateIntegrityError(
                f"Deployment state record for adapter '{adapter_id}' has invalid 'processed_packages': {processed!r}."
            )

        for pkg in processed:
            if not isinstance(pkg, str) or not pkg.strip():
                raise DeploymentStateIntegrityError(
                    f"Deployment state record for adapter '{adapter_id}' contains invalid package ID in history: {pkg!r}."
                )

        last_deployed = record.get("last_deployed_utc")
        if last_deployed is not None and (not isinstance(last_deployed, str) or not last_deployed.strip()):
            raise DeploymentStateIntegrityError(
                f"Deployment state record for adapter '{adapter_id}' has invalid 'last_deployed_utc': {last_deployed!r}."
            )


class AntiReplayTracker:
    """
    Manages local deployment state tracking to prevent replay attacks in offline/edge scenarios.

    CONTROLLED RECOVERY PROCEDURE FOR CORRUPTED REPLAY STATE
    =========================================================
    If AntiReplayTracker raises DeploymentStateIntegrityError due to state corruption,
    malformed JSON, or an unreadable state file:
      1. DO NOT delete, truncate, or overwrite the existing state file automatically.
         Automatic reset enables replay attacks where an adversary corrupts state to replay
         revoked packages.
      2. The security operator must investigate and quarantine the damaged file:
           mv outputs/.deployment_state.json outputs/.deployment_state.json.corrupt.<timestamp>
      3. Reconstruct deployment history from audited offline deployment ledgers or provenance
         logs to identify the highest deployed sequence number for each adapter.
      4. Perform an explicit, authorized administrative initialization using
         tracker.initialize_state(force=True, initial_adapters={...}).
    """

    _thread_locks: Dict[Path, threading.RLock] = {}
    _meta_lock = threading.Lock()

    def __init__(self, state_file_path: Optional[Path] = None):
        self.state_file_path = Path(state_file_path or DEFAULT_STATE_FILE).resolve()

    @classmethod
    def _get_thread_lock(cls, path: Path) -> threading.RLock:
        canonical = path.resolve()
        with cls._meta_lock:
            if canonical not in cls._thread_locks:
                cls._thread_locks[canonical] = threading.RLock()
            return cls._thread_locks[canonical]

    @contextmanager
    def _state_lock(self):
        """
        Acquires both process-level thread lock and OS-level file lock (via fcntl.flock)
        to prevent race conditions between concurrent threads or processes.
        """
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.state_file_path.with_name(f"{self.state_file_path.name}.lock")
        thread_lock = self._get_thread_lock(self.state_file_path)

        with thread_lock:
            with open(lock_file, "a+", encoding="utf-8") as f:
                try:
                    if _HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    yield
                finally:
                    if _HAS_FCNTL:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except OSError:
                            pass

    def _load_state(self) -> Dict[str, Any]:
        """
        Loads and validates deployment state.

        Distinguishes:
          - Missing file (first-time deployment): returns initial state.
          - Existing file:
            * unreadable -> raises DeploymentStateIntegrityError
            * empty / whitespace -> raises DeploymentStateIntegrityError
            * malformed JSON -> raises DeploymentStateIntegrityError
            * invalid schema/types -> raises DeploymentStateIntegrityError

        Never resets corrupt state to an empty dictionary.
        """
        if not self.state_file_path.exists():
            return {
                "schema_version": 1,
                "adapters": {},
            }

        try:
            raw_text = self.state_file_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise DeploymentStateIntegrityError(
                f"Deployment state file '{self.state_file_path}' exists but cannot be read: {exc}"
            ) from exc

        if not raw_text.strip():
            raise DeploymentStateIntegrityError(
                f"Deployment state file '{self.state_file_path}' exists but is empty or truncated."
            )

        try:
            state = json.loads(raw_text)
        except Exception as exc:
            raise DeploymentStateIntegrityError(
                f"Deployment state file '{self.state_file_path}' is corrupted or malformed JSON: {exc}"
            ) from exc

        validate_deployment_state_schema(state)
        return state

    def _save_state(self, state: Dict[str, Any]) -> None:
        """
        Atomically commits validated state to disk using a unique temporary file,
        flush, fsync, and atomic rename.
        """
        validate_deployment_state_schema(state)
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file_path.with_name(
            f"{self.state_file_path.name}.tmp.{os.getpid()}_{uuid.uuid4().hex}"
        )
        try:
            data = json.dumps(state, indent=2, sort_keys=True)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_file_path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            logger.error("Failed to commit deployment state to %s: %s", self.state_file_path, exc)
            raise

    def initialize_state(
        self,
        force: bool = False,
        initial_adapters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Explicit, safe initialization path for anti-replay state.

        If state already exists, requires force=True to overwrite.
        """
        with self._state_lock():
            if self.state_file_path.exists() and not force:
                raise DeploymentStateIntegrityError(
                    f"Deployment state file '{self.state_file_path}' already exists. "
                    "Refusing to overwrite without explicit force=True."
                )
            new_state = {
                "schema_version": 1,
                "adapters": initial_adapters or {},
            }
            self._save_state(new_state)

    def get_next_sequence_number(self, adapter_id: str) -> int:
        with self._state_lock():
            state = self._load_state()
            adapters_state = state.get("adapters", {})
            adapter_record = adapters_state.get(adapter_id, {"last_sequence": 0})
            return adapter_record.get("last_sequence", 0) + 1

    def check_and_update(
        self,
        manifest: Dict[str, Any],
        target_base_model_id: Optional[str] = None,
        target_adapter_id: Optional[str] = None,
        now_utc: Optional[datetime] = None,
    ) -> None:
        """
        Validates timestamp expiration, target model ID, target adapter ID, and anti-replay status.
        Updates state atomically upon success under an exclusive lock.
        """
        if not isinstance(manifest, dict):
            raise ManifestSchemaError("Manifest must be a dictionary.")

        # A. Check Timestamp Expiration
        current_time = now_utc or datetime.now(timezone.utc)
        exp_str = manifest.get("expiration_timestamp")
        if exp_str:
            try:
                exp_dt = datetime.fromisoformat(exp_str)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                if current_time > exp_dt:
                    raise PackageExpiredError(
                        f"Package expired on {exp_str} (current time: {current_time.isoformat()})."
                    )
            except ValueError as e:
                if not isinstance(e, PackageExpiredError):
                    raise ManifestSchemaError(f"Invalid timestamp format in manifest: {exp_str}") from e
                raise

        # B. Check Intended Base Model ID
        if target_base_model_id:
            pkg_model = manifest.get("base_model_id", "")
            if pkg_model != target_base_model_id:
                raise ModelMismatchError(
                    f"Package model ID '{pkg_model}' does not match target model ID '{target_base_model_id}'."
                )

        # C. Check Intended Adapter ID
        if target_adapter_id:
            pkg_adapter = manifest.get("adapter_id", "")
            if pkg_adapter != target_adapter_id:
                raise AdapterMismatchError(
                    f"Package adapter ID '{pkg_adapter}' does not match target adapter ID '{target_adapter_id}'."
                )

        # D. Monotonic Anti-Replay Validation under exclusive lock
        adapter_id = manifest.get("adapter_id")
        package_id = manifest.get("package_id")
        sequence_num = manifest.get("sequence_number")

        if not adapter_id or not isinstance(adapter_id, str):
            raise ManifestSchemaError(f"Invalid adapter_id in manifest: {adapter_id!r}")
        if not package_id or not isinstance(package_id, str):
            raise ManifestSchemaError(f"Invalid package_id in manifest: {package_id!r}")
        if (
            sequence_num is None
            or isinstance(sequence_num, bool)
            or not isinstance(sequence_num, int)
            or sequence_num < 1
        ):
            raise ManifestSchemaError(f"Invalid sequence_number in manifest: {sequence_num!r}")

        with self._state_lock():
            state = self._load_state()
            adapters_state = state.get("adapters", {})
            adapter_record = adapters_state.get(adapter_id, {"last_sequence": 0, "processed_packages": []})

            if package_id in adapter_record.get("processed_packages", []):
                raise ReplayAttackError(
                    f"Replay attack detected! Package ID '{package_id}' has already been deployed."
                )

            last_seq = adapter_record.get("last_sequence", 0)
            if sequence_num <= last_seq:
                raise ReplayAttackError(
                    f"Replay attack detected! Monotonic sequence number {sequence_num} is stale "
                    f"(last deployed sequence for '{adapter_id}' was {last_seq})."
                )

            # Record deployment state
            processed = list(adapter_record.get("processed_packages", []))
            processed.append(package_id)
            adapters_state[adapter_id] = {
                "last_sequence": sequence_num,
                "processed_packages": processed,
                "last_deployed_utc": current_time.isoformat(),
            }
            state["adapters"] = adapters_state
            self._save_state(state)
            logger.info(
                "Anti-replay check PASSED & state recorded (adapter=%s, seq=%d, pkg_id=%s).",
                adapter_id, sequence_num, package_id
            )
