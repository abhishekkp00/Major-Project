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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.common.exceptions import (
    AdapterMismatchError,
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

    if manifest["kdf_version"] != "hkdf-sha256-v1":
        raise ManifestSchemaError(f"Unsupported KDF version: {manifest['kdf_version']}")

    if manifest["encryption_version"] != "aes-256-gcm-v1":
        raise ManifestSchemaError(f"Unsupported encryption version: {manifest['encryption_version']}")

    if manifest["signature_algorithm"] != "rsa-pss-2048-sha256":
        raise ManifestSchemaError(f"Unsupported signature algorithm: {manifest['signature_algorithm']}")

    if manifest["digest_algorithm"] != "sha256":
        raise ManifestSchemaError(f"Unsupported digest algorithm: {manifest['digest_algorithm']}")

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

class AntiReplayTracker:
    """
    Manages local deployment state tracking to prevent replay attacks in offline/edge scenarios.
    """
    def __init__(self, state_file_path: Optional[Path] = None):
        self.state_file_path = state_file_path or DEFAULT_STATE_FILE

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_file_path.exists():
            return {"adapters": {}}
        try:
            return json.loads(self.state_file_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupted deployment state file; re-initializing.")
            return {"adapters": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_file_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def get_next_sequence_number(self, adapter_id: str) -> int:
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
        Updates state atomically upon success.
        """
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

        # D. Monotonic Anti-Replay Validation
        adapter_id = manifest["adapter_id"]
        package_id = manifest["package_id"]
        sequence_num = manifest["sequence_number"]

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
        processed = adapter_record.get("processed_packages", [])
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
