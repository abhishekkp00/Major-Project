"""
provenance_anti_replay_experiments.py
======================================
Runs the 10 Provenance and Anti-Replay Attack Simulations for SecureLoRA.

Measures rejection latency, detection gate, and security results.
Outputs: outputs/evaluation/provenance_anti_replay_matrix.json
"""

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List

from src.security import (
    encrypt_adapter,
    get_fingerprint_hash,
    derive_key_from_env,
    generate_dev_keypair,
    compute_sha256,
    save_hash,
    AntiReplayTracker,
)
from src.phase3.package_builder import build_package
from src.phase4.package_validator import validate_package_provenance
from src.phase4.device_auth import verify_device_binding
from src.common.exceptions import (
    ManifestSchemaError,
    SignatureValidationError,
    IntegrityValidationError,
    ReplayAttackError,
    PackageExpiredError,
    ModelMismatchError,
    AdapterMismatchError,
    DeviceAuthorizationError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("secure_lora.provenance_experiments")

OUTPUT_JSON_PATH = Path("outputs/evaluation/provenance_anti_replay_matrix.json")


def setup_valid_baseline_package(temp_dir: Path) -> Dict[str, Any]:
    """Helper to create a valid baseline adapter package and RSA keys."""
    keys_dir = temp_dir / "keys"
    pkg_dir = temp_dir / "pkg_v1"
    raw_adapter = temp_dir / "raw_adapter"
    raw_adapter.mkdir(parents=True, exist_ok=True)
    (raw_adapter / "adapter_model.bin").write_bytes(b"VALID_LORA_WEIGHT_BYTES_1234567890")

    priv_key = keys_dir / "private.pem"
    pub_key = keys_dir / "public.pem"
    generate_dev_keypair(priv_key, pub_key, key_size=2048)

    salt = "test_deployment_salt_999"
    fp_hash = get_fingerprint_hash()
    key = derive_key_from_env(fp_hash, salt)

    enc_meta = encrypt_adapter(
        adapter_input=raw_adapter,
        output_enc_path=pkg_dir / "adapter.enc",
        key=key,
        fingerprint_hash=fp_hash,
        metadata_path=pkg_dir / "metadata.json",
    )
    digest = compute_sha256(pkg_dir / "adapter.enc")
    save_hash(digest, pkg_dir / "adapter.hash")

    manifest = build_package(
        package_dir=pkg_dir,
        adapter_id="medical-lora-v1",
        model_reference="distilbert-base-uncased",
        fingerprint_hash=fp_hash,
        package_version="1.0.0",
        enc_metadata=enc_meta,
        public_key_src=pub_key,
        private_key_src=priv_key,
        sequence_number=10,
    )

    return {
        "pkg_dir": pkg_dir,
        "keys_dir": keys_dir,
        "priv_key": priv_key,
        "pub_key": pub_key,
        "salt": salt,
        "fp_hash": fp_hash,
        "manifest": manifest,
    }


def run_provenance_experiments() -> List[Dict[str, Any]]:
    """Runs all 10 attack simulation scenarios and measures rejection latency."""
    logger.info("Executing 10 Provenance and Anti-Replay Attack Simulations...")

    results = []

    with tempfile.TemporaryDirectory(prefix="provenance_exp_") as tmp_root:
        root_path = Path(tmp_root)
        base_setup = setup_valid_baseline_package(root_path)

        state_file = root_path / "test_deployment_state.json"
        tracker = AntiReplayTracker(state_file_path=state_file)

        # Baseline seed registration: register sequence 5 so seq 10 is valid
        tracker.check_and_update(
            manifest={
                "package_id": "seed-package-uuid-00",
                "adapter_id": "medical-lora-v1",
                "sequence_number": 5,
            }
        )

        scenarios = [
            ("modified_manifest", "Modified package_manifest.json (tampered model_id/sequence)"),
            ("modified_adapter", "Modified ciphertext (tampered adapter.enc byte)"),
            ("copied_valid_package", "Copied valid package re-executed (duplicate package_id)"),
            ("old_package_replay", "Stale package replay (sequence_number <= last_seen)"),
            ("wrong_model_id", "Wrong base model ID target mismatch"),
            ("wrong_adapter_id", "Wrong adapter ID target mismatch"),
            ("expired_package", "Expired package (creation_time + TTL passed)"),
            ("invalid_signature", "Corrupted RSA-2048-PSS signature (adapter.sig)"),
            ("malicious_package_attacker_key", "Malicious package signed with attacker private key"),
            ("valid_package_unauthorized_device", "Valid package executed on unauthorized device"),
        ]

        for sc_id, sc_name in scenarios:
            t0 = time.perf_counter()

            # Create test package copy
            test_pkg = root_path / f"pkg_{sc_id}"
            shutil.copytree(base_setup["pkg_dir"], test_pkg)

            gate = "UNKNOWN"
            outcome = "FAILED_TO_DETECT"
            rejected = False
            error_msg = ""

            try:
                if sc_id == "modified_manifest":
                    man_p = test_pkg / "package_manifest.json"
                    m_data = json.loads(man_p.read_text())
                    m_data["base_model_id"] = "malicious-llama-3-8b"
                    man_p.write_text(json.dumps(m_data, indent=2))

                    gate = "Step 3: Signature Validation"
                    validate_package_provenance(test_pkg)

                elif sc_id == "modified_adapter":
                    enc_p = test_pkg / "adapter.enc"
                    data = bytearray(enc_p.read_bytes())
                    data[10] ^= 0xFF  # Flip byte
                    enc_p.write_bytes(bytes(data))

                    gate = "Step 4: Digest Validation"
                    validate_package_provenance(test_pkg)

                elif sc_id == "copied_valid_package":
                    # First run valid check
                    m, d = validate_package_provenance(test_pkg)
                    tracker.check_and_update(m)
                    # Second run (duplicate package_id)
                    gate = "Step 5: Replay / Version Validation"
                    tracker.check_and_update(m)

                elif sc_id == "old_package_replay":
                    gate = "Step 5: Replay / Version Validation"
                    man_p = test_pkg / "package_manifest.json"
                    m_data = json.loads(man_p.read_text())
                    m_data["package_id"] = "stale-pkg-uuid-777"
                    m_data["sequence_number"] = 4  # Stale vs last_seq=10
                    man_p.write_text(json.dumps(m_data, indent=2))

                    m, d = validate_package_provenance(test_pkg)
                    tracker.check_and_update(m)

                elif sc_id == "wrong_model_id":
                    gate = "Step 5: Replay / Version Validation"
                    m, d = validate_package_provenance(test_pkg)
                    tracker.check_and_update(m, target_base_model_id="gpt-2-xl")

                elif sc_id == "wrong_adapter_id":
                    gate = "Step 5: Replay / Version Validation"
                    m, d = validate_package_provenance(test_pkg)
                    tracker.check_and_update(m, target_adapter_id="unauthorized-adapter-xyz")

                elif sc_id == "expired_package":
                    gate = "Step 5: Replay / Version Validation"
                    man_p = test_pkg / "package_manifest.json"
                    m_data = json.loads(man_p.read_text())
                    m_data["expiration_timestamp"] = (
                        datetime.now(timezone.utc) - timedelta(days=1)
                    ).isoformat()
                    man_p.write_text(json.dumps(m_data, indent=2))

                    m, d = validate_package_provenance(test_pkg)
                    tracker.check_and_update(m)


                elif sc_id == "invalid_signature":
                    sig_p = test_pkg / "adapter.sig"
                    sig_p.write_bytes(b"CORRUPTED_SIGNATURE_BYTES_0000000000")

                    gate = "Step 3: Signature Validation"
                    validate_package_provenance(test_pkg)

                elif sc_id == "malicious_package_attacker_key":
                    # Attacker signs with their own key but victim's public.pem is in package
                    att_priv = root_path / "att_priv.pem"
                    att_pub = root_path / "att_pub.pem"
                    generate_dev_keypair(att_priv, att_pub)

                    man_p = test_pkg / "package_manifest.json"
                    m_data = json.loads(man_p.read_text())
                    # Attacker rebuilds signature with attacker key
                    from src.security import compute_canonical_manifest_digest, sign_digest, save_signature
                    can_d = compute_canonical_manifest_digest(m_data, m_data["encrypted_adapter_digest"])
                    att_sig = sign_digest(can_d, att_priv)
                    save_signature(att_sig, test_pkg / "adapter.sig")

                    gate = "Step 3: Signature Validation"
                    validate_package_provenance(test_pkg)

                elif sc_id == "valid_package_unauthorized_device":
                    gate = "Step 6: Device Authorization"
                    verify_device_binding(
                        expected_fingerprint_hash="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                        mock_fingerprint="1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
                    )

            except (
                ManifestSchemaError,
                SignatureValidationError,
                IntegrityValidationError,
                ReplayAttackError,
                PackageExpiredError,
                ModelMismatchError,
                AdapterMismatchError,
                DeviceAuthorizationError,
            ) as exc:
                rejected = True
                outcome = "BLOCKED_AND_REJECTED"
                error_msg = str(exc)

            latency_ms = round((time.perf_counter() - t0) * 1000, 3)

            results.append({
                "scenario_id": sc_id,
                "scenario_name": sc_name,
                "detection_gate": gate,
                "rejected": rejected,
                "outcome": outcome,
                "rejection_latency_ms": latency_ms,
                "error_details": error_msg,
            })

    # Save output report
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Provenance experiment matrix saved → %s", OUTPUT_JSON_PATH)

    _print_matrix_table(results)
    return results


def _print_matrix_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print("  Adapter Provenance and Anti-Replay Deployment Security Matrix")
    print("=" * 90)
    print(f"| {'Attack Scenario':<36} | {'Detection Gate':<32} | {'Result':<18} | {'Latency (ms)':<12} |")
    print("| " + "-" * 36 + " | " + "-" * 32 + " | " + "-" * 18 + " | " + "-" * 12 + " |")

    for r in results:
        res_str = "REJECTED" if r["rejected"] else "ALLOWED"
        print(f"| {r['scenario_id']:<36} | {r['detection_gate']:<32} | {res_str:<18} | {r['rejection_latency_ms']:<12.3f} |")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_provenance_experiments()
