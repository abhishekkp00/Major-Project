"""
main.py
=======
Phase 4 Secure Deployment & Verification Gateway for SecureLoRA.

Implements the strict 9-step verification order:
  1. Package Completeness Check
  2. Manifest Schema Validation
  3. Signature Validation (RSA-2048-PSS over canonical manifest digest)
  4. Digest Validation (SHA-256 ciphertext match)
  5. Replay & Version Validation (Monotonic anti-replay state tracking, timestamps, model IDs)
  6. Device Authorization (Adaptive Device-Bound State Machine)
  7. Key Derivation (HKDF-SHA256)
  8. Decryption (AES-256-GCM in volatile RAM context)
  9. Adapter Load & Inference Validation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.phase4.config import Phase4Config
from src.phase4.package_loader import PackageLoader
from src.phase4.package_validator import validate_package_provenance
from src.phase4.device_auth import verify_device_binding, get_device_bound_key
from src.phase4.decryptor import DecryptedAdapterContext
from src.phase4.adapter_loader import load_base_model_and_tokenizer, load_peft_adapter
from src.phase4.inference_runner import run_side_by_side_inference
from src.phase4.validation_report import generate_validation_reports
from src.security import AntiReplayTracker
from src.common.exceptions import (
    IncompletePackageError,
    InvalidArchiveError,
    IntegrityValidationError,
    SignatureValidationError,
    DeviceAuthorizationError,
    ManifestSchemaError,
    ReplayAttackError,
    PackageExpiredError,
    ModelMismatchError,
    AdapterMismatchError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("secure_lora.phase4.main")


def run_deployment_pipeline(
    package_path: Path,
    salt: str,
    base_model_name: str,
    prompt: str,
    output_dir: Path,
    target_adapter_id: Optional[str] = None,
    admin_reauth_token: Optional[str] = None,
) -> int:
    """
    Executes the strict 9-step deployment verification pipeline.
    Never decrypts or loads the adapter before authenticity and policy checks succeed.
    """
    logger.info("======================================================================")
    logger.info("   STARTING PHASE 4: SECURE DEPLOYMENT & PROVENANCE PIPELINE           ")
    logger.info("======================================================================")

    steps_status = {
        "Step 1: Package Completeness": "PENDING",
        "Step 2: Manifest Schema Validation": "PENDING",
        "Step 3: Signature Validation": "PENDING",
        "Step 4: Digest Validation": "PENDING",
        "Step 5: Replay & Version Validation": "PENDING",
        "Step 6: Device Authorization": "PENDING",
        "Step 7: Key Derivation": "PENDING",
        "Step 8: Decryption & Extraction": "PENDING",
        "Step 9: Adapter Load & Inference": "PENDING",
    }

    manifest: Dict[str, Any] = {}
    fingerprint_hash = ""
    verification_success = False
    inference_result: Dict[str, Any] = {
        "prompt": prompt,
        "base_output": "[N/A - PIPELINE FAILED]",
        "peft_output": "[N/A - PIPELINE FAILED]",
        "adapter_active": False,
    }

    try:
        from dotenv import load_dotenv
        load_dotenv()

        # ── Step 1: Package Completeness ────────────────────────────────────
        logger.info("[1/9] Verifying package completeness...")
        try:
            loader = PackageLoader(package_path, max_bytes=Phase4Config.MAX_PACKAGE_BYTES)
            with loader as extracted_dir:
                steps_status["Step 1: Package Completeness"] = "PASSED"
                logger.info("[1/9] PASS — Package completeness verified.")

                manifest_path = extracted_dir / "package_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

                # ── Step 2, 3, 4: Schema, Signature & Digest Validation ──────
                logger.info("[2-4/9] Validating manifest schema, RSA-PSS signature & SHA-256 digest...")
                try:
                    manifest, ciphertext_digest = validate_package_provenance(extracted_dir)
                    steps_status["Step 2: Manifest Schema Validation"] = "PASSED"
                    steps_status["Step 3: Signature Validation"] = "PASSED"
                    steps_status["Step 4: Digest Validation"] = "PASSED"
                    logger.info("[2-4/9] PASS — Schema, signature & ciphertext digest verified.")
                except ManifestSchemaError:
                    steps_status["Step 2: Manifest Schema Validation"] = "FAILED"
                    steps_status["Step 3: Signature Validation"] = "SKIPPED"
                    steps_status["Step 4: Digest Validation"] = "SKIPPED"
                    raise
                except SignatureValidationError:
                    steps_status["Step 2: Manifest Schema Validation"] = "PASSED"
                    steps_status["Step 3: Signature Validation"] = "FAILED"
                    steps_status["Step 4: Digest Validation"] = "SKIPPED"
                    raise
                except IntegrityValidationError:
                    steps_status["Step 2: Manifest Schema Validation"] = "PASSED"
                    steps_status["Step 3: Signature Validation"] = "PASSED"
                    steps_status["Step 4: Digest Validation"] = "FAILED"
                    raise

                # ── Step 5: Replay & Version Validation ───────────────────────
                logger.info("[5/9] Validating anti-replay status, timestamps, and model IDs...")
                try:
                    tracker = AntiReplayTracker()
                    tracker.check_and_update(
                        manifest=manifest,
                        target_base_model_id=base_model_name,
                        target_adapter_id=target_adapter_id,
                    )
                    steps_status["Step 5: Replay & Version Validation"] = "PASSED"
                    logger.info("[5/9] PASS — Package is fresh, un-replayed, and model-matched.")
                except (ReplayAttackError, PackageExpiredError, ModelMismatchError, AdapterMismatchError) as e:
                    steps_status["Step 5: Replay & Version Validation"] = "FAILED"
                    raise

                # ── Step 6: Device Authorization ────────────────────────────
                logger.info("[6/9] Evaluating policy-driven device authorization...")
                try:
                    expected_fp_hash = manifest.get("device_fingerprint_hash_ref", "")
                    expected_features = manifest.get("deployment_policy", {}).get("expected_features")
                    auth_res = verify_device_binding(
                        expected_fingerprint_hash=expected_fp_hash,
                        expected_features=expected_features,
                        admin_reauth_token=admin_reauth_token,
                    )
                    steps_status["Step 6: Device Authorization"] = "PASSED"
                    logger.info("[6/9] PASS — Host machine authorized (state=%s).", auth_res.state.value)
                except DeviceAuthorizationError:
                    steps_status["Step 6: Device Authorization"] = "FAILED"
                    raise

                # ── Step 7: Key Derivation ──────────────────────────────────
                logger.info("[7/9] Deriving decryption key via HKDF-SHA256...")
                try:
                    kdf_ver = manifest.get("kdf_version") or manifest.get("encryption", {}).get("kdf_version")
                    key = get_device_bound_key(salt, kdf_version=kdf_ver)
                    steps_status["Step 7: Key Derivation"] = "PASSED"
                    logger.info("[7/9] PASS — Key derived transiently in volatile memory.")
                except Exception as e:
                    steps_status["Step 7: Key Derivation"] = "FAILED"
                    raise ValueError(f"Key derivation failed: {e}") from e

                # ── Step 8: Decryption & Extraction ────────────────────────
                logger.info("[8/9] Decrypting adapter archive via AES-256-GCM...")
                try:
                    enc_path = extracted_dir / "adapter.enc"
                    decryptor = DecryptedAdapterContext(enc_path, key)
                    with decryptor as decrypted_adapter_dir:
                        steps_status["Step 8: Decryption & Extraction"] = "PASSED"
                        logger.info("[8/9] PASS — Adapter decrypted to temporary context.")

                        # ── Step 9: Adapter Load & Inference ──────────────────
                        logger.info("[9/9] Loading PEFT adapter and executing inference test...")
                        try:
                            base_model, tokenizer = load_base_model_and_tokenizer(base_model_name)
                            peft_model = load_peft_adapter(base_model, decrypted_adapter_dir)
                            inference_result = run_side_by_side_inference(
                                base_model=base_model,
                                peft_model=peft_model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                            )
                            steps_status["Step 9: Adapter Load & Inference"] = "PASSED"
                            verification_success = True
                            logger.info("[9/9] PASS — Inference completed.")
                        except Exception as e:
                            steps_status["Step 9: Adapter Load & Inference"] = "FAILED"
                            raise

                except Exception:
                    if steps_status["Step 8: Decryption & Extraction"] == "PENDING":
                        steps_status["Step 8: Decryption & Extraction"] = "FAILED"
                    raise

        except (IncompletePackageError, InvalidArchiveError):
            steps_status["Step 1: Package Completeness"] = "FAILED"
            raise

    except Exception as exc:
        logger.error("VERIFICATION FAILED: %s", exc)
        for step in steps_status:
            if steps_status[step] == "PENDING":
                steps_status[step] = "SKIPPED"
        verification_success = False

    try:
        json_report, md_report = generate_validation_reports(
            output_dir=output_dir,
            manifest=manifest,
            fingerprint_hash=fingerprint_hash or "UNKNOWN",
            steps_status=steps_status,
            verification_success=verification_success,
            inference_result=inference_result,
        )
        logger.info("Validation report generated → %s", md_report)
    except Exception as e:
        logger.warning("Failed to generate validation report: %s", e)

    return 0 if verification_success else 1


def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Phase 4 Verification Gateway")
    parser.add_argument("--package", type=str, required=True, help="Path to package directory or tar.gz")
    parser.add_argument("--salt", type=str, required=True, help="Deployment secret salt")
    parser.add_argument("--model", type=str, default=Phase4Config.BASE_MODEL_NAME, help="Base model ID")
    parser.add_argument("--prompt", type=str, default="What are the symptoms of diabetes?", help="Test prompt")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Output directory")

    args = parser.parse_args()
    rc = run_deployment_pipeline(
        package_path=Path(args.package),
        salt=args.salt,
        base_model_name=args.model,
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
