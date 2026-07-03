"""
phase4/main.py
--------------
Main entrypoint and orchestration pipeline for Phase 4 deployment verification.
Integrates all modules (loading, validation, device auth, decryption, model load,
inference, and reporting) into a cohesive, secure execution pipeline.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

from .config import Phase4Config
from .package_loader import PackageLoader, IncompletePackageError, InvalidArchiveError
from .package_validator import validate_package_integrity, IntegrityValidationError, SignatureValidationError
from .device_auth import verify_device_binding, get_device_bound_key, DeviceAuthorizationError
from .decryptor import DecryptedAdapterContext
from .adapter_loader import load_base_model_and_tokenizer, load_peft_adapter
from .inference_runner import run_side_by_side_inference
from .validation_report import generate_validation_reports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase4.main")

def run_deployment_pipeline(
    package_path: Path,
    salt: str,
    base_model_name: str,
    prompt: str,
    output_dir: Path
) -> int:
    """
    Orchestrates the entire Phase 4 verification and secure inference pipeline.
    Returns 0 on success, non-zero on failure.
    """
    logger.info("======================================================================")
    logger.info("   STARTING PHASE 4: SECURE DEPLOYMENT AND VERIFICATION PIPELINE      ")
    logger.info("======================================================================")
    
    # Initialize pipeline steps tracking
    steps_status = {
        "Step 1: Package Completeness": "PENDING",
        "Step 2: Integrity Verification": "PENDING",
        "Step 3: Signature Verification": "PENDING",
        "Step 4: Device Authorization": "PENDING",
        "Step 5: Key Derivation": "PENDING",
        "Step 6: Decryption & Extraction": "PENDING",
        "Step 7: PEFT Model Loading": "PENDING",
        "Step 8: Inference Validation": "PENDING"
    }
    
    manifest: Dict[str, Any] = {}
    fingerprint_hash = ""
    verification_success = False
    inference_result: Dict[str, Any] = {
        "prompt": prompt,
        "base_output": "[N/A - PIPELINE FAILED]",
        "peft_output": "[N/A - PIPELINE FAILED]",
        "adapter_active": False
    }

    try:
        # Load environment variables if not loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        # ── Step 1: Package Completeness ────────────────────────────────────
        logger.info("[1/8] Loading and extracting protected package...")
        try:
            # We instantiate PackageLoader as a context manager
            loader = PackageLoader(package_path, max_bytes=Phase4Config.MAX_PACKAGE_BYTES)
            with loader as extracted_dir:
                steps_status["Step 1: Package Completeness"] = "PASSED"
                logger.info("[1/8] PASS — Package completeness verified.")
                
                # Load manifest file to extract metadata
                import json
                manifest_path = extracted_dir / "package_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_fp_hash = manifest.get("device_fingerprint_hash_ref", "")
                
                # ── Step 2 & 3: Integrity and Signature Verification ─────────
                logger.info("[2-3/8] Verifying SHA-256 integrity and RSA-PSS signature...")
                try:
                    fingerprint_hash = validate_package_integrity(extracted_dir)
                    steps_status["Step 2: Integrity Verification"] = "PASSED"
                    steps_status["Step 3: Signature Verification"] = "PASSED"
                    logger.info("[2-3/8] PASS — Integrity and signature verified successfully.")
                except IntegrityValidationError as e:
                    steps_status["Step 2: Integrity Verification"] = "FAILED"
                    steps_status["Step 3: Signature Verification"] = "SKIPPED"
                    raise
                except SignatureValidationError as e:
                    steps_status["Step 2: Integrity Verification"] = "PASSED"
                    steps_status["Step 3: Signature Verification"] = "FAILED"
                    raise

                # ── Step 4: Device Authorization ────────────────────────────
                logger.info("[4/8] Performing device authorization check...")
                try:
                    verify_device_binding(expected_fp_hash)
                    steps_status["Step 4: Device Authorization"] = "PASSED"
                    logger.info("[4/8] PASS — Host machine matches target fingerprint.")
                except DeviceAuthorizationError:
                    steps_status["Step 4: Device Authorization"] = "FAILED"
                    raise

                # ── Step 5: Key Derivation ──────────────────────────────────
                logger.info("[5/8] Deriving device-bound decryption key...")
                try:
                    key = get_device_bound_key(salt)
                    steps_status["Step 5: Key Derivation"] = "PASSED"
                    logger.info("[5/8] PASS — AES-256 key derived dynamically in-memory.")
                except Exception as e:
                    steps_status["Step 5: Key Derivation"] = "FAILED"
                    raise ValueError(f"Key derivation failed: {e}") from e

                # ── Step 6: Decryption & Extraction ────────────────────────
                logger.info("[6/8] Decrypting adapter archive in secure context...")
                try:
                    enc_path = extracted_dir / "adapter.enc"
                    # Initialize context manager for decryption
                    decryptor = DecryptedAdapterContext(enc_path, key)
                    with decryptor as decrypted_adapter_dir:
                        steps_status["Step 6: Decryption & Extraction"] = "PASSED"
                        logger.info("[6/8] PASS — Adapter decrypted and extracted to temp folder.")

                        # ── Step 7: PEFT Model Loading ────────────────────────
                        logger.info("[7/8] Loading base model and binding PEFT adapter...")
                        try:
                            base_model, tokenizer = load_base_model_and_tokenizer(base_model_name)
                            peft_model = load_peft_adapter(base_model, decrypted_adapter_dir)
                            steps_status["Step 7: PEFT Model Loading"] = "PASSED"
                            logger.info("[7/8] PASS — Model and PEFT adapter loaded in memory.")
                        except Exception as e:
                            steps_status["Step 7: PEFT Model Loading"] = "FAILED"
                            raise

                        # ── Step 8: Inference Validation ──────────────────────
                        logger.info("[8/8] Running comparative inference test...")
                        try:
                            inference_result = run_side_by_side_inference(
                                base_model=base_model,
                                peft_model=peft_model,
                                tokenizer=tokenizer,
                                prompt=prompt
                            )
                            steps_status["Step 8: Inference Validation"] = "PASSED"
                            verification_success = True
                            logger.info("[8/8] PASS — Inference completed and compared.")
                        except Exception as e:
                            steps_status["Step 8: Inference Validation"] = "FAILED"
                            raise

                except Exception as e:
                    if steps_status["Step 6: Decryption & Extraction"] == "PENDING":
                        steps_status["Step 6: Decryption & Extraction"] = "FAILED"
                    raise

        except (IncompletePackageError, InvalidArchiveError) as e:
            steps_status["Step 1: Package Completeness"] = "FAILED"
            raise

    except Exception as exc:
        logger.error("VERIFICATION FAILED: %s", exc)
        # Remaining steps set to SKIPPED if they were pending
        for step in steps_status:
            if steps_status[step] == "PENDING":
                steps_status[step] = "SKIPPED"
        verification_success = False

    # Generate final reports (always written, serving as audit trail of success or failure)
    try:
        json_report, md_report = generate_validation_reports(
            output_dir=output_dir,
            manifest=manifest,
            fingerprint_hash=fingerprint_hash or "UNKNOWN",
            steps_status=steps_status,
            verification_success=verification_success,
            inference_result=inference_result
        )
        logger.info("Reports saved to: %s", output_dir)
    except Exception as e:
        logger.error("Failed to generate validation reports: %s", e)

    if verification_success:
        logger.info("======================================================================")
        logger.info("   PHASE 4 PIPELINE COMPLETED SUCCESSFULLY - DEPLOYMENT AUTHORIZED     ")
        logger.info("======================================================================")
        return 0
    else:
        logger.info("======================================================================")
        logger.info("   PHASE 4 PIPELINE FAILED - DEPLOYMENT REJECTED                      ")
        logger.info("======================================================================")
        return 1

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phase4",
        description="Phase 4: Secure Deployment, Verification, and Inference Validation"
    )
    parser.add_argument(
        "-p", "--package",
        default=str(Phase4Config.PACKAGE_PATH),
        help="Path to the protected adapter package directory or tar.gz archive."
    )
    parser.add_argument(
        "-s", "--salt",
        default=os.environ.get("P3_DEVICE_SALT", ""),
        help="Secret salt for device-bound key derivation. (Reads P3_DEVICE_SALT by default)"
    )
    parser.add_argument(
        "-m", "--base-model",
        default=Phase4Config.DEFAULT_BASE_MODEL,
        help="Base model to load (default: JackFram/llama-68m)."
    )
    parser.add_argument(
        "--prompt",
        default="Explain the concept of device-bound security in machine learning.",
        help="Prompt to run for inference comparison."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=str(Phase4Config.DEPLOYMENT_OUTPUT_DIR),
        help="Directory to save the validation reports."
    )
    
    args = parser.parse_args(argv)
    
    if not args.salt:
        logger.error("Missing secret device salt. Pass --salt or set the P3_DEVICE_SALT environment variable.")
        return 1
        
    return run_deployment_pipeline(
        package_path=Path(args.package),
        salt=args.salt,
        base_model_name=args.base_model,
        prompt=args.prompt,
        output_dir=Path(args.output_dir)
    )

if __name__ == "__main__":
    sys.exit(main())
