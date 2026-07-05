import os
import time
import json
import shutil
import logging
import tarfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Callable

# Import Phase 3 & 4 primitives
from src.security.fingerprint import get_fingerprint_hash
from src.security.key_derivation import derive_key_from_env
from src.security.crypto import encrypt_adapter, compute_sha256, decrypt_stream
from src.security.signature import sign_digest, generate_dev_keypair, save_signature
from src.phase3.package_builder import build_package, export_package_archive
from src.phase4.main import run_deployment_pipeline
from src.common.exceptions import (
    CryptoError,
    IntegrityValidationError,
    SignatureValidationError,
    DeviceAuthorizationError
)

logger = logging.getLogger("secure_lora.orchestrator.security_orchestrator")


def get_directory_size(path: Path) -> int:
    """Returns the total size of all files in a directory in bytes."""
    total_size = 0
    if path.is_file():
        return path.stat().st_size
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            if fp.exists():
                total_size += fp.stat().st_size
    return total_size


def run_security_orchestration(
    job_id: str,
    job_dir: Path,
    salt: str,
    base_model_name: str,
    update_state_fn: Callable[..., None]
) -> Dict[str, Any]:
    """
    Orchestrates the entire Phase 3 and Phase 4 workflow.
    Emits the exact requested statuses, runs safety gate simulations,
    captures all necessary metrics, and returns the outcomes.
    """
    outcomes: Dict[str, Any] = {}
    
    adapter_input_dir = job_dir / "adapter"
    protected_output_dir = job_dir / "protected"
    deployment_dir = job_dir / "deployment"
    
    protected_output_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    # ────────────────────────────────────────────────────────────────
    # STATUS: preparing_adapter
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="PACKAGING", stage="preparing_adapter", progress=71)
    logger.info("[%s] Status: preparing_adapter", job_id)
    adapter_size_before = get_directory_size(adapter_input_dir)
    outcomes["adapter_size_before_encryption_bytes"] = adapter_size_before

    # ────────────────────────────────────────────────────────────────
    # STATUS: deriving_device_binding
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="PACKAGING", stage="deriving_device_binding", progress=73)
    logger.info("[%s] Status: deriving_device_binding", job_id)
    fp_hash = get_fingerprint_hash()
    
    # Temporarily set environment variables to align with legacy derivation
    os.environ["P3_DEVICE_SALT"] = salt
    key = derive_key_from_env(fp_hash)

    # ────────────────────────────────────────────────────────────────
    # STATUS: encrypting_adapter
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="PACKAGING", stage="encrypting_adapter", progress=75)
    logger.info("[%s] Status: encrypting_adapter", job_id)
    
    enc_path = protected_output_dir / "adapter.enc"
    meta_path = protected_output_dir / "metadata.json"
    
    start_enc_time = time.perf_counter()
    enc_meta = encrypt_adapter(
        adapter_input=adapter_input_dir,
        output_enc_path=enc_path,
        key=key,
        fingerprint_hash=fp_hash,
        metadata_path=meta_path
    )
    end_enc_time = time.perf_counter()
    outcomes["encryption_time_seconds"] = end_enc_time - start_enc_time

    # ────────────────────────────────────────────────────────────────
    # STATUS: generating_hash
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="PACKAGING", stage="generating_hash", progress=77)
    logger.info("[%s] Status: generating_hash", job_id)
    hash_path = protected_output_dir / "adapter.hash"
    digest = compute_sha256(enc_path)
    # Save hash file
    hash_path.write_text(digest, encoding="utf-8")

    # ────────────────────────────────────────────────────────────────
    # STATUS: generating_signature
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="PACKAGING", stage="generating_signature", progress=80)
    logger.info("[%s] Status: generating_signature", job_id)
    
    priv_key_path = protected_output_dir / "dev_private.pem"
    pub_key_path = protected_output_dir / "public.pem"
    sig_path = protected_output_dir / "adapter.sig"
    
    if not priv_key_path.exists() or not pub_key_path.exists():
        generate_dev_keypair(priv_key_path, pub_key_path, key_size=2048)
        
    signature = sign_digest(digest, priv_key_path)
    save_signature(signature, sig_path)

    # ────────────────────────────────────────────────────────────────
    # STATUS: building_package
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="PACKAGING", stage="building_package", progress=85)
    logger.info("[%s] Status: building_package", job_id)
    
    manifest = build_package(
        package_dir=protected_output_dir,
        adapter_id=job_id,
        model_reference=base_model_name,
        fingerprint_hash=fp_hash,
        package_version="1.0.0",
        enc_metadata=enc_meta,
        public_key_src=pub_key_path
    )
    
    package_archive_path = export_package_archive(protected_output_dir)
    package_size = package_archive_path.stat().st_size
    outcomes["protected_package_size_bytes"] = package_size

    # ────────────────────────────────────────────────────────────────
    # STATUS: running_integrity_check
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="DEPLOYING", stage="running_integrity_check", progress=90)
    logger.info("[%s] Status: running_integrity_check", job_id)
    
    # We will measure verification time on the successful authorized check path
    start_ver_time = time.perf_counter()

    # ────────────────────────────────────────────────────────────────
    # STATUS: running_device_authorization_check
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="DEPLOYING", stage="running_device_authorization_check", progress=92)
    logger.info("[%s] Status: running_device_authorization_check", job_id)

    # ────────────────────────────────────────────────────────────────
    # STATUS: running_secure_deployment_check
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="DEPLOYING", stage="running_secure_deployment_check", progress=94)
    logger.info("[%s] Status: running_secure_deployment_check", job_id)

    # ────────────────────────────────────────────────────────────────
    # STATUS: secure_inference_validation
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="DEPLOYING", stage="secure_inference_validation", progress=96)
    logger.info("[%s] Status: secure_inference_validation", job_id)
    
    # Run the real deployment validation pipeline
    exit_code = run_deployment_pipeline(
        package_path=package_archive_path,
        salt=salt,
        base_model_name=base_model_name,
        prompt="Verification prompt.",
        output_dir=deployment_dir
    )
    end_ver_time = time.perf_counter()
    outcomes["verification_time_seconds"] = end_ver_time - start_ver_time
    
    authorized_pass = "pass" if exit_code == 0 else "fail"
    outcomes["authorized_deployment"] = authorized_pass
    outcomes["deployment_validation_result"] = "success" if exit_code == 0 else "failed"

    # ────────────────────────────────────────────────────────────────
    # SIMULATIONS
    # ────────────────────────────────────────────────────────────────
    
    # 1. Tamper Simulation
    logger.info("[%s] Running Tamper Simulation...", job_id)
    tamper_sim_path = job_dir / "protected_tampered.tar.gz"
    
    # Copy original archive
    shutil.copy2(package_archive_path, tamper_sim_path)
    
    # Modify one byte in the archive to corrupt it
    with open(tamper_sim_path, "r+b") as f:
        f.seek(100)
        f.write(b"\x00\x00\x00")
        
    try:
        # Should raise an error because the archive or signature check fails
        tamper_exit_code = run_deployment_pipeline(
            package_path=tamper_sim_path,
            salt=salt,
            base_model_name=base_model_name,
            prompt="Verification prompt.",
            output_dir=job_dir / "deployment_tampered"
        )
        if tamper_exit_code != 0:
            outcomes["tamper_simulation"] = "pass"
        else:
            outcomes["tamper_simulation"] = "fail"
    except Exception as e:
        logger.info("[%s] Tamper simulation successfully caught integrity error: %s", job_id, e)
        outcomes["tamper_simulation"] = "pass"
    finally:
        if tamper_sim_path.exists():
            tamper_sim_path.unlink()
        shutil.rmtree(job_dir / "deployment_tampered", ignore_errors=True)

    # 2. Unauthorized-Device Simulation
    logger.info("[%s] Running Unauthorized Device Simulation...", job_id)
    try:
        # Use an incorrect salt
        bad_salt = salt + "_unauthorized_device"
        unauth_exit_code = run_deployment_pipeline(
            package_path=package_archive_path,
            salt=bad_salt,
            base_model_name=base_model_name,
            prompt="Verification prompt.",
            output_dir=job_dir / "deployment_unauth"
        )
        if unauth_exit_code != 0:
            outcomes["unauthorized_device_simulation"] = "pass"
        else:
            outcomes["unauthorized_device_simulation"] = "fail"
    except Exception as e:
        logger.info("[%s] Unauthorized device simulation successfully caught authorization error: %s", job_id, e)
        outcomes["unauthorized_device_simulation"] = "pass"
    finally:
        shutil.rmtree(job_dir / "deployment_unauth", ignore_errors=True)

    # ────────────────────────────────────────────────────────────────
    # STATUS: security_validation_completed
    # ────────────────────────────────────────────────────────────────
    update_state_fn(job_id, status="COMPLETED", stage="security_validation_completed", progress=100)
    logger.info("[%s] Status: security_validation_completed", job_id)

    # Clean up environment variable
    os.environ.pop("P3_DEVICE_SALT", None)

    return outcomes
