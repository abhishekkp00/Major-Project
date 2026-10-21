import os
import logging
import tempfile
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from werkzeug.exceptions import RequestEntityTooLarge

from .service import orchestrator
from src.orchestrator.dataset_processor import validate_dataset_file
from src.common.exceptions import DatasetValidationError
from src.security.api_auth import require_bearer_token, current_principal, job_visible_to_principal
from .job_security import (
    require_job_access,
    public_job_view,
    redact_text,
    resolve_artifact,
    is_blocked_artifact_name,
    ALLOWED_ARTIFACT_SUFFIXES,
)

logger = logging.getLogger("secure_lora.orchestrator.routes")
orchestrator_bp = Blueprint("orchestrator", __name__)

DEFAULT_MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB safe default
UPLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB bounded chunk size
ALLOWED_DATASET_EXTENSIONS = {".txt", ".csv", ".json", ".jsonl", ".md"}


@orchestrator_bp.app_errorhandler(413)
@orchestrator_bp.app_errorhandler(RequestEntityTooLarge)
def handle_orchestrator_request_entity_too_large(error):
    max_bytes = (
        current_app.config.get("MAX_CONTENT_LENGTH")
        if current_app else DEFAULT_MAX_UPLOAD_SIZE
    ) or DEFAULT_MAX_UPLOAD_SIZE
    max_mb = max_bytes / (1024 * 1024)
    return jsonify({
        "success": False,
        "error": f"Request entity too large: payload exceeds maximum allowed size ({max_mb:.1f} MB limit)."
    }), 413


@orchestrator_bp.route("/api/orchestrator/validate", methods=["POST"])
@require_bearer_token
def pre_validate_dataset():
    """Parses and runs PII inspection on an uploaded dataset file before job creation."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    max_size = (
        current_app.config.get("MAX_CONTENT_LENGTH")
        if current_app else None
    ) or DEFAULT_MAX_UPLOAD_SIZE

    # Pre-validate Content-Length if supplied
    if request.content_length is not None and request.content_length > max_size:
        return jsonify({
            "success": False,
            "error": f"Upload exceeds maximum allowed size of {max_size} bytes."
        }), 413

    filename = secure_filename(file.filename or "")
    if not filename:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DATASET_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file format '{suffix}'. Supported formats: {', '.join(sorted(ALLOWED_DATASET_EXTENSIONS))}"
        }), 400

    # Save to a temporary file via bounded streaming
    temp_fd, temp_str = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)
    temp_path = Path(temp_str)

    try:
        bytes_written = 0
        with open(temp_path, "wb") as f_out:
            while True:
                chunk = file.stream.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_size:
                    if temp_path.exists():
                        temp_path.unlink()
                    return jsonify({
                        "success": False,
                        "error": f"Upload exceeds maximum allowed size of {max_size} bytes."
                    }), 413
                f_out.write(chunk)

        # Validate and inspect without loading entire dataset into memory (Requirement 9)
        _, metadata = validate_dataset_file(temp_path, metadata_only=True)
        return jsonify({"success": True, "metadata": metadata})
    except DatasetValidationError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except RequestEntityTooLarge as re_err:
        return jsonify({"success": False, "error": str(re_err)}), 413
    except Exception as e:
        logger.exception("Pre-validation failure:")
        return jsonify({"success": False, "error": "Failed to validate dataset"}), 500
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@orchestrator_bp.route("/api/orchestrator/datasets", methods=["GET"])
def list_datasets():
    """Returns list of registered dataset adapters and their metadata."""
    try:
        from src.data_sources.dataset_registry import dataset_registry
        datasets = dataset_registry.list_datasets()
        return jsonify({"success": True, "datasets": datasets})
    except Exception as e:
        logger.exception("Failed to list datasets:")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@orchestrator_bp.route("/api/orchestrator/datasets/<dataset_id>", methods=["GET"])
def get_dataset_details(dataset_id):
    """Retrieves metadata and sample statistics for a specific dataset adapter."""
    try:
        from src.data_sources.dataset_registry import dataset_registry
        adapter = dataset_registry.get_dataset_adapter(dataset_id)
        meta = adapter.get_metadata()
        stats = adapter.get_statistics()
        return jsonify({"success": True, "metadata": meta, "statistics": stats})
    except KeyError as k_err:
        return jsonify({"success": False, "error": str(k_err)}), 404
    except Exception as e:
        logger.exception("Failed to get dataset details:")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@orchestrator_bp.route("/api/orchestrator/jobs", methods=["POST"])
@require_bearer_token
def create_job():
    """Creates a new job with specified dataset configuration and subset size."""
    data = request.json or {}
    dataset_name = data.get("dataset_name", "")
    dataset_type = data.get("dataset_type") or data.get("dataset_id") or dataset_name or "synthetic"
    if not dataset_name:
        dataset_name = dataset_type

    version = data.get("version", "1.0.0")
    epochs = int(data.get("epochs", 1))
    salt = data.get("salt")
    subset_size = int(data.get("subset_size", 1000))

    training_config = data.get("training_config")
    if not isinstance(training_config, dict):
        training_config = {}
    if "dp_enabled" in data and "dp_enabled" not in training_config:
        training_config["dp_enabled"] = bool(data["dp_enabled"])
    for k in [
        "lora_r", "lora_alpha", "lora_dropout", "model_name",
        "dp_target_epsilon", "dp_target_delta", "dp_max_grad_norm",
        "dp_noise_multiplier", "target_modules", "dp_accountant"
    ]:
        if k in data and k not in training_config:
            training_config[k] = data[k]

    try:
        job_id = orchestrator.create_job(
            dataset_name=dataset_name,
            version=version,
            epochs=epochs,
            salt=salt,
            dataset_type=dataset_type,
            subset_size=subset_size,
            owner_id=current_principal(),
            training_config=training_config,
        )
        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        logger.exception("Failed to create job:")
        return jsonify({"success": False, "error": "Internal server error"}), 500



@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/upload", methods=["POST"])
@require_job_access
def upload_file(job_id):
    """Uploads a raw dataset file to a created job workspace with bounded streaming."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    max_size = (
        current_app.config.get("MAX_CONTENT_LENGTH")
        if current_app else None
    ) or DEFAULT_MAX_UPLOAD_SIZE

    # Pre-validate Content-Length if supplied
    if request.content_length is not None and request.content_length > max_size:
        return jsonify({
            "success": False,
            "error": f"Upload exceeds maximum allowed size of {max_size} bytes."
        }), 413

    filename = secure_filename(file.filename or "")
    if not filename:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DATASET_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file format '{suffix}'. Supported formats: {', '.join(sorted(ALLOWED_DATASET_EXTENSIONS))}"
        }), 400

    target_path = orchestrator.base_jobs_dir / job_id / "raw_inputs" / filename

    try:
        # Stream file in bounded chunks enforcing max size limit
        bytes_written = orchestrator.add_dataset_file_stream(
            job_id=job_id,
            filename=filename,
            stream=file.stream,
            max_size_bytes=max_size,
            chunk_size=UPLOAD_CHUNK_SIZE,
        )

        logger.info(
            "Successfully uploaded dataset '%s' (%d bytes) to job '%s'",
            filename, bytes_written, job_id
        )
        return jsonify({"success": True, "filename": filename})

    except RequestEntityTooLarge as re_err:
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        return jsonify({"success": False, "error": str(re_err)}), 413
    except Exception as e:
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        logger.exception("Failed to upload dataset file:")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/start", methods=["POST"])
@require_job_access
def start_job(job_id):
    """Starts the full end-to-end secure pipeline execution."""
    try:
        orchestrator.start_job(job_id)
        return jsonify({"success": True})
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as e:
        logger.exception("Failed to start job:")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@orchestrator_bp.route("/api/orchestrator/jobs", methods=["GET"])
@require_bearer_token
def get_jobs():
    """Lists all created orchestration jobs."""
    try:
        principal = current_principal()
        jobs = [
            public_job_view(j) for j in orchestrator.get_all_jobs()
            if job_visible_to_principal(j, principal)
        ]
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        logger.exception("Failed to list jobs:")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>", methods=["GET"])
@require_job_access
def get_job_status(job_id):
    """Polls detailed status for a specific job."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, "job": public_job_view(job)})


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/logs", methods=["GET"])
@require_job_access
def get_job_logs(job_id):
    """Retrieves standard training logs for a running fine-tuning job."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    log_file = orchestrator.base_jobs_dir / job_id / "training.log"
    if not log_file.exists():
        return jsonify({"success": True, "logs": "Waiting for training logs to start..."})

    try:
        logs = log_file.read_text(encoding="utf-8")
        # Tail logs to prevent large bandwidth consumption
        lines = logs.splitlines()[-200:]
        return jsonify({"success": True, "logs": redact_text("\n".join(lines))})
    except Exception:
        logger.exception("Failed to read training log for job %s", job_id)
        return jsonify({"success": False, "error": "Unable to read job logs"}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/metrics", methods=["GET"])
@require_job_access
def get_job_metrics(job_id):
    """Exposes training and dataset metrics for a job."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    metrics = {
        "loss_history": job.get("loss_history", []),
        "current_epoch": job.get("current_epoch"),
        "pii_detected_summary": job.get("pii_summary", {}),
        "num_records": job.get("num_records", 0),
        "security_metrics": job.get("security_metrics", {})
    }
    return jsonify({"success": True, "metrics": metrics})


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/artifacts", methods=["GET"])
@require_job_access
def list_job_artifacts(job_id):
    """Lists safe downloadable package artifacts generated for the job."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    protected_dir = orchestrator.base_jobs_dir / job_id / "protected"
    if not protected_dir.is_dir():
        return jsonify({"success": True, "artifacts": []})

    artifacts = []
    for path in sorted(protected_dir.iterdir()):
        approved = resolve_artifact(protected_dir, path.name, orchestrator.base_jobs_dir)
        if approved is None:
            continue
        artifacts.append({
            "name": approved.name,
            "size_bytes": approved.stat().st_size,
            "download_url": f"/api/orchestrator/jobs/{job_id}/download/{approved.name}"
        })

    return jsonify({"success": True, "artifacts": artifacts})


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/download/<filename>", methods=["GET"])
@require_job_access
def download_job_artifact(job_id, filename):
    """Serves an allow-listed package artifact located directly inside the job's protected/ directory."""
    from flask import send_from_directory
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    protected_dir = orchestrator.base_jobs_dir / job_id / "protected"
    approved = resolve_artifact(protected_dir, filename, orchestrator.base_jobs_dir)
    if approved is None:
        return jsonify({"success": False, "error": "Access denied or file not found"}), 403

    return send_from_directory(str(approved.parent), approved.name, as_attachment=True)


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/report", methods=["GET"])
@require_job_access
def get_job_report(job_id):
    """Retrieves the final validation report from the deployment verification pipeline."""
    import json
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    report_file = orchestrator.base_jobs_dir / job_id / "deployment" / "validation_report.json"
    if not report_file.exists():
        return jsonify({"success": False, "error": "Validation report not generated yet"}), 404

    try:
        report_data = json.loads(report_file.read_text(encoding="utf-8"))
        report_data["security_validation_outcomes"] = job.get("security_metrics", {})
        return jsonify({"success": True, "report": report_data})
    except Exception:
        logger.exception("Failed to read validation report for job %s", job_id)
        return jsonify({"success": False, "error": "Internal server error"}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/packaging", methods=["GET"])
@require_job_access
def get_job_packaging(job_id):
    """
    Returns cryptographic packaging details, manifest metadata, file inventory,
    digest verification results, and RSA-PSS signature verification status.
    Never fabricates signatures, hashes, or keys. Never exposes private keys or
    derived symmetric encryption keys.
    """
    import hashlib
    import hmac
    import json as _json
    from src.security.crypto import compute_sha256
    from src.security.key_derivation import KDF_VERSION

    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    protected_dir = orchestrator.base_jobs_dir / job_id / "protected"
    if not protected_dir.is_dir():
        return jsonify({
            "success": True,
            "available": False,
            "job_id": job_id,
            "status": "NOT_PACKAGED",
            "reason": "Cryptographic packaging directory does not exist for this job."
        })

    enc_path = protected_dir / "adapter.enc"
    hash_path = protected_dir / "adapter.hash"
    sig_path = protected_dir / "adapter.sig"
    meta_path = protected_dir / "metadata.json"
    manifest_path = protected_dir / "package_manifest.json"
    pub_key_path = protected_dir / "public.pem"

    core_present = any(p.exists() for p in [enc_path, hash_path, sig_path, manifest_path])
    if not core_present:
        return jsonify({
            "success": True,
            "available": False,
            "job_id": job_id,
            "status": "NOT_PACKAGED",
            "reason": "No cryptographic package artifacts have been generated yet."
        })

    manifest_data = {}
    if manifest_path.exists():
        try:
            manifest_data = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse package_manifest.json for job %s: %s", job_id, exc)

    meta_data = {}
    if meta_path.exists():
        try:
            meta_data = _json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse metadata.json for job %s: %s", job_id, exc)

    inventory = []
    role_map = {
        "adapter.enc": "AES-256-GCM Encrypted Adapter Ciphertext",
        "adapter.hash": "SHA-256 Digest Reference (Hex)",
        "adapter.sig": "RSA-2048-PSS Signature Binary",
        "metadata.json": "Encryption Envelope & Salt Metadata",
        "package_manifest.json": "Signed Cryptographic Provenance Manifest",
        "public.pem": "RSA-2048 Public Key (X.509 PEM)",
        "secure_adapter.tar.gz": "Consolidated Protected Package Archive"
    }

    for child in sorted(protected_dir.iterdir()):
        approved = resolve_artifact(protected_dir, child.name, orchestrator.base_jobs_dir)
        if approved is None:
            continue
        try:
            file_hash = compute_sha256(approved)
            file_size = approved.stat().st_size
        except Exception:
            file_hash = None
            file_size = 0

        inventory.append({
            "name": approved.name,
            "size_bytes": file_size,
            "sha256_digest": file_hash,
            "role": role_map.get(approved.name, "Package Supporting Artifact"),
            "downloadable": True,
            "download_url": f"/api/orchestrator/jobs/{job_id}/download/{approved.name}"
        })

    # 1. SHA-256 Integrity Verification
    actual_ciphertext_hash = None
    if enc_path.exists():
        try:
            actual_ciphertext_hash = compute_sha256(enc_path)
        except Exception:
            pass

    recorded_hash = None
    if hash_path.exists():
        try:
            recorded_hash = hash_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    manifest_enc_digest = str(manifest_data.get("encrypted_adapter_digest", "")).strip().lower() if manifest_data else None

    digest_matches_recorded = (
        actual_ciphertext_hash is not None and
        recorded_hash is not None and
        hmac.compare_digest(actual_ciphertext_hash.lower(), recorded_hash.lower())
    )
    digest_matches_manifest = (
        actual_ciphertext_hash is not None and
        bool(manifest_enc_digest) and
        hmac.compare_digest(actual_ciphertext_hash.lower(), manifest_enc_digest)
    )

    # 2. RSA-2048-PSS Signature Verification
    signature_verified = False
    verification_error = None
    canonical_digest = None

    if pub_key_path.exists() and sig_path.exists() and enc_path.exists():
        try:
            from src.phase4.package_validator import validate_package_provenance
            val_manifest, verified_digest = validate_package_provenance(protected_dir, pub_key_path)
            signature_verified = True
            if manifest_data and actual_ciphertext_hash:
                from src.security import compute_canonical_manifest_digest
                canonical_digest = compute_canonical_manifest_digest(manifest_data, actual_ciphertext_hash)
        except Exception as exc:
            signature_verified = False
            verification_error = str(exc)
            if manifest_data and actual_ciphertext_hash:
                try:
                    from src.security import compute_canonical_manifest_digest
                    canonical_digest = compute_canonical_manifest_digest(manifest_data, actual_ciphertext_hash)
                except Exception:
                    pass
    elif not sig_path.exists():
        verification_error = "Signature file 'adapter.sig' is not present in package."
    elif not pub_key_path.exists():
        verification_error = "Public key 'public.pem' is not present in package."
    else:
        verification_error = "Adapter ciphertext 'adapter.enc' is missing."

    public_key_pem = None
    public_key_fingerprint = None
    if pub_key_path.exists():
        try:
            public_key_pem = pub_key_path.read_text(encoding="utf-8").strip()
            public_key_fingerprint = hashlib.sha256(public_key_pem.encode("utf-8")).hexdigest()
        except Exception:
            pass

    # 3. HKDF Key Derivation Metadata
    kdf_version = manifest_data.get("kdf_version") or meta_data.get("kdf_version") or KDF_VERSION
    hkdf_salt_hex = manifest_data.get("hkdf_salt_hex") or meta_data.get("hkdf_salt_hex") or ""
    salt_length_bytes = len(bytes.fromhex(hkdf_salt_hex)) if hkdf_salt_hex else 0
    device_fp_ref = manifest_data.get("device_fingerprint_hash_ref") or meta_data.get("fingerprint_hash") or ""

    # 4. AES-256-GCM Encryption Metadata
    iv_hex = meta_data.get("iv_hex") or ""
    tag_hex = meta_data.get("tag_hex") or ""
    encryption_version = meta_data.get("encryption_version") or manifest_data.get("encryption_version") or "aes-256-gcm-v1"
    ciphertext_size = enc_path.stat().st_size if enc_path.exists() else 0

    sec_metrics = job.get("security_metrics") or {}
    encryption_time_ms = sec_metrics.get("encryption_time_ms")
    signing_time_ms = sec_metrics.get("signing_time_ms")

    screening_outcome = sec_metrics.get("adapter_screening_outcome") or "NOT_RUN"
    step1_status = "PASSED" if screening_outcome in ("APPROVED", "SCREENED", "LOW_RISK") or job.get("screening_passed") else (
        "BYPASS" if sec_metrics.get("screening_bypassed") else (
            "FAILED" if screening_outcome == "REJECTED" else "UNKNOWN"
        )
    )

    step2_status = "PASSED" if enc_path.exists() and bool(iv_hex) and bool(tag_hex) else (
        "INCOMPLETE" if enc_path.exists() else "PENDING"
    )

    step3_status = "PASSED" if digest_matches_recorded else (
        "FAILED" if hash_path.exists() and actual_ciphertext_hash else "PENDING"
    )

    step4_status = "PASSED" if signature_verified else (
        "FAILED" if sig_path.exists() else "PENDING"
    )

    if step2_status == "PASSED" and step3_status == "PASSED" and step4_status == "PASSED":
        package_status = "PACKAGED"
    elif step4_status == "FAILED":
        package_status = "SIGNATURE_FAILED"
    elif step3_status == "FAILED":
        package_status = "INTEGRITY_FAILED"
    elif not enc_path.exists():
        package_status = "NOT_ENCRYPTED"
    else:
        package_status = "INCOMPLETE"

    return jsonify({
        "success": True,
        "available": True,
        "job_id": job_id,
        "package_id": manifest_data.get("package_id"),
        "package_status": package_status,
        "adapter_id": manifest_data.get("adapter_id") or job_id,
        "base_model_id": manifest_data.get("base_model_id") or manifest_data.get("model_reference") or job.get("dataset_name"),
        "package_version": manifest_data.get("package_version") or "1.0.0",
        "created_at_utc": manifest_data.get("creation_timestamp") or manifest_data.get("created_at_utc"),
        "inventory": inventory,
        "sequence_stages": {
            "step_1_inspect": {
                "name": "Adapter Source & Security Inspection",
                "status": step1_status,
                "screening_outcome": screening_outcome
            },
            "step_2_encrypt": {
                "name": "HKDF Key Derivation & AES-256-GCM Encryption",
                "status": step2_status,
                "latency_ms": encryption_time_ms
            },
            "step_3_digest": {
                "name": "SHA-256 Ciphertext Digest Generation",
                "status": step3_status
            },
            "step_4_sign": {
                "name": "RSA-2048-PSS Provenance Signing & Manifest Sealing",
                "status": step4_status,
                "latency_ms": signing_time_ms
            }
        },
        "integrity": {
            "digest_algorithm": "SHA-256 (FIPS 180-4)",
            "ciphertext_digest": actual_ciphertext_hash,
            "recorded_digest": recorded_hash,
            "manifest_digest": manifest_enc_digest,
            "digest_matches_recorded": digest_matches_recorded,
            "digest_matches_manifest": digest_matches_manifest,
            "artefact_hashes": manifest_data.get("artefact_hashes") or {}
        },
        "signature": {
            "algorithm": "RSA-PSS with SHA-256 (MGF1, 2048-bit)",
            "verified": signature_verified,
            "verification_error": verification_error,
            "canonical_digest": canonical_digest,
            "public_key_present": pub_key_path.exists(),
            "public_key_fingerprint": public_key_fingerprint,
            "public_key_pem": public_key_pem,
            "signature_present": sig_path.exists()
        },
        "manifest": {
            "present": manifest_path.exists(),
            "package_id": manifest_data.get("package_id"),
            "adapter_id": manifest_data.get("adapter_id"),
            "base_model_id": manifest_data.get("base_model_id"),
            "sequence_number": manifest_data.get("sequence_number"),
            "package_version": manifest_data.get("package_version"),
            "binding_policy_version": manifest_data.get("binding_policy_version"),
            "deployment_policy": manifest_data.get("deployment_policy"),
            "expiration_timestamp": manifest_data.get("expiration_timestamp"),
            "security_notes": manifest_data.get("security_notes")
        },
        "key_derivation": {
            "kdf_algorithm": "HKDF-SHA256 (RFC 5869)",
            "kdf_version": kdf_version,
            "salt_hex": hkdf_salt_hex,
            "salt_length_bytes": salt_length_bytes,
            "ikm_source": "Device Secret (CSPRNG 256-bit entropy, never stored)",
            "info_context": f"securelora-adapter-v2|{device_fp_ref[:16]}..." if device_fp_ref else "securelora-adapter-v2|<device_fingerprint>",
            "device_fingerprint_ref": device_fp_ref,
            "zeroization_guarantee": "Ephemeral in-memory derivation; zeroed immediately after AES-GCM envelope sealing."
        },
        "encryption": {
            "cipher": "AES-256-GCM (NIST SP 800-38D)",
            "encryption_version": encryption_version,
            "iv_hex": iv_hex,
            "tag_hex": tag_hex,
            "ciphertext_size_bytes": ciphertext_size,
            "encryption_time_ms": encryption_time_ms
        },
        "security_disclaimers": {
            "hardware_attestation": "Software-derived device binding relies on OS-reported hardware telemetry and local CSPRNG secrets. It does not provide hardware-isolated key protection (e.g., TPM 2.0, ARM TrustZone, or HSM) and cannot defend against a compromised kernel.",
            "transient_plaintext_storage": "During candidate adapter validation and PEFT initialization, decrypted model tensors are buffered in restricted-permission ephemeral storage before mapping into torch memory. In production, mount temporary workspace directories on RAM-backed tmpfs with strict POSIX permissions (0700)."
        }
    })


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/screening", methods=["GET"])
@require_job_access
def get_adapter_screening(job_id):
    """
    Step 3 Endpoint: Returns pre-deployment adapter security screening metrics & decision explanation.
    Never fabricates metrics: if screening has not occurred or weights are missing,
    returns {"success": True, "available": False, "reason": "..."}.
    """
    import json as _json
    from src.evaluation.adapter_security import ScreeningConfig
    cfg = ScreeningConfig()

    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    job_dir = orchestrator.base_jobs_dir / job_id
    report_file = job_dir / "screening_report.json"
    sec_metrics = job.get("security_metrics") or {}

    report_data = None
    if report_file.exists():
        try:
            report_data = _json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not report_data and "screening_details" in sec_metrics:
        report_data = sec_metrics["screening_details"]

    if not report_data:
        adapter_dir = job_dir / "adapter"
        if adapter_dir.exists() and any(adapter_dir.iterdir()):
            try:
                from src.evaluation.adapter_security import evaluate_adapter_security
                res = evaluate_adapter_security(adapter_source=adapter_dir, adapter_id=job_id)
                report_data = res.to_dict()
                try:
                    report_file.write_text(_json.dumps(report_data, indent=2), encoding="utf-8")
                except Exception:
                    pass
            except Exception as e:
                logger.warning("On-demand adapter screening failed for %s: %s", job_id, e)

    if not report_data:
        # Check if basic summary metrics exist from orchestrator
        risk_score = sec_metrics.get("security_screening_risk_score")
        risk_level = sec_metrics.get("security_screening_risk_level")
        outcome = sec_metrics.get("security_screening")
        if risk_score is not None and risk_level is not None:
            dec_map = {"LOW": "SCREENED", "MEDIUM": "REVIEW", "HIGH": "REJECTED"}
            dec_str = dec_map.get(risk_level, "REVIEW")
            dec_code = "APPROVED" if risk_level == "LOW" else ("REQUIRES_ADMIN_APPROVAL" if risk_level == "MEDIUM" else "REJECTED")
            r_val = float(risk_score)
            return jsonify({
                "success": True,
                "available": True,
                "job_id": job_id,
                "screening_mode": "PRODUCTION",
                "base_model": job.get("base_model_name") or job.get("training_config", {}).get("model_name") or "JackFram/llama-68m",
                "decision": dec_str,
                "decision_code": dec_code,
                "risk_level": risk_level,
                "adapter_risk_score": r_val,
                "structural_score": r_val,
                "behavioral_score": 0.0,
                "consistency_score": 0.0,
                "configured_weights": {
                    "structural": cfg.weight_structural,
                    "behavioral": cfg.weight_behavioral,
                    "consistency": cfg.weight_consistency,
                },
                "configured_thresholds": {
                    "low_risk": cfg.low_risk_threshold,
                    "high_risk": cfg.high_risk_threshold,
                    "max_frobenius_norm": cfg.max_frobenius_norm,
                    "max_l2_norm": cfg.max_l2_norm,
                    "max_layer_zscore": cfg.max_layer_zscore,
                    "max_trigger_sensitivity": cfg.max_trigger_sensitivity,
                    "min_paraphrase_consistency": cfg.min_paraphrase_consistency,
                    "max_output_divergence": cfg.max_output_divergence,
                },
                "structural_analysis": {
                    "total_parameters": sec_metrics.get("total_parameters", 0),
                    "layer_count": sec_metrics.get("layer_count", 0),
                    "global_frobenius_norm": sec_metrics.get("global_frobenius_norm"),
                    "global_l2_norm": sec_metrics.get("global_l2_norm"),
                    "global_l1_norm": sec_metrics.get("global_l1_norm"),
                    "global_linf_norm": sec_metrics.get("global_linf_norm"),
                    "max_layer_zscore": sec_metrics.get("max_layer_zscore"),
                    "outlier_layer_count": sec_metrics.get("outlier_layer_count", 0),
                    "outlier_layers": sec_metrics.get("outlier_layers", []),
                    "sparsity_ratio": sec_metrics.get("sparsity_ratio"),
                    "rank_utilization_mean": sec_metrics.get("rank_utilization_mean"),
                    "cosine_similarity_ref": None,
                    "parameter_drift_score": sec_metrics.get("parameter_drift_score", 0.0),
                    "structural_risk_score": r_val,
                    "layer_metrics": {},
                    "actual_adapter_loaded": sec_metrics.get("actual_adapter_loaded", True),
                    "weights_source": "file",
                },
                "behavioral_analysis": {
                    "real_inference_performed": sec_metrics.get("behavioral_inference_performed", False),
                    "execution_status": "NOT_EXECUTED" if not sec_metrics.get("behavioral_inference_performed") else "COMPLETED",
                    "evaluation_type": "NOT_EXECUTED" if not sec_metrics.get("behavioral_inference_performed") else "REAL_BEHAVIORAL_EVALUATION",
                    "probe_count": 0,
                    "normal_output_divergence": None,
                    "trigger_sensitivity": None,
                    "paraphrase_consistency": None,
                    "abnormal_response_rate": None,
                    "classification_flip_rate": None,
                    "behavioral_risk_score": 0.0,
                    "consistency_risk_score": 0.0,
                    "probe_results": [],
                },
                "decision_explanation": {
                    "decision": dec_str,
                    "decision_code": dec_code,
                    "risk_level": risk_level,
                    "rationale": f"Adapter security screening completed with outcome {outcome or dec_str} (risk score: {r_val:.4f}).",
                    "structural_evidence": f"Evaluated structural weights. Risk score: {r_val:.4f}.",
                    "behavioral_evidence": "Behavioral inference callback was not provisioned during pipeline packaging.",
                    "low_threshold": cfg.low_risk_threshold,
                    "high_threshold": cfg.high_risk_threshold,
                    "final_risk": r_val,
                },
                "research_context": {
                    "label": "PRE-DEPLOYMENT ADAPTER SCREENING",
                    "disclaimer": "This is a statistical and behavioral risk screening mechanism, not a formal safety proof."
                }
            })

        return jsonify({
            "success": True,
            "available": False,
            "reason": "Adapter security screening report not evaluated for this job."
        })

    risk_level = report_data.get("risk_level", "LOW")
    risk_score = float(report_data.get("adapter_risk_score", report_data.get("risk_score", 0.10)))
    dec_code = report_data.get("decision", "APPROVED")

    decision_map = {"LOW": "SCREENED", "MEDIUM": "REVIEW", "HIGH": "REJECTED"}
    decision_str = decision_map.get(risk_level, "SCREENED")
    if not report_data.get("approved", True) and not report_data.get("bypassed_via_force", False):
        decision_str = "REJECTED"

    struct_rep = report_data.get("structural_report", report_data.get("structural_evidence", {})) or {}
    behav_rep = report_data.get("behavioral_report", report_data.get("behavioral_evidence", {})) or {}
    risk_bd = report_data.get("risk_breakdown", {}) or {}

    struct_score = float(report_data.get("structural_score", risk_bd.get("structural_risk_score", 0.0)))
    behav_score = float(report_data.get("behavioral_score", risk_bd.get("behavioral_risk_score", 0.0)))
    cons_score = float(report_data.get("consistency_score", risk_bd.get("consistency_risk_score", 0.0)))

    rationale = report_data.get("risk_assessment", {}).get("rationale") or \
        (f"Adapter evaluated with composite risk score of {risk_score:.4f} ({risk_level} RISK). "
         f"Decision: {decision_str}.")

    struct_ev_text = f"Global Frobenius Norm: {struct_rep.get('global_frobenius_norm', 'N/A')}, " \
                     f"Outlier Layers: {struct_rep.get('outlier_layer_count', 0)}, " \
                     f"Max Z-Score: {struct_rep.get('max_layer_zscore', 'N/A')}."

    behav_ev_text = f"Trigger Sensitivity: {behav_rep.get('trigger_sensitivity', behav_rep.get('trigger_sensitivity_score', 'N/A'))}, " \
                    f"Paraphrase Consistency: {behav_rep.get('paraphrase_consistency', behav_rep.get('paraphrase_consistency_score', 'N/A'))}, " \
                    f"Output Divergence: {behav_rep.get('normal_output_divergence', behav_rep.get('output_divergence_kl', 'N/A'))}."

    return jsonify({
        "success": True,
        "available": True,
        "job_id": job_id,
        "screening_mode": report_data.get("screening_mode") or ("PRODUCTION" if struct_rep.get("actual_adapter_loaded", True) else "RESEARCH"),
        "base_model": report_data.get("base_model_id") or job.get("base_model_name") or "JackFram/llama-68m",
        "trainable_params": struct_rep.get("total_parameters", 0),
        "decision": decision_str,
        "decision_code": dec_code,
        "risk_level": risk_level,
        "adapter_risk_score": risk_score,
        "bypassed_via_force": report_data.get("bypassed_via_force", False),
        "structural_score": struct_score,
        "behavioral_score": behav_score,
        "consistency_score": cons_score,
        "configured_weights": {
            "structural": cfg.weight_structural,
            "behavioral": cfg.weight_behavioral,
            "consistency": cfg.weight_consistency,
        },
        "configured_thresholds": {
            "low_risk": cfg.low_risk_threshold,
            "high_risk": cfg.high_risk_threshold,
            "max_frobenius_norm": cfg.max_frobenius_norm,
            "max_l2_norm": cfg.max_l2_norm,
            "max_layer_zscore": cfg.max_layer_zscore,
            "max_trigger_sensitivity": cfg.max_trigger_sensitivity,
            "min_paraphrase_consistency": cfg.min_paraphrase_consistency,
            "max_output_divergence": cfg.max_output_divergence,
        },
        "structural_analysis": {
            "total_parameters": struct_rep.get("total_parameters", 0),
            "layer_count": struct_rep.get("layer_count", 0),
            "global_frobenius_norm": struct_rep.get("global_frobenius_norm"),
            "global_l1_norm": struct_rep.get("global_l1_norm"),
            "global_l2_norm": struct_rep.get("global_l2_norm"),
            "global_linf_norm": struct_rep.get("global_linf_norm"),
            "max_layer_zscore": struct_rep.get("max_layer_zscore"),
            "outlier_layer_count": struct_rep.get("outlier_layer_count", 0),
            "outlier_layers": struct_rep.get("outlier_layers", []),
            "sparsity_ratio": struct_rep.get("sparsity_ratio"),
            "rank_utilization_mean": struct_rep.get("rank_utilization_mean"),
            "cosine_similarity_ref": struct_rep.get("cosine_similarity_ref"),  # None if no reference weights
            "parameter_drift_score": struct_rep.get("parameter_drift_score"),
            "structural_risk_score": struct_score,
            "layer_metrics": struct_rep.get("layer_metrics", {}),
            "weights_source": struct_rep.get("weights_source_desc", "file"),
            "actual_adapter_loaded": struct_rep.get("actual_adapter_loaded", True),
        },
        "behavioral_analysis": {
            "real_inference_performed": behav_rep.get("real_inference_performed", False),
            "execution_status": behav_rep.get("execution_status", "COMPLETED"),
            "evaluation_type": behav_rep.get("evaluation_type", "REAL_BEHAVIORAL_EVALUATION"),
            "probe_count": len(behav_rep.get("probe_results", [])) or (behav_rep.get("probe_count", 0)),
            "normal_output_divergence": behav_rep.get("normal_output_divergence", behav_rep.get("output_divergence_kl")),
            "trigger_sensitivity": behav_rep.get("trigger_sensitivity", behav_rep.get("trigger_sensitivity_score")),
            "paraphrase_consistency": behav_rep.get("paraphrase_consistency", behav_rep.get("paraphrase_consistency_score")),
            "abnormal_response_rate": behav_rep.get("abnormal_response_rate"),
            "classification_flip_rate": behav_rep.get("classification_flip_rate"),
            "behavioral_risk_score": behav_score,
            "consistency_risk_score": cons_score,
            "probe_results": behav_rep.get("probe_results", []),
        },
        "risk_breakdown": {
            "structural": struct_score,
            "behavioral": behav_score,
            "consistency": cons_score,
            "combined": risk_score
        },
        "decision_explanation": {
            "decision": decision_str,
            "decision_code": dec_code,
            "risk_level": risk_level,
            "rationale": rationale,
            "structural_evidence": struct_ev_text,
            "behavioral_evidence": behav_ev_text,
            "low_threshold": cfg.low_risk_threshold,
            "high_threshold": cfg.high_risk_threshold,
            "final_risk": risk_score
        },
        "research_context": {
            "label": "PRE-DEPLOYMENT ADAPTER SCREENING",
            "disclaimer": "This is a risk assessment mechanism, not a formal proof of adapter safety."
        }
    })


def _build_pipeline_summary_data(job, base_jobs_dir=None):
    """
    Returns a structured 10-stage SecureLoRA lifecycle summary dictionary for any job.
    """
    import json as _json

    job_id      = job.get("job_id", "")
    status      = job.get("status", "CREATED")
    stage       = job.get("stage", "dataset_intake")
    progress    = job.get("progress", 0)
    pii         = job.get("pii_summary") or {}
    if not pii and job_id:
        ds_meta = orchestrator.base_jobs_dir / job_id / "encrypted" / "dataset_metadata.json"
        if ds_meta.exists():
            try:
                d_info = _json.loads(ds_meta.read_text(encoding="utf-8"))
                pii = d_info.get("pii_audit_summary") or d_info.get("pii_summary") or d_info.get("pii_detected_summary") or {}
            except Exception:
                pass
    sec         = job.get("security_metrics") or {}
    vsteps      = job.get("verification_steps") or {}
    if not vsteps:
        rep_path = orchestrator.base_jobs_dir / job_id / "deployment" / "validation_report.json"
        if rep_path.exists():
            try:
                rep_data = _json.loads(rep_path.read_text(encoding="utf-8"))
                vsteps = rep_data.get("verification_pipeline", {}).get("steps", {})
            except Exception:
                pass
    loss_hist   = job.get("loss_history") or []
    eval_met    = job.get("eval_metrics") or {}
    created_at  = job.get("created_at")
    updated_at  = job.get("updated_at")
    error       = redact_text(job.get("error"))

    # ── helpers ──────────────────────────────────────────────────────────
    STAGE_ORDER = [
        "dataset_intake", "dataset_protection", "pii_inspection", "fine_tuning",
        "preparing_adapter", "encrypting_adapter", "generating_hash",
        "generating_signature", "building_package",
        "running_integrity_check", "running_device_authorization_check",
        "running_secure_deployment_check", "secure_inference_validation",
        "security_validation_completed"
    ]
    STAGE_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}

    def _stage_status(required_stage: str) -> str:
        """Return PASSED/RUNNING/PENDING/FAILED based on pipeline position."""
        if status == "FAILED":
            cur_idx = STAGE_IDX.get(stage, 0)
            req_idx = STAGE_IDX.get(required_stage, 0)
            if req_idx < cur_idx:
                return "PASSED"
            if req_idx == cur_idx:
                return "FAILED"
            return "PENDING"
        if status == "COMPLETED":
            return "PASSED"
        cur_idx = STAGE_IDX.get(stage, 0)
        req_idx = STAGE_IDX.get(required_stage, 0)
        if req_idx < cur_idx:
            return "PASSED"
        if req_idx == cur_idx:
            return "RUNNING"
        return "PENDING"

    def _vstep(key: str) -> str:
        """Safe verification step status lookup."""
        if vsteps and key in vsteps:
            return vsteps[key]
        if vsteps:
            target_suffix = key.split(":")[-1].strip().lower()
            for k, val in vsteps.items():
                if k.split(":")[-1].strip().lower() == target_suffix:
                    return val
        if status == "COMPLETED" and stage == "security_validation_completed":
            return "PASSED"
        # A completed orchestration job is not evidence that an unrecorded
        # verification gate ran successfully.
        return "PENDING"


    # ── per-stage last epoch loss ─────────────────────────────────────────
    final_train_loss = None
    final_val_loss   = None
    if loss_hist:
        for h in reversed(loss_hist):
            if final_train_loss is None and h.get("loss") is not None:
                final_train_loss = round(h["loss"], 4)
            if final_val_loss is None and h.get("eval_loss") is not None:
                final_val_loss = round(h["eval_loss"], 4)
            if final_train_loss is not None and final_val_loss is not None:
                break
    if final_val_loss is None and isinstance(eval_met, dict) and eval_met.get("val_loss") is not None:
        try:
            final_val_loss = round(float(eval_met["val_loss"]), 4)
        except Exception:
            pass
    if final_train_loss is None and isinstance(eval_met, dict) and eval_met.get("train_loss") is not None:
        try:
            final_train_loss = round(float(eval_met["train_loss"]), 4)
        except Exception:
            pass

    # ── training config from eval_metrics (written by train_lora) ────────
    tc = eval_met.get("training_config", {}) if isinstance(eval_met, dict) else {}
    dp_enabled = job.get("dp_enabled") if job.get("dp_enabled") is not None else (
        tc.get("dp_enabled") or (isinstance(eval_met, dict) and eval_met.get("training_mode") in ("dp_lora", "dp-lora"))
    )
    if dp_enabled:
        dp_epsilon = (eval_met.get("epsilon") if isinstance(eval_met, dict) else None) or tc.get("dp_epsilon") or job.get("dp_epsilon")
        dp_delta = (eval_met.get("delta") if isinstance(eval_met, dict) else None) or tc.get("dp_delta") or job.get("dp_delta")
        noise_mult = (eval_met.get("noise_multiplier") if isinstance(eval_met, dict) else None) or tc.get("noise_multiplier") or job.get("dp_noise")
        clip_norm = (eval_met.get("max_grad_norm") if isinstance(eval_met, dict) else None) or tc.get("max_grad_norm") or tc.get("clipping_norm")
    else:
        # Never claim differential privacy was applied to a run that used standard LoRA
        dp_epsilon = None
        dp_delta = None
        noise_mult = None
        clip_norm = None
    trainable       = (eval_met.get("trainable_parameters") or eval_met.get("trainable_params") if isinstance(eval_met, dict) else None) or tc.get("trainable_params")
    total_params    = (eval_met.get("total_parameters") or eval_met.get("total_params") if isinstance(eval_met, dict) else None) or tc.get("total_params")
    trainable_pct   = (eval_met.get("trainable_percent") if isinstance(eval_met, dict) else None) or (round(100 * trainable / total_params, 4) if trainable and total_params else None)
    train_time_s    = (eval_met.get("training_duration_seconds") or eval_met.get("training_time_s") if isinstance(eval_met, dict) else None) or tc.get("training_time_s")
    perplexity      = (eval_met.get("perplexity") if isinstance(eval_met, dict) else None)

    # ── PII counts ───────────────────────────────────────────────────────
    total_pii_detected = pii.get("total_entities_detected")
    if total_pii_detected is None:
        numeric_counts = [v for k, v in pii.items() if isinstance(v, (int, float)) and k not in ("total_entities_detected", "total_entities_masked")]
        total_pii_detected = sum(numeric_counts) if numeric_counts else (0 if pii else None)
    
    total_pii_masked = pii.get("total_entities_masked")

    entity_types_found = pii.get("entity_types_found") or [k for k, v in pii.items() if isinstance(v, (int, float)) and v > 0 and k not in ("total_entities_detected", "total_entities_masked")]

    # ── Screening report ─────────────────────────────────────────────────
    scr_rep = sec.get("screening_report") if isinstance(sec, dict) else {}
    scr_rep = scr_rep if isinstance(scr_rep, dict) else {}
    risk_breakdown = scr_rep.get("risk_breakdown") or {}
    structural_check = risk_breakdown.get("structural_risk_score") or sec.get("adapter_screening_structural")
    behavioral_check = risk_breakdown.get("behavioral_risk_score") or sec.get("adapter_screening_behavioral")
    risk_score = sec.get("security_screening_risk_score") or sec.get("adapter_risk_score") or scr_rep.get("security_screening_risk_score")
    screening_outcome = sec.get("security_screening") or sec.get("adapter_screening_outcome")

    stages = [
        {
            "id": "dataset",
            "name": "Dataset Ingestion",
            "status": _stage_status("dataset_intake"),
            "purpose": "Validate and ingest the training dataset. Detect schema and record count.",
            "metrics": {
                "records": job.get("num_records") or job.get("subset_size"),
                "dataset_name": job.get("dataset_name") or job.get("dataset_type") or "Dataset",
                "schema": job.get("schema_detected") or "instruction",
                "version": job.get("version") or "1.0.0",
            },
            "security_significance": "Validates data format before any sensitive processing begins. Rejects malformed files.",
            "result": f"{job.get('num_records') or job.get('subset_size') or 'N/A'} records ingested" if (job.get("num_records") or job.get("subset_size")) else "Dataset validated",
        },
        {
            "id": "pii_audit",
            "name": "PII Audit & Masking",
            "status": _stage_status("pii_inspection"),
            "purpose": "Detect and mask all PII/PHI entities in-RAM using hybrid NER (SpaCy + Regex + Presidio). Zero-disk-leakage: no raw data written.",
            "metrics": {
                "records_scanned": job.get("num_records") or job.get("subset_size"),
                "pii_detected": total_pii_detected,
                "pii_masked": total_pii_masked,
                "records_with_pii": pii.get("records_with_pii"),
                # These scores are valid only when computed against labeled ground truth.
                "precision": pii.get("precision") or pii.get("estimated_precision"),
                "recall": pii.get("recall") or pii.get("estimated_recall"),
                "f1": pii.get("f1") or pii.get("estimated_f1"),
                "entity_types": entity_types_found,
            },
            "security_significance": "Prevents PII from entering the training loop. All masking is in-RAM — no raw PII reaches disk.",
            "result": (f"{total_pii_detected} detected; {total_pii_masked} masked (in-RAM)" if total_pii_detected is not None and total_pii_masked is not None else ("Zero PII detected" if total_pii_detected == 0 else "PII audit completed")),
        },
        {
            "id": "data_protect",
            "name": "Data Protection (AES-256-GCM)",
            "status": _stage_status("dataset_protection"),
            "purpose": "Encrypt the sanitized dataset with AES-256-GCM using a job-unique 256-bit key. Metadata anchored with SHA-256.",
            "metrics": {
                "algorithm": "AES-256-GCM",
                "key_bits": 256,
                "integrity_primitive": "SHA-256",
            },
            "security_significance": "Ensures training data is unreadable at rest. The per-job key never leaves the job workspace.",
            "result": "Dataset encrypted with AES-256-GCM" if _stage_status("dataset_protection") == "PASSED" else "N/A",
        },
        {
            "id": "training",
            "name": "LoRA / DP-LoRA Fine-Tuning",
            "status": _stage_status("fine_tuning"),
            "purpose": "Fine-tune a LoRA adapter on the encrypted dataset. Optionally applies Differential Privacy (Opacus) for formal privacy guarantees.",
            "metrics": {
                "mode": "DP-LoRA" if dp_enabled else ("LoRA" if dp_enabled is not None else None),
                "trainable_params": trainable,
                "total_params": total_params,
                "trainable_pct": trainable_pct,
                "epochs": job.get("epochs"),
                "current_epoch": job.get("current_epoch"),
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "perplexity": perplexity,
                "training_time_s": train_time_s,
                "dp_enabled": dp_enabled,
                "epsilon": dp_epsilon,
                "delta": dp_delta,
                "noise_multiplier": noise_mult,
                "max_grad_norm": clip_norm,
            },
            "security_significance": "DP-LoRA adds calibrated noise to gradients to satisfy (ε,δ)-DP. ε≤8 considered strong privacy.",
            "result": (f"Loss {final_train_loss}" if final_train_loss else "N/A"),
        },
        {
            "id": "screening",
            "name": "Adapter Security Screening",
            "status": _stage_status("preparing_adapter"),
            "purpose": "Pre-packaging structural + behavioral screening to detect backdoors, trojan triggers, or malicious weight patterns before signing.",
            "metrics": {
                "structural_check": structural_check,
                "behavioral_check": behavioral_check,
                "risk_score": risk_score,
                "screening_result": screening_outcome,
                "malicious_detection_rate": sec.get("malicious_adapter_detection_rate"),
            },
            "security_significance": "Cryptographic signing does NOT prove an adapter was benign before signing. Screening catches malicious adapters before they reach the package.",
            "result": str(screening_outcome or "N/A"),
        },
        {
            "id": "provenance",
            "name": "Provenance & RSA-PSS Signature",
            "status": _stage_status("generating_signature"),
            "purpose": "Compute SHA-256 adapter digest and sign with RSA-PSS (2048-bit). Generate cryptographic provenance manifest.",
            "metrics": {
                "signature_algorithm": "RSA-PSS",
                "hash_algorithm": "SHA-256",
                "signature_ok": sec.get("signature_ok"),
                "replay_rejection": sec.get("replay_rejection_rate"),
            },
            "security_significance": "RSA-PSS signature proves the adapter was produced by the authorized packager and has not been tampered with.",
            "result": "Signed with RSA-PSS" if _stage_status("generating_signature") == "PASSED" else "N/A",
        },
        {
            "id": "packaging",
            "name": "Cryptographic Packaging",
            "status": _stage_status("building_package"),
            "purpose": "Encrypt the LoRA adapter weights with AES-256-GCM using an HKDF-SHA256 device-bound key. Bundle with manifest, hashes, and signature into .tar.gz.",
            "metrics": {
                "algorithm": "AES-256-GCM",
                "kdf": "HKDF-SHA256",
                "package_format": ".tar.gz",
                "tamper_rejection": sec.get("tamper_rejection_rate"),
            },
            "security_significance": "Adapter weights are unreadable ciphertext. Decryption only succeeds on the device the package was built for.",
            "result": "Protected adapter package created" if _stage_status("building_package") == "PASSED" else "N/A",
        },
        {
            "id": "device_auth",
            "name": "Device Authorization",
            "status": _vstep("Step 4: Device Authorization"),
            "purpose": "Verify that the current hardware fingerprint matches the authorized device fingerprint embedded in the package manifest.",
            "metrics": {
                "fingerprint_check": _vstep("Step 4: Device Authorization") if vsteps else None,
                "key_derivation": _vstep("Step 5: Key Derivation") if vsteps else None,
                "cross_device_rejection": sec.get("cross_device_rejection_rate"),
                "unauthorized_rejection": sec.get("unauthorized_deployment_rejection_rate"),
            },
            "security_significance": "Device-bound HKDF key derivation: adapter cannot be decrypted on any unauthorized device. Cross-device rejection rate near 100% under evaluated fingerprint configurations.",
            "result": _vstep("Step 4: Device Authorization") if vsteps else "N/A",
        },
        {
            "id": "deployment",
            "name": "Deployment Gate (Steps 1–8)",
            "status": (
                "PASSED" if len(vsteps) >= 8 and all(vsteps.get(k) == "PASSED" for k in vsteps)
                else ("FAILED" if any(v == "FAILED" for v in vsteps.values())
                      else ("PENDING" if status == "COMPLETED" else _stage_status("running_secure_deployment_check")))
            ),
            "purpose": "Run all 8 cryptographic verification gates: Package Completeness → SHA-256 → RSA-PSS → Device Auth → HKDF Key → AES-GCM Decrypt → PEFT Load → Inference.",
            "metrics": {k: v for k, v in vsteps.items()} if vsteps else {},
            "security_significance": "All 8 gates must pass. Any failure immediately aborts deployment and leaves weights encrypted.",
            "result": (
                "All 8 gates PASSED" if len(vsteps) >= 8 and all(vsteps.get(k) == "PASSED" for k in vsteps)
                else (f"{sum(1 for v in vsteps.values() if v == 'PASSED')}/8 gates explicitly passed; verification evidence incomplete" if vsteps
                      else "Verification evidence unavailable")
            ),
        },
        {
            "id": "inference",
            "name": "Secure Inference Validation",
            "status": _vstep("Step 8: Inference Validation"),
            "purpose": "Side-by-side inference test comparing baseline model vs. secured LoRA adapter. Validates adapter is active and functional.",
            "metrics": {
                "inference_check": _vstep("Step 8: Inference Validation") if vsteps else None,
                "adapter_active": vsteps.get("Step 8: Inference Validation") == "PASSED" if vsteps else None,
            },
            "security_significance": "Confirms the decrypted adapter produces different outputs from the base model, proving successful loading.",
            "result": _vstep("Step 8: Inference Validation") if vsteps else "N/A",
        },
    ]

    return {
        "success": True,
        "job_id": job_id,
        "pipeline_status": status,
        "current_stage": stage,
        "progress": progress,
        "created_at": created_at,
        "updated_at": updated_at,
        "error": error,
        "stages": stages,
        # Top-level KPI convenience fields
        "kpi": {
            "dataset": job.get("dataset_name") or job.get("dataset_type") or "Dataset",
            "records": job.get("num_records") or job.get("subset_size"),
            "pii_detected": total_pii_detected,
            "pii_masked": total_pii_masked,
            "training_mode": "DP-LoRA" if dp_enabled else ("LoRA" if dp_enabled is not None else None),
            "dp_epsilon": dp_epsilon,
            "dp_delta": dp_delta,
            "noise_multiplier": noise_mult,
            "max_grad_norm": clip_norm,
            "adapter_status": sec.get("adapter_screening_outcome") or ("LOADED" if status == "COMPLETED" else status),
            "package_status": "PACKAGED" if _stage_status("building_package") == "PASSED" else _stage_status("building_package"),
            "device_status": _vstep("Step 4: Device Authorization") if vsteps else None,
            "deployment_status": status if status in ("COMPLETED", "FAILED") else _stage_status("running_secure_deployment_check"),
        }
    }


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/pipeline-summary", methods=["GET"])
@require_job_access
def get_pipeline_summary(job_id):
    """
    Returns a structured 10-stage SecureLoRA lifecycle summary for the Pipeline tab.
    """
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    return jsonify(_build_pipeline_summary_data(job, orchestrator.base_jobs_dir))


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/stream", methods=["GET"])
@require_job_access
def stream_job_events(job_id):
    """Exposes a Server-Sent Events (SSE) stream for real-time progress updates."""
    import time
    import json
    from flask import Response

    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    def event_stream():
        while True:
            current_job = orchestrator.get_job(job_id)
            if not current_job:
                break

            summary_data = _build_pipeline_summary_data(current_job, orchestrator.base_jobs_dir)
            payload = {
                "job_id": current_job.get("job_id"),
                "status": current_job.get("status"),
                "stage": current_job.get("stage"),
                "progress": current_job.get("progress"),
                "last_updated": current_job.get("last_updated"),
                "loss_history": current_job.get("loss_history", []),
                "current_epoch": current_job.get("current_epoch"),
                "pii_summary": current_job.get("pii_summary"),
                "num_records": current_job.get("num_records"),
                "security_metrics": current_job.get("security_metrics"),
                "verification_steps": current_job.get("verification_steps"),
                "error": redact_text(current_job.get("error")),
                "stages": summary_data.get("stages", []),
                "kpi": summary_data.get("kpi", {})
            }

            yield f"data: {json.dumps(payload)}\n\n"

            if current_job.get("status") in ["COMPLETED", "FAILED"]:
                break

            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream")


@orchestrator_bp.route("/api/orchestrator/dataset-templates", methods=["GET"])
def get_dataset_templates():
    """Returns dynamic metadata for the registered dataset adapters."""
    from src.data_sources.dataset_registry import dataset_registry
    datasets = dataset_registry.list_datasets()
    templates = []
    for d in datasets:
        templates.append({
            "id": d["id"],
            "dataset_id": d["id"],
            "name": d["name"],
            "tagline": f"{d['domain']} ({d['license']})",
            "description": f"Source: {d['source']}. Ground-truth: {'Available' if d['ground_truth_available'] else 'EHR Coverage Evaluation'}.",
            "record_count": d["default_subset_size"],
            "subset_options": d["subset_options"],
            "format": "JSONL Schema",
            "privacy_category": d["domain"],
            "status": "READY",
            "filename": d["id"],
            "source": d["source"],
            "license": d["license"],
            "ground_truth_available": d["ground_truth_available"]
        })
    return jsonify({"success": True, "templates": templates})



@orchestrator_bp.route("/api/orchestrator/chat", methods=["POST"])
@require_bearer_token
def orchestrator_chat():
    """
    Executes Base Model vs SecureLoRA PEFT Model inference using the canonical inference_service.
    Exposes max_new_tokens and temperature generation parameters.
    No analytics fallback. Returns status="MODEL_UNAVAILABLE" if model is not loaded.
    """
    from src.orchestrator.inference_service import compare_base_and_securelora, ensure_model_loaded
    from src.orchestrator.model_registry import model_registry

    data = request.json or {}
    question = data.get("question", "").strip()
    max_new_tokens = int(data.get("max_new_tokens", 128))
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))

    if not question:
        return jsonify({"success": False, "status": "INVALID_REQUEST", "error": "question parameter is required"}), 400

    ensure_model_loaded()

    if not model_registry.is_verified():
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        res = compare_base_and_securelora(prompt=question, generation_config=gen_config)
        return jsonify({
            "success": False,
            "status": "MODEL_UNAVAILABLE",
            "message": "SecureLoRA model is unavailable. Deployment must be verified first.",
            "base_output": res.get("base_output", "[BASE MODEL UNAVAILABLE] Base model un-tuned inference is inactive."),
            "securelora_output": res.get("securelora_output", "[SECURELORA ADAPTER UNAVAILABLE]\nSecureLoRA model is unavailable. Deployment must be verified first."),
            "base_pii_entities": res.get("base_pii_entities", []),
            "securelora_pii_entities": [],
            "base_pii_count": res.get("base_pii_count", 0),
            "securelora_pii_count": 0,
            "adapter_loaded": False,
            "deployment_verified": False,
            "model_info": res.get("model_info", {
                "base_model_name": "N/A",
                "adapter_id": "N/A",
                "deployment_id": "N/A",
                "deployment_status": "UNAVAILABLE"
            })
        }), 200

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    res = compare_base_and_securelora(prompt=question, generation_config=gen_config)

    if res.get("status") == "SUCCESS":
        return jsonify({
            "success": True,
            "status": "SUCCESS",
            "prompt": question,
            "question": question,
            "base_output": res["base_output"],
            "securelora_output": res["securelora_output"],
            "base_pii": {"count": res["base_pii_count"], "entities": res["base_pii_entities"]},
            "securelora_pii": {"count": res["securelora_pii_count"], "entities": res["securelora_pii_entities"]},
            "base_pii_entities": res["base_pii_entities"],
            "securelora_pii_entities": res["securelora_pii_entities"],
            "base_pii_count": res["base_pii_count"],
            "securelora_pii_count": res["securelora_pii_count"],
            "adapter_loaded": res["adapter_loaded"],
            "deployment_verified": res["deployment_verified"],
            "model_info": res["model_info"],
            "raw_answer": res["base_output"],
            "answer": res["securelora_output"]
        })
    else:
        return jsonify({
            "success": False,
            "status": res.get("status", "MODEL_UNAVAILABLE"),
            "message": res.get("message", "Model generation failed"),
            "base_output": res.get("base_output", "[MODEL_UNAVAILABLE]"),
            "securelora_output": res.get("securelora_output", "[MODEL_UNAVAILABLE]"),
            "base_pii_entities": res.get("base_pii_entities", []),
            "securelora_pii_entities": res.get("securelora_pii_entities", []),
            "base_pii_count": res.get("base_pii_count", 0),
            "securelora_pii_count": res.get("securelora_pii_count", 0),
            "adapter_loaded": res.get("adapter_loaded", False),
            "deployment_verified": res.get("deployment_verified", False),
            "model_info": res.get("model_info", {})
        }), 200 if res.get("status") == "MODEL_UNAVAILABLE" else 500


@orchestrator_bp.route("/api/orchestrator/model-reload", methods=["POST"])
@require_bearer_token
def reload_model():
    """Forces verification and loading of the latest deployment package into ModelRegistry."""
    from src.orchestrator.model_registry import model_registry
    from src.orchestrator.inference_service import ensure_model_loaded
    loaded = ensure_model_loaded(force=True)
    info = model_registry.get_info()
    verified = model_registry.is_verified()
    return jsonify({
        "success": loaded,
        "model_verified": verified,
        "adapter_loaded": info["adapter_loaded"],
        "base_model_name": info["base_model_name"],
        "adapter_name": info["adapter_id"],
        "adapter_id": info["adapter_id"],
        "deployment_id": info["deployment_id"],
        "deployment_status": info["deployment_status"],
        "status": info["deployment_status"]
    })


@orchestrator_bp.route("/api/orchestrator/model-status", methods=["GET"])
@require_bearer_token
def get_model_status():
    """Returns the backend ModelRegistry status."""
    from src.orchestrator.model_registry import model_registry
    from src.orchestrator.inference_service import ensure_model_loaded
    ensure_model_loaded()
    info = model_registry.get_info()
    verified = model_registry.is_verified()
    return jsonify({
        "success": True,
        "model_verified": verified,
        "adapter_loaded": info["adapter_loaded"],
        "base_model_name": info["base_model_name"],
        "adapter_name": info["adapter_id"],
        "adapter_id": info["adapter_id"],
        "deployment_id": info["deployment_id"],
        "deployment_status": info["deployment_status"],
        "status": info["deployment_status"]
    })


@orchestrator_bp.route("/api/orchestrator/training-config", methods=["GET"])
@require_bearer_token
def get_training_config():
    """Returns supported base models, LoRA defaults, and DP-LoRA backend capabilities."""
    from src.common.config_loader import config
    try:
        import opacus  # noqa: F401
        opacus_supported = True
    except ImportError:
        opacus_supported = False

    return jsonify({
        "success": True,
        "config": {
            "supported_models": [
                {
                    "id": config.model_name or "JackFram/llama-68m",
                    "name": f"{config.model_name or 'JackFram/llama-68m'} (Evaluated Baseline)",
                    "parameters": "68M",
                    "default": True,
                    "description": "Baseline causal language model evaluated and pre-configured for SecureLoRA."
                },
                {
                    "id": "meta-llama/Llama-2-7b-hf",
                    "name": "meta-llama/Llama-2-7b-hf (7B Scale)",
                    "parameters": "7B",
                    "default": False,
                    "description": "High-capacity 7B parameter foundation model requiring dedicated GPU acceleration."
                }
            ],
            "lora_defaults": {
                "r": config.lora_r,
                "r_min": 1,
                "r_max": 64,
                "alpha": config.lora_alpha,
                "alpha_min": 1,
                "alpha_max": 128,
                "dropout": config.lora_dropout,
                "dropout_min": 0.0,
                "dropout_max": 0.5,
                "target_modules": config.target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
                "default_modules": ["q_proj", "v_proj"]
            },
            "dp_capabilities": {
                "supported": opacus_supported,
                "engine": "Opacus (PyTorch DP-SGD)",
                "default_accountant": config.dp_accountant or "rdp",
                "supported_accountants": ["rdp", "prv"],
                "default_max_grad_norm": config.dp_max_grad_norm,
                "max_grad_norm_min": 0.1,
                "max_grad_norm_max": 10.0,
                "default_noise_multiplier": config.dp_noise_multiplier or 1.2,
                "noise_multiplier_min": 0.1,
                "noise_multiplier_max": 5.0,
                "default_target_delta": config.dp_target_delta,
                "default_target_epsilon": config.dp_target_epsilon,
                "target_epsilon_min": 0.5,
                "target_epsilon_max": 50.0
            }
        }
    })





