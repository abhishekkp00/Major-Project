import os
import logging
import tempfile
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from .service import orchestrator
from src.orchestrator.dataset_processor import validate_dataset_file
from src.common.exceptions import DatasetValidationError

logger = logging.getLogger("secure_lora.orchestrator.routes")
orchestrator_bp = Blueprint("orchestrator", __name__)


@orchestrator_bp.route("/api/orchestrator/validate", methods=["POST"])
def pre_validate_dataset():
    """Parses and runs PII inspection on an uploaded dataset file before job creation."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()

    # Save to a temporary file
    temp_fd, temp_str = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)
    temp_path = Path(temp_str)

    try:
        file.save(temp_path)
        # Validate and inspect
        _, metadata = validate_dataset_file(temp_path)
        return jsonify({"success": True, "metadata": metadata})
    except DatasetValidationError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as e:
        logger.exception("Pre-validation failure:")
        return jsonify({"success": False, "error": f"Failed to validate dataset: {str(e)}"}), 500
    finally:
        if temp_path.exists():
            temp_path.unlink()


@orchestrator_bp.route("/api/orchestrator/jobs", methods=["POST"])
def create_job():
    """Creates a new job with specified configuration."""
    data = request.json or {}
    dataset_name = data.get("dataset_name", "")
    if not dataset_name:
        return jsonify({"success": False, "error": "dataset_name is required"}), 400

    version = data.get("version", "1.0.0")
    epochs = int(data.get("epochs", 1))
    salt = data.get("salt")

    try:
        job_id = orchestrator.create_job(
            dataset_name=dataset_name,
            version=version,
            epochs=epochs,
            salt=salt
        )
        return jsonify({"success": True, "job_id": job_id})
    except Exception as e:
        logger.exception("Failed to create job:")
        return jsonify({"success": False, "error": str(e)}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/upload", methods=["POST"])
def upload_file(job_id):
    """Uploads a raw dataset file to a created job workspace."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    try:
        content = file.read()
        orchestrator.add_dataset_file(job_id, filename, content)
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        logger.exception("Failed to upload dataset file:")
        return jsonify({"success": False, "error": str(e)}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/start", methods=["POST"])
def start_job(job_id):
    """Starts the full end-to-end secure pipeline execution."""
    try:
        orchestrator.start_job(job_id)
        return jsonify({"success": True})
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as e:
        logger.exception("Failed to start job:")
        return jsonify({"success": False, "error": str(e)}), 500


@orchestrator_bp.route("/api/orchestrator/jobs", methods=["GET"])
def get_jobs():
    """Lists all created orchestration jobs."""
    try:
        jobs = orchestrator.get_all_jobs()
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        logger.exception("Failed to list jobs:")
        return jsonify({"success": False, "error": str(e)}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Polls detailed status for a specific job."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, "job": job})


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/logs", methods=["GET"])
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
        return jsonify({"success": True, "logs": "\n".join(lines)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/metrics", methods=["GET"])
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
def list_job_artifacts(job_id):
    """Lists safe downloadable package artifacts generated for the job."""
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    protected_dir = orchestrator.base_jobs_dir / job_id / "protected"
    if not protected_dir.exists():
        return jsonify({"success": True, "artifacts": []})

    artifacts = []
    # Exclude secret keys (e.g. .pem, .key)
    safe_extensions = [".enc", ".hash", ".sig", ".json", ".gz", ".pem"]
    for path in protected_dir.iterdir():
        if path.is_file() and path.suffix in safe_extensions:
            # Never expose private key
            if "private" in path.name or (path.name.endswith(".pem") and path.name != "public.pem"):
                continue
            artifacts.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "download_url": f"/api/orchestrator/jobs/{job_id}/download/{path.name}"
            })

    return jsonify({"success": True, "artifacts": artifacts})


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/download/<filename>", methods=["GET"])
def download_job_artifact(job_id, filename):
    """Serves a specific safe package artifact for download."""
    from flask import send_from_directory
    job = orchestrator.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    protected_dir = orchestrator.base_jobs_dir / job_id / "protected"
    safe_extensions = [".enc", ".hash", ".sig", ".json", ".gz", ".pem"]
    target_path = protected_dir / filename

    if not target_path.exists() or target_path.suffix not in safe_extensions:
        return jsonify({"success": False, "error": "Access denied or file not found"}), 403

    if "private" in filename or (filename.endswith(".pem") and filename != "public.pem"):
        return jsonify({"success": False, "error": "Access denied"}), 403

    return send_from_directory(str(protected_dir), filename, as_attachment=True)


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/report", methods=["GET"])
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
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@orchestrator_bp.route("/api/orchestrator/jobs/<job_id>/stream", methods=["GET"])
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
                "error": current_job.get("error")
            }

            yield f"data: {json.dumps(payload)}\n\n"

            if current_job.get("status") in ["COMPLETED", "FAILED"]:
                break

            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream")
