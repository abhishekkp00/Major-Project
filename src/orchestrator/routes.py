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
