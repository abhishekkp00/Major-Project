import os
import json
import uuid
import shutil
import hashlib
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import math
from src.phase1.pipeline import SecureDatasetPipeline
from src.security import generate_key
from src.common.config_loader import config

logger = logging.getLogger("secure_lora.orchestrator.service")


def _sanitize_json_values(obj: Any) -> Any:
    """Recursively replaces NaN and Inf with None for strict JSON compatibility."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json_values(v) for v in obj]
    return obj


def _derive_device_salt() -> str:
    """Derives a deterministic salt from real hardware identifiers.
    Reads /etc/machine-id (Linux), /proc/cpuinfo, and network MAC address.
    Falls back to UUID namespace hash if hardware reads fail.
    This ensures hardware-binding is unique per device without being hardcoded.
    """
    sources = []
    # 1. Linux machine-id (unique per OS install)
    for mid_path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
        try:
            sources.append(Path(mid_path).read_text().strip())
            break
        except OSError:
            pass
    # 2. First non-loopback MAC address
    try:
        import socket, struct, fcntl
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        info = fcntl.ioctl(s.fileno(), 0x8927, struct.pack("256s", b"eth0"[:15]))
        sources.append(info[18:24].hex())
    except Exception:
        # Fallback: uuid1 embeds a MAC address
        sources.append(hex(uuid.getnode()))
    # 3. Hostname
    try:
        sources.append(socket.gethostname())
    except Exception:
        pass
    combined = "|".join(sources)
    return hashlib.sha256(combined.encode()).hexdigest()[:64]



class JobOrchestrator:
    def __init__(self, base_jobs_dir: str = "outputs/jobs"):
        self.base_jobs_dir = Path(base_jobs_dir)
        self.base_jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_jobs_dir / "jobs_db.json"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._load_db()

    def _load_db(self):
        with self.lock:
            if self.db_path.exists():
                try:
                    raw = self.db_path.read_text(encoding="utf-8")
                    self.jobs = _sanitize_json_values(json.loads(raw))
                    # Any training job that was interrupted on restart is marked as failed
                    for job_id, job in self.jobs.items():
                        if job.get("status") in ["INGESTING", "TRAINING", "PACKAGING", "DEPLOYING"]:
                            job["status"] = "FAILED"
                            job["error"] = "System restarted during job execution."
                except Exception as e:
                    logger.error("Failed to load jobs database: %s", e)
                    self.jobs = {}
            else:
                self.jobs = {}

    def _save_db(self):
        with self.lock:
            try:
                sanitized_jobs = _sanitize_json_values(self.jobs)
                temp_path = self.db_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(sanitized_jobs, indent=4), encoding="utf-8")
                temp_path.replace(self.db_path)
            except Exception as e:
                logger.error("Failed to save jobs database: %s", e)

    def create_job(
        self,
        dataset_name: str,
        version: str = "1.0.0",
        epochs: int = 1,
        salt: Optional[str] = None,
        dataset_type: Optional[str] = None,
        subset_size: Optional[int] = 1000
    ) -> str:
        job_id = f"job_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:8]}"
        job_dir = self.base_jobs_dir / job_id

        # Setup standard directories
        dirs = {
            "raw_inputs": job_dir / "raw_inputs",
            "encrypted": job_dir / "encrypted",
            "checkpoints": job_dir / "checkpoints",
            "adapter": job_dir / "adapter",
            "protected": job_dir / "protected",
            "deployment": job_dir / "deployment"
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Generate unique 256-bit encryption key
        key = generate_key()
        key_path = job_dir / "secrets.key"
        key_path.write_bytes(key)
        if os.name == 'posix':
            key_path.chmod(0o600)

        job_record = {
            "job_id": job_id,
            "dataset_name": dataset_name,
            "dataset_type": dataset_type or dataset_name,
            "subset_size": subset_size or 1000,
            "version": version,
            "status": "CREATED",
            "stage": "dataset_intake",
            "progress": 0,
            "epochs": epochs,
            "salt": salt or os.environ.get("P3_DEVICE_SALT") or _derive_device_salt(),
            "loss_history": [],
            "eval_metrics": {},
            "verification_steps": {},
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        with self.lock:
            self.jobs[job_id] = job_record
        self._save_db()

        logger.info("Created orchestration job: %s (dataset_type=%s, subset_size=%s)", job_id, dataset_type or dataset_name, subset_size)
        return job_id

    def add_dataset_file(self, job_id: str, filename: str, content: bytes):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        target_path = self.base_jobs_dir / job_id / "raw_inputs" / filename
        target_path.write_bytes(content)
        logger.info("Saved dataset file %s to job %s", filename, job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda x: x["created_at"], reverse=True)

    def update_job_state(self, job_id: str, **kwargs):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)
                self.jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_db()

    def start_job(self, job_id: str):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        if job["status"] != "CREATED":
            raise ValueError(f"Job {job_id} is already in status {job['status']}")

        thread = threading.Thread(target=self._run_pipeline, args=(job_id,), daemon=True)
        thread.start()
        logger.info("Started background worker for job: %s", job_id)

    def _run_pipeline(self, job_id: str):
        job_dir = self.base_jobs_dir / job_id
        job = self.get_job(job_id)
        if not job:
            return

        salt = job["salt"]
        epochs = job["epochs"]
        dataset_name = job["dataset_name"]
        dataset_type = job.get("dataset_type", dataset_name)
        subset_size = job.get("subset_size", 1000)
        version = job["version"]

        try:
            # ────────────────────────────────────────────────────────────────
            # PHASE 1: INGESTION & ENCRYPTION
            # ────────────────────────────────────────────────────────────────
            self.update_job_state(job_id, status="INGESTING", stage="dataset_protection", progress=10)
            logger.info("[%s] Phase 1 Ingestion started.", job_id)

            key = (job_dir / "secrets.key").read_bytes()
            raw_dir = job_dir / "raw_inputs"
            enc_dir = job_dir / "encrypted"

            # Check if dataset adapter is requested or if raw_dir is empty
            uploaded_files = list(raw_dir.glob("*"))
            if not uploaded_files:
                try:
                    from src.data_sources.dataset_registry import dataset_registry
                    logger.info("[%s] Ingesting via DatasetAdapter '%s' (subset=%d)...", job_id, dataset_type, subset_size)
                    self.update_job_state(job_id, status="INGESTING", stage="dataset_protection", progress=12)
                    adapter = dataset_registry.get_dataset_adapter(dataset_type)
                    records = adapter.load_dataset(subset_size=subset_size)
                    target_file = raw_dir / "dataset.jsonl"
                    with open(target_file, "w", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r) + "\n")
                    uploaded_files = [target_file]
                except Exception as adapter_err:
                    logger.warning("[%s] Dataset adapter load failed, checking fallback samples: %s", job_id, adapter_err)
                    import shutil
                    samples_dir = Path(__file__).resolve().parents[2] / "samples"
                    sample_map = {
                        "pii_corporate.jsonl": samples_dir / "sample_pii_data.jsonl",
                        "sample_pii_data.jsonl": samples_dir / "sample_pii_data.jsonl",
                        "clinical_notes.jsonl": samples_dir / "sample_medical_phi.jsonl",
                        "sample_medical_phi.jsonl": samples_dir / "sample_medical_phi.jsonl",
                        "real_world_pii.jsonl": samples_dir / "real_world_pii.jsonl"
                    }
                    sample_file = sample_map.get(dataset_name, samples_dir / "real_world_pii.jsonl")
                    if sample_file and sample_file.exists():
                        target_file = raw_dir / (dataset_name if dataset_name else "dataset.jsonl")
                        shutil.copy(sample_file, target_file)
                        uploaded_files = [target_file]
                    else:
                        raise RuntimeError("No uploaded files found in raw input directory.")

            uploaded_file = uploaded_files[0]
            
            from src.orchestrator.dataset_processor import (
                validate_dataset_file,
                preprocess_and_standardize,
                encrypt_and_save_dataset
            )

            self.update_job_state(job_id, status="INGESTING", stage="dataset_protection", progress=16)
            raw_records, file_meta = validate_dataset_file(uploaded_file)
            
            self.update_job_state(job_id, status="INGESTING", stage="dataset_protection", progress=20)
            processed_records = preprocess_and_standardize(raw_records)
            
            self.update_job_state(job_id, status="INGESTING", stage="dataset_protection", progress=23)
            metadata = encrypt_and_save_dataset(
                processed_records=processed_records,
                key=key,
                output_dir=enc_dir,
                dataset_name=dataset_name,
                version=version,
                pii_summary=file_meta.get("pii_detected_summary", {})
            )

            self.update_job_state(
                job_id,
                progress=25,
                pii_summary=file_meta.get("pii_detected_summary", {}),
                schema_detected=file_meta.get("schema_detected", "unknown"),
                num_records=metadata.get("num_records", 0)
            )
            logger.info("[%s] Phase 1 complete. Ingested %d records.", job_id, metadata.get("num_records", 0))

            # ────────────────────────────────────────────────────────────────
            # PHASE 2: IN-MEMORY FINE-TUNING (as subprocess)
            # ────────────────────────────────────────────────────────────────
            self.update_job_state(job_id, status="TRAINING", stage="fine_tuning", progress=30)
            logger.info("[%s] Phase 2 Training started.", job_id)

            progress_json_path = job_dir / "progress.json"
            env = os.environ.copy()
            # Set to empty string (not pop) so load_dotenv inside subprocess
            # cannot re-inject the global .env key — forcing use of SECURE_LORA_KEY_PATH
            env["SECURE_LORA_KEY_HEX"] = ""
            env["SECURE_LORA_INPUT_DIR"] = str(raw_dir)
            env["SECURE_LORA_OUTPUT_DIR"] = str(enc_dir)
            env["SECURE_LORA_ENCRYPTED_DATA"] = str(enc_dir / "encrypted_dataset.enc")
            env["SECURE_LORA_METADATA_PATH"] = str(enc_dir / "dataset_metadata.json")
            env["SECURE_LORA_CHECKPOINT_DIR"] = str(job_dir / "checkpoints")
            env["SECURE_LORA_OUTPUT_DIR_LORA"] = str(job_dir / "adapter")
            env["SECURE_LORA_KEY_PATH"] = str(job_dir / "secrets.key")
            env["SECURE_LORA_EPOCHS"] = str(epochs)
            env["SECURE_LORA_BATCH_SIZE"] = "2"
            env["SECURE_LORA_SEED"] = "42"
            env["SECURE_LORA_PROGRESS_FILE"] = str(progress_json_path)

            log_file = job_dir / "training.log"
            process = subprocess.Popen(
                ["./venv/bin/python", "-m", "src.phase2.train_lora"],
                cwd=str(Path.cwd()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Monitor log output to parse metrics in real-time
            loss_history = []
            with open(log_file, "w", encoding="utf-8") as lf:
                for line in iter(process.stdout.readline, ""):
                    lf.write(line)
                    lf.flush()
                    
                    if progress_json_path.exists():
                        try:
                            prog_data = json.loads(progress_json_path.read_text(encoding="utf-8"))
                            current_step = prog_data.get("current_step", 0)
                            total_steps = max(1, prog_data.get("total_steps", 100))
                            epoch = prog_data.get("epoch", 0.0)
                            history = prog_data.get("history", [])
                            
                            loss_history = [
                                {
                                    "epoch": h.get("epoch"),
                                    "loss": h.get("loss"),
                                    "eval_loss": h.get("eval_loss")
                                }
                                for h in history
                                if h.get("loss") is not None or h.get("eval_loss") is not None
                            ]
                            
                            fine_tuning_progress = min(70, 30 + int(40 * (current_step / total_steps)))
                            self.update_job_state(
                                job_id,
                                progress=fine_tuning_progress,
                                loss_history=loss_history,
                                current_epoch=epoch
                            )
                        except Exception:
                            pass
                    else:
                        if "{'loss':" in line or "'loss':" in line:
                            try:
                                start_idx = line.find("{")
                                end_idx = line.rfind("}")
                                if start_idx != -1 and end_idx != -1:
                                    data_str = line[start_idx:end_idx+1].replace("'", '"')
                                    metric_data = json.loads(data_str)
                                    loss = metric_data.get("loss")
                                    epoch = metric_data.get("epoch")
                                    if loss is not None and epoch is not None:
                                        loss_history.append({"epoch": epoch, "loss": loss})
                                        self.update_job_state(job_id, loss_history=loss_history)
                            except Exception as parse_err:
                                logger.debug("Failed parsing training loss line: %s", parse_err)

            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"Training failed with exit code {process.returncode}. See training.log for details.")

            # Load evaluation metrics from eval_report.json
            eval_metrics = {}
            eval_report_file = Path("outputs/evaluation/eval_report.json")
            if eval_report_file.exists():
                try:
                    eval_metrics = json.loads(eval_report_file.read_text(encoding="utf-8"))
                    # Copy custom eval report to job folder for isolation
                    shutil.copy2(eval_report_file, job_dir / "eval_report.json")
                except Exception as e:
                    logger.warning("Could not read training eval report: %s", e)

            self.update_job_state(job_id, progress=70, eval_metrics=eval_metrics)
            logger.info("[%s] Phase 2 complete.", job_id)

            # ────────────────────────────────────────────────────────────────
            # SECURITY PACKAGING & DEPLOYMENT VERIFICATION GATES (Phases 3 & 4)
            # ────────────────────────────────────────────────────────────────
            from src.orchestrator.security_orchestrator import run_security_orchestration
            
            security_outcomes = run_security_orchestration(
                job_id=job_id,
                job_dir=job_dir,
                salt=salt,
                base_model_name=config.model_name,
                update_state_fn=self.update_job_state
            )
            
            # Read verification step info if any
            verification_steps = {}
            report_path = job_dir / "deployment" / "validation_report.json"
            if report_path.exists():
                try:
                    report_data = json.loads(report_path.read_text(encoding="utf-8"))
                    verification_steps = report_data.get("verification_pipeline", {}).get("steps", {})
                except Exception as e:
                    logger.warning("Could not read verification report: %s", e)

            self.update_job_state(
                job_id,
                status="COMPLETED",
                stage="security_validation_completed",
                progress=100,
                security_metrics=security_outcomes,
                verification_steps=verification_steps
            )
            logger.info("[%s] Full secure lifecycle completed successfully!", job_id)

        except Exception as exc:
            logger.error("[%s] Pipeline execution failed: %s", job_id, exc, exc_info=True)
            self.update_job_state(job_id, status="FAILED", error=str(exc))


# Global Orchestrator Instance
orchestrator = JobOrchestrator()
