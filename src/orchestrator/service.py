import os
import json
import uuid
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.phase1.pipeline import SecureDatasetPipeline
from src.security import generate_key

logger = logging.getLogger("secure_lora.orchestrator.service")


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
                    self.jobs = json.loads(self.db_path.read_text(encoding="utf-8"))
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
                temp_path = self.db_path.with_suffix(".tmp")
                temp_path.write_text(json.dumps(self.jobs, indent=4), encoding="utf-8")
                temp_path.replace(self.db_path)
            except Exception as e:
                logger.error("Failed to save jobs database: %s", e)

    def create_job(
        self,
        dataset_name: str,
        version: str = "1.0.0",
        epochs: int = 1,
        salt: Optional[str] = None
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
            "version": version,
            "status": "CREATED",
            "stage": "dataset_intake",
            "progress": 0,
            "epochs": epochs,
            "salt": salt or "demo-integration-salt-abc123xyz",
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

        logger.info("Created orchestration job: %s", job_id)
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

            # Find uploaded file in raw_dir
            uploaded_files = list(raw_dir.glob("*"))
            if not uploaded_files:
                raise RuntimeError("No uploaded files found in raw input directory.")
            
            uploaded_file = uploaded_files[0]
            
            from src.orchestrator.dataset_processor import (
                validate_dataset_file,
                preprocess_and_standardize,
                encrypt_and_save_dataset
            )

            raw_records, file_meta = validate_dataset_file(uploaded_file)
            processed_records = preprocess_and_standardize(raw_records)
            
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
            # PHASE 3: PROTECTION & HARDWARE BINDING
            # ────────────────────────────────────────────────────────────────
            self.update_job_state(job_id, status="PACKAGING", stage="hardware_binding", progress=75)
            logger.info("[%s] Phase 3 Protection started.", job_id)

            env_p3 = os.environ.copy()
            env_p3["P3_DEVICE_SALT"] = salt
            env_p3["P3_ADAPTER_INPUT_DIR"] = str(job_dir / "adapter")
            env_p3["P3_PROTECTED_OUTPUT_DIR"] = str(job_dir / "protected")
            env_p3["P3_RSA_PRIVATE_KEY_PATH"] = str(job_dir / "protected" / "dev_private.pem")
            env_p3["P3_RSA_PUBLIC_KEY_PATH"] = str(job_dir / "protected" / "public.pem")

            proc_p3 = subprocess.run(
                ["./venv/bin/python", "-m", "src.phase3.main", "protect", "--archive"],
                cwd=str(Path.cwd()),
                env=env_p3,
                capture_output=True,
                text=True
            )
            if proc_p3.returncode != 0:
                raise RuntimeError(f"Phase 3 protect package builder failed: {proc_p3.stderr or proc_p3.stdout}")

            self.update_job_state(job_id, progress=90)
            logger.info("[%s] Phase 3 complete.", job_id)

            # ────────────────────────────────────────────────────────────────
            # PHASE 4: VERIFICATION & SECURE DEPLOYMENT VALIDATION
            # ────────────────────────────────────────────────────────────────
            self.update_job_state(job_id, status="DEPLOYING", stage="deployment_validation", progress=95)
            logger.info("[%s] Phase 4 Deployment started.", job_id)

            # Run Phase 4 validator pipeline
            from src.phase4.main import run_deployment_pipeline
            package_path = job_dir / "protected" / "protected_adapter.tar.gz"
            # Fallback if with_suffix renamed it
            if not package_path.exists():
                package_path = job_dir / "protected" / "protected.tar.gz"
            if not package_path.exists():
                package_path = job_dir / "protected"

            # Execute validation pipeline
            exit_code = run_deployment_pipeline(
                package_path=package_path,
                salt=salt,
                base_model_name=config.model_name,
                prompt="Verify device binding.",
                output_dir=job_dir / "deployment"
            )

            # Read Phase 4 step details from validation_report.json
            verification_steps = {}
            report_path = job_dir / "deployment" / "validation_report.json"
            if report_path.exists():
                try:
                    report_data = json.loads(report_path.read_text(encoding="utf-8"))
                    verification_steps = report_data.get("verification_pipeline", {}).get("steps", {})
                except Exception as e:
                    logger.warning("Could not read verification report: %s", e)

            if exit_code != 0:
                raise RuntimeError("Phase 4 deployment verification pipeline failed.")

            self.update_job_state(
                job_id,
                status="COMPLETED",
                stage="completed",
                progress=100,
                verification_steps=verification_steps
            )
            logger.info("[%s] Full lifecycle completed successfully!", job_id)

        except Exception as exc:
            logger.error("[%s] Pipeline execution failed: %s", job_id, exc, exc_info=True)
            self.update_job_state(job_id, status="FAILED", error=str(exc))


# Global Orchestrator Instance
orchestrator = JobOrchestrator()
