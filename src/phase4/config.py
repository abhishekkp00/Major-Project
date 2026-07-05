import os
import logging
from pathlib import Path
from src.common.config_loader import config

logger = logging.getLogger("secure_lora.phase4.config")

REQUIRED_PACKAGE_FILES = [
    "adapter.enc",
    "adapter.hash",
    "adapter.sig",
    "metadata.json",
    "package_manifest.json",
]


class Phase4Config:
    PACKAGE_PATH: Path = Path(os.environ.get("P4_PACKAGE_PATH", "outputs/protected_adapter"))
    PUBLIC_KEY_PATH: Path = Path(os.environ.get("P4_PUBLIC_KEY_PATH", "outputs/protected_adapter/public.pem"))

    DEVICE_SALT: str = config.device_salt

    DEPLOYMENT_OUTPUT_DIR: Path = Path(os.environ.get("P4_OUTPUT_DIR", "outputs/deployment_validation"))

    VALIDATION_REPORT_JSON: Path = DEPLOYMENT_OUTPUT_DIR / "validation_report.json"
    VALIDATION_REPORT_MD:   Path = DEPLOYMENT_OUTPUT_DIR / "validation_report.md"
    INFERENCE_LOG_JSON:     Path = DEPLOYMENT_OUTPUT_DIR / "inference_log.json"
    DEMO_SUMMARY_JSON:      Path = DEPLOYMENT_OUTPUT_DIR / "demo_summary.json"

    DEFAULT_BASE_MODEL: str = config.model_name
    MAX_NEW_TOKENS: int = 64
    MAX_PACKAGE_BYTES: int = 512 * 1024 * 1024
    REQUIRED_PACKAGE_FILES = REQUIRED_PACKAGE_FILES

    @classmethod
    def validate(cls) -> None:
        config.validate_phase4()
        cls.DEPLOYMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
