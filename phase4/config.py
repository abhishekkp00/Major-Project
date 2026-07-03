"""
phase4/config.py
----------------
Centralised, environment-driven configuration for Phase 4 deployment.

All secrets are read exclusively from environment variables so the module
stays free of hard-coded sensitive values.  The Phase 3 salt (P3_DEVICE_SALT)
is re-used here — both phases must share the same salt for key derivation to
produce the same device-bound key that was used during encryption.

Nothing in this module logs or prints secret material.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Required files that MUST be present in any valid protected package ─────────
REQUIRED_PACKAGE_FILES = [
    "adapter.enc",
    "adapter.hash",
    "adapter.sig",
    "metadata.json",
    "package_manifest.json",
]


class Phase4Config:
    # ── Package source ─────────────────────────────────────────────────────────
    #: Default location of the protected adapter package (folder or .tar.gz).
    PACKAGE_PATH: Path = Path(
        os.environ.get("P4_PACKAGE_PATH", "outputs/protected_adapter")
    )

    #: Path to the RSA public key used for signature verification.
    PUBLIC_KEY_PATH: Path = Path(
        os.environ.get("P4_PUBLIC_KEY_PATH", "outputs/protected_adapter/public.pem")
    )

    # ── Device binding ─────────────────────────────────────────────────────────
    #: Secret salt shared with Phase 3 — must be set in the environment.
    DEVICE_SALT: str = os.environ.get("P3_DEVICE_SALT", "")

    # ── Output paths ───────────────────────────────────────────────────────────
    DEPLOYMENT_OUTPUT_DIR: Path = Path(
        os.environ.get("P4_OUTPUT_DIR", "outputs/deployment_validation")
    )

    VALIDATION_REPORT_JSON: Path = DEPLOYMENT_OUTPUT_DIR / "validation_report.json"
    VALIDATION_REPORT_MD:   Path = DEPLOYMENT_OUTPUT_DIR / "validation_report.md"
    INFERENCE_LOG_JSON:     Path = DEPLOYMENT_OUTPUT_DIR / "inference_log.json"
    DEMO_SUMMARY_JSON:      Path = DEPLOYMENT_OUTPUT_DIR / "demo_summary.json"

    # ── Model loading ──────────────────────────────────────────────────────────
    #: Fallback base-model override; normally read from the package manifest.
    DEFAULT_BASE_MODEL: str = os.environ.get(
        "P4_DEFAULT_BASE_MODEL", "JackFram/llama-68m"
    )

    #: Maximum new tokens produced during inference validation.
    MAX_NEW_TOKENS: int = int(os.environ.get("P4_MAX_NEW_TOKENS", "64"))

    # ── Safety limits ──────────────────────────────────────────────────────────
    #: Largest package we will extract (bytes).  Guards against zip-bombs.
    MAX_PACKAGE_BYTES: int = int(
        os.environ.get("P4_MAX_PACKAGE_BYTES", str(512 * 1024 * 1024))  # 512 MB
    )

    # ── Validation behaviour ───────────────────────────────────────────────────
    REQUIRED_PACKAGE_FILES = REQUIRED_PACKAGE_FILES

    @classmethod
    def validate(cls) -> None:
        """
        Enforces mandatory pre-conditions before Phase 4 begins.

        Raises
        ------
        EnvironmentError
            If the device salt is missing.
        FileNotFoundError
            If the package path does not exist.
        """
        if not cls.DEVICE_SALT:
            raise EnvironmentError(
                "P3_DEVICE_SALT is not set. "
                "Export it as an environment variable before running Phase 4."
            )

        pkg = cls.PACKAGE_PATH
        if not pkg.exists():
            raise FileNotFoundError(
                f"Package path not found: {pkg}. "
                "Run Phase 3 first or set P4_PACKAGE_PATH."
            )

        cls.DEPLOYMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Phase4Config validated successfully.")
