"""
phase3/config.py
----------------
Centralised, environment-driven configuration for Phase 3.

Every secret and every tunable knob lives here so the rest of the modules
stay free of hard-coded values.  Secrets are pulled exclusively from
environment variables — the caller is responsible for populating them
(e.g. via a .env file loaded before import or via CI/CD secrets injection).

Nothing in this module logs or prints secret material.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Phase3Config:
    # ---------------------------------------------------------------------------
    # Input / output paths
    # ---------------------------------------------------------------------------
    #: Directory produced by Phase 2 containing the raw LoRA adapter files.
    ADAPTER_INPUT_DIR: Path = Path(
        os.environ.get("P3_ADAPTER_INPUT_DIR", "outputs/final_adapter")
    )

    #: Directory where all protected-package artefacts are written.
    PROTECTED_OUTPUT_DIR: Path = Path(
        os.environ.get("P3_PROTECTED_OUTPUT_DIR", "outputs/protected_adapter")
    )

    # ---------------------------------------------------------------------------
    # Cryptographic tuning
    # ---------------------------------------------------------------------------
    #: Salt for HKDF/SHA-256-based key derivation.  MUST be set via env var in
    #: production; the fallback is deliberately weak to force explicit configuration.
    DEVICE_SALT: str = os.environ.get("P3_DEVICE_SALT", "")

    #: RSA key size used for test/dev key generation (bits).
    RSA_KEY_BITS: int = int(os.environ.get("P3_RSA_KEY_BITS", "2048"))

    #: Path where the dev/test RSA private key is persisted.
    RSA_PRIVATE_KEY_PATH: Path = Path(
        os.environ.get("P3_RSA_PRIVATE_KEY_PATH", "outputs/protected_adapter/dev_private.pem")
    )

    #: Path where the RSA public key is stored (included in the package).
    RSA_PUBLIC_KEY_PATH: Path = Path(
        os.environ.get("P3_RSA_PUBLIC_KEY_PATH", "outputs/protected_adapter/public.pem")
    )

    # ---------------------------------------------------------------------------
    # Package metadata
    # ---------------------------------------------------------------------------
    #: Human-readable adapter identifier embedded in the package manifest.
    ADAPTER_ID: str = os.environ.get("P3_ADAPTER_ID", "lora-adapter-v1")

    #: Base-model reference for the manifest (informational only).
    MODEL_REFERENCE: str = os.environ.get("P3_MODEL_REFERENCE", "JackFram/llama-68m")

    #: Package schema version (semver).
    PACKAGE_VERSION: str = "3.0.0"

    # ---------------------------------------------------------------------------
    # Paths for individual package artefacts
    # ---------------------------------------------------------------------------
    @classmethod
    def enc_path(cls) -> Path:
        return cls.PROTECTED_OUTPUT_DIR / "adapter.enc"

    @classmethod
    def hash_path(cls) -> Path:
        return cls.PROTECTED_OUTPUT_DIR / "adapter.hash"

    @classmethod
    def sig_path(cls) -> Path:
        return cls.PROTECTED_OUTPUT_DIR / "adapter.sig"

    @classmethod
    def metadata_path(cls) -> Path:
        return cls.PROTECTED_OUTPUT_DIR / "metadata.json"

    @classmethod
    def manifest_path(cls) -> Path:
        return cls.PROTECTED_OUTPUT_DIR / "package_manifest.json"

    # ---------------------------------------------------------------------------
    # Validation helper
    # ---------------------------------------------------------------------------
    @classmethod
    def validate(cls) -> None:
        """
        Enforce mandatory pre-conditions before any Phase 3 operation begins.

        Raises
        ------
        EnvironmentError
            If the device salt has not been set via ``P3_DEVICE_SALT``.
        FileNotFoundError
            If the Phase 2 adapter input directory does not exist.
        """
        if not cls.DEVICE_SALT:
            raise EnvironmentError(
                "P3_DEVICE_SALT is not set. "
                "Export it as an environment variable before running Phase 3."
            )

        if not cls.ADAPTER_INPUT_DIR.exists():
            raise FileNotFoundError(
                f"Phase 2 adapter directory not found: {cls.ADAPTER_INPUT_DIR}. "
                "Run Phase 2 first, or point P3_ADAPTER_INPUT_DIR at the correct path."
            )

        cls.PROTECTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Phase3Config validated successfully.")
