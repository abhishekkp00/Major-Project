import os
import logging
from pathlib import Path
from src.common.config_loader import config

logger = logging.getLogger("secure_lora.phase3.config")


class Phase3Config:
    ADAPTER_INPUT_DIR: Path = Path(os.environ.get("P3_ADAPTER_INPUT_DIR", "outputs/final_adapter"))
    PROTECTED_OUTPUT_DIR: Path = Path(os.environ.get("P3_PROTECTED_OUTPUT_DIR", "outputs/protected_adapter"))

    DEVICE_SALT: str = config.device_salt
    RSA_KEY_BITS: int = config.rsa_key_bits
    RSA_PRIVATE_KEY_PATH: Path = config.rsa_private_key_path
    RSA_PUBLIC_KEY_PATH: Path = config.rsa_public_key_path

    ADAPTER_ID: str = config.adapter_id
    MODEL_REFERENCE: str = config.model_reference
    PACKAGE_VERSION: str = config.package_version

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

    @classmethod
    def validate(cls) -> None:
        config.validate_phase3()
