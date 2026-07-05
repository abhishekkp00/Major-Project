import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.common.exceptions import ConfigError

logger = logging.getLogger("secure_lora.config")


def find_project_root() -> Path:
    """Walks up from this file to locate the project root containing 'config' or '.env'."""
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "config").exists() or (parent / ".env").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()

# Load env variables from root path
load_dotenv(PROJECT_ROOT / ".env")


class ConfigLoader:
    def __init__(self):
        self.root = PROJECT_ROOT
        self.config_dir = self.root / "config"

        # Load Yaml files if they exist, else default to empty dicts
        self.app = self._load_yaml(self.config_dir / "app.yaml")
        self.training = self._load_yaml(self.config_dir / "training.yaml")
        self.security = self._load_yaml(self.config_dir / "security.yaml")
        self.deployment = self._load_yaml(self.config_dir / "deployment.yaml")

    def _load_yaml(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to parse config file %s: %s. Using empty defaults.", path.name, e)
                return {}
        return {}

    # ── App configs ──────────────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return os.environ.get("SECURE_LORA_MODEL_NAME", self.app.get("model_name", "JackFram/llama-68m"))

    @property
    def seed(self) -> int:
        return int(os.environ.get("SECURE_LORA_SEED", self.app.get("seed", 42)))

    @property
    def max_seq_length(self) -> int:
        return int(os.environ.get("SECURE_LORA_MAX_LEN", self.app.get("max_seq_length", 256)))

    # ── Training configs ──────────────────────────────────────────────────────
    @property
    def batch_size(self) -> int:
        return int(os.environ.get("SECURE_LORA_BATCH_SIZE", self.training.get("batch_size", 2)))

    @property
    def gradient_accumulation_steps(self) -> int:
        return int(os.environ.get("SECURE_LORA_GRAD_ACCUM", self.training.get("gradient_accumulation_steps", 4)))

    @property
    def learning_rate(self) -> float:
        return float(os.environ.get("SECURE_LORA_LR", self.training.get("learning_rate", 2e-4)))

    @property
    def num_epochs(self) -> int:
        return int(os.environ.get("SECURE_LORA_EPOCHS", self.training.get("num_epochs", 3)))

    @property
    def lora_r(self) -> int:
        return int(os.environ.get("SECURE_LORA_R", self.training.get("lora_r", 8)))

    @property
    def lora_alpha(self) -> int:
        return int(os.environ.get("SECURE_LORA_ALPHA", self.training.get("lora_alpha", 16)))

    @property
    def lora_dropout(self) -> float:
        return float(os.environ.get("SECURE_LORA_DROPOUT", self.training.get("lora_dropout", 0.05)))

    @property
    def lora_bias(self) -> str:
        return os.environ.get("SECURE_LORA_BIAS", self.training.get("lora_bias", "none"))

    @property
    def target_modules(self) -> list:
        return self.training.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

    @property
    def dataset_input_dir(self) -> Path:
        return Path(os.environ.get("SECURE_LORA_INPUT_DIR", "real_data_inputs"))

    @property
    def dataset_output_dir(self) -> Path:
        return Path(os.environ.get("SECURE_LORA_OUTPUT_DIR", "encrypted_real_data"))

    @property
    def encrypted_dataset_path(self) -> Path:
        return Path(os.environ.get("SECURE_LORA_ENCRYPTED_DATA", self.dataset_output_dir / "encrypted_dataset.enc"))

    @property
    def metadata_path(self) -> Path:
        return Path(os.environ.get("SECURE_LORA_METADATA_PATH", self.dataset_output_dir / "dataset_metadata.json"))

    @property
    def checkpoint_dir(self) -> Path:
        return Path(os.environ.get("SECURE_LORA_CHECKPOINT_DIR", "checkpoints"))

    @property
    def lora_output_dir(self) -> Path:
        return Path(os.environ.get("SECURE_LORA_OUTPUT_DIR_LORA", "lora_adapters"))

    # ── Security configs ──────────────────────────────────────────────────────
    @property
    def secure_lora_key_hex(self) -> str:
        return os.environ.get("SECURE_LORA_KEY_HEX", "")

    @property
    def secure_lora_key_path(self) -> Path:
        path_str = os.environ.get("SECURE_LORA_KEY_PATH", self.security.get("secure_lora_key_path", "secrets.key"))
        return Path(path_str) if path_str else Path("secrets.key")

    @property
    def device_salt(self) -> str:
        return os.environ.get("P3_DEVICE_SALT", "")

    @property
    def rsa_key_bits(self) -> int:
        return int(os.environ.get("P3_RSA_KEY_BITS", self.security.get("rsa_key_bits", 2048)))

    @property
    def rsa_private_key_path(self) -> Path:
        return Path(os.environ.get("P3_RSA_PRIVATE_KEY_PATH", self.security.get("rsa_private_key_path", "outputs/protected_adapter/dev_private.pem")))

    @property
    def rsa_public_key_path(self) -> Path:
        return Path(os.environ.get("P3_RSA_PUBLIC_KEY_PATH", self.security.get("rsa_public_key_path", "outputs/protected_adapter/public.pem")))

    @property
    def adapter_id(self) -> str:
        return os.environ.get("P3_ADAPTER_ID", self.security.get("adapter_id", "lora-adapter-v1"))

    @property
    def model_reference(self) -> str:
        return os.environ.get("P3_MODEL_REFERENCE", self.security.get("model_reference", "JackFram/llama-68m"))

    @property
    def package_version(self) -> str:
        return self.security.get("package_version", "3.0.0")

    # ── Deployment configs ────────────────────────────────────────────────────
    @property
    def package_path(self) -> Path:
        return Path(os.environ.get("P4_PACKAGE_PATH", self.deployment.get("package_path", "outputs/protected_adapter")))

    @property
    def p4_public_key_path(self) -> Path:
        return Path(os.environ.get("P4_PUBLIC_KEY_PATH", self.deployment.get("public_key_path", "outputs/protected_adapter/public.pem")))

    @property
    def deployment_output_dir(self) -> Path:
        return Path(os.environ.get("P4_OUTPUT_DIR", self.deployment.get("output_dir", "outputs/deployment_validation")))

    @property
    def max_new_tokens(self) -> int:
        return int(os.environ.get("P4_MAX_NEW_TOKENS", self.deployment.get("max_new_tokens", 64)))

    @property
    def max_package_bytes(self) -> int:
        return int(os.environ.get("P4_MAX_PACKAGE_BYTES", self.deployment.get("max_package_bytes", 512 * 1024 * 1024)))

    @property
    def required_package_files(self) -> list:
        return self.deployment.get("required_package_files", ["adapter.enc", "adapter.hash", "adapter.sig", "metadata.json", "package_manifest.json"])

    def validate_phase1(self) -> None:
        """Enforces conditions for Phase 1 (Ingestion & Protection)."""
        # Ensure input directory exists
        if not self.dataset_input_dir.exists():
            self.dataset_input_dir.mkdir(parents=True, exist_ok=True)

    def validate_phase2(self) -> None:
        """Enforces conditions for Phase 2 (Fine-tuning)."""
        self.lora_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.encrypted_dataset_path.exists():
            raise FileNotFoundError(f"Missing encrypted dataset file: {self.encrypted_dataset_path}")

        # Check workspace for leaks
        for path in self.root.glob("*.jsonl"):
            if "audit" not in path.name and "temp" in path.name:
                raise ConfigError(
                    f"Plaintext file leak detected: '{path.name}'. "
                    "Remove plaintext files from workspace before training."
                )

    def validate_phase3(self) -> None:
        """Enforces conditions for Phase 3 (Adapter Protection)."""
        if not self.device_salt:
            raise ConfigError(
                "P3_DEVICE_SALT is not set. Export it as an environment variable before running Phase 3."
            )
        # Check if phase 2 adapter dir exists
        adapter_input = Path(os.environ.get("P3_ADAPTER_INPUT_DIR", "outputs/final_adapter"))
        if not adapter_input.exists():
            if self.lora_output_dir.exists():
                os.environ["P3_ADAPTER_INPUT_DIR"] = str(self.lora_output_dir)
            else:
                raise FileNotFoundError(
                    f"LoRA adapter directory not found. Run Phase 2 first."
                )

        protected_out = Path(os.environ.get("P3_PROTECTED_OUTPUT_DIR", "outputs/protected_adapter"))
        protected_out.mkdir(parents=True, exist_ok=True)

    def validate_phase4(self) -> None:
        """Enforces conditions for Phase 4 (Deployment)."""
        if not self.device_salt:
            raise ConfigError(
                "P3_DEVICE_SALT is not set. Export it as an environment variable before running Phase 4."
            )
        if not self.package_path.exists():
            raise FileNotFoundError(
                f"Package path not found: {self.package_path}. Run Phase 3 first or set P4_PACKAGE_PATH."
            )
        self.deployment_output_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = ConfigLoader()
