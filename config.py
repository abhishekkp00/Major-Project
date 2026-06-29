import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TrainingConfig:
    # Model parameters
    MODEL_NAME = os.environ.get("SECURE_LORA_MODEL_NAME", "JackFram/llama-68m")
    
    # Dataset files
    ENCRYPTED_DATASET_PATH = Path(os.environ.get("SECURE_LORA_ENCRYPTED_DATA", "encrypted_real_data/encrypted_dataset.enc"))
    METADATA_PATH = Path(os.environ.get("SECURE_LORA_METADATA_PATH", "encrypted_real_data/dataset_metadata.json"))
    
    # Training hyperparameters
    BATCH_SIZE = int(os.environ.get("SECURE_LORA_BATCH_SIZE", 2))
    GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("SECURE_LORA_GRAD_ACCUM", 4))
    LEARNING_RATE = float(os.environ.get("SECURE_LORA_LR", 2e-4))
    NUM_EPOCHS = int(os.environ.get("SECURE_LORA_EPOCHS", 3))
    MAX_SEQ_LENGTH = int(os.environ.get("SECURE_LORA_MAX_LEN", 256))
    SEED = int(os.environ.get("SECURE_LORA_SEED", 42))
    
    # Output directories
    OUTPUT_DIR = Path(os.environ.get("SECURE_LORA_OUTPUT_DIR", "lora_adapters"))
    CHECKPOINT_DIR = Path(os.environ.get("SECURE_LORA_CHECKPOINT_DIR", "checkpoints"))
    
    # LoRA config settings
    LORA_R = int(os.environ.get("SECURE_LORA_R", 8))
    LORA_ALPHA = int(os.environ.get("SECURE_LORA_ALPHA", 16))
    LORA_DROPOUT = float(os.environ.get("SECURE_LORA_DROPOUT", 0.05))
    LORA_BIAS = os.environ.get("SECURE_LORA_BIAS", "none")
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    @classmethod
    def validate(cls):
        """Verify dataset paths and enforce safety policies before starting training."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        if not cls.ENCRYPTED_DATASET_PATH.exists():
            raise FileNotFoundError(f"Missing encrypted dataset file: {cls.ENCRYPTED_DATASET_PATH}")
            
        # Check workspace for stray plaintext jsonl data dumps
        for path in Path(".").glob("*.jsonl"):
            if "audit" not in path.name and "temp" in path.name:
                raise PermissionError(
                    f"Plaintext file leak detected: '{path.name}'. "
                    "Remove plaintext files from workspace before training."
                )
