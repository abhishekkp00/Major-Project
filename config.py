import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TrainingConfig:
    """Manages all configuration parameters for the Secure LoRA Fine-Tuning Pipeline."""
    
    # Model Configuration
    # Using a lightweight, fast backbone (JackFram/llama-68m) for standard correctness
    # Can be replaced with Phi-3, Mistral, Llama-3, etc.
    MODEL_NAME = os.environ.get("SECURE_LORA_MODEL_NAME", "JackFram/llama-68m")
    
    # Dataset Ingestion Configuration
    ENCRYPTED_DATASET_PATH = Path(os.environ.get("SECURE_LORA_ENCRYPTED_DATA", "encrypted_real_data/encrypted_dataset.enc"))
    METADATA_PATH = Path(os.environ.get("SECURE_LORA_METADATA_PATH", "encrypted_real_data/dataset_metadata.json"))
    
    # Training Parameters
    BATCH_SIZE = int(os.environ.get("SECURE_LORA_BATCH_SIZE", 2))
    GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("SECURE_LORA_GRAD_ACCUM", 4))
    LEARNING_RATE = float(os.environ.get("SECURE_LORA_LR", 2e-4))
    NUM_EPOCHS = int(os.environ.get("SECURE_LORA_EPOCHS", 3))
    MAX_SEQ_LENGTH = int(os.environ.get("SECURE_LORA_MAX_LEN", 256))
    SEED = int(os.environ.get("SECURE_LORA_SEED", 42))
    
    # Directory Layout
    OUTPUT_DIR = Path(os.environ.get("SECURE_LORA_OUTPUT_DIR", "lora_adapters"))
    CHECKPOINT_DIR = Path(os.environ.get("SECURE_LORA_CHECKPOINT_DIR", "checkpoints"))
    
    # LoRA Parameters
    LORA_R = int(os.environ.get("SECURE_LORA_R", 8))
    LORA_ALPHA = int(os.environ.get("SECURE_LORA_ALPHA", 16))
    LORA_DROPOUT = float(os.environ.get("SECURE_LORA_DROPOUT", 0.05))
    LORA_BIAS = os.environ.get("SECURE_LORA_BIAS", "none")
    
    # Target modules for LoRA injection (standard self-attention projections)
    # Automatically adjusted based on base model type
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    @classmethod
    def validate(cls):
        """Ensures directories and settings are correct before training begins."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Security validation: Verify encrypted file exists
        if not cls.ENCRYPTED_DATASET_PATH.exists():
            raise FileNotFoundError(f"Encrypted dataset not found at {cls.ENCRYPTED_DATASET_PATH}")
            
        # Security safety check: Assert no plaintext temporary datasets are sitting in the root workspace
        for path in Path(".").glob("*.jsonl"):
            if "audit" not in path.name and "temp" in path.name:
                raise PermissionError(f"Security Alert: Unsecured plaintext files ({path.name}) found in workspace. Remove before starting training.")
