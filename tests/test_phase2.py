import os
import json
import shutil
import logging
import subprocess
from pathlib import Path
from secure_lora.security import generate_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_phase2")

def create_mock_real_data(input_dir: Path):
    """Creates a mock dataset containing realistic corporate training samples."""
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard instruction-response Q&A dataset
    qa_data = [
        {
            "instruction": "What is the corporate data storage policy?",
            "input": "",
            "output": "All corporate datasets must be encrypted using AES-256-GCM before writing to persistent storage."
        },
        {
            "instruction": "Explain how to handle in-memory fine-tuning.",
            "input": "Secure LoRA Framework",
            "output": "Decrypt the dataset into memory streamingly or via temporary unlinked files, tokenize, and delete the temporary files before launching training."
        },
        {
            "instruction": "Who is authorized to access the device keys?",
            "input": "",
            "output": "Only the device-bound key manager and verified local training jobs can access cryptographic keys."
        },
        {
            "instruction": "What is the rank parameter in LoRA?",
            "input": "",
            "output": "The rank 'r' defines the low-rank factorization dimension of the adapter weights, typically set to 8 or 16."
        }
    ]
    
    with open(input_dir / "corporate_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_data, f)
    logger.info(f"Mock training dataset generated in {input_dir}")

def run_phase2_tests():
    logger.info("Starting Phase 2 Secure Fine-Tuning Verification...")
    
    base_path = Path("tests/temp_phase2_test")
    base_path.mkdir(parents=True, exist_ok=True)
    
    input_dir = base_path / "raw_inputs"
    output_dir = base_path / "encrypted_outputs"
    checkpoint_dir = base_path / "checkpoints"
    adapter_dir = base_path / "lora_adapters"
    
    # 1. Create Mock Dataset
    create_mock_real_data(input_dir)
    
    # 2. Write key to .env or load existing env key
    # Ensure env variable overrides point to this temporary test space
    os.environ["SECURE_LORA_INPUT_DIR"] = str(input_dir)
    os.environ["SECURE_LORA_OUTPUT_DIR"] = str(output_dir)
    os.environ["SECURE_LORA_CHECKPOINT_DIR"] = str(checkpoint_dir)
    os.environ["SECURE_LORA_OUTPUT_DIR_LORA"] = str(adapter_dir) # Custom mapping
    
    # Overrides for config.py
    os.environ["SECURE_LORA_ENCRYPTED_DATA"] = str(output_dir / "encrypted_dataset.enc")
    os.environ["SECURE_LORA_METADATA_PATH"] = str(output_dir / "dataset_metadata.json")
    os.environ["SECURE_LORA_EPOCHS"] = "1"
    os.environ["SECURE_LORA_LR"] = "5e-4"
    os.environ["SECURE_LORA_BATCH_SIZE"] = "1"
    os.environ["SECURE_LORA_GRAD_ACCUM"] = "1"
    os.environ["SECURE_LORA_R"] = "4"
    
    # Ensure key exists in env or create temporary key in .env
    if "SECURE_LORA_KEY_HEX" not in os.environ:
        # Load it from local .env if available, otherwise write a temporary test key
        if Path(".env").exists():
            logger.info("Loading encryption key from active local .env file.")
        else:
            temp_key_hex = generate_key().hex()
            os.environ["SECURE_LORA_KEY_HEX"] = temp_key_hex
            logger.info("No active .env file found; generated temporary environment key for test run.")
            
    # 3. Encrypt the dataset using Phase 1 CLI tool
    logger.info("Encrypting mock dataset using Phase 1 CLI...")
    enc_cmd = [
        "python3", "-m", "secure_lora.cli", "encrypt",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-n", "TestPhase2Dataset"
    ]
    subprocess.run(enc_cmd, check=True)
    
    enc_file = output_dir / "encrypted_dataset.enc"
    meta_file = output_dir / "dataset_metadata.json"
    assert enc_file.exists(), "Encrypted file was not created by CLI!"
    assert meta_file.exists(), "Metadata file was not created by CLI!"
    logger.info("Verification: Mock dataset GCM-encrypted successfully.")
    
    # 4. Trigger Training Run via train_lora.py
    # We execute train_lora.py using our virtual environment python to isolate dependencies
    logger.info("Triggering secure fine-tuning training loop (this will load model and run 1 epoch)...")
    train_cmd = ["venv/bin/python3", "train_lora.py"]
    
    # Update config.py target paths using env vars before running subprocess
    env_copy = os.environ.copy()
    env_copy["SECURE_LORA_OUTPUT_DIR"] = str(adapter_dir) # Ensure trainer outputs to test adapter folder
    
    # Run training
    subprocess.run(train_cmd, env=env_copy, check=True)
    
    # 5. Assert checkings
    logger.info("Verifying training outcomes and adapter packaging...")
    assert adapter_dir.exists(), "Final adapter directory was not created!"
    
    # PEFT config and weight files must exist
    peft_config_file = adapter_dir / "adapter_config.json"
    assert peft_config_file.exists(), "PEFT adapter_config.json is missing!"
    
    peft_weights_safetensors = adapter_dir / "adapter_model.safetensors"
    peft_weights_bin = adapter_dir / "adapter_model.bin"
    assert peft_weights_safetensors.exists() or peft_weights_bin.exists(), "PEFT weight files (adapter_model.safetensors / adapter_model.bin) are missing!"
    
    # Verify evaluation report
    report_file = Path("eval_report.json")
    assert report_file.exists(), "eval_report.json report is missing!"
    with open(report_file, "r") as f:
        report = json.load(f)
    
    logger.info("Evaluation report summary:")
    logger.info(f"  Validation Loss: {report.get('validation_loss')}")
    logger.info(f"  Perplexity: {report.get('perplexity')}")
    logger.info(f"  Pre-train gen sample: {report.get('pre_training_generation')}")
    logger.info(f"  Post-train gen sample: {report.get('post_training_generation')}")
    
    # 6. Verify Checkpoint management
    logger.info("Verifying checkpoint folder contents...")
    assert checkpoint_dir.exists(), "Checkpoint directory was not created during training!"
    checkpoint_folders = list(checkpoint_dir.glob("checkpoint-*"))
    logger.info(f"Found {len(checkpoint_folders)} checkpoint folders.")
    assert len(checkpoint_folders) <= 2, "Checkpoint rotation failed to limit checkpoints to max_to_keep=2!"
    
    # 7. Security Cleanups: Confirm no temporary plaintext JSONL files remain in the workspace
    logger.info("Verifying plaintext leakage defense...")
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".jsonl") and f.startswith("tmp"):
                assert False, f"Plaintext leakage: found un-shredded temporary file {f} under {root}"
                
    logger.info("Verification: Plaintext dataset file was securely shredded from persistent disk before training ended.")
    
    # Clean up test directories
    shutil.rmtree(base_path, ignore_errors=True)
    if report_file.exists():
        report_file.unlink()
        
    logger.info("==========================================")
    logger.info("PHASE 2 VALIDATION COMPLETED SUCCESSFULLY! (100%)")
    logger.info("==========================================")

if __name__ == "__main__":
    run_phase2_tests()
