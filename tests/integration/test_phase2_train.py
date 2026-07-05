import os
import json
import shutil
import pytest
from pathlib import Path

from src.security import generate_key
from src.phase1.pipeline import SecureDatasetPipeline
from src.phase2.train_lora import run_training


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def create_mock_real_data(input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
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


def test_phase2_training(tmp_dir):
    input_dir = tmp_dir / "raw_inputs"
    output_dir = tmp_dir / "encrypted_outputs"
    checkpoint_dir = tmp_dir / "checkpoints"
    adapter_dir = tmp_dir / "lora_adapters"

    create_mock_real_data(input_dir)

    temp_key_hex = generate_key().hex()
    os.environ["SECURE_LORA_KEY_HEX"] = temp_key_hex

    # Run encryption pipeline
    pipeline = SecureDatasetPipeline(bytes.fromhex(temp_key_hex))
    pipeline.encrypt_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_name="TestPhase2Dataset"
    )

    # Set environment variables for config_loader
    os.environ["SECURE_LORA_INPUT_DIR"] = str(input_dir)
    os.environ["SECURE_LORA_OUTPUT_DIR"] = str(output_dir)
    os.environ["SECURE_LORA_CHECKPOINT_DIR"] = str(checkpoint_dir)
    os.environ["SECURE_LORA_OUTPUT_DIR_LORA"] = str(adapter_dir)

    os.environ["SECURE_LORA_ENCRYPTED_DATA"] = str(output_dir / "encrypted_dataset.enc")
    os.environ["SECURE_LORA_METADATA_PATH"] = str(output_dir / "dataset_metadata.json")
    os.environ["SECURE_LORA_EPOCHS"] = "1"
    os.environ["SECURE_LORA_LR"] = "5e-4"
    os.environ["SECURE_LORA_BATCH_SIZE"] = "1"
    os.environ["SECURE_LORA_GRAD_ACCUM"] = "1"
    os.environ["SECURE_LORA_R"] = "4"

    try:
        # Run training loop in-process
        run_training()

        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists()

        report_file = Path("eval_report.json")
        assert report_file.exists()
        report = json.loads(report_file.read_text())
        assert "validation_loss" in report
        assert "perplexity" in report

        # Verify no un-shredded tmp files
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith(".jsonl") and f.startswith("tmp"):
                    assert False, f"Plaintext leak: found {f} in {root}"
    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if Path("eval_report.json").exists():
            Path("eval_report.json").unlink()
        for k in [
            "SECURE_LORA_INPUT_DIR", "SECURE_LORA_OUTPUT_DIR", "SECURE_LORA_CHECKPOINT_DIR",
            "SECURE_LORA_OUTPUT_DIR_LORA", "SECURE_LORA_ENCRYPTED_DATA", "SECURE_LORA_METADATA_PATH",
            "SECURE_LORA_EPOCHS", "SECURE_LORA_LR", "SECURE_LORA_BATCH_SIZE", "SECURE_LORA_GRAD_ACCUM",
            "SECURE_LORA_R", "SECURE_LORA_KEY_HEX"
        ]:
            os.environ.pop(k, None)
