import os
import json
import shutil
import pytest
from pathlib import Path

from src.security import generate_key, compute_sha256
from src.phase1.pipeline import SecureDatasetPipeline


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def create_mock_dataset(input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)

    txt_content = (
        "Organizational policy 101: Never share device-bound keys.\n"
        "Organizational policy 102: Always encrypt training datasets at rest.\n"
    )
    (input_dir / "policy.txt").write_text(txt_content, encoding="utf-8")

    md_content = (
        "# Security Standard Operating Procedure\n\n"
        "LoRA adapter training must run locally on verified compute units.\n\n"
        "In-memory decryption prevents intermediate plaintext leakages.\n"
    )
    (input_dir / "procedures.md").write_text(md_content, encoding="utf-8")

    csv_content = (
        "instruction,input,output\n"
        "\"What is the model name?\",\"\",\"The framework is called Secure Device-Bound LoRA.\"\n"
        "\"How to handle encryption?\",\"\",\"Use AES-256-GCM chunked encryption stream.\"\n"
    )
    (input_dir / "qa.csv").write_text(csv_content, encoding="utf-8")

    json_data = [
        {"prompt": "Define PEFT.", "response": "PEFT stands for Parameter-Efficient Fine-Tuning."},
        {"text": "Random loose causal document content for language model pretraining."}
    ]
    with open(input_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f)


def test_pipeline_integration(tmp_dir):
    input_dir = tmp_dir / "raw_inputs"
    output_dir = tmp_dir / "encrypted_outputs"
    decrypted_dir = tmp_dir / "decrypted_outputs"
    decrypted_dir.mkdir()

    create_mock_dataset(input_dir)

    key = generate_key()
    key_file = tmp_dir / "secret.key"
    key_file.write_bytes(key)

    pipeline = SecureDatasetPipeline(key)

    metadata = pipeline.encrypt_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_name="TestCorpDataset",
        version="1.0.0"
    )

    enc_file = output_dir / "encrypted_dataset.enc"
    meta_file = output_dir / "dataset_metadata.json"

    assert enc_file.exists()
    assert meta_file.exists()

    assert metadata["dataset_name"] == "TestCorpDataset"
    assert metadata["encryption_status"] == "encrypted"
    assert metadata["encryption_algorithm"] == "AES-256-GCM"
    assert metadata["num_records"] > 0
    assert "encrypted_file_sha256" in metadata
    assert "encrypted_file_size_bytes" in metadata

    # Decrypt and verify
    decrypted_records = list(pipeline.decrypt_records(enc_file))
    assert len(decrypted_records) == metadata["num_records"]

    instruction_records = [r for r in decrypted_records if "instruction" in r]
    causal_records = [r for r in decrypted_records if "text" in r]

    assert len(instruction_records) > 0
    assert len(causal_records) > 0

    # Temp file decrypt
    with pipeline.decrypted_temp_file(enc_file) as temp_plaintext_file:
        assert temp_plaintext_file.exists()
        lines = temp_plaintext_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == metadata["num_records"]

    assert not temp_plaintext_file.exists()

    # Tamper test
    corrupted_enc_file = tmp_dir / "corrupted_dataset.enc"
    shutil.copy(enc_file, corrupted_enc_file)

    with open(corrupted_enc_file, "r+b") as f:
        f.seek(32)
        byte = f.read(1)
        corrupted_byte = bytes([byte[0] ^ 0xFF])
        f.seek(32)
        f.write(corrupted_byte)

    with pytest.raises(ValueError):
        list(pipeline.decrypt_records(corrupted_enc_file))

    # Wrong key test
    wrong_key = generate_key()
    wrong_pipeline = SecureDatasetPipeline(wrong_key)
    with pytest.raises(ValueError):
        list(wrong_pipeline.decrypt_records(enc_file))

    # Shredding raw test
    raw_shred_dir = tmp_dir / "raw_shred"
    create_mock_dataset(raw_shred_dir)

    shred_output_dir = tmp_dir / "shred_outputs"
    pipeline.encrypt_dataset(
        input_dir=raw_shred_dir,
        output_dir=shred_output_dir,
        dataset_name="ShrededTest",
        shred_raw=True
    )

    remaining_files = list(raw_shred_dir.glob("*"))
    for f in remaining_files:
        assert not f.is_file()
