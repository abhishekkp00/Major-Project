import os
import json
import shutil
import pytest
from pathlib import Path

from src.orchestrator.dataset_processor import (
    validate_dataset_file,
    preprocess_and_standardize,
    encrypt_and_save_dataset
)
from src.common.exceptions import DatasetValidationError
from src.security.crypto import generate_key, decrypt_stream


@pytest.fixture()
def tmp_dir(tmp_path: Path):
    return tmp_path


def test_validation_unsupported_format(tmp_dir):
    bad_file = tmp_dir / "data.xlsx"
    bad_file.write_text("dummy content")
    with pytest.raises(DatasetValidationError, match="Unsupported file format"):
        validate_dataset_file(bad_file)


def test_validation_malformed_json(tmp_dir):
    bad_json = tmp_dir / "data.json"
    bad_json.write_text("{malformed: json")
    with pytest.raises(DatasetValidationError, match="Malformed JSON"):
        validate_dataset_file(bad_json)


def test_pii_detection_and_parsing(tmp_dir):
    data = [
        {"instruction": "Mask PII", "output": "Email is alice@gmail.com and phone is 123-456-7890."}
    ]
    json_file = tmp_dir / "dataset.json"
    json_file.write_text(json.dumps(data))

    records, meta = validate_dataset_file(json_file)
    assert len(records) == 1
    assert meta["num_raw_records"] == 1
    assert meta["pii_detected_summary"]["email"] == 1
    assert meta["pii_detected_summary"]["phone"] == 1
    assert meta["has_pii"] is True
    assert meta["schema_detected"] == "instruction"


def test_preprocessing_standardization():
    raw_data = [
        {"prompt": "Say hello", "response": "Hello!"},
        {"text": "Just plain causal text content."}
    ]
    processed = preprocess_and_standardize(raw_data)
    assert len(processed) == 2
    assert "instruction" in processed[0]
    assert "output" in processed[0]
    assert processed[0]["instruction"] == "Say hello"
    assert "text" in processed[1]
    assert processed[1]["text"] == "Just plain causal text content."


def test_encryption_and_decryption_flow(tmp_dir):
    processed = [
        {"instruction": "Query", "input": "", "output": "Response text"}
    ]
    key = generate_key()
    out_dir = tmp_dir / "encrypted_workspace"

    meta = encrypt_and_save_dataset(
        processed_records=processed,
        key=key,
        output_dir=out_dir,
        dataset_name="secured_dataset",
        version="1.1.0",
        pii_summary={"email": 0, "phone": 0}
    )

    assert meta["dataset_name"] == "secured_dataset"
    assert meta["version"] == "1.1.0"
    assert meta["encryption_status"] == "encrypted"

    enc_file = out_dir / "encrypted_dataset.enc"
    assert enc_file.exists()

    # Verify we can decrypt stream back to records
    dec_file = tmp_dir / "decrypted.jsonl"
    with open(enc_file, "rb") as fin, open(dec_file, "wb") as fout:
        decrypt_stream(fin, fout, key)

    decrypted_lines = dec_file.read_text().splitlines()
    assert len(decrypted_lines) == 1
    assert json.loads(decrypted_lines[0])["output"] == "Response text"
