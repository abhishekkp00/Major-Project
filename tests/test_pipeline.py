import os
import json
import logging
import shutil
import tempfile
from pathlib import Path
from secure_lora.security import generate_key, compute_sha256, secure_delete_file
from secure_lora.pipeline import SecureDatasetPipeline

# Set up logging for output transparency
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_suite")

def create_mock_dataset(input_dir: Path):
    """Creates mock data files in .txt, .csv, .json, and .md formats."""
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Text File
    txt_content = (
        "Organizational policy 101: Never share device-bound keys.\n"
        "Organizational policy 102: Always encrypt training datasets at rest.\n"
    )
    (input_dir / "policy.txt").write_text(txt_content, encoding="utf-8")
    
    # 2. Markdown File
    md_content = (
        "# Security Standard Operating Procedure\n\n"
        "LoRA adapter training must run locally on verified compute units.\n\n"
        "In-memory decryption prevents intermediate plaintext leakages.\n"
    )
    (input_dir / "procedures.md").write_text(md_content, encoding="utf-8")
    
    # 3. CSV File (with instruction/response columns)
    csv_content = (
        "instruction,input,output\n"
        "\"What is the model name?\",\"\",\"The framework is called Secure Device-Bound LoRA.\"\n"
        "\"How to handle encryption?\",\"\",\"Use AES-256-GCM chunked encryption stream.\"\n"
    )
    (input_dir / "qa.csv").write_text(csv_content, encoding="utf-8")
    
    # 4. JSON File (list of objects)
    json_data = [
        {"prompt": "Define PEFT.", "response": "PEFT stands for Parameter-Efficient Fine-Tuning."},
        {"text": "Random loose causal document content for language model pretraining."}
    ]
    with open(input_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f)

def run_tests():
    logger.info("Starting Secure LoRA Framework Test Suite...")
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as base_temp:
        base_path = Path(base_temp)
        input_dir = base_path / "raw_inputs"
        output_dir = base_path / "encrypted_outputs"
        decrypted_dir = base_path / "decrypted_outputs"
        decrypted_dir.mkdir()
        
        # 1. Create mock data
        create_mock_dataset(input_dir)
        logger.info(f"Mock raw files generated in {input_dir}")
        
        # 2. Generate secure key
        key = generate_key()
        key_file = base_path / "secret.key"
        key_file.write_bytes(key)
        logger.info("Key generated and stored temporarily.")
        
        # 3. Initialize pipeline
        pipeline = SecureDatasetPipeline(key)
        
        # 4. Encrypt dataset
        logger.info("Running encryption pipeline...")
        metadata = pipeline.encrypt_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_name="TestCorpDataset",
            version="1.0.0"
        )
        
        # 5. Assert files were created
        enc_file = output_dir / "encrypted_dataset.enc"
        meta_file = output_dir / "dataset_metadata.json"
        
        assert enc_file.exists(), "Encrypted file was not created!"
        assert meta_file.exists(), "Metadata file was not created!"
        logger.info("Verified: Encrypted file and metadata json exist.")
        
        # Check metadata attributes
        logger.info("Validating metadata schema structure...")
        assert metadata["dataset_name"] == "TestCorpDataset"
        assert metadata["encryption_status"] == "encrypted"
        assert metadata["encryption_algorithm"] == "AES-256-GCM"
        assert metadata["num_records"] > 0
        assert "encrypted_file_sha256" in metadata
        assert "encrypted_file_size_bytes" in metadata
        logger.info("Verified: Metadata matches expected parameters.")
        
        # 6. Streamingly decrypt and verify contents
        logger.info("Running streaming decryption validation (no plaintext output on disk)...")
        decrypted_records = list(pipeline.decrypt_records(enc_file))
        
        # Print records to inspect structure
        logger.info("Decrypted records sample:")
        for idx, rec in enumerate(decrypted_records[:3]):
            logger.info(f"  Record {idx+1}: {rec}")
            
        assert len(decrypted_records) == metadata["num_records"], "Mismatch in records count!"
        logger.info("Verified: Decrypted record count matches metadata record count.")
        
        # Validate that we successfully parsed the instruction format
        instruction_records = [r for r in decrypted_records if "instruction" in r]
        causal_records = [r for r in decrypted_records if "text" in r]
        
        assert len(instruction_records) > 0, "Failed to ingest instruction-tuning records!"
        assert len(causal_records) > 0, "Failed to ingest causal/plain text records!"
        logger.info(f"Verified: Found {len(instruction_records)} instruction-tuning and {len(causal_records)} causal records.")
        
        # 7. Validate decrypted temporary file context manager
        logger.info("Validating temporary decryption context manager...")
        with pipeline.decrypted_temp_file(enc_file) as temp_plaintext_file:
            assert temp_plaintext_file.exists(), "Temp plaintext file was not yielded!"
            # Read size/lines
            lines = temp_plaintext_file.read_text(encoding="utf-8").splitlines()
            assert len(lines) == metadata["num_records"]
            logger.info(f"Verified: Temp plaintext file exists with {len(lines)} lines during context block.")
            
        # Verify the file is now gone
        assert not temp_plaintext_file.exists(), "Temp plaintext file was NOT cleaned up!"
        logger.info("Verified: Temp plaintext file was shredded and removed automatically upon block exit.")
        
        # 8. Test Tamper Detection
        logger.info("Testing tamper detection and authenticated decryption constraints...")
        # Create a copy of the encrypted file
        corrupted_enc_file = base_path / "corrupted_dataset.enc"
        shutil.copy(enc_file, corrupted_enc_file)
        
        # Corrupt a single byte of the ciphertext (skip magic headers)
        with open(corrupted_enc_file, "r+b") as f:
            f.seek(32) # Seek past headers
            byte = f.read(1)
            # Flip bits
            corrupted_byte = bytes([byte[0] ^ 0xFF])
            f.seek(32)
            f.write(corrupted_byte)
            
        # Try to decrypt the corrupted file
        try:
            list(pipeline.decrypt_records(corrupted_enc_file))
            assert False, "Decryption succeeded on a corrupted payload! Critical vulnerability!"
        except ValueError as e:
            logger.info(f"Verified: Authenticated GCM successfully caught tamper. Threw expected error: {e}")
            
        # 9. Test key verification failure (wrong key)
        logger.info("Testing decryption with incorrect key...")
        wrong_key = generate_key()
        wrong_pipeline = SecureDatasetPipeline(wrong_key)
        try:
            list(wrong_pipeline.decrypt_records(enc_file))
            assert False, "Decryption succeeded with a wrong key! Critical vulnerability!"
        except ValueError as e:
            logger.info(f"Verified: Decryption failed as expected with wrong key: {e}")

        # 10. Test raw files shredding option
        logger.info("Testing raw files shredding option...")
        # Create another test pipeline with shred_raw=True
        raw_shred_dir = base_path / "raw_shred"
        create_mock_dataset(raw_shred_dir)
        
        shred_output_dir = base_path / "shred_outputs"
        pipeline.encrypt_dataset(
            input_dir=raw_shred_dir,
            output_dir=shred_output_dir,
            dataset_name="ShrededTest",
            shred_raw=True
        )
        
        # Confirm that the raw files inside raw_shred_dir are deleted/shredded
        remaining_files = list(raw_shred_dir.glob("*"))
        # Only subdirs could remain, but all ingested files should be removed
        for f in remaining_files:
            assert not f.is_file(), f"File {f.name} was not shredded and removed!"
            
        logger.info("Verified: All ingested raw files in input folder were shredded and deleted after encryption pipeline.")

    logger.info("==========================================")
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY! (10/10)")
    logger.info("==========================================")

if __name__ == "__main__":
    run_tests()
