import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Generator, Union
from contextlib import contextmanager

from src.phase1.ingestion import ingest_directory
from src.phase1.preprocessing import preprocess_dataset
from src.security import (
    encrypt_stream,
    decrypt_stream,
    decrypt_generator,
    decrypted_temporary_file,
    shred_file,
    compute_sha256
)

logger = logging.getLogger("secure_lora.phase1.pipeline")


class SecureDatasetPipeline:
    """
    Coordinates the ingestion, preprocessing, encryption, storage, and cleanup
    of datasets for the Secure LoRA Framework.
    """

    def __init__(self, key: bytes):
        """Initializes the pipeline with a 256-bit AES key."""
        if len(key) != 32:
            raise ValueError("Encryption key must be exactly 32 bytes (256 bits).")
        self.key = key

    def encrypt_dataset(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        dataset_name: str,
        version: str = "1.0.0",
        shred_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Runs the full secure dataset preparation pipeline.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        enc_file_path = output_path / "encrypted_dataset.enc"
        meta_file_path = output_path / "dataset_metadata.json"

        # Temp file path in secure manner
        import tempfile
        temp_fd, temp_jsonl_str = tempfile.mkstemp(suffix=".jsonl")
        os.close(temp_fd)
        temp_jsonl_path = Path(temp_jsonl_str)

        raw_count = 0
        processed_count = 0
        sources = set()
        schema_type = "unknown"

        try:
            logger.info("Step 1: Ingesting raw directory...")
            raw_records = ingest_directory(input_path)
            raw_count = len(raw_records)

            if raw_count == 0:
                raise ValueError(f"No valid raw files found in {input_dir}.")

            logger.info("Step 2: Preprocessing and standardizing records...")
            processed_records = preprocess_dataset(raw_records)
            processed_count = len(processed_records)

            if processed_count == 0:
                raise ValueError("No records left after preprocessing filter.")

            # Determine schema format
            sample = processed_records[0]
            if "instruction" in sample and "output" in sample:
                schema_type = "instruction"
            elif "text" in sample:
                schema_type = "causal_lm"

            for rec in processed_records:
                if "source_file" in rec:
                    sources.add(rec["source_file"])

            logger.info("Step 3: Writing preprocessed records to temporary JSONL...")
            with open(temp_jsonl_path, "w", encoding="utf-8") as f:
                for record in processed_records:
                    f.write(json.dumps(record) + "\n")

            logger.info("Step 4: Encrypting temporary file chunk-by-chunk...")
            # Write to a temporary file before atomic rename to prevent half-written file leaks
            temp_enc_fd, temp_enc_str = tempfile.mkstemp(suffix=".enc")
            os.close(temp_enc_fd)
            temp_enc_path = Path(temp_enc_str)

            try:
                with open(temp_jsonl_path, "rb") as fin, open(temp_enc_path, "wb") as fout:
                    encrypt_stream(fin, fout, self.key)

                # Atomic replace
                if enc_file_path.exists():
                    enc_file_path.unlink()
                temp_enc_path.rename(enc_file_path)
            finally:
                if temp_enc_path.exists():
                    temp_enc_path.unlink()

            logger.info("Step 5: Generating metadata...")
            file_size = enc_file_path.stat().st_size
            checksum = compute_sha256(enc_file_path)

            metadata = {
                "dataset_name": dataset_name,
                "version": version,
                "encryption_status": "encrypted",
                "encryption_algorithm": "AES-256-GCM",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "encrypted_file_size_bytes": file_size,
                "encrypted_file_sha256": checksum,
                "num_records": processed_count,
                "raw_records_ingested": raw_count,
                "schema_type": schema_type,
                "source_files": sorted(list(sources))
            }

            # Atomic write of metadata
            temp_meta_fd, temp_meta_str = tempfile.mkstemp(suffix=".json")
            os.close(temp_meta_fd)
            temp_meta_path = Path(temp_meta_str)
            try:
                with open(temp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
                if meta_file_path.exists():
                    meta_file_path.unlink()
                temp_meta_path.rename(meta_file_path)
            finally:
                if temp_meta_path.exists():
                    temp_meta_path.unlink()

            logger.info("Encryption pipeline completed successfully.")

            # Step 7: Optionally shred original raw inputs
            if shred_raw:
                logger.info("Shredding raw inputs as requested...")
                for path in input_path.glob("**/*"):
                    if path.is_file() and path.name in sources:
                        shred_file(path)

            return metadata

        finally:
            # Step 6: Guarantee temporary file is shredded
            logger.info("Executing secure cleanup of intermediate plaintext files...")
            shred_file(temp_jsonl_path)

    def decrypt_dataset_to_file(
        self,
        encrypted_file_path: Union[str, Path],
        output_file_path: Union[str, Path]
    ) -> None:
        """
        Decrypts an encrypted dataset file back to a plaintext file.
        """
        enc_path = Path(encrypted_file_path)
        out_path = Path(output_file_path)

        with open(enc_path, "rb") as fin, open(out_path, "wb") as fout:
            decrypt_stream(fin, fout, self.key)
        logger.info("Dataset successfully decrypted to: %s", out_path)

    def decrypt_records(self, encrypted_file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """
        Returns a Python Generator yielding decrypted records.
        """
        yield from decrypt_generator(encrypted_file_path, self.key)

    @contextmanager
    def decrypted_temp_file(self, encrypted_file_path: Union[str, Path]):
        """
        Returns a context manager yielding a temporary plaintext file path.
        """
        with decrypted_temporary_file(encrypted_file_path, self.key) as temp_path:
            yield temp_path
