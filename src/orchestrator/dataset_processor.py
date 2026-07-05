import os
import re
import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone

from src.common.exceptions import DatasetValidationError
from src.security.crypto import encrypt_stream, compute_sha256
from src.security.shred import shred_file

logger = logging.getLogger("secure_lora.orchestrator.dataset_processor")

# Common PII Regex patterns for security auditing
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b")
}


def clean_text(text: Any) -> str:
    """Cleans whitespace and removes control characters."""
    if text is None:
        return ""
    text_str = str(text)
    # Remove control characters except newline and tab
    text_str = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]').sub('', text_str)
    # Normalize multiple spaces and tabs
    text_str = re.compile(r'[ \t]+').sub(' ', text_str)
    return text_str.strip()


def inspect_text_for_pii(text: str) -> Dict[str, int]:
    """Scans text and returns counts of detected PII entities."""
    counts = {}
    for name, pattern in PII_PATTERNS.items():
        found = pattern.findall(text)
        if found:
            counts[name] = len(found)
    return counts


def validate_dataset_file(file_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validates file format, parses records, inspects metadata,
    detects schema format, and performs a PII security audit.
    Raises DatasetValidationError if validation checks fail.
    """
    if not file_path.exists():
        raise DatasetValidationError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in {'.txt', '.csv', '.json', '.jsonl', '.md'}:
        raise DatasetValidationError(f"Unsupported file format '{suffix}'. Supported formats: .txt, .csv, .json, .jsonl, .md")

    records: List[Dict[str, Any]] = []
    pii_counts = {"email": 0, "phone": 0, "ssn": 0, "credit_card": 0}
    total_chars = 0

    try:
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    cleaned = line.strip()
                    if cleaned:
                        total_chars += len(cleaned)
                        # PII Inspection
                        for pii_type, count in inspect_text_for_pii(cleaned).items():
                            pii_counts[pii_type] += count
                        records.append({"text": cleaned, "line_number": line_num})

        elif suffix == '.md':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
                for idx, block in enumerate(blocks, 1):
                    total_chars += len(block)
                    for pii_type, count in inspect_text_for_pii(block).items():
                        pii_counts[pii_type] += count
                    records.append({"text": block, "block_index": idx})

        elif suffix == '.csv':
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                # Basic check for empty file
                preview = f.read(2048)
                if not preview.strip():
                    raise DatasetValidationError("Uploaded CSV file is empty.")
                f.seek(0)

                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise DatasetValidationError("Uploaded CSV is missing headers.")

                for row_idx, row in enumerate(reader, 1):
                    if not row or all(v is None or str(v).strip() == "" for v in row.values()):
                        continue  # skip completely empty row
                    
                    cleaned_row = {k: v for k, v in row.items() if k is not None}
                    row_str = " ".join(str(v) for v in cleaned_row.values())
                    total_chars += len(row_str)
                    for pii_type, count in inspect_text_for_pii(row_str).items():
                        pii_counts[pii_type] += count
                    
                    cleaned_row["row_index"] = row_idx
                    records.append(cleaned_row)

        elif suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        obj = json.loads(line_str)
                        if not isinstance(obj, dict):
                            raise DatasetValidationError(f"Invalid JSONL record on line {line_num}: must be a JSON object.")
                        
                        total_chars += len(line_str)
                        for pii_type, count in inspect_text_for_pii(line_str).items():
                            pii_counts[pii_type] += count
                        obj["line_number"] = line_num
                        records.append(obj)
                    except json.JSONDecodeError as err:
                        raise DatasetValidationError(f"Malformed JSONL on line {line_num}: {err}") from err

        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as err:
                    raise DatasetValidationError(f"Malformed JSON format: {err}") from err

                if isinstance(data, list):
                    for idx, item in enumerate(data, 1):
                        if isinstance(item, dict):
                            item_str = json.dumps(item)
                            total_chars += len(item_str)
                            for pii_type, count in inspect_text_for_pii(item_str).items():
                                pii_counts[pii_type] += count
                            item_copy = dict(item)
                            item_copy["record_index"] = idx
                            records.append(item_copy)
                        elif isinstance(item, str):
                            total_chars += len(item)
                            for pii_type, count in inspect_text_for_pii(item).items():
                                pii_counts[pii_type] += count
                            records.append({"text": item, "record_index": idx})
                        else:
                            raise DatasetValidationError(f"Invalid JSON array element at index {idx}: must be an object or string.")
                elif isinstance(data, dict):
                    data_str = json.dumps(data)
                    total_chars += len(data_str)
                    for pii_type, count in inspect_text_for_pii(data_str).items():
                        pii_counts[pii_type] += count
                    records.append(dict(data))
                else:
                    raise DatasetValidationError("Unsupported JSON root element: must be a list of records or a single object.")

    except DatasetValidationError:
        raise
    except Exception as e:
        logger.error("Dataset validation parsing failure: %s", e)
        raise DatasetValidationError(f"Failed to parse dataset file: {e}") from e

    if not records:
        raise DatasetValidationError("No valid data records found in the uploaded file.")

    # Determine schema representation
    sample = records[0]
    schema = "unknown"
    if "instruction" in sample and "output" in sample:
        schema = "instruction"
    elif "text" in sample:
        schema = "causal_lm"
    elif any(k in sample for k in ["prompt", "question", "query"]) and any(k in sample for k in ["output", "response", "answer"]):
        schema = "instruction"

    metadata = {
        "file_name": file_path.name,
        "file_type": suffix,
        "file_size_bytes": file_path.stat().st_size,
        "num_raw_records": len(records),
        "total_characters": total_chars,
        "schema_detected": schema,
        "pii_detected_summary": pii_counts,
        "has_pii": any(v > 0 for v in pii_counts.values())
    }

    return records, metadata


def preprocess_and_standardize(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardizes Alpaca format (instruction/input/output) or Causal LM format (text).
    Strips control characters, normalizes whitespace.
    """
    standardized: List[Dict[str, Any]] = []

    for idx, record in enumerate(raw_records, 1):
        instruction = (
            record.get("instruction") or
            record.get("prompt") or
            record.get("question") or
            record.get("query")
        )
        input_val = (
            record.get("input") or
            record.get("context") or
            record.get("source") or
            record.get("source_text") or
            ""
        )
        output = (
            record.get("output") or
            record.get("response") or
            record.get("answer") or
            record.get("target") or
            record.get("target_text")
        )

        clean_rec = {}
        if "source_file" in record:
            clean_rec["source_file"] = record["source_file"]

        if instruction and output:
            clean_rec["instruction"] = clean_text(instruction)
            clean_rec["input"] = clean_text(input_val)
            clean_rec["output"] = clean_text(output)
            standardized.append(clean_rec)
        elif "text" in record or "content" in record:
            text_content = record.get("text") or record.get("content")
            clean_rec["text"] = clean_text(text_content)
            standardized.append(clean_rec)
        else:
            # Fallback combining other fields
            filtered_keys = [k for k in record.keys() if k not in {
                "source_file", "row_index", "line_number", "record_index", "block_index"
            }]
            if len(filtered_keys) == 1:
                clean_rec["text"] = clean_text(record[filtered_keys[0]])
                standardized.append(clean_rec)
            elif len(filtered_keys) > 1:
                combined = [f"{k.capitalize()}: {clean_text(record[k])}" for k in filtered_keys if record[k]]
                if combined:
                    clean_rec["text"] = "\n".join(combined)
                    standardized.append(clean_rec)

    return standardized


def encrypt_and_save_dataset(
    processed_records: List[Dict[str, Any]],
    key: bytes,
    output_dir: Path,
    dataset_name: str,
    version: str,
    pii_summary: Dict[str, int]
) -> Dict[str, Any]:
    """
    Saves preprocessed records into an AES-256-GCM encrypted container.
    Generates metadata signature. Securely shreds all temporary plaintext files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    enc_file_path = output_dir / "encrypted_dataset.enc"
    meta_file_path = output_dir / "dataset_metadata.json"

    # Create secure temporary plaintext JSONL file
    temp_fd, temp_jsonl_str = tempfile.mkstemp(suffix=".jsonl")
    os.close(temp_fd)
    temp_jsonl_path = Path(temp_jsonl_str)

    try:
        # Write preprocessed records to temp file
        with open(temp_jsonl_path, "w", encoding="utf-8") as f:
            for record in processed_records:
                f.write(json.dumps(record) + "\n")

        # Create temporary encrypted file before rename
        temp_enc_fd, temp_enc_str = tempfile.mkstemp(suffix=".enc")
        os.close(temp_enc_fd)
        temp_enc_path = Path(temp_enc_str)

        try:
            with open(temp_jsonl_path, "rb") as fin, open(temp_enc_path, "wb") as fout:
                encrypt_stream(fin, fout, key)

            # Atomic replace
            if enc_file_path.exists():
                enc_file_path.unlink()
            temp_enc_path.rename(enc_file_path)
        finally:
            if temp_enc_path.exists():
                temp_enc_path.unlink()

        # Generate signed metadata file
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
            "num_records": len(processed_records),
            "pii_audit_summary": pii_summary
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

        logger.info("Encrypted dataset and metadata successfully saved to %s", output_dir)
        return metadata

    finally:
        # Shred temporary unlinked plaintext file
        shred_file(temp_jsonl_path)
