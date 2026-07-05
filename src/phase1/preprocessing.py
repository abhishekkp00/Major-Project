import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger("secure_lora.phase1.preprocessing")


def clean_text(text: Any) -> str:
    """
    Cleans text by stripping whitespace, normalizing multiple spaces,
    and removing control characters.
    """
    if text is None:
        return ""
    text_str = str(text)
    # Remove control characters (except newline and tab)
    text_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text_str)
    # Normalize multiple spaces and tabs to a single space
    text_str = re.sub(r'[ \t]+', ' ', text_str)
    return text_str.strip()


def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single record and standardizes its schema for LLM fine-tuning.
    Prefers:
      - Alpaca Format: {'instruction': ..., 'input': ..., 'output': ...}
      - Causal LM Format: {'text': ...}
    """
    standardized = {}

    # Extract original source tracking metadata
    if "source_file" in record:
        standardized["source_file"] = record["source_file"]

    # Attempt to extract instruction-tuning fields
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
        record.get("source_text")
    )
    if input_val == record.get("source_file"):
        input_val = ""

    output = (
        record.get("output") or
        record.get("response") or
        record.get("answer") or
        record.get("target") or
        record.get("target_text")
    )

    if instruction and output:
        standardized["instruction"] = clean_text(instruction)
        standardized["input"] = clean_text(input_val) if input_val else ""
        standardized["output"] = clean_text(output)
    elif "text" in record or "content" in record:
        text_content = record.get("text") or record.get("content")
        standardized["text"] = clean_text(text_content)
    else:
        filtered_keys = [k for k in record.keys() if k not in {"source_file", "row_index", "line_number", "record_index", "block_index"}]

        if len(filtered_keys) == 1:
            standardized["text"] = clean_text(record[filtered_keys[0]])
        elif len(filtered_keys) > 1:
            combined = []
            for k in filtered_keys:
                val = record[k]
                if val:
                    combined.append(f"{k.capitalize()}: {clean_text(val)}")
            if combined:
                standardized["text"] = "\n".join(combined)

    return standardized


def preprocess_dataset(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters and normalizes raw dataset records, removing empty lines or empty records.
    """
    processed_records = []

    for record in raw_records:
        proc = preprocess_record(record)
        has_instruction_content = proc.get("instruction") and proc.get("output")
        has_text_content = proc.get("text")

        if has_instruction_content or has_text_content:
            processed_records.append(proc)

    logger.info("Preprocessed dataset: %d raw records -> %d clean records", len(raw_records), len(processed_records))
    return processed_records
