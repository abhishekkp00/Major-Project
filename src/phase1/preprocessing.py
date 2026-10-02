import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("secure_lora.phase1.preprocessing")

from src.security.pii_engine import deobfuscate_text

# ---------------------------------------------------------------------------
# Lazy singleton for the PII masker — mirrors the orchestrator's _get_pii_masker()
# pattern so startup is never blocked by model loading.
# ---------------------------------------------------------------------------
_pii_masker = None


def _get_pii_masker():
    """Lazy-loads mask_pii_advanced once and caches it for the process lifetime."""
    global _pii_masker
    if _pii_masker is None:
        from src.security.pii_engine import mask_pii_advanced  # noqa: PLC0415
        _pii_masker = mask_pii_advanced
    return _pii_masker


def _mask(text: str) -> str:
    """
    Applies the canonical HybridPIIEngine masking policy to *text*.
    Returns the masked string; falls back to the original on error so that
    a transient masker failure never silently drops records.
    """
    if not text:
        return text
    try:
        masked, _counts = _get_pii_masker()(text)
        return masked
    except Exception as exc:  # pragma: no cover
        logger.warning("PII masking failed — returning field as-is: %s", exc)
        return text


def clean_text(text: Any) -> str:
    """
    Cleans text by stripping whitespace, normalizing multiple spaces,
    removing control characters, and de-obfuscating hidden PII tokens.
    """
    if text is None:
        return ""
    text_str = str(text)
    # Deobfuscate unicode, URL encoding, [at]/[dot] patterns
    text_str = deobfuscate_text(text_str)
    # Normalize multiple spaces and tabs to a single space
    text_str = re.sub(r'[ \t]+', ' ', text_str)
    return text_str.strip()


def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single record and standardizes its schema for LLM fine-tuning.
    Prefers:
      - Alpaca Format: {'instruction': ..., 'input': ..., 'output': ...}
      - Causal LM Format: {'text': ...}

    PII sanitization policy (mirrors orchestrator/dataset_processor.py):
      1. De-obfuscate hidden tokens (clean_text / deobfuscate_text).
      2. Normalize whitespace (clean_text).
      3. Run HybridPIIEngine masking via mask_pii_advanced (_mask).
    Masking happens BEFORE the record is handed off to any encryption step.
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
        # clean_text first (deobfuscate + whitespace), then mask PII
        standardized["instruction"] = _mask(clean_text(instruction))
        standardized["input"]       = _mask(clean_text(input_val)) if input_val else ""
        standardized["output"]      = _mask(clean_text(output))
    elif "text" in record or "content" in record:
        text_content = record.get("text") or record.get("content")
        standardized["text"] = _mask(clean_text(text_content))
    else:
        filtered_keys = [k for k in record.keys() if k not in {"source_file", "row_index", "line_number", "record_index", "block_index"}]

        if len(filtered_keys) == 1:
            standardized["text"] = _mask(clean_text(record[filtered_keys[0]]))
        elif len(filtered_keys) > 1:
            combined = []
            for k in filtered_keys:
                val = record[k]
                if val:
                    combined.append(f"{k.capitalize()}: {clean_text(val)}")
            if combined:
                standardized["text"] = _mask("\n".join(combined))

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
