"""
chat_engine.py
==============
Privacy-Preserving Chat Engine for SecureLoRA.

Two-mode operation:
  MODE A (Model Available): Uses the fine-tuned PEFT adapter loaded by Phase 4
           to generate real LLM responses to questions. All responses are
           passed through a regex PII guardrail before display.

  MODE B (No Model): Falls back to aggregate statistical analysis over the
           uploaded dataset records — still privacy-safe and useful.

Privacy guardrail (always active):
  - Blocks any question that seeks individual-level PII (intent classifier)
  - Runs regex-based PII masking on every generated answer before it is returned
"""

import re
import json
import logging
import collections
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("secure_lora.chat_engine")

# ---------------------------------------------------------------------------
# 0. Thread-safe model registry — populated by Phase 4 adapter loader
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_chat_model   = None   # The loaded PEFT model (or None)
_chat_tokenizer = None  # Matching tokenizer (or None)
_chat_model_name = None  # Base model name string


def register_model(model, tokenizer, model_name: str):
    """Called by Phase 4 after a successful deployment verification."""
    global _chat_model, _chat_tokenizer, _chat_model_name
    with _model_lock:
        _chat_model = model
        _chat_tokenizer = tokenizer
        _chat_model_name = model_name
    logger.info("Chat engine: fine-tuned model registered (%s)", model_name)


def get_registered_model():
    with _model_lock:
        return _chat_model, _chat_tokenizer, _chat_model_name


# ---------------------------------------------------------------------------
# 1. Lightweight regex-only PII guardrail (no ML needed — fast, synchronous)
# ---------------------------------------------------------------------------

try:
    from src.security.pii_engine import ENTITIES_PATTERNS, deobfuscate_text as _deobfuscate

    def _regex_mask(text: str):
        cleaned = _deobfuscate(text)
        counts = {}
        masked = cleaned
        for entity_type, (pattern, replacement) in ENTITIES_PATTERNS.items():
            found = pattern.findall(masked)
            if found:
                counts[entity_type] = len(found)
                masked = pattern.sub(replacement, masked)
        return masked, counts

except Exception:
    # Bare-minimum fallback if pii_engine can't be imported
    _BARE_PATTERNS = [
        (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[EMAIL]"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        (re.compile(r"\b(?:\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "[TEL]"),
    ]

    def _regex_mask(text: str):
        masked, counts = text, {}
        for pat, repl in _BARE_PATTERNS:
            hits = pat.findall(masked)
            if hits:
                counts[repl] = len(hits)
                masked = pat.sub(repl, masked)
        return masked, counts


# ---------------------------------------------------------------------------
# 2. PII-seeking intent classifier
# ---------------------------------------------------------------------------

_INDIVIDUAL_PII_PATTERNS = [
    re.compile(r"\b(tell me|give me|show me|reveal|expose|leak|find|lookup|get)\b.{0,30}\b(ssn|social security|passport|credit card|card number|password|private key|secret key)\b", re.I),
    re.compile(r"\b(what is|tell me)\b.{0,30}\b(ssn|social security|passport|credit card|password)\b.{0,30}\b(of|for)\b", re.I),
    re.compile(r"\b(reveal|expose|leak|show)\b.{0,30}\b(personal info|private data|contact details)\b.{0,30}\b(of|for)\b\s*[A-Z][a-z]+", re.I),
    re.compile(r"\bidentity of\b\s*[A-Z][a-z]+", re.I),
]

_PRIVACY_BLOCK_RESPONSE = (
    "🔒 **Privacy Guardrail Activated**\n\n"
    "This question appears to request personally identifiable information (PII) "
    "about a specific individual. SecureLoRA's privacy policy prohibits revealing "
    "individual-level sensitive data.\n\n"
    "You can ask **aggregate questions** instead, such as:\n"
    "- _What is the most common medical condition in this dataset?_\n"
    "- _How many patient records are in the dataset?_\n"
    "- _What trends exist across the full dataset?_"
)


def _is_pii_seeking(question: str) -> bool:
    for pat in _INDIVIDUAL_PII_PATTERNS:
        if pat.search(question):
            return True
    return False


# ---------------------------------------------------------------------------
# 3. Dataset loader helpers
# ---------------------------------------------------------------------------

def load_records_from_job(job_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for candidate in [
        job_dir / "processed_records.jsonl",
        job_dir / "raw_inputs",
    ]:
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            except Exception as e:
                logger.warning("Could not load records from %s: %s", candidate, e)
            if records:
                return records
        elif candidate.is_dir():
            for fpath in candidate.iterdir():
                if fpath.is_file() and fpath.suffix in {".jsonl", ".json", ".txt", ".csv"}:
                    try:
                        for line in fpath.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    records.append(json.loads(line))
                                except json.JSONDecodeError:
                                    records.append({"text": line})
                    except Exception as e:
                        logger.warning("Could not read %s: %s", fpath, e)
    return records


def load_records_from_jsonl(raw_jsonl: str) -> List[Dict[str, Any]]:
    records = []
    for line in raw_jsonl.strip().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"text": line})
    return records


# ---------------------------------------------------------------------------
# 4. Aggregate analytics (Mode B — no model)
# ---------------------------------------------------------------------------

def _extract_text_fields(record: Dict[str, Any]) -> str:
    parts = []
    for v in record.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, dict)):
            parts.append(json.dumps(v))
    return " ".join(parts)


def compute_dataset_analytics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    total = len(records)
    field_keys = list(records[0].keys()) if records else []
    value_freq: Dict[str, collections.Counter] = {}
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, str) and len(v) < 300:
                value_freq.setdefault(k, collections.Counter())[v] += 1
    masked_tag_counts: collections.Counter = collections.Counter()
    tag_pattern = re.compile(r"\[([A-Z_]+)\]")
    for rec in records:
        text = _extract_text_fields(rec)
        for tag in tag_pattern.findall(text):
            masked_tag_counts[tag] += 1
    top_values: Dict[str, List] = {}
    for field, counter in value_freq.items():
        if any(kw in field.lower() for kw in ["name", "email", "phone", "ssn", "address", "dob", "password", "token"]):
            continue
        top_values[field] = counter.most_common(5)
    return {
        "total_records": total,
        "fields": field_keys,
        "top_values_per_field": top_values,
        "masked_entity_tag_counts": dict(masked_tag_counts),
    }


_INTENT_KEYWORDS = {
    "condition":    ["condition", "diagnosis", "disease", "illness", "ailment", "medical", "health", "treatment", "prescription", "medication"],
    "distribution": ["distribution", "spread", "breakdown", "majority", "most", "common", "frequent", "popular", "dominant", "average"],
    "count":        ["how many", "count", "number of", "total", "records", "patients", "employees", "users"],
    "fields":       ["what fields", "what columns", "what data", "dataset contain", "what info", "what information"],
    "pii_masked":   ["pii", "masked", "redacted", "sensitive", "personal", "private", "protected", "hidden"],
    "tags":         ["tags", "entities", "entity types", "masked types", "what was masked"],
}


def _classify_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"


def _clean_entity_text(text: str) -> str:
    """
    Cleans raw training record text by removing instruction headers,
    raw PII placeholders like [MASKED_NAME], [MASKED_MRN], [MASKED_DATE],
    and extracting the underlying natural medical/corporate topics.
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Remove instruction preambles
    text = re.sub(r"^(Redact|Scrub|Mask)\s+(PHI|PII|HIPAA|patient|data|identifiers)\s*(from|in)?\s*(this|the)?\s*(clinical record|email|record|data)?:\s*", "", text, flags=re.I)
    text = re.sub(r"^Patient\s+data:\s*", "", text, flags=re.I)

    # 2. Remove [MASKED_...] and [REDACTED_...] placeholder tokens
    text = re.sub(r"\[(MASKED_[A-Z0-9_]+|REDACTED_[A-Z0-9_]+|[A-Z_]+)\]", "", text)

    # 3. Clean up extra punctuation/spaces
    text = re.sub(r"\s+([,.:;])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.:;-")

    # 4. Extract domain-specific medical/clinical/corporate terms dynamically
    medical_patterns = [
        r"\b(acute coronary syndrome|covid-19|hypertension|diabetes|asthma|cancer|arrhythmia|pneumonia|lactic acidosis|lisinopril|myocardial infarction|stroke)\b",
        r"\b(project aurora|wire transfer|quarterly audit|financial report|security incident|system outage|data breach)\b"
    ]
    extracted = []
    for pat in medical_patterns:
        matches = re.findall(pat, text, flags=re.I)
        for m in matches:
            formatted = m.title() if m.lower() != "covid-19" else "COVID-19"
            if formatted not in extracted:
                extracted.append(formatted)
    
    if extracted:
        return ", ".join(extracted)

    # If no specific key terms matched, clean remaining descriptive text
    cleaned_clause = re.sub(r"\b(Patient|Case|Discharged|MRN|born|admitted on|scheduled at|tested positive for)\b", "", text, flags=re.I)
    cleaned_clause = re.sub(r"\s+", " ", cleaned_clause).strip(" ,.:;-")

    if cleaned_clause and len(cleaned_clause) > 3:
        return cleaned_clause[:100]

    return "Clinical / Enterprise Record"


def _answer_from_analytics(question: str, records: List[Dict[str, Any]]) -> str:
    analytics = compute_dataset_analytics(records)
    intent = _classify_intent(question)
    parts = []

    if intent == "count":
        parts.append(f"📊 **Dataset Record Count**\n\nThis dataset contains **{analytics['total_records']} records** in total.")

    elif intent == "fields":
        fields = analytics.get("fields", [])
        safe_fields = [f for f in fields if not any(kw in f.lower() for kw in ["name", "email", "phone", "ssn", "password"])]
        parts.append(f"📋 **Dataset Structure**\n\nThe dataset contains **{len(fields)} fields**.")
        if safe_fields:
            parts.append("Safe (non-PII) fields: `" + "`, `".join(safe_fields[:10]) + "`.")
        parts.append("Fields containing PII have been **masked** before training.")

    elif intent in ("pii_masked", "tags"):
        tag_counts = analytics.get("masked_entity_tag_counts", {})
        if tag_counts:
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            parts.append("🛡️ **PII Masking Summary**\n\nSensitive entity types detected and masked:\n")
            for tag, cnt in sorted_tags:
                parts.append(f"  - `[{tag}]` — **{cnt}** occurrences masked")
        else:
            parts.append("🛡️ No masked PII tags found in visible text fields.")

    elif intent in ("condition", "distribution", "general"):
        top_vals = analytics.get("top_values_per_field", {})
        priority_fields = ["condition", "diagnosis", "output", "instruction", "text", "clinical_note", "category", "label", "department", "status"]
        answered = False
        for pf in priority_fields:
            for field, counter in top_vals.items():
                if pf in field.lower() and counter:
                    top = counter[:5]
                    parts.append(f"📈 **Distribution of Clinical & Analytical Topics across {analytics['total_records']} records:**\n")
                    for val, cnt in top:
                        pct = round(100 * cnt / analytics["total_records"], 1)
                        clean_topic = _clean_entity_text(str(val))
                        parts.append(f"  - **{clean_topic}** — {cnt} records ({pct}%)")
                    answered = True
                    break
            if answered:
                break
        if not answered and top_vals:
            field, counter = next(iter(top_vals.items()))
            parts.append(f"📈 **Top Topic Distribution** (out of {analytics['total_records']} records):\n")
            for val, cnt in counter[:5]:
                clean_topic = _clean_entity_text(str(val))
                parts.append(f"  - **{clean_topic}** — {cnt} records")
        if not parts:
            parts.append(
                f"📊 This dataset has **{analytics['total_records']} records** with {len(analytics['fields'])} fields.\n\n"
                "Try asking about specific fields, trends, or PII masking statistics."
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 5. Mode A — LLM inference using the fine-tuned PEFT adapter
# ---------------------------------------------------------------------------

def _answer_with_model(question: str, records: List[Dict[str, Any]]) -> str:
    """Runs the fine-tuned model on the question and returns the sanitized output."""
    import torch
    from src.common.config_loader import config

    model, tokenizer, model_name = get_registered_model()
    if model is None or tokenizer is None:
        return _answer_from_analytics(question, records)

    # Build a brief dataset context summary (aggregate only — no raw PII)
    analytics = compute_dataset_analytics(records)
    context_lines = [f"Dataset: {analytics.get('total_records', 'unknown')} records."]
    top_vals = analytics.get("top_values_per_field", {})
    for field, counter in list(top_vals.items())[:3]:
        top = counter[:3]
        context_lines.append(f"Top {field}: " + ", ".join(f"{v}({c})" for v, c in top))
    context = " | ".join(context_lines)

    prompt = (
        f"You are a privacy-preserving data analyst. "
        f"Dataset context: {context}\n"
        f"Question: {question}\n"
        f"Answer (aggregate statistics only, no individual PII):"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        max_new_tokens = int(config.max_new_tokens)

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.1,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if not generated:
            return _answer_from_analytics(question, records)

        return generated

    except Exception as e:
        logger.warning("LLM inference failed (%s), falling back to analytics.", e)
        return _answer_from_analytics(question, records)


# ---------------------------------------------------------------------------
# 6. Public API
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    records: List[Dict[str, Any]]
) -> Tuple[str, str, bool]:
    """
    Returns (answer_text, privacy_status, was_blocked).

    privacy_status: 'SAFE' | 'BLOCKED' | 'GUARDED'
    was_blocked:    True if a PII-seeking question was rejected
    """
    if not question.strip():
        return "Please ask a question about your dataset.", "SAFE", False

    # Step 1: Block PII-seeking questions at the gate
    if _is_pii_seeking(question):
        return _PRIVACY_BLOCK_RESPONSE, "BLOCKED", True

    if not records:
        return (
            "No dataset is currently loaded. Please upload a dataset via the Pipeline tab first.",
            "SAFE",
            False,
        )

    # Step 2: Generate answer — use fine-tuned model if available, else analytics
    model, tokenizer, _ = get_registered_model()
    if model is not None:
        raw_answer = _answer_with_model(question, records)
    else:
        raw_answer = _answer_from_analytics(question, records)

    # Step 3: Final regex PII guardrail on the output — catches any accidental leakage
    guarded_answer, leaked_counts = _regex_mask(raw_answer)
    privacy_status = "GUARDED" if leaked_counts else "SAFE"

    return guarded_answer, privacy_status, False
