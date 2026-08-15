# Sample Datasets

This directory contains pre-configured sample datasets used for demonstration, testing, and Q&A evaluation:

| File | Description | PII Types Included |
|------|-------------|-------------------|
| `sample_medical_phi.jsonl` | Hospital clinical notes with protected health information (PHI) | Names, Diagnoses, MRN, Dates |
| `sample_pii_data.jsonl` | Corporate HR records | Names, Emails, Phone numbers, SSNs |
| `real_world_pii.jsonl` | Real-world benchmark records with PII redaction labels | Mixed corporate & personal PII |

---
*Note: Any dataset file passed into the SecureLoRA pipeline is automatically pre-processed using the `HybridPIIEngine` to mask sensitive entities prior to AES-256-GCM encryption and LoRA fine-tuning.*
