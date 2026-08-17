# Sample Datasets

> **Dataset Safety Statement**:  
> "This dataset contains synthetic fictional identities and is not derived from real individuals."

This directory contains pre-configured sample datasets used for demonstration, testing, and Q&A evaluation:

| File | Description | PII Types Included | Synthetic Flag |
|------|-------------|-------------------|:---:|
| `sample_medical_phi.jsonl` | Hospital clinical notes with protected health information (PHI) | Names, Diagnoses, MRN, Dates | `synthetic: true` |
| `sample_pii_data.jsonl` | Corporate HR records | Names, Emails, Phone numbers, SSNs | `synthetic: true` |
| `synthetic_pii_benchmark.jsonl` | Synthetic benchmark records with PII redaction labels sampled from AI4Privacy generator | Mixed corporate & personal PII | `synthetic: true` |

---
*Note: Any dataset file passed into the SecureLoRA pipeline is automatically pre-processed using the `HybridPIIEngine` to mask sensitive entities prior to AES-256-GCM encryption and LoRA fine-tuning.*
