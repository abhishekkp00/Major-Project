# SECURELORA: DATASET & PRIVACY SAFETY AUDIT REPORT (PHASE 4)

**Date of Audit**: August 16, 2026  
**Repository**: `https://github.com/abhishekkp00/Major-Project`  
**Audit Scope**: Complete inspection of committed datasets, sample files, test fixtures, secret detection scan, external dataset governance, and synthetic data verification.

---

## 1. Executive Summary

A comprehensive dataset safety and secret audit was conducted across all committed repository files. **Zero real-world PII, zero production credentials, and zero private cryptographic keys exist in this repository.**

All sample benchmark files committed to the repository have been verified as **100% synthetic fictional data**. To eliminate ambiguity, files previously labeled `real_world_pii.jsonl` have been audited, updated with explicit `"synthetic": true` record flags, and aligned with `synthetic_pii_benchmark.jsonl`.

---

## 2. Datasets Inspected

Every data file, sample fixture, script, and dataset loader in the repository was inspected:

| File / Location | File Type | Record Count | Data Domain | Audit Finding | Action Taken |
|---|---|:---:|---|---|---|
| `samples/sample_pii_data.jsonl` | JSONL | 3 | Corporate HR / Emails | 100% Synthetic test fixture | Added `"synthetic": true` flag |
| `samples/sample_medical_phi.jsonl` | JSONL | 3 | Hospital Clinical Notes | 100% Synthetic test fixture | Added `"synthetic": true` flag |
| `synthetic_pii_benchmark.jsonl` | JSONL | 10 | AI4Privacy Generator Sample | 100% Synthetic benchmark | Created & added `"synthetic": true` flag |
| `samples/synthetic_pii_benchmark.jsonl` | JSONL | 10 | AI4Privacy Generator Sample | 100% Synthetic benchmark | Created & added `"synthetic": true` flag |
| `real_world_pii.jsonl` (legacy) | JSONL | 10 | AI4Privacy Generator Sample | 100% Synthetic benchmark | Updated with `"synthetic": true` flag |
| `src/data_sources/ai4privacy_loader.py` | Python Adapter | Dynamic | HuggingFace Open Web PII | External dataset loader | Configured HF API streaming + offline fallback |
| `src/data_sources/synthea_loader.py` | Python Adapter | Dynamic | Synthea Synthetic Patient Records | External synthetic loader | Configured streaming EHR parser |
| `src/data_sources/synthetic_loader.py` | Python Adapter | Dynamic | Controlled Synthetic Generator | Internal synthetic benchmark | Configured deterministic random generator |

---

## 3. Special Case Audit: `real_world_pii.jsonl` Verification

### Audit Determination
Records inside `real_world_pii.jsonl` were inspected line-by-line. The records contain randomized obfuscation strings (e.g. `vtpkbqcutaxb799@yahoo.com`, `8c75:fe4b:fa9e:b914:fca8:fbad:d89:4e6f`, `evxnizmsc1999`) originating from the AI4Privacy open synthetic benchmark generator. **They do NOT contain data from real individuals.**

### Actions Taken
1. Created `synthetic_pii_benchmark.jsonl` as the primary unambiguous benchmark dataset file.
2. Updated every record in `synthetic_pii_benchmark.jsonl` (and legacy fallback copies) to explicitly contain `"synthetic": true`.
3. Added the mandatory **Dataset Safety Statement** across dataset documentation (`samples/README.md`, `docs/DATASETS.md`):
   > *"This dataset contains synthetic fictional identities and is not derived from real individuals."*
4. Refactored Python dataset loaders (`src/data_sources/synthetic_loader.py`, `src/data_sources/ai4privacy_loader.py`), orchestration routing (`src/orchestrator/service.py`), script generators (`scripts/download_real_pii.py`, `scripts/download_training_data.py`), and web dashboard static routes (`src/evaluation/dashboard.py`) to reference `synthetic_pii_benchmark.jsonl`.

---

## 4. External Dataset Governance & Licensing

### 4.1 AI4Privacy PII-Masking Benchmark (`ai4privacy/pii-masking-300k`)
- **Official Source**: Hugging Face Hub — [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)
- **Licensing**: Apache-2.0 / CC-BY-4.0
- **Intended Use**: Academic and research evaluation of PII detection precision, recall, and F1-score.
- **Repository Storage Governance**: **Raw 300k dataset files are NEVER committed to this repository.** The dataset adapter streams records in-RAM via the Hugging Face `datasets` API or uses local synthetic fallback in offline test environments.

### 4.2 Synthea™ Synthetic Mass Clinical Records
- **Official Source**: MITRE Synthea™ Synthetic Patient Generator — [`https://synthea.mitre.org`](https://synthea.mitre.org)
- **Licensing**: Apache License 2.0 (MITRE Corporation)
- **Intended Use**: Evaluation of healthcare clinical note sanitization coverage and EHR structure extraction.
- **Repository Storage Governance**: Raw patient record databases are not committed. Synthea records are synthesized dynamically or loaded via lightweight synthetic templates.

---

## 5. Security & Secret Detection Scan Results

A full automated scan was conducted across the codebase for sensitive credentials:

| Secret Category | Search Patterns | Instances Detected | Status | Remediation / Verification |
|---|---|:---:|:---:|---|
| **API Keys & Tokens** | `api_key`, `API_KEY`, `sk_live_`, `bearer`, `hf_` | 0 genuine keys | **CLEAN** | Generator template string placeholders only |
| **Private Cryptographic Keys** | `BEGIN PRIVATE KEY`, `BEGIN RSA PRIVATE KEY` | 0 keys | **CLEAN** | RSA key pairs generated dynamically in volatile RAM |
| **Environment File Secrets** | `.env`, `.env.local`, `SECURE_LORA_KEY_HEX` | 0 exposed secrets | **CLEAN** | Documentation provides `secrets.token_hex(32)` setup commands |
| **Passwords & Credentials** | `password`, `PASS`, `db_pass` | 0 production credentials | **CLEAN** | Synthetic PII label placeholders only |
| **Real Government IDs** | Real SSN/Passport/Driver License patterns | 0 real IDs | **CLEAN** | All numbers generated via synthetic regex patterns |

---

## 6. Remaining Risks & Governance Controls

1. **Risk**: External network requests to Hugging Face Hub during live evaluations.  
   - **Control**: Mandatory `HF_HUB_OFFLINE=1` environment flag enforced in test suites (`tests/test_dataset_adapters.py`). When offline, adapters automatically fall back to local synthetic benchmark files.
2. **Risk**: Misinterpreting synthetic sample data as real personal data.  
   - **Control**: Every local JSONL record carries `"synthetic": true`, and all documentation headers enforce the mandatory Dataset Safety Statement.
