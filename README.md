<div align="center">

# 🔐 SecureLoRA

**Hardware-Bound & Privacy-Preserving LoRA Fine-Tuning Pipeline for Large Language Models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Security](https://img.shields.io/badge/Encryption-AES--256--GCM-critical)](#)
[![Signature](https://img.shields.io/badge/Signature-RSA--PSS%202048-blue)](#)
[![Tests](https://img.shields.io/badge/Unit%20%26%20Security%20Tests-69%2F69%20PASS-success)](#)

</div>

---

## Executive Overview

**SecureLoRA** is an end-to-end, zero-leakage security framework designed for privacy-preserving Low-Rank Adaptation (LoRA) fine-tuning and hardware-bound deployment of Large Language Models (LLMs). It mitigates two fundamental vulnerabilities in enterprise machine learning pipelines:

1. **Unprotected Adapter Theft & Illegal Relocation** — Standard PEFT/LoRA weight artifacts (`adapter_model.safetensors`) are plaintext files susceptible to exfiltration and unauthorized execution on foreign nodes. SecureLoRA cryptographically binds adapter weights to the authorized target machine's unique hardware fingerprint using ephemeral HKDF key derivation.
2. **Personal Identifiable Information (PII) Leakage** — Raw training datasets in healthcare (PHI), finance, or enterprise support contain sensitive entities (SSNs, medical record numbers, API keys, credentials) that LLMs risk memorizing. SecureLoRA enforces an in-transit hybrid redaction engine combining SpaCy Transformer Named Entity Recognition (NER) with RFC/ISO regex pattern matching prior to model ingestion.

---

## Architectural Workflow

The system operates across a **four-phase zero-trust lifecycle**:

```
Phase 1: Hybrid PII Audit    Phase 2: LoRA Fine-Tuning    Phase 3: Hardware Packaging   Phase 4: Gate Validation
─────────────────────────    ─────────────────────────    ───────────────────────────   ─────────────────────────
Raw User Input / Dataset     Redacted Dataset JSONL       Trained LoRA Weights          Encrypted .tar.gz Package
            │                            │                            │                             │
            ▼                            ▼                            ▼                             ▼
SpaCy NER + ISO Regex        PEFT / HuggingFace Trainer   Derive Target HW Fingerprint  1. Package Completeness Check
Mask SSN, Phone, Email,      Train on Frozen Base Model   HKDF-SHA256 Key Derivation    2. SHA-256 Digest Verification
Secret Keys, Credit Cards    Save Ephemeral Weights       AES-256-GCM Encryption        3. RSA-2048-PSS Signature Check
            │                            │                RSA-2048-PSS Signature        4. HW Fingerprint Matching
            ▼                            ▼                            │                 5. In-Memory Decryption & Load
Encrypted JSONL Payload      Encrypted Adapter Output     Signed .tar.gz Package        6. Side-by-Side Inference
```

### Core Cryptographic Guarantee
Decryption keys are **never stored on disk or embedded in environment configs**. Decryption keys are derived transiently in volatile RAM at boot time from the executing machine's immutable hardware identifiers (`/etc/machine-id`, MAC address, hostname). If an encrypted adapter package is exfiltrated to an unauthorized node, HKDF key derivation produces a mismatched key, causing AES-256-GCM authentication tag failure and **instant termination before model weight extraction**.

---

## Technical Module Specifications

### 1. Hybrid PII Detection & Masking Engine (`src/security/pii_engine.py`)
- **Dual Engine Architecture**: Combines contextual SpaCy NLP entity extraction with high-precision regular expressions.
- **Entity Coverage**: Automatically redacts Names (`NAME_PHRASE`), Social Security Numbers (`SSN`), Email Addresses (`EMAIL`), Phone Numbers (`PHONE`), IP Addresses (`IP_ADDRESS`), Credit Card Numbers (`CREDIT_CARD`), and Secret API Keys (`API_KEY`).
- **Canonical Anchoring**: Calculates SHA-256 digests before and after redaction to anchor record lineage into immutable integrity traces.

### 2. Ephemeral Hardware Fingerprinting & Key Derivation (`src/security/fingerprint.py`, `key_derivation.py`)
- **Hardware Tuple**: Extracts machine-specific entropy from OS Machine ID (`/etc/machine-id`), network interface MAC addresses, and node hostnames.
- **HKDF-SHA256 Key Derivation**: Feeds the hardware fingerprint and a dynamic cryptographic salt into HKDF (RFC 5869) to derive 256-bit symmetric keys. Keys exist exclusively in ephemeral RAM during execution.

### 3. Multi-Gate Cryptographic Package Vault (`src/phase3/`, `src/phase4/`)
- **AES-256-GCM Authenticated Encryption**: Adapter weights are encrypted in streaming blocks with Galois/Counter Mode (GCM), providing confidentiality and tamper-proofing via 128-bit authentication tags.
- **RSA-2048-PSS Digital Signatures**: Every deployment package is signed with an RSA-2048 private key using Probabilistic Signature Scheme (PSS) padding, ensuring non-repudiation and origin verification.
- **3-Pass Secure File Shredding**: Transient plaintext chunks created during processing are overwritten using DoD 5220.22-M 3-pass zeroing before filesystem unlinking.

### 4. Interactive Fail-Fast Security Workbench (`src/evaluation/dashboard.py`, `dashboard.js`)
- **Interactive Attack Simulator**: Real-time simulation suite enabling direct injection of:
  - *Adapter Theft Attack*: Simulates unauthorized deployment on foreign hardware nodes.
  - *In-Transit Payload Corruption*: Simulates active man-in-the-middle payload tampering.
  - *Prompt Jailbreak Attack*: Simulates malicious prompt injection attempts.
- **Live Visual Pipeline**: Real-time visual tracking of PII redaction spans, SHA-256 checksum validations, and side-by-side model outputs (Baseline vs. Secured LoRA).

### 5. Dynamic Physics-Based SDG-13 Environmental Model (`src/orchestrator/transparency.py`)
- **FLOPs Computation**: Evaluates exact Floating Point Operations based on model parameters ($68.12\text{M}$ base params + $98.3\text{K}$ LoRA params), input token length, and fine-tuning epochs:
  $$\text{FLOPs} = 6 \times (\text{BaseParams} + \text{LoRAParams}) \times \text{Tokens} \times \text{Epochs}$$
- **IPCC 2023 Carbon Emissions Factor**: Calculates GPU energy consumption ($300\text{W}$ TDP) and maps compute savings to real-world carbon emission reduction:
  $$\text{CO}_2\text{e}_{\text{grams}} = \text{Energy}_{\text{kWh}} \times 475.0\text{ gCO}_2\text{e/kWh}$$

---

## Detailed Project Structure

```
SecureLoRA/
├── config/                         # System & Deployment YAML Configurations
│   ├── app.yaml                    # Dashboard server settings and routes
│   ├── deployment.yaml             # Phase 4 validation gate thresholds
│   ├── security.yaml               # Fingerprint entropy sources & HKDF parameters
│   └── training.yaml               # Hyperparameters for PEFT/LoRA fine-tuning
│
├── src/                            # Core Source Code Package
│   ├── security/                   # Low-Level Cryptographic Primitives
│   │   ├── crypto.py               # AES-256-GCM authenticated encryption/decryption
│   │   ├── fingerprint.py          # Machine ID, MAC address & hardware hashing
│   │   ├── key_derivation.py       # HKDF-SHA256 ephemeral key derivation
│   │   ├── pii_engine.py           # Hybrid SpaCy NER + Regex PII redaction engine
│   │   ├── shred.py                # 3-pass secure RAM & disk shredding
│   │   └── signature.py            # RSA-2048-PSS digital signing & verification
│   │
│   ├── phase1/                     # Phase 1: PII Audit & Dataset Ingestion
│   │   ├── cli.py                  # Phase 1 command-line interface
│   │   ├── ingestion.py            # Multi-format reader (JSONL, CSV, TXT, MD)
│   │   ├── pipeline.py             # Ingest → Redact → Encrypt workflow orchestrator
│   │   └── preprocessing.py        # Record normalization & schema validation
│   │
│   ├── phase2/                     # Phase 2: In-Memory LoRA Fine-Tuning
│   │   └── train_lora.py           # PEFT adapter fine-tuning on redacted payloads
│   │
│   ├── phase3/                     # Phase 3: Hardware Binding & Packaging
│   │   ├── main.py                 # Phase 3 packaging execution script
│   │   ├── package_builder.py      # Encrypted .tar.gz bundle construction
│   │   └── verifier.py             # Pre-packaging completeness validator
│   │
│   ├── phase4/                     # Phase 4: Secure Deployment Gateway
│   │   ├── adapter_loader.py       # Decrypted PEFT weight injection into base LLM
│   │   ├── decryptor.py            # In-RAM stream decryptor
│   │   ├── device_auth.py          # Dynamic hardware fingerprint matching
│   │   ├── inference_runner.py     # Comparative inference evaluation runner
│   │   ├── main.py                 # Phase 4 deployment execution entry point
│   │   └── package_validator.py    # 8-step multi-gate security verifier
│   │
│   ├── orchestrator/               # Backend Service & Chat Engines
│   │   ├── chat_engine.py          # Dual model interface (Baseline vs. Secured LoRA)
│   │   ├── dataset_processor.py    # Streaming dataset intake & validation
│   │   ├── routes.py               # REST API endpoint definitions
│   │   ├── security_orchestrator.py# End-to-end automated pipeline executor
│   │   ├── service.py              # Background job manager & state database
│   │   └── transparency.py         # SDG-13 physics model & PII inspection trace
│   │
│   ├── evaluation/                 # Metrics & Security Benchmarking
│   │   ├── baseline_comparison.py  # Comparative evaluation vs non-encrypted baseline
│   │   ├── crypto_benchmark.py     # AES-GCM & HKDF latency benchmarks
│   │   ├── dashboard.py            # Interactive Flask monitoring server
│   │   ├── pii_metrics.py          # Precision, Recall & F1 evaluation for PII
│   │   ├── threat_model.py         # Automated security attack simulator
│   │   ├── templates/              # HTML5 responsive workbench templates
│   │   └── static/                 # Dynamic CSS theme & JavaScript engine
│   │
│   ├── common/                     # Common Framework Utilities
│   │   ├── config_loader.py        # Validated YAML configuration loader
│   │   └── exceptions.py           # System exception class definitions
│   │
│   └── utils/                      # Helper & Logging Utilities
│       ├── checkpoint_utils.py     # Checkpoint management tools
│       └── logging_utils.py        # Structured JSON logging setup
│
├── tests/                          # Test Suite (69 Automated Tests)
│   ├── unit/                       # Unit tests for individual modules
│   ├── integration/                # Full pipeline phase integration tests
│   └── security/                   # Cryptographic security & fail-fast tests
│
├── outputs/                        # Evaluation Results & Model Checkpoints
│   ├── evaluation/                 # Fine-tuning metric reports
│   ├── notebook_adapter/           # Trained PEFT LoRA adapter weights
│   ├── notebook_encrypted/         # Encrypted dataset artifacts
│   └── paper_results/              # Serialized benchmark metrics
│
├── paper_figures/                  # High-Resolution Architectural Figures
├── samples/                        # Demonstration PII Records (Medical, Finance, Corporate)
├── scripts/                        # Automated Evaluation & Figure Generation Scripts
├── train_and_evaluate_lora.ipynb   # Interactive Cell-by-Cell Notebook Demo
├── PROJECT_STRUCTURE.md            # Comprehensive File Taxonomy Document
└── README.md                       # Project Documentation Specification
```

---

## Security Guarantees & Threat Model Matrix

| Threat Vector | Attack Mechanism | SecureLoRA Defence | Security Outcome |
| :--- | :--- | :--- | :--- |
| **Adapter Theft** | Adversary copies `.tar.gz` package to foreign machine | HKDF key derived on foreign machine produces invalid 256-bit AES key | **GCM Auth Tag Failure**: Instant termination before weight extraction |
| **In-Transit Corruption** | MITM adversary alters encrypted payload bits during network transit | AES-256-GCM authentication tag validation | **Tamper Detected**: Integrity gate blocks execution |
| **Package Tampering** | Adversary modifies package manifest or signature | RSA-2048-PSS digital signature verification | **Signature Mismatch**: Gate 3 rejects deployment package |
| **PII Memorization** | LLM memorizes SSNs, emails, or phone numbers from training data | Phase 1 Hybrid SpaCy NER + ISO Regex Masking | **Zero Leakage**: Model trains exclusively on `[REDACTED_*]` tokens |
| **Cold-Boot Memory Extraction** | Adversary inspects disk for residual plaintext adapter weights | In-RAM decryption + 3-pass DoD 5220.22-M file shredding | **Zero Disk Residue**: Plaintext weights exist only in volatile RAM |

---

## Environmental & Compute Impact Physics Formulas

SecureLoRA integrates a dynamic physics model mapping computational operations directly to energy consumption and carbon reduction:

1. **Floating Point Operations (FLOPs)**:
   $$\text{FLOPs}_{\text{total}} = 6 \times (N_{\text{base}} + N_{\text{lora}}) \times T_{\text{tokens}} \times E_{\text{epochs}}$$
   *Where $N_{\text{base}} = 68.12\text{M}$, $N_{\text{lora}} = 98.3\text{K}$.*

2. **Compute Duration ($t_{\text{seconds}}$)**:
   $$t_{\text{seconds}} = \frac{\text{FLOPs}_{\text{total}}}{65.0 \times 10^{12} \times 0.35}$$
   *Based on NVIDIA T4 profile ($65\text{ TFLOPS}$) at $35\%$ Model FLOPs Utilization (MFU).*

3. **Energy & IPCC Carbon Impact**:
   $$\text{Energy}_{\text{kWh}} = \frac{300\text{W} \times (t_{\text{seconds}} / 3600)}{1000}$$
   $$\text{Carbon Saved (gCO}_2\text{e)} = \text{Energy}_{\text{kWh}} \times 475.0\text{ gCO}_2\text{e/kWh}$$
