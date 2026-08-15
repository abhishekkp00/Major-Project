<div align="center">

# 🔐 SecureLoRA

**Hardware-Bound LoRA Fine-Tuning Framework for LLMs**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Security](https://img.shields.io/badge/Encryption-AES--256--GCM-critical)](#)
[![Tests](https://img.shields.io/badge/Attack%20Simulations-6%2F6%20PASS-success)](#)

</div>

---

## What is SecureLoRA?

SecureLoRA is a security framework for fine-tuning and deploying Large Language Models using LoRA (Low-Rank Adaptation). It solves two critical problems in enterprise and edge ML deployments:

1. **Adapter Theft** — Standard LoRA adapters are unprotected files that can be copied and run on any machine. SecureLoRA cryptographically binds adapter weights to the specific hardware they were deployed on.

2. **PII Leakage** — Training datasets often contain sensitive personal data (SSNs, emails, phone numbers) that LLMs can memorize. SecureLoRA automatically scrubs PII before the model ever sees it.

---

## How It Works

The system runs as a **four-phase pipeline**:

```
Phase 1: PII Audit          Phase 2: LoRA Training      Phase 3: Packaging          Phase 4: Deployment
─────────────────────       ────────────────────────     ──────────────────────      ────────────────────
Raw dataset                 Encrypted dataset            Trained adapter             Signed package
    │                           │                            │                           │
    ▼                           ▼                            ▼                           ▼
Scan & mask PII         Decrypt in RAM only          Collect HW fingerprint      Verify SHA-256 hash
(SSN, email, phone,     Train LoRA on frozen         Derive AES key via HKDF     Verify RSA signature
 IP, API keys, cards)   base model                   AES-256-GCM encrypt         Match HW fingerprint
    │                   Save adapter weights          RSA-PSS sign package        Decrypt in RAM only
    ▼                       │                             │                       Load into model
Encrypted JSONL         Encrypted adapter            Signed .tar.gz
```

**The key security guarantee:** The decryption key is never stored anywhere. It is derived at runtime from the device's own hardware identifiers (machine ID, CPU model, disk UUID) using HKDF — meaning the adapter can only be decrypted on the exact machine it was packaged for.

---

## Project Structure

```
SecureLoRA/
│
├── config/                         # YAML configuration (model, security, deployment)
│   ├── training.yaml               # LoRA hyperparameters, base model, batch size
│   ├── security.yaml               # Fingerprint sources, key derivation settings
│   ├── deployment.yaml             # Phase 4 gateway settings
│   └── app.yaml                    # Dashboard and logging settings
│
├── src/
│   ├── security/                   # Core cryptographic primitives
│   │   ├── fingerprint.py          # Hardware fingerprint collection & SHA-256
│   │   ├── key_derivation.py       # HKDF key derivation (ephemeral, never stored)
│   │   ├── crypto.py               # AES-256-GCM streaming + block encryption
│   │   ├── signature.py            # RSA-PSS sign & verify
│   │   └── shred.py                # 3-pass secure file overwrite & delete
│   │
│   ├── phase1/                     # PII Audit & Dataset Encryption
│   │   ├── ingestion.py            # Multi-format file reader (txt, csv, json, md)
│   │   ├── preprocessing.py        # Record normalization and schema standardization
│   │   ├── pipeline.py             # Full Phase 1 orchestration
│   │   └── cli.py                  # Command-line interface
│   │
│   ├── phase2/                     # In-Memory LoRA Fine-Tuning
│   │   └── train_lora.py           # Training loop (PEFT, HuggingFace Trainer)
│   │
│   ├── phase3/                     # Hardware Binding & Packaging
│   │   ├── main.py                 # Phase 3 entry point
│   │   ├── package_builder.py      # Builds signed .tar.gz deployment archive
│   │   └── verifier.py             # Pre-packaging integrity checks
│   │
│   ├── phase4/                     # Secure Deployment Gateway
│   │   ├── main.py                 # Phase 4 entry point
│   │   ├── package_validator.py    # Multi-gate cryptographic validation
│   │   ├── device_auth.py          # Hardware fingerprint match check
│   │   ├── decryptor.py            # In-RAM AES-256-GCM decryption
│   │   ├── adapter_loader.py       # Loads decrypted weights into base model
│   │   └── inference_runner.py     # Before/after inference comparison
│   │
│   ├── orchestrator/               # Job Management & Dashboard Backend
│   │   ├── service.py              # Job queue and background runner
│   │   ├── security_orchestrator.py# End-to-end Phase 3+4 orchestration
│   │   ├── dataset_processor.py    # Upload → validate → encrypt pipeline
│   │   └── routes.py               # Flask API routes
│   │
│   ├── evaluation/                 # Evaluation & Monitoring
│   │   ├── dashboard.py            # Real-time Flask monitoring dashboard
│   │   ├── pii_metrics.py          # PII detection precision/recall/F1
│   │   ├── crypto_benchmark.py     # Cryptographic performance benchmarks
│   │   ├── baseline_comparison.py  # Security comparison vs other approaches
│   │   └── threat_model.py         # Attack simulations against real crypto code
│   │
│   ├── common/                     # Shared Utilities
│   │   ├── config_loader.py        # YAML config reader with validation
│   │   └── exceptions.py           # Custom exception hierarchy
│   │
│   └── utils/                      # Helper Modules
│       ├── logging_utils.py        # Structured logger setup
│       └── checkpoint_utils.py     # Training checkpoint management
│
├── tests/                          # Test Suite
│   ├── unit/                       # Per-module unit tests
│   ├── integration/                # Phase-level integration tests
│   └── security/                   # Cryptographic security tests
│
├── outputs/                        # Runtime-generated files
│   ├── paper_results/              # Evaluation outputs (benchmarks, metrics)
│   ├── evaluation/                 # Training evaluation report
│   └── deployment_validation/      # Phase 4 gate validation reports
│
├── paper_figures/                  # Generated charts and architecture diagrams
├── samples/                        # Demonstration datasets (PHI, corporate PII, real-world)
├── scripts/                        # Evaluation, benchmarking, and paper figure generators
│   └── run_paper_evaluation.py     # Runs all evaluation modules end-to-end
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── train_and_evaluate_lora.ipynb   # Interactive Jupyter notebook demo
```


---

## Security Properties

| Property | Mechanism |
|----------|-----------|
| Adapter only decryptable on authorized device | HKDF key derived from hardware fingerprint |
| Any file modification is detectable | SHA-256 hash + AES-GCM authentication tag |
| Adapter origin is verifiable | RSA-2048-PSS digital signature |
| Plaintext weights never written to disk | Decryption in volatile RAM + 3-pass file shredding |
| PII removed before model training | Phase 1 regex scanner (SSN, email, phone, IP, keys, cards) |
| No specialist hardware required | Software-only — no TPM chip or SGX enclave needed |

---

## Setup

```bash
# Clone and install
git clone https://github.com/abhishekkp00/Major-Project.git
cd Major-Project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set P3_DEVICE_SALT and SECURE_LORA_KEY_HEX

# Launch dashboard
python -m src.evaluation.dashboard
# → http://localhost:5001
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `P3_DEVICE_SALT` | ✅ | Random salt for HKDF key derivation. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `SECURE_LORA_KEY_HEX` | ✅ | 64-character hex AES-256 key for dataset encryption |
| `SECURE_LORA_KEY_PATH` | Alternative | Path to binary key file (used if `KEY_HEX` is not set) |
| `DASHBOARD_PORT` | ❌ | Dashboard port (default: `5001`) |

---

## Tests

```bash
pytest tests/ -v                          # Full suite
pytest tests/security/ -v                # Security tests only
pytest tests/ --cov=src                  # With coverage
```
