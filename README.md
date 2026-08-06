<div align="center">

# 🔐 SecureLoRA

### Hardware-Bound LoRA Fine-Tuning Framework for LLMs

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blueviolet)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-Research-green)](LICENSE)
[![Security](https://img.shields.io/badge/Encryption-AES--256--GCM-critical)](src/security/crypto.py)
[![Tests](https://img.shields.io/badge/Attack%20Simulations-6%2F6%20PASS-success)](src/evaluation/threat_model.py)

> **An enterprise-grade, software-only pipeline that cryptographically binds LoRA adapter weights to specific hardware — preventing theft, tampering, and PII leakage throughout the entire MLOps lifecycle.**

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Contributions](#-key-contributions)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Running Each Phase](#-running-each-phase)
- [Security Evaluation](#-security-evaluation)
- [Evaluation Results](#-evaluation-results)
- [Research & Publication](#-research--publication)
- [Dataset & Licenses](#-dataset--licenses)

---

## 🎯 Problem Statement

Standard LoRA adapters are **unprotected portable weight matrices** — easily cloned, reverse-engineered, or deployed on unauthorized devices. Additionally, training datasets often contain **PII/PHI** that LLMs can memorize and reproduce during inference, creating GDPR, CCPA, and HIPAA liabilities.

**SecureLoRA solves both problems** in a single, integrated four-phase MLOps pipeline.

---

## 🔑 Key Contributions

| Contribution | Description |
|---|---|
| **Hardware Binding** | AES-256-GCM key derived via HKDF from `machine-id + CPU + disk UUID` — decryption only works on the authorized device |
| **Zero-Plaintext-at-Rest** | Decrypted weights live exclusively in volatile RAM; all temp files are 3-pass shredded |
| **Supply Chain Integrity** | RSA-2048-PSS digital signature over SHA-256 digest — detects any adapter tampering or injection |
| **PII/PHI Scrubbing** | Phase 1 regex engine masks SSN, email, phone, IP, API keys, and credit card numbers before any tokenization |
| **Software-Only** | No TPM chip or SGX enclave required — works on any commodity Linux machine |
| **Paper-Ready Evaluation** | Real benchmark suite: crypto timing, PII metrics (F1: 0.957), 6 live attack simulations (6/6 PASS) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SecureLoRA Pipeline                         │
├──────────────┬──────────────┬──────────────────┬────────────────────┤
│   PHASE 1    │   PHASE 2    │     PHASE 3      │      PHASE 4       │
│  PII Audit   │  LoRA Train  │  HW Bind & Pack  │  Secure Deploy     │
├──────────────┼──────────────┼──────────────────┼────────────────────┤
│ Raw JSONL    │ In-memory    │ Fingerprint:     │ Extract archive    │
│     ↓        │ tokenization │ machine-id +     │     ↓              │
│ PII Scanner  │     ↓        │ cpu + disk UUID  │ SHA-256 hash check │
│ (SSN/Email/  │ LoRA adapts  │     ↓ HKDF       │     ↓              │
│  Phone/IP/   │ frozen base  │ AES-256-GCM enc  │ RSA-PSS verify     │
│  Keys/Cards) │ model        │     ↓            │     ↓              │
│     ↓        │     ↓        │ RSA-PSS sign     │ HW fingerprint     │
│ Sanitized    │ Adapter      │     ↓            │ match check        │
│ JSONL →      │ weights →    │ .tar.gz package  │     ↓              │
│ AES-256-GCM  │ AES-256-GCM  │                  │ In-RAM decrypt     │
│ encrypted    │ encrypted    │                  │ + PEFT load        │
│ dataset      │ + shredded   │                  │ + shred temp       │
└──────────────┴──────────────┴──────────────────┴────────────────────┘
```

### Cryptographic Stack

```
Hardware Identifiers ─────────────────────────────────────────────────┐
  /etc/machine-id                                                      │
  /proc/cpuinfo (model name)         SHA-256                          │
  /dev/disk/by-uuid (first UUID)  ─────────► fingerprint_hash         │
                                                     │                 │
                                              HKDF-SHA256             │
                                            (IKM=fingerprint,         │
                                             salt=P3_DEVICE_SALT)     │
                                                     │                 │
                                              AES-256-GCM key         │
                                                     │                 │
Adapter weights ──────────────────────────► AES-256-GCM encrypt ◄────┘
                                              (AAD = fingerprint)
                                                     │
                                              SHA-256 digest
                                                     │
Developer private key ───────────────────► RSA-2048-PSS sign
                                                     │
                                            .tar.gz package
                                       (enc + hash + sig + pubkey)
```

---

## 📁 Project Structure

```
SecureLoRA/
│
├── 📄 README.md                        ← You are here
├── 📄 requirements.txt                 ← All Python dependencies
├── 📄 .env.example                     ← Environment variable template
├── 📄 conftest.py                      ← Pytest shared fixtures
│
├── 🚀 run_paper_evaluation.py          ← Master: runs all 4 evaluation modules
│
├── ⚙️  config/                          ← YAML configuration files
│   ├── app.yaml                        ← Application-level settings
│   ├── training.yaml                   ← LoRA hyperparameters & model config
│   ├── security.yaml                   ← Crypto & fingerprint settings
│   └── deployment.yaml                 ← Phase 4 deployment config
│
├── 🧠 src/                             ← All source code
│   │
│   ├── common/                         ← Shared utilities
│   │   ├── config_loader.py            ← Reads YAML configs, validates fields
│   │   └── exceptions.py              ← Custom exception hierarchy
│   │
│   ├── security/                       ← 🔐 Core cryptographic primitives
│   │   ├── fingerprint.py              ← Hardware fingerprint collection & SHA-256
│   │   ├── key_derivation.py           ← HKDF key derivation (ephemeral, no disk)
│   │   ├── crypto.py                   ← AES-256-GCM streaming + block encryption
│   │   ├── signature.py                ← RSA-PSS sign & verify
│   │   └── shred.py                    ← 3-pass secure file overwrite + delete
│   │
│   ├── phase1/                         ← 📥 Ingestion & PII Audit
│   │   ├── ingestion.py                ← Multi-format file ingestor (txt/csv/json/md)
│   │   ├── preprocessing.py            ← Record normalization & schema standardization
│   │   ├── pipeline.py                 ← Full Phase 1: ingest → scrub → encrypt dataset
│   │   └── cli.py                      ← CLI entry point for Phase 1
│   │
│   ├── phase2/                         ← 🏋️  In-Memory LoRA Fine-Tuning
│   │   └── train_lora.py               ← Full training loop (PEFT, Trainer, eval)
│   │
│   ├── phase3/                         ← 📦 Packaging & Hardware Binding
│   │   ├── main.py                     ← Phase 3 orchestrator entry point
│   │   ├── adapter_encryptor.py        ← Adapter encryption wrapper
│   │   ├── package_builder.py          ← Builds signed .tar.gz deployment archive
│   │   ├── verifier.py                 ← Pre-packaging verification checks
│   │   └── config.py                   ← Phase 3 config loader
│   │
│   ├── phase4/                         ← 🚪 Secure Deployment Gateway
│   │   ├── main.py                     ← Phase 4 orchestrator entry point
│   │   ├── package_loader.py           ← Extracts and validates .tar.gz archive
│   │   ├── package_validator.py        ← Multi-gate cryptographic validation
│   │   ├── device_auth.py              ← Hardware fingerprint match check
│   │   ├── decryptor.py                ← In-RAM AES-256-GCM decryption
│   │   ├── adapter_loader.py           ← Binds decrypted weights to base model
│   │   ├── inference_runner.py         ← Before/after inference comparison
│   │   ├── validation_report.py        ← Generates gate-by-gate validation report
│   │   └── config.py                   ← Phase 4 config loader
│   │
│   ├── orchestrator/                   ← 🎛️  Dashboard & Job Orchestration
│   │   ├── service.py                  ← Job management & background runner
│   │   ├── security_orchestrator.py    ← Phase 3+4 end-to-end orchestration
│   │   ├── dataset_processor.py        ← Upload → validate → encrypt dataset
│   │   └── routes.py                   ← Flask API routes
│   │
│   ├── evaluation/                     ← 📊 Paper Evaluation Suite
│   │   ├── dashboard.py                ← Flask real-time monitoring dashboard
│   │   ├── pii_metrics.py              ← PII detection precision/recall/F1
│   │   ├── crypto_benchmark.py         ← AES/HKDF/RSA/SHA-256 timing benchmarks
│   │   ├── baseline_comparison.py      ← Security matrix vs. 5 baseline approaches
│   │   ├── threat_model.py             ← STRIDE threat model + 6 attack simulations
│   │   ├── static/                     ← Dashboard CSS & JavaScript
│   │   └── templates/                  ← Dashboard HTML template
│   │
│   └── utils/                          ← 🛠️  Helper Utilities
│       ├── logging_utils.py            ← Structured logger setup
│       └── checkpoint_utils.py         ← Training checkpoint management
│
├── 🧪 tests/                           ← Full test suite (pytest)
│   ├── unit/                           ← Unit tests (per-module)
│   ├── integration/                    ← Integration tests (phase-level)
│   └── security/                       ← Security-specific tests (crypto, shred)
│
├── 📊 outputs/                         ← Generated at runtime (gitignored where sensitive)
│   ├── paper_results/                  ← ✅ Evaluation outputs (committed)
│   │   ├── paper_evaluation_results.json
│   │   ├── paper_evaluation_summary.md
│   │   └── benchmarks/                 ← Per-module JSON reports
│   ├── evaluation/                     ← eval_report.json from training
│   └── deployment_validation/          ← Phase 4 gate validation reports
│
├── 📈 paper_figures/                   ← Publication-ready figures (PDF + PNG)
│   ├── fig1_loss_curve.*
│   ├── fig2_parameter_efficiency.*
│   ├── fig3_security_timing.*
│   ├── fig4_security_gates.*
│   ├── fig5_perplexity.*
│   ├── fig6_architecture.*
│   └── fig7_pii_detection.*
│
├── 📚 dataset_licenses_and_assets/     ← Dataset documentation & licenses
└── 📓 train_and_evaluate_lora.ipynb    ← Interactive Jupyter notebook demo
```

---

## ⚡ Quick Start

### 1. Prerequisites

```bash
python --version   # 3.10+
git clone https://github.com/abhishekkp00/Major-Project.git
cd Major-Project
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env — set P3_DEVICE_SALT and AES_KEY_HEX (see below)
```

### 4. Run the Full Pipeline (Dashboard)

```bash
source venv/bin/activate
python -m src.evaluation.dashboard
# Open http://localhost:5001 in your browser
```

### 5. Run Paper Evaluation Suite

```bash
python run_paper_evaluation.py
# Results → outputs/paper_results/
```

---

## ⚙️ Configuration

All configuration lives in `config/`. Edit YAML files — no code changes required.

| File | Controls |
|------|----------|
| `config/training.yaml` | Base model, LoRA rank/alpha/dropout, batch size, learning rate, epochs |
| `config/security.yaml` | Hardware fingerprint sources, key derivation salt, RSA key size |
| `config/deployment.yaml` | Package output paths, verification gate settings |
| `config/app.yaml` | Dashboard port, log level, job workspace root |

**Required `.env` variables:**

```bash
# Minimum required
P3_DEVICE_SALT=your_random_salt_here     # Used in HKDF key derivation
AES_KEY_HEX=                             # Auto-generated if blank

# Optional
SECURE_LORA_SAMPLE_PROMPT="Mask all PII in the text."
SECURE_LORA_PROGRESS_FILE=outputs/progress.json
```

---

## 🚀 Running Each Phase

### Phase 1 — PII Audit & Dataset Encryption

```bash
# Encrypt a directory of raw training files
python -m src.phase1.cli encrypt \
  --input-dir real_data_inputs/ \
  --output-dir encrypted_real_data/ \
  --dataset-name my_dataset
```

### Phase 2 — LoRA Fine-Tuning

```bash
# Trains on the encrypted dataset; decrypts only in volatile RAM
python -m src.phase2.train_lora
# Outputs: outputs/final_adapter/  +  outputs/evaluation/eval_report.json
```

### Phase 3 — Hardware Binding & Packaging

```bash
# Encrypts adapter with hardware fingerprint + signs with RSA key
python -m src.phase3.main \
  --adapter-dir outputs/final_adapter/ \
  --output-dir outputs/protected_adapter/
```

### Phase 4 — Secure Deployment Gateway

```bash
# Validates all gates, decrypts in RAM, runs inference
python -m src.phase4.main \
  --package outputs/protected_adapter.tar.gz \
  --prompt "Mask all PII from the following text."
```

### Interactive Dashboard

```bash
python -m src.evaluation.dashboard
# Real-time pipeline monitor at http://localhost:5001
```

---

## 🛡️ Security Evaluation

### Attack Simulations

All 6 attack simulations are **live tests against real crypto code** — not mocked:

| ID | Attack Scenario | Countermeasure | Result |
|----|-----------------|----------------|--------|
| SIM-01 | Physical disk clone → foreign device | HKDF fingerprint binding | ✅ PASS |
| SIM-02 | Bit-level ciphertext corruption | SHA-256 hash + GCM auth tag | ✅ PASS |
| SIM-03 | RSA-PSS signature forgery | Wrong public key rejected | ✅ PASS |
| SIM-04 | Salt replay + spoofed hardware | HKDF two-factor binding | ✅ PASS |
| SIM-05 | GCM authentication tag corruption | AEAD tag verification | ✅ PASS |
| SIM-06 | Wrong salt → wrong key | HKDF deterministic mismatch | ✅ PASS |

Run them:

```bash
python -m src.evaluation.threat_model
```

### Run All Benchmarks

```bash
# PII detection metrics
python -m src.evaluation.pii_metrics

# Crypto timing benchmarks
python -m src.evaluation.crypto_benchmark

# Baseline comparison
python -m src.evaluation.baseline_comparison

# Full paper evaluation (all 4 modules)
python run_paper_evaluation.py
```

---

## 📊 Evaluation Results

> All results are **genuinely measured** on this machine. Run `python run_paper_evaluation.py` to reproduce.

### PII Detection (48-sample labeled corpus)

| PII Type | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| SSN | 1.000 | 1.000 | 1.000 |
| Email | 1.000 | 1.000 | 1.000 |
| Phone | 0.857 | 1.000 | 0.923 |
| IP Address | 1.000 | 1.000 | 1.000 |
| API Key | 1.000 | 0.833 | 0.909 |
| Credit Card | 1.000 | 0.833 | 0.909 |
| **Macro Avg** | **0.976** | **0.944** | **0.957** |

### Cryptographic Overhead (512 KB adapter payload)

| Approach | Mean Latency | Notes |
|----------|-------------|-------|
| Unprotected LoRA | 0.60 ms | No security |
| Password-Only AES | 14.69 ms | No hardware binding |
| Filesystem ACL | 0.74 ms | Defeated by disk clone |
| TPM-Bound (reference) | 24.69 ms | Requires TPM hardware |
| **SecureLoRA (this work)** | **1.31 ms** | **Software-only, full protection** |

### Security Score (8-dimension matrix, 0.0–1.0)

| Method | Aggregate Score |
|--------|----------------|
| Unprotected LoRA | 0.125 |
| Password-Only AES | 0.450 |
| Filesystem ACL | 0.138 |
| TPM-Bound | 0.656 |
| **SecureLoRA** | **0.969** |

### LoRA Training Results

| Metric | Value |
|--------|-------|
| Base Model | `JackFram/llama-68m` |
| Trainable Parameters | 98,304 / 68M (0.144%) |
| Validation Loss | 2.641 |
| Perplexity | 14.02 |
| Training Status | Completed ✅ |

---

## 📄 Research & Publication

### Suggested Paper Title

> *"SecureLoRA: Hardware-Bound Parameter-Efficient Fine-Tuning with Zero-Plaintext-at-Rest Enforcement for Edge LLM Deployment"*

### Target Venues

| Venue | Type | Suitability |
|-------|------|-------------|
| **IEEE Access** | Open Access Journal | 🟢 Best fit |
| **Elsevier Computers & Security** | Journal | 🟡 Strong |
| **IEEE TrustCom** | Conference | 🟡 Good |
| **NeurIPS/ICML Security Workshop** | Workshop | 🟢 Entry point |

### Generating Paper Figures

```bash
python generate_paper_figures.py
# Outputs: paper_figures/ (PDF + PNG, publication-ready)
```

### Paper-Ready Results

Pre-generated results are in `outputs/paper_results/`:
- `paper_evaluation_summary.md` — ready-to-paste Markdown tables
- `paper_evaluation_results.json` — full structured data
- `benchmarks/*.json` — per-module raw data

---

## 📚 Dataset & Licenses

Training data sourced from openly licensed PII masking datasets. See [`dataset_licenses_and_assets/`](dataset_licenses_and_assets/) for:
- Dataset README and distribution statistics
- LLaMA community license agreements
- Data split documentation

---

## 🗂️ Running Tests

```bash
# Full test suite
pytest tests/ -v

# By category
pytest tests/unit/ -v
pytest tests/security/ -v
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

<div align="center">

**SecureLoRA** — Enterprise-grade LoRA protection, software-only.  
Built for research publication and real-world edge deployment.

</div>
