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

1. **Unprotected Adapter Theft & Illegal Relocation** — Standard PEFT/LoRA weight artifacts (`adapter_model.safetensors`) are plaintext files susceptible to exfiltration and unauthorized execution on foreign nodes. SecureLoRA cryptographically binds adapter weights to the authorized target machine's software-derived device identity using HKDF-SHA256 key derivation.
2. **Personal Identifiable Information (PII) Leakage** — Raw training datasets in healthcare (PHI), finance, or enterprise support contain sensitive entities (SSNs, medical record numbers, API keys, credentials) that LLMs risk memorizing. SecureLoRA enforces a hybrid redaction engine (SpaCy Transformer NER + RFC/ISO regex) as a *first-line control*, and optionally reinforces it with Differentially Private SGD (DP-SGD via Opacus) to provide rigorous membership inference resistance.

> **Claim accuracy**: DP-LoRA is an established research direction (Li et al. 2021, Yu et al. 2021). Our research contribution is the *measured interaction* of training-data privacy (DP-SGD), device-bound adapter protection (HKDF + AES-GCM), and deployment security (RSA-PSS) within a single unified pipeline. DP addresses training-data privacy. Device binding addresses adapter theft. These are orthogonal threat surfaces.

---

## Architectural Workflow

The system operates across a **four-phase zero-trust lifecycle**:

Mode A — Standard LoRA (default):

```
Phase 1: Hybrid PII Audit    Phase 2A: LoRA Fine-Tuning    Phase 3: Secure Packaging    Phase 4: Gate Validation
─────────────────────────    ──────────────────────────    ─────────────────────────    ─────────────────────────
Raw Dataset                  Redacted Dataset JSONL         Trained LoRA Weights         Encrypted .tar.gz Package
     │                               │                              │                              │
     ▼                               ▼                              ▼                              ▼
SpaCy NER + ISO Regex         HuggingFace Trainer             HKDF-SHA256 Key               1. Completeness Check
Mask SSN, Phone, Email,       LoRA on Frozen Base Model       AES-256-GCM Encrypt           2. SHA-256 Integrity
Secret Keys, Credit Cards     Save Adapter Weights            RSA-2048-PSS Sign             3. RSA-PSS Signature
     │                               │                              │                       4. Fingerprint Match
     ▼                               ▼                              ▼                       5. HKDF Key Derivation
Encrypted JSONL Payload        Adapter Weights                Signed Package                6. AES-GCM Decrypt & Load
```

Mode B — DP-LoRA (optional, `DP_ENABLED=1`):

```
Phase 1: Hybrid PII Audit    Phase 2B: DP-LoRA               Phase 3/4: (same as Mode A)
─────────────────────────    ────────────────────────────
Raw Dataset                  Redacted Dataset JSONL
     │                               │
     ▼                               ▼
SpaCy NER + ISO Regex         LoRA on Frozen Base Model
[First-line PII control]      Per-Example Gradient Clip (C)
                              + Gaussian Noise (σ = noise_multiplier)
                              Privacy Budget: (ε, δ)-DP
                              Accountant: Rényi DP / PRV
                              ε computed post-hoc by accountant
```

### Core Cryptographic Guarantee
Decryption keys are **never stored on disk or embedded in environment configs**. Keys are derived transiently in volatile RAM from the executing machine's software-derived device identity (OS machine-id, CPU model, disk UUID) using HKDF-SHA256 (RFC 5869). If an encrypted adapter package is exfiltrated to an unauthorized node, HKDF key derivation produces a mismatched key, causing AES-256-GCM authentication tag failure and **instant termination before model weight extraction**.

> **Security scope**: The device fingerprint is a *software-derived identity*, not a hardware root of trust. An attacker with root access to the authorised machine can reproduce all three entropy sources. The deployment salt (`P3_DEVICE_SALT`) is the only true secret material. VM cloning or OS migration to equivalent hardware breaks binding by changing `/etc/machine-id`.

---

## Technical Module Specifications

### 1. Hybrid PII Detection & Masking Engine (`src/security/pii_engine.py`)
- **Dual Engine Architecture**: Combines contextual SpaCy NLP entity extraction with high-precision regular expressions.
- **Entity Coverage**: Automatically redacts Names (`NAME_PHRASE`), Social Security Numbers (`SSN`), Email Addresses (`EMAIL`), Phone Numbers (`PHONE`), IP Addresses (`IP_ADDRESS`), Credit Card Numbers (`CREDIT_CARD`), and Secret API Keys (`API_KEY`).
- **Canonical Anchoring**: Calculates SHA-256 digests before and after redaction to anchor record lineage into immutable integrity traces.

### 2. Differentially Private LoRA Fine-Tuning (`src/phase2/dp_trainer.py`, `train_lora.py`)

SecureLoRA supports two training modes with identical dataset, seed, model, split, and evaluation so results are scientifically comparable:

**Mode A — Standard LoRA** (default): HuggingFace Trainer, no privacy mechanism.

**Mode B — DP-LoRA** (`DP_ENABLED=1`): Opacus 1.5.x PrivacyEngine wraps only the trainable LoRA parameters (frozen base model parameters are never touched by the DP mechanism).

| DP component | Implementation | Reference |
|---|---|---|
| Per-example gradients | `opacus.GradSampleModule` | Goodfellow 2015 |
| Gradient clipping | $\tilde{g}_i = g_i \cdot \min(1, C/\|g_i\|_2)$ | Abadi et al. 2016 |
| Gaussian noise | $\mathcal{N}(0, \sigma^2 C^2 I)$ per step | Abadi et al. 2016 |
| Privacy accountant | Rényi DP (rdp) or PRV/f-DP (prv) | Gopi et al. 2021 |
| ε computation | `privacy_engine.get_epsilon(delta)` — real accountant, never hardcoded | Mironov 2017 |

**Configuration** (env-vars or `config/training.yaml`):

```bash
export DP_ENABLED=1              # activate DP-LoRA
export DP_TARGET_EPSILON=8.0     # target ε (actual ε from accountant may differ)
export DP_TARGET_DELTA=1e-5      # failure probability δ
export DP_MAX_GRAD_NORM=1.0      # per-example clipping norm C
export DP_ACCOUNTANT=rdp         # rdp (Rényi) or prv (PRV)
```

> **Important**: DP-LoRA addresses *training-data privacy* (membership inference resistance). It does NOT protect the adapter from theft — that is the role of device-bound encryption (Phase 3/4). These are orthogonal mechanisms.

### 3. Adaptive Device-Bound Adapter Authorization System (`src/security/fingerprint.py`, `device_auth_policy.py`, `key_derivation.py`)

Rather than relying on fragile static matching, SecureLoRA features an **Adaptive, Policy-Driven Device Authorization Engine** that classifies device entropy attributes into stability tiers:

- **Stable Identity Features** (`machine_id`, `cpu_model`): Attributes expected to survive reboots, process restarts, and network topology changes. Any mutation in a stable feature constitutes a *Sensitive Event* (e.g., hypervisor migration, VM cloning, CPU swap) and immediately triggers the `UNAUTHORIZED` state.
- **Semi-Stable Features** (`disk_uuid`, `hostname`, `network_interface`): Attributes that may mutate during legitimate system maintenance.

#### Policy-Driven State Machine

```
              ┌──────────────────────────────────────┐
              │             AUTHORIZED               │
              └──────────────────┬───────────────────┘
                                 │ Semi-stable feature changed
                                 │ (within policy allowed limits)
                                 ▼
              ┌──────────────────────────────────────┐
              │      REAUTHORIZATION_REQUIRED        │
              └──────────────────┬───────────────────┘
                                 │ Valid Admin Token Provided
                                 ▼
                      [ Back to AUTHORIZED ]

           (Stable feature changed OR policy disallowed change)
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │             UNAUTHORIZED             │
              └──────────────────────────────────────┘
```

#### Authorization Policy & Configuration (`config/security.yaml` or env vars)

```yaml
binding_policy:
  strictness: "high"                    # high, medium, low
  allowed_feature_changes:
    network_interface: true             # Allow MAC changes under REAUTHORIZATION_REQUIRED
    hostname: false                     # Block unapproved hostname changes
    machine_id: false                   # Block machine-id changes
    disk_uuid: false                    # Block storage disk swaps
```

- **Administrative Reauthorization Workflow**: Controlled recovery from `REAUTHORIZATION_REQUIRED` requires providing a cryptographically verified admin token (`P3_ADMIN_REAUTH_TOKEN`). Direct or silent auto-recovery to `AUTHORIZED` is strictly prohibited.
- **HKDF-SHA256 Key Derivation**: Feeds the fingerprint digest and a secret deployment salt (`P3_DEVICE_SALT`) into HKDF (RFC 5869) with context string `securelora-adapter-v1` to derive 256-bit symmetric keys transiently in volatile RAM.
- **KDF Versioning**: Every package manifest records `kdf_version` (`hkdf-sha256-v1`). The deployment gate rejects packages with unsupported KDF versions.

#### Experimental Evaluation Matrix (10 Operational Scenarios)

| Scenario | State | Stability | Security Impact | Recovery Action |
| :--- | :---: | :---: | :--- | :--- |
| **Same device across reboot** | `AUTHORIZED` | `STABLE` | Zero impact; expected operation | N/A (Automatic Authorization) |
| **Network interface change** | `REAUTHORIZATION_REQUIRED` | `SEMI_STABLE_CHANGED` | Low; network interface swap allowed | Admin Token (`P3_ADMIN_REAUTH_TOKEN`) Approved |
| **Hostname change** | `UNAUTHORIZED` | `SEMI_STABLE_CHANGED` | Medium; unapproved hostname mutation blocked | Re-package adapter or update policy |
| **Disk replacement** | `UNAUTHORIZED` | `SEMI_STABLE_CHANGED` | High; storage volume swap detected & blocked | Re-package on new disk baseline |
| **Machine-id replacement** | `UNAUTHORIZED` | `UNSTABLE_STABLE_CHANGED` | Critical; OS installation identity replaced | Rejected; full re-registration required |
| **VM clone execution** | `UNAUTHORIZED` | `UNSTABLE_STABLE_CHANGED` | Critical; image cloning onto foreign CPU rejected | Rejected; target node authorization failed |
| **Unmapped container execution** | `UNAUTHORIZED` | `UNSTABLE_STABLE_CHANGED` | High; isolated container runtime rejected | Mount host `/etc/machine-id` volume |
| **Simulated foreign hardware** | `UNAUTHORIZED` | `UNSTABLE_STABLE_CHANGED` | Critical; total hardware mismatch (adapter theft) | Blocked (AES-GCM Auth Tag Failure) |
| **Spoofed fingerprint values** | `UNAUTHORIZED` | `UNSTABLE_STABLE_CHANGED` | High; spoofed strings fail hash check | Deployment Salt & Fingerprint Match Required |
| **Missing entropy sources** | `UNAUTHORIZED` | `MISSING_IDENTIFIERS` | High; environment lacking all identifiers rejected | Restore OS access to `/etc/machine-id` |

> **Security Limitation Explicit Statement**: Device binding provides *software-derived device identity binding*, NOT hardware-backed attestation (e.g., TPM / SGX enclave). An attacker with root privilege can spoof machine attributes. Security relies on the combined secrecy of `P3_DEVICE_SALT` and HKDF-SHA256 authentication.


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
| **Adapter Theft** | Adversary copies `.tar.gz` package to foreign machine | HKDF key derived on foreign machine produces invalid 256-bit AES key | **GCM Auth Tag Failure**: Instant termination before weight extraction. *Caveat: VM cloning or root access to the authorised machine can reproduce the fingerprint.* |
| **In-Transit Corruption** | MITM adversary alters encrypted payload bits during network transit | AES-256-GCM authentication tag validation | **Tamper Detected**: Integrity gate blocks execution |
| **Package Tampering** | Adversary modifies package manifest or signature | RSA-2048-PSS digital signature verification | **Signature Mismatch**: Gate 3 rejects deployment package |
| **PII Memorization** | LLM memorizes SSNs, emails, or phone numbers from training data | Phase 1 Hybrid SpaCy NER + ISO Regex Masking | **Reduced leakage risk**: Model trains on `[REDACTED_*]` tokens. Masking accuracy is not 100%. |
| **Cold-Boot Memory Extraction** | Adversary inspects disk for residual plaintext adapter weights | AES-GCM decryption to temp dir + 3-pass file shredding on context exit | **Minimised disk residue**: Plaintext exists in a temporary directory during PEFT loading; shredded immediately after. Swap, CoW filesystems, or suspend may retain fragments. |

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
