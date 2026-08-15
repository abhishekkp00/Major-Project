# 🏗️ SecureLoRA Project Structure

This document details the modular layout of the **SecureLoRA** codebase.

```
MAJOR_PROJECT/
│
├── ⚙️ config/                           # YAML Configuration System
│   ├── app.yaml                        # Application & Dashboard settings
│   ├── training.yaml                   # LoRA hyperparameters (rank, alpha, lr, epochs, base model)
│   ├── security.yaml                   # Cryptographic algorithms & key derivation specs
│   └── deployment.yaml                 # Phase 4 deployment validation rules
│
├── 📦 src/                             # Core Python Package Source
│   ├── common/                         # Configuration reader & exception definitions
│   │   ├── config_loader.py            # Centralized YAML + environment configuration loader
│   │   └── exceptions.py               # Enterprise exception hierarchy
│   │
│   ├── security/                       # Cryptographic & Privacy Bedrock
│   │   ├── pii_engine.py               # HybridPIIEngine (SpaCy/Presidio NER + ISO/RFC regex patterns)
│   │   ├── crypto.py                   # Streaming AES-256-GCM encryption & decryption
│   │   ├── fingerprint.py              # Hardware identification & HKDF salt generation
│   │   ├── key_derivation.py           # Hardware-bound key derivation (RAM-only)
│   │   ├── signature.py                # RSA-2048-PSS digital signing & verification
│   │   └── shred.py                    # Multi-pass DoD 5220.22-M secure file shredding
│   │
│   ├── phase1/                         # Data Protection & Ingestion
│   │   ├── ingestion.py                # Multi-format data reader (.jsonl, .csv, .txt, .json)
│   │   ├── preprocessing.py            # Schema standardization & cleaning
│   │   └── pipeline.py                 # RAM-only PII masking & dataset encryption
│   │
│   ├── phase2/                         # In-Memory LoRA Fine-Tuning
│   │   └── train_lora.py               # PyTorch + HuggingFace PEFT training loop
│   │
│   ├── phase3/                         # Hardware Binding & Package Generation
│   │   ├── package_builder.py          # Builds signed deployment archive
│   │   ├── adapter_encryptor.py        # Hardware-bound adapter weight encryption
│   │   └── verifier.py                 # Pre-packaging security verification
│   │
│   ├── phase4/                         # Secure Deployment Gateway
│   │   ├── package_validator.py        # 8-step cryptographic gate verifier
│   │   ├── device_auth.py              # Hardware fingerprint authorization
│   │   ├── decryptor.py                # In-RAM adapter weight decryption context
│   │   ├── adapter_loader.py           # Loads PEFT adapter into base LLM
│   │   └── inference_runner.py         # Side-by-side inference engine
│   │
│   ├── orchestrator/                   # Job Orchestration & Analytical Chat
│   │   ├── service.py                  # Multi-job workflow runner
│   │   ├── dataset_processor.py        # Ingestion, validation, and PII masking pipeline
│   │   ├── chat_engine.py              # Dual-mode Q&A engine (Fine-tuned LLM + Aggregate analytics)
│   │   ├── security_orchestrator.py    # Pipeline status manager
│   │   └── routes.py                   # Flask orchestration endpoints
│   │
│   ├── evaluation/                     # Monitoring & Research Metrics
│   │   ├── dashboard.py                # Interactive web dashboard server
│   │   ├── pii_metrics.py              # PII detection precision, recall, F1 computation
│   │   ├── crypto_benchmark.py         # Cryptographic latency & throughput benchmark suite
│   │   ├── baseline_comparison.py      # Security comparison against alternative frameworks
│   │   └── threat_model.py             # STRIDE threat model & attack simulation suite
│   │
│   └── utils/                          # Common Utilities
│       ├── checkpoint_utils.py         # Checkpoint rotation & rotation policy
│       └── logging_utils.py            # Structured logging setup
│
├── 🧪 tests/                           # Pytest Test Suite
│   ├── unit/                           # Unit tests for crypto, PII, and config
│   ├── integration/                    # End-to-end pipeline phase tests
│   └── security/                       # Hardware-binding & attack tests
│
├── 📂 samples/                         # Demonstration Datasets
│   ├── sample_medical_phi.jsonl        # Clinical health notes with PHI
│   ├── sample_pii_data.jsonl           # Corporate employee records
│   └── real_world_pii.jsonl            # Real-world benchmark records
│
├── 📜 scripts/                         # Helper & Evaluation Executables
│   ├── run_paper_evaluation.py         # Master evaluation runner
│   ├── generate_paper_figures.py       # IEEE publication diagram generator
│   └── download_real_pii.py            # Benchmark dataset fetcher
│
├── 📓 train_and_evaluate_lora.ipynb    # Interactive End-to-End Jupyter Notebook
├── 📋 README.md                        # Master Project Documentation
├── 📄 SECURITY.md                      # Security Policy & Vulnerability Reporting
├── 🐍 requirements.txt                 # Project Dependencies
└── ⚙️ .env.example                     # Environment Template
```
