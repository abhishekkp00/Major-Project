# Secure Device-Bound LoRA Fine-Tuning Framework for LLMs

This framework implements a secure, device-bound dataset ingestion, preprocessing, encryption, and Parameter-Efficient Fine-Tuning (PEFT) LoRA pipeline for Large Language Models. 

The security architecture guarantees a zero-plaintext-at-rest policy. Raw training text is encrypted via AES-256-GCM, loaded and decrypted streamingly into memory during training sessions, and automatically shredded at the byte level when training completes or crashes. Only adapter weights are exported.

---

## 1. Directory Structure

```
MAJOR_PROJECT/
│
├── secure_lora/               # Crypto & Ingestion Engine (Phase 1)
│   ├── __init__.py
│   ├── cli.py                 # Command line tools
│   ├── ingestion.py           # Ingestion for CSV, TXT, JSON, MD
│   ├── pipeline.py            # Phase 1 manager
│   ├── preprocessing.py       # Data cleansing
│   └── security.py            # AES-256-GCM and shredding utils
│
├── utils/                     # Training Helpers (Phase 2)
│   ├── checkpoint_utils.py    # Rotations & crash recovery
│   └── logging_utils.py       # Structured logging
│
├── tests/
│   ├── test_pipeline.py       # Phase 1 integration tests
│   └── test_phase2.py         # End-to-end training tests
│
├── config.py                  # Environment-driven training config
├── train_lora.py              # Main training workflow orchestrator
├── dashboard.py               # Flask-based web-based validation dashboard
├── ingest_openpii.py          # Hugging Face OpenPII dataset ingestion helper
├── requirements.txt           # Dependency specifications
├── .env                       # Local environment secrets (Git-ignored)
└── .gitignore                 # Directory and secret file exclusions
```

---

## 2. Environment Variables (`.env`)

Configure default parameters in a `.env` file at the root level to run commands without repetitive parameters:

```env
# Cryptography
SECURE_LORA_KEY_HEX=your_secure_hex_key_here

# Dataset & Paths
SECURE_LORA_ENCRYPTED_DATA=encrypted_real_data/encrypted_dataset.enc
SECURE_LORA_METADATA_PATH=encrypted_real_data/dataset_metadata.json
SECURE_LORA_OUTPUT_DIR=lora_adapters
SECURE_LORA_CHECKPOINT_DIR=checkpoints

# Hyperparameters
SECURE_LORA_MODEL_NAME=JackFram/llama-68m
SECURE_LORA_BATCH_SIZE=2
SECURE_LORA_GRAD_ACCUM=4
SECURE_LORA_LR=2e-4
SECURE_LORA_EPOCHS=3
SECURE_LORA_MAX_LEN=256
SECURE_LORA_SEED=42
SECURE_LORA_R=8
SECURE_LORA_ALPHA=16
SECURE_LORA_DROPOUT=0.05

# Dashboard
SECURE_LORA_DASHBOARD_PORT=5005
```

---

## 3. Quickstart Guide

### 1. Ingest and Encrypt OpenPII Masking Dataset
Fetch OpenPII samples from Hugging Face, format them to instruction-response targets, and encrypt them using AES-256-GCM:
```bash
# Ingest first 150 samples (configurable with -l)
PYTHONPATH=. venv/bin/python3 ingest_openpii.py --limit 150
```

### 2. Run Secure Fine-Tuning
Execute the fine-tuning run. The model loads base LLM weights in CPU memory, streamingly decrypts data directly to RAM, injects LoRA adapters, and trains.
```bash
PYTHONPATH=. venv/bin/python3 train_lora.py
```
*If interrupted, launching the script again will automatically detect the latest checkpoint in `checkpoints/` and resume training.*

### 3. Launch Web Validation Dashboard
Run the dashboard to inspect metrics (Val Loss, Perplexity) and generate side-by-side completions from the baseline base model and the fine-tuned LoRA model:
```bash
PYTHONPATH=. venv/bin/python3 dashboard.py
```
Navigate to **`http://127.0.0.1:5005`** in your browser.

---

## 4. Key Security & Operational Implementations

### A. Ephemeral Decryption
Plaintext data is decrypted only into transient memory buffers inside `decrypted_temporary_file`. Once the tokenization block finishes, the temporary storage is overwritten with random bytes (`os.urandom`) and unlinked. 

### B. Safe Checkpoint Rotations
To prevent disk exhaustion during training, the `SecureCheckpointCallback` checks checkpoints after every epoch and calls `rotate_checkpoints`, keeping only the latest two checkpoints and cleanly unlinking older iterations.

### C. Evaluation & Reports
After training concludes, a validation evaluation runs to compute validation loss and perplexity metrics. Results are saved to `eval_report.json` and immediately rendered in the pipeline dashboard.
