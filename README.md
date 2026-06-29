# Secure Device-Bound LoRA Fine-Tuning Framework

A lightweight, secure fine-tuning framework for training Large Language Models locally. This project enables organizations to run Parameter-Efficient Fine-Tuning (PEFT/LoRA) on private datasets while maintaining a strict zero-plaintext-at-rest security policy. 

All training data remains encrypted on disk using AES-256-GCM. During a training session, data is streamingly decrypted in memory, tokenized, and the temporary plaintext buffers are immediately shredded to prevent disk leaks. The output is a lightweight set of LoRA adapter weights, keeping the base model frozen and unmodified.

---

## Features

- **Zero-Plaintext-at-Rest**: Private datasets are processed and stored encrypted. Decryption only happens dynamically in memory.
- **PEFT LoRA Training**: Keeps the base LLM completely frozen and only trains lightweight adapter weights.
- **Crash-Safe & Resumable**: Automatically checkpoints the training state, rotating checkpoints to save disk space, and resumes seamlessly if interrupted.
- **Interactive Validation Dashboard**: A local web interface to run training, monitor validation metrics, and compare responses between the base model and the fine-tuned adapter side by side.
- **Dataset Ingestion Tools**: Built-in support to download, format, and encrypt instruction datasets (like Hugging Face OpenPII).

---

## Directory Layout

```
MAJOR_PROJECT/
│
├── secure_lora/               # Crypto & ingestion backend engine
├── utils/                     # Logging and checkpoint helpers
├── tests/                     # Validation & pipeline integration tests
├── config.py                  # Environment settings and validations
├── train_lora.py              # Training workflow manager
├── dashboard.py               # Local metrics and validation web portal
├── ingest_openpii.py          # Hugging Face OpenPII dataset helper
├── requirements.txt           # Unified dependency specifications
└── README.md                  # Project overview (This file)
```

---

## Quickstart Guide

### 1. Setup Environment
Initialize your virtual environment and install the package requirements:
```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### 2. Ingest OpenPII Masking Dataset
Download a sample dataset from Hugging Face, format it, and save it as an encrypted dataset:
```bash
PYTHONPATH=. venv/bin/python3 ingest_openpii.py --limit 150
```

### 3. Run Fine-Tuning
Start the local fine-tuning run on the encrypted dataset:
```bash
PYTHONPATH=. venv/bin/python3 train_lora.py
```
*If interrupted, restarting the command will automatically pick up from the latest saved checkpoint.*

### 4. Open Validation Dashboard
Launch the web interface to check training logs, evaluate validation perplexity, and run side-by-side prompt comparisons:
```bash
PYTHONPATH=. venv/bin/python3 dashboard.py
```
Navigate to **`http://127.0.0.1:5005`** in your browser.
