# SecureLoRA Setup & Installation Guide

This document provides explicit, step-by-step instructions for installing, configuring, and verifying the SecureLoRA framework on a fresh developer system.

---

## 1. System Requirements

*   **Operating System**: Linux (Ubuntu 20.04 LTS / 22.04 LTS recommended)
*   **Python**: Python 3.10, 3.11, or 3.12
*   **RAM**: 8 GB minimum (16 GB recommended for multi-model evaluation)
*   **Storage**: 5 GB available disk space
*   **GPU**: Optional (CPU execution fully supported for lightweight 68M model evaluation)

---

## 2. Environment Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/abhishekkp00/Major-Project.git
cd Major-Project
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify core libraries:
```bash
python -c "import torch, transformers, peft, cryptography, opacus; print('All core dependencies imported successfully!')"
```

---

## 3. Environment Variable Configuration

Copy the environment configuration template:
```bash
cp .env.example .env
```

Generate cryptographically secure keys for encryption and hardware binding:
```bash
# Generate 32-byte (64-char hex) dataset encryption key
export SECURE_LORA_KEY_HEX=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Generate 32-byte secret salt for HKDF hardware binding
export P3_DEVICE_SALT=$(python3 -c "import secrets; print(secrets.token_hex(32))")
```

Update your `.env` file with these values:
```env
SECURE_LORA_KEY_HEX=<your_hex_key>
P3_DEVICE_SALT=<your_salt_key>
SECURE_LORA_INPUT_DIR=real_data_inputs
SECURE_LORA_OUTPUT_DIR=encrypted_real_data
P3_ADAPTER_INPUT_DIR=outputs/final_adapter
P3_PROTECTED_OUTPUT_DIR=outputs/protected_adapter
P3_RSA_KEY_BITS=2048
P3_RSA_PRIVATE_KEY_PATH=outputs/protected_adapter/dev_private.pem
P3_RSA_PUBLIC_KEY_PATH=outputs/protected_adapter/public.pem
P3_ADAPTER_ID=lora-adapter-v1
P3_MODEL_REFERENCE=JackFram/llama-68m
DASHBOARD_PORT=5005
```

---

## 4. Output Directories Initialization

Ensure all required runtime and evaluation directories exist:
```bash
mkdir -p encrypted_real_data real_data_inputs outputs/final_adapter outputs/protected_adapter outputs/evaluation/privacy outputs/evaluation/screening outputs/evaluation/adaptive_evasion outputs/evaluation/device_binding outputs/evaluation/model_scale outputs/evaluation/statistics outputs/evaluation/archive logs
```

---

## 5. System Verification

Run the test suite to confirm installation:
```bash
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```

If all tests pass, system setup is complete and fully verified.
