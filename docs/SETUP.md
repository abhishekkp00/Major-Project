# SecureLoRA Setup, Enrollment & Operational Lifecycle Guide

This document provides explicit, step-by-step instructions for installing, configuring, enrolling device secrets, training, packaging, deploying, recovering from failures, and verifying the SecureLoRA framework.

---

## 1. System Requirements

* **Operating System**: Linux (Ubuntu 20.04 LTS / 22.04 LTS / 24.04 LTS recommended; POSIX filesystem with file locking support)
* **Python**: Python 3.10, 3.11, or 3.12
* **RAM**: 8 GB minimum (16 GB recommended for multi-model evaluation)
* **Storage**: 5 GB available disk space
* **GPU**: Optional (CPU execution fully supported for lightweight 68M model evaluation)

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

Generate cryptographically secure keys for encryption and device binding:
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

Ensure all runtime directories exist:
```bash
mkdir -p encrypted_real_data real_data_inputs outputs/final_adapter outputs/protected_adapter outputs/evaluation/privacy outputs/evaluation/screening outputs/evaluation/adaptive_evasion outputs/evaluation/device_binding outputs/evaluation/model_scale outputs/evaluation/statistics outputs/evaluation/archive logs
```

---

## 4. Secret & Device Enrollment

Before training or packaging, initialize the host device secret and generate RSA provenance signing keys:

### 4.1 Initialize Host Device Secret
```bash
# Enrolls the local host secret (.device_secret file with POSIX permissions 0600)
python3 -c "
from src.security.device_secret import initialize_device_secret
path = initialize_device_secret(force=False)
print(f'Device secret enrolled at: {path}')
"
```

### 4.2 Generate RSA-2048 Provenance Keypair
```bash
# Generate 2048-bit RSA private and public keys for package signing
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out outputs/protected_adapter/dev_private.pem
openssl rsa -pubout -in outputs/protected_adapter/dev_private.pem -out outputs/protected_adapter/public.pem
chmod 600 outputs/protected_adapter/dev_private.pem
```

---

## 5. Training Lifecycle (Phase 1 PII Redaction + Phase 2 DP-LoRA)

Train a privacy-preserving low-rank adapter on benchmark data with Opacus DP-SGD:

```bash
python3 -m src.phase2.train_lora \
  --model_name "JackFram/llama-68m" \
  --data_path "synthetic_pii_benchmark.jsonl" \
  --output_dir "outputs/final_adapter" \
  --batch_size 4 \
  --num_epochs 1 \
  --lr 5e-4 \
  --target_epsilon 2.5 \
  --target_delta 1e-5 \
  --max_grad_norm 1.0 \
  --noise_multiplier 1.2
```

Outputs:
* `outputs/final_adapter/adapter_model.safetensors` (or `adapter_model.bin`)
* `outputs/final_adapter/adapter_config.json`

---

## 6. Packaging Lifecycle (Phase 3 Screening + Binding + Signing)

Screen the adapter weights, derive host-bound keys, and create the cryptographically signed `.tar.gz` package:

```bash
python3 -m src.phase3.main \
  --adapter-id "lora-adapter-v1" \
  --adapter-path "outputs/final_adapter" \
  --model-ref "JackFram/llama-68m" \
  --private-key "outputs/protected_adapter/dev_private.pem" \
  --output-dir "outputs/protected_adapter" \
  --device-salt "$P3_DEVICE_SALT"
```

Package Outputs:
* `outputs/protected_adapter/secure_lora_package.tar.gz` containing:
  * `package_manifest.json` (canonical metadata, sequence number, digest, RSA-PSS signature)
  * `adapter_payload.bin` (AES-256-GCM encrypted adapter archive)

---

## 7. Deployment & Ephemeral Inference (Phase 4)

Authorize the host environment, verify anti-replay sequence tracking, and execute ephemeral inference:

```bash
python3 -m src.phase4.inference_runner \
  --package-path "outputs/protected_adapter/secure_lora_package.tar.gz" \
  --public-key "outputs/protected_adapter/public.pem" \
  --device-salt "$P3_DEVICE_SALT" \
  --prompt "Summarize medical intake notes for patient."
```

During execution:
1. Manifest signature is verified via RSA-2048-PSS over canonical metadata and ciphertext digest.
2. Anti-replay sequence number is verified and committed to `outputs/.deployment_state.json`.
3. Host fingerprint is evaluated against the deployment authorization policy.
4. Ephemeral adapter is unpacked into a temporary directory (POSIX `0700` directory, `0600` files).
5. Weights are loaded into `PeftModel` and immediately overwritten with 3 passes of random bytes and unlinked.

---

## 8. Failure Recovery Procedures

### 8.1 Corrupted Anti-Replay State Recovery
If deployment fails with `DeploymentStateIntegrityError` due to a truncated or corrupted `outputs/.deployment_state.json`:

```bash
# 1. Quarantine the damaged state file (NEVER delete or wipe automatically)
mv outputs/.deployment_state.json outputs/.deployment_state.json.corrupt.$(date +%s)

# 2. Inspect deployment audit logs to verify the highest sequence number deployed per adapter
```

Re-initialize state explicitly with authorization:
```python
from src.security.provenance import AntiReplayTracker
tracker = AntiReplayTracker()
tracker.initialize_state(
    force=True,
    initial_adapters={
        "lora-adapter-v1": {"last_sequence": 4, "processed_packages": ["pkg-004"]}
    }
)
```

### 8.2 Device Drift Recovery under Adaptive Policy
If an authorized machine undergoes legitimate hardware maintenance (e.g. CPU or network interface replacement):
1. Under **Adaptive Policy**, partial drift within configured tolerance thresholds is accepted automatically, updating the enrolled host profile.
2. If drift exceeds policy thresholds, the deployment will fail fast. An administrator must generate a new deployment package bound to the updated device fingerprint with an incremented `sequence_number`.

---

## 9. Test Suite Execution Commands

Run the test suites corresponding to your operational tier:

```bash
# 1. Mandatory Application Security & Cryptographic Gate (78 tests)
PYTHONPATH=. ./venv/bin/pytest tests/security/ -v

# 2. Unit & Research Evaluator Suite (485 tests)
PYTHONPATH=. ./venv/bin/pytest tests/unit/ tests/test_*.py -v

# 3. Pipeline Integration Tests (10 tests)
PYTHONPATH=. ./venv/bin/pytest tests/integration/ -v

# 4. Complete Test Suite (575 tests)
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```
