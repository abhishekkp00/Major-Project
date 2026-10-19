# SecureLoRA Technical Assurance & Threat Model Specification

**Target System**: SecureLoRA (`abhishekkp00/Major-Project`)  
**Auditor & Systems Architecture Review**: October 19, 2026  
**Audited Commit**: `2e576957388aa5c1d63e9f45ec8c61e4663675a6`  
**Automated Test Suite**: 575 / 575 Passing Tests (`pytest tests/`)  
**Continuous Integration Gate**: GitHub Actions [Run #39218383185](https://github.com/abhishekkp00/Major-Project/actions/runs/39218383185) (100% Green across Python 3.10, 3.11, 3.12)

---

## 1. Traceability Matrix: Documented Lifecycle Steps to Implementation

Each documented lifecycle step in SecureLoRA is mapped to its concrete implementation module, entry point, configuration parameters, and test verification suite:

| Lifecycle Step | Implementation Module | Primary Entry Point | Configuration Keys | Test Suite Reference |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0: Ingestion & Schema Normalization** | `src/orchestrator/service.py`<br>`src/data_sources/` | `OrchestratorService.create_job()`<br>`OrchestratorService.add_dataset_file_stream()` | `ALLOWED_DATASET_EXTENSIONS`<br>`MAX_UPLOAD_SIZE_BYTES` (50MB) | `tests/unit/test_dataset_upload_hardening.py`<br>`tests/test_dataset_adapters.py` |
| **Stage 1: PII Audit & Masking** | `src/phase1/pii_scrubber.py`<br>`src/phase1/spacy_engine.py`<br>`src/phase1/presidio_engine.py` | `PIIScrubber.scrub_text()`<br>`HybridPIIEngine.redact()` | `config/pii_config.yaml`<br>`PII_CONFIDENCE_THRESHOLD` | `tests/unit/test_pii_name_masking.py`<br>`tests/unit/test_pii_fail_closed.py` |
| **Stage 2: Dataset Cryptographic Protection** | `src/security/crypto.py`<br>`src/phase1/main.py` | `encrypt_dataset_at_rest()`<br>`AESGCMEncryptor.encrypt()` | `SECURE_LORA_KEY_HEX`<br>`SECURE_LORA_OUTPUT_DIR` | `tests/security/test_crypto.py`<br>`tests/integration/test_phase1_pipeline.py` |
| **Stage 3: DP-LoRA Privacy Fine-Tuning** | `src/phase2/train_lora.py`<br>`src/phase2/dp_optimizer.py` | `run_training()`<br>`DPOptimizer.step()` | `config/dp_config.yaml`<br>`target_epsilon` (2.5), `max_grad_norm` (1.0), `noise_multiplier` (1.2) | `tests/unit/test_phase2_dp.py`<br>`tests/integration/test_phase2_train.py` |
| **Stage 4: Pre-Packaging Security Screening** | `src/security/adapter_screening/`<br>`src/evaluation/adapter_security.py` | `pre_packaging_screening_gate()`<br>`ScreeningPipeline.screen_adapter()` | `risk_threshold` ($\tau=0.35$)<br>`_SCREENING_MODE="production"`<br>`enable_screening=True` | `tests/unit/test_adapter_screening.py`<br>`tests/unit/test_phase3_screening_integration.py` |
| **Stage 5: Cryptographic Provenance Signing** | `src/security/signature.py`<br>`src/security/provenance.py` | `sign_manifest()`<br>`compute_canonical_manifest_digest()` | `P3_RSA_PRIVATE_KEY_PATH`<br>`P3_RSA_KEY_BITS` (2048) | `tests/security/test_signature.py`<br>`tests/unit/test_provenance_anti_replay.py` |
| **Stage 6: Device-Bound Packaging** | `src/phase3/package_builder.py`<br>`src/security/kdf.py`<br>`src/security/fingerprint.py` | `package_adapter()`<br>`derive_device_key()` | `P3_DEVICE_SALT`<br>`device_secret` (`.device_secret`) | `tests/security/test_device_bound_key_lifecycle.py`<br>`tests/security/test_kdf_hkdf.py` |
| **Stage 7: Host Authorization Verification** | `src/security/device_auth_policy.py`<br>`src/phase4/device_authorizer.py` | `DeviceAuthorizationPolicy.evaluate()`<br>`authorize_host_environment()` | `binding_policy_version`<br>`drift_tolerance_threshold` | `tests/unit/test_device_auth_policy.py`<br>`tests/unit/test_phase4_device_auth.py` |
| **Stage 8: Package Integrity & Anti-Replay Gate** | `src/phase4/package_validator.py`<br>`src/security/provenance.py` | `validate_package()`<br>`AntiReplayTracker.record_deployment()` | `outputs/.deployment_state.json`<br>`manifest["sequence_number"]` | `tests/unit/test_phase4_package_validator.py`<br>`tests/unit/test_provenance_anti_replay.py` |
| **Stage 9: Secure Ephemeral Adapter Loading** | `src/phase4/adapter_loader.py`<br>`src/phase4/inference_runner.py` | `DecryptedAdapterContext`<br>`InferenceRunner.run_inference()` | `SECURELORA_TEMP_DIR`<br>`shred_passes=3` | `tests/unit/test_phase4_adapter_loader.py`<br>`tests/security/test_shred.py` |

---

## 2. Technical Assurance: Rigorous Clarification of Security Claims

To preserve academic and engineering integrity, the following technical clarifications delineate the exact security guarantees from common misconceptions:

### 2.1 Ephemeral Plaintext vs. "RAM-Only Loading"
* **Technical Reality**: `peft.PeftModel.from_pretrained()` and standard PyTorch backends require a directory path on the local filesystem containing configuration (`adapter_config.json`) and weight matrices (`adapter_model.safetensors` or `adapter_model.bin`).
* **Implementation Mechanism**: Decrypted adapter weights are written to a temporary directory created with restricted POSIX permissions (mode `0700` for directory, `0600` for files). This plaintext window exists strictly for the duration of the Python `DecryptedAdapterContext` context manager block.
* **Explicit Boundary**: SecureLoRA does **NOT** perform pure in-RAM zero-disk adapter loading. Plaintext exists transiently on disk during the model loading phase before being shredded and unlinked. Operators requiring memory-only execution must mount `SECURELORA_TEMP_DIR` to a `tmpfs` RAM disk (`/dev/shm`).

### 2.2 Best-Effort Overwrite vs. "Guaranteed Physical Shredding"
* **Technical Reality**: SecureLoRA's `shred_file()` executes a 3-pass overwrite with random bytes (`os.urandom`) followed by `os.unlink()`.
* **Hardware & Filesystem Limitations**:
  - **Flash Translation Layer (FTL)**: Modern solid-state drives (SSDs, NVMe) utilize wear leveling and bad-block remapping. In-place software writes are written to new physical flash cells, leaving data in older cells until garbage collected.
  - **Copy-on-Write (CoW) Filesystems**: On Btrfs, ZFS, and APFS, overwrites allocate new blocks rather than overwriting existing data.
  - **Journaling & Snapshots**: Ext4 with `data=journal`, LVM snapshots, or cloud volume snapshots preserve prior block states.
  - **OS Swap**: If the OS pages volatile process memory or disk buffers to an unencrypted swap partition, plaintext remnants can persist.
* **Explicit Boundary**: Overwrite-before-unlink is a defense-in-depth mitigation against simple forensic file recovery on magnetic media. It is **NOT** a certified physical sanitization guarantee on modern SSDs or snapshotting filesystems.

### 2.3 Software-Derived Fingerprinting vs. "Hardware-Rooted / TPM Attestation"
* **Technical Reality**: Host environment identity is derived using `src/security/fingerprint.py`, which collects OS-accessible attributes:
  - `/etc/machine-id` (or `/var/lib/dbus/machine-id`)
  - `/proc/cpuinfo` (CPU model name and core count)
  - Primary network interface MAC address
  - Root storage filesystem UUID
  - Combined with a high-entropy secret salt (`P3_DEVICE_SALT`) via HKDF-SHA256
* **Explicit Boundary**:
  - There is **NO** hardware root of trust, TPM 2.0 PCR sealing, Intel SGX, or AMD SEV remote attestation implemented.
  - Software attributes can be inspected, altered, or spoofed by root users, kernel modules, or hypervisors.
  - VM cloning of an authorized virtual machine duplicates `/etc/machine-id` and MAC address, allowing clone execution unless hypervisor-level UUID tracking is maintained.

### 2.4 Conditional & Configurable Behavioral Screening vs. "Mandatory Behavioral Vetting"
* **Technical Reality**: Pre-packaging security screening evaluates both structural anomaly metrics (SVD rank drift, Frobenius norm deviations) and behavioral subspace shifts.
* **Operational Execution Path**:
  - **Structural Screening**: Evaluates static weight matrices and is always executed when screening is enabled.
  - **Behavioral Screening**: Requires a live model inference callable (`candidate_model_fn`). When `candidate_model_fn` is `None` during offline packaging, behavioral screening cannot execute live probes and marks `real_callable_used=False`.
  - **Bypass Modes**: Administrative override tokens (`APPROVED_WITH_OVERRIDE`) or the CLI `--force` flag allow operators to package adapters flagged as high or medium risk.
* **Explicit Boundary**: Behavioral screening is **conditional** upon having a live model runtime and is not mathematically unbypassable; administrative bypass mechanisms exist for research workflows.

### 2.5 Local Monotonic Sequence Tracking vs. "Global Network Anti-Replay Guarantees"
* **Technical Reality**: `AntiReplayTracker` (`src/security/provenance.py`) enforces strict monotonic sequence incrementing (`manifest["sequence_number"] > last_sequence`) and records package IDs in a local state store (`outputs/.deployment_state.json`) protected by `fcntl.flock` and `threading.RLock`.
* **Explicit Boundary**:
  - **Local Host Scope**: The guarantee protects against replay attacks on the *same host* (or nodes mounting the same shared filesystem).
  - **Decentralized Deployments**: It does **NOT** provide distributed Byzantine consensus across independent edge nodes. If a package is replayed against a separate node that has never seen sequence number $N$, that node's local state has no knowledge of prior deployments unless a centralized revocation ledger is deployed.
  - **Fail-Closed Behavior**: If `.deployment_state.json` is corrupted or unreadable, `AntiReplayTracker` raises `DeploymentStateIntegrityError` and aborts all deployments. It refuses to automatically reset state.

---

## 3. Verification Rigor Classification

The table below classifies every capability in SecureLoRA by its empirical verification level:

| System Capability | Classification | Verification Details & Evidence |
| :--- | :--- | :--- |
| **Hybrid PII Redaction** | Implemented, Unit-Tested, Experimentally Measured | Micro-F1 = 0.9610 on synthetic test records (`outputs/benchmarks/pii_metrics.json`). |
| **Generative LLM Memorization** | **Not-Yet-Executed** | Marked `NOT_EXECUTED` in `outputs/evaluation/privacy/comparison.json`. No live generative token extraction was run in offline CI. |
| **DP-LoRA Training ($\epsilon \le 2.5$)** | Implemented, Integration-Tested, Experimentally Measured | Verified $\epsilon=2.4430, \delta=10^{-5}$ using Opacus RDP accountant (`outputs/research/runs/EXP_E9_seed_42.json`). |
| **Structural Anomaly Screening** | Implemented, Unit-Tested, Experimentally Measured | 100% detection on Level 0/1; 0% detection on Level 2/3 adaptive evasion (`outputs/evaluation/adaptive_evasion/comparison.json`). |
| **Joint Adapter Screening** | Implemented, Unit-Tested, Experimentally Measured | F1 = 1.0000 on synthetic 40-sample multi-seed Trojan benchmark suite (`outputs/evaluation/adaptive_evasion/comparison.json`). |
| **Software Device Authorization** | Implemented, Unit-Tested, Experimentally Measured | 100% foreign device rejection; adaptive policy reduces FRR from 80% to 20% under simulated drift (`outputs/evaluation/device_binding/comparison.json`). |
| **Cryptographic Packaging Overhead** | Implemented, Unit-Tested, Experimentally Measured | AES-256-GCM (0.210 ms), Decrypt+HKDF (0.192 ms), RSA-PSS verify (0.051 ms) on 68M-tier (`outputs/research/runs/EXP_E9_seed_42.json`). |
| **Threat Model Attack Scenarios** | **Simulated** | SIM-01 through SIM-08 in `src/evaluation/threat_model.py` simulate attacks against synthetic test fixtures. |
| **SQLite Concurrent Job Store** | Implemented, Unit-Tested | Multi-threaded / multi-process SQLite transactional store verified with 10 concurrent workers (`tests/unit/test_job_store_concurrency.py`). |
| **Upload Streaming & Path Containment** | Implemented, Unit-Tested | Early extension checks and `is_relative_to` containment verified (`tests/security/test_appsec_hardening_regressions.py`). |
| **Malicious Pickle Exploit Blocking** | Implemented, Unit-Tested | `torch.load(..., weights_only=True)` blocks `__reduce__` exploit unpickling (`tests/security/test_appsec_hardening_regressions.py`). |

---

## 4. Synthetic Benchmark Scope & Generalization Limits

1. **Benchmark Size & Composition**:
   - The reported PII redaction metrics (Precision = 0.9737, Recall = 0.9487, Micro-F1 = 0.9610) were evaluated on `synthetic_pii_benchmark.jsonl`, consisting of $N=48$ annotated synthetic records.
   - The dataset contains synthetic entities (names, email addresses, SSNs, credit card numbers, medical record numbers, phone numbers).
2. **Generalization Constraints**:
   - These scores reflect performance on structured synthetic benchmark distributions.
   - They do **NOT** guarantee identical precision or recall across noisy, unconstrained, multi-lingual, or unstructured clinical and enterprise documents.
   - Zero-PII leakage into trained weights cannot be formally proven by redaction alone; formal privacy relies on DP-LoRA $(\epsilon, \delta)$ differential privacy bounds.

---

## 5. End-to-End Operational Runbook

### 5.1 Environment Setup
```bash
# 1. Clone repository and enter directory
git clone https://github.com/abhishekkp00/Major-Project.git
cd Major-Project

# 2. Initialize Python 3.10+ virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install locked dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Secret & Device Enrollment
Before training or packaging, initialize the host device secret and environment keys:
```bash
# 1. Create .env from template
cp .env.example .env

# 2. Generate cryptographically strong symmetric dataset key and device salt
export SECURE_LORA_KEY_HEX=$(python3 -c "import secrets; print(secrets.token_hex(32))")
export P3_DEVICE_SALT=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# 3. Initialize host device secret file (.device_secret with mode 0600)
python3 -c "
from src.security.device_secret import initialize_device_secret
path = initialize_device_secret(force=False)
print(f'Device secret enrolled at: {path}')
"

# 4. Generate RSA-2048 signing keypair for manifest provenance
mkdir -p outputs/protected_adapter
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out outputs/protected_adapter/dev_private.pem
openssl rsa -pubout -in outputs/protected_adapter/dev_private.pem -out outputs/protected_adapter/public.pem
chmod 600 outputs/protected_adapter/dev_private.pem
```

### 5.3 Training Execution (Phase 1 PII Redaction + Phase 2 DP-LoRA)
```bash
# Execute DP-LoRA training pipeline on synthetic benchmark data
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

### 5.4 Packaging Execution (Phase 3 Screening + Signing + Binding)
```bash
# Package the trained adapter with pre-packaging screening, RSA-PSS signature, and HKDF binding
python3 -m src.phase3.main \
  --adapter-id "lora-adapter-v1" \
  --adapter-path "outputs/final_adapter" \
  --model-ref "JackFram/llama-68m" \
  --private-key "outputs/protected_adapter/dev_private.pem" \
  --output-dir "outputs/protected_adapter" \
  --device-salt "$P3_DEVICE_SALT"
```

### 5.5 Deployment & Inference Validation (Phase 4)
```bash
# Verify host authorization, anti-replay, and execute ephemeral inference
python3 -m src.phase4.inference_runner \
  --package-path "outputs/protected_adapter/secure_lora_package.tar.gz" \
  --public-key "outputs/protected_adapter/public.pem" \
  --device-salt "$P3_DEVICE_SALT" \
  --prompt "Summarize medical intake notes for patient."
```

### 5.6 Failure Recovery Procedures

#### Procedure A: Anti-Replay Corrupted State File Recovery
If deployment fails with `DeploymentStateIntegrityError` due to a truncated or malformed `outputs/.deployment_state.json`:
1. **Quarantine damaged state**:
   ```bash
   mv outputs/.deployment_state.json outputs/.deployment_state.json.corrupt.$(date +%s)
   ```
2. **Inspect deployment logs**: Determine the highest valid `sequence_number` successfully deployed for each adapter.
3. **Explicitly initialize clean state**:
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

#### Procedure B: Device Drift under Adaptive Policy
If authorized hardware components change (e.g. CPU or network card replacement):
1. Under **Adaptive Policy** (`src/security/device_auth_policy.py`), partial component drift within `drift_tolerance_threshold` is permitted while updating the device profile.
2. If drift exceeds policy limits, the package author must issue an updated manifest re-bound to the new host fingerprint digest with incremented `sequence_number`.

### 5.7 Test Execution Commands
```bash
# 1. Mandatory AppSec & Cryptographic Regression Tests (78 tests)
PYTHONPATH=. ./venv/bin/pytest tests/security/ -v

# 2. Core Unit & Research Evaluation Tests (485 tests)
PYTHONPATH=. ./venv/bin/pytest tests/unit/ tests/test_*.py -v

# 3. Pipeline Integration Tests (10 tests)
PYTHONPATH=. ./venv/bin/pytest tests/integration/ -v

# 4. Complete Test Suite (575 tests)
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```

---

## 6. Audit Metadata & Provenance

* **Audit Date**: October 19, 2026
* **Audited Commit SHA**: `2e576957388aa5c1d63e9f45ec8c61e4663675a6`
* **Test Command Executed**: `./venv/bin/pytest tests/`
* **Audit Test Results**: **575 passed**, 0 failures, 14 deprecation warnings in 221 seconds.
* **AppSec Findings Remediated**:
  - `SEC-01`: Enforced `weights_only=True` in `src/evaluation/adapter_security.py` (CWE-502).
  - `SEC-02`: Replaced `pickle.dump` with PEFT `save_pretrained` and `safetensors` in `src/phase2/train_lora.py` (CWE-502).
  - `SEC-03`: Added early extension checks before chunk streaming in `src/orchestrator/routes.py` (CWE-400 / CWE-20).
  - `SEC-04`: Added `is_relative_to` storage containment check in `src/orchestrator/service.py` (CWE-22).
  - `SEC-05`: Added `.gitignore` rules for SQLite databases, WAL journals, and decrypted folders (CWE-532).
