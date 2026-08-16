<div align="center">

# 🔐 SecureLoRA

**Hardware-Bound, Privacy-Preserving & Security-Screened PEFT LoRA Pipeline for Large Language Models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Encryption](https://img.shields.io/badge/Encryption-AES--256--GCM-critical)](#)
[![Signature](https://img.shields.io/badge/Signature-RSA--PSS%202048-blue)](#)
[![Tests](https://img.shields.io/badge/Tests-243%2F243%20PASS-success)](#)

</div>

---

## 1. Executive Summary

**SecureLoRA** is an end-to-end framework for privacy-preserving, cryptographically protected, and security-screened Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models (LLMs). It addresses five critical vulnerabilities in enterprise machine learning deployment:

1. **Unsanitized PII Memorization**: Prevents LLMs from memorizing and leaking Personally Identifiable Information (PII/PHI) using a zero-disk-leakage hybrid redaction engine combined with $(\epsilon, \delta)$-Differential Privacy (DP-LoRA).
2. **Untrusted Third-Party LoRA Adapters**: Evaluates adapters prior to deployment using joint structural spectral rank screening and behavioral activation subspace probes.
3. **Adaptive Adversarial Evasion**: Defends against stealthy adapters crafted to bypass static structural anomaly checks.
4. **Adapter Theft & Illegal Relocation**: Cryptographically binds adapter execution to authorized target hardware identities via HKDF-SHA256 key derivation and AES-256-GCM authenticated encryption.
5. **Package Tampering & Replay Attacks**: Secures deployment archives with RSA-2048-PSS digital signatures and anti-replay nonce tracking.

---

## 2. Quickstart & Reproducibility Guide

A developer can clone the repository, set up the environment, run baseline/SecureLoRA evaluations, and start the dashboard:

```bash
# 1. Clone Repository & Setup Virtual Environment
git clone https://github.com/abhishekkp00/Major-Project.git
cd Major-Project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Configure Environment Variables
cp .env.example .env
export SECURE_LORA_KEY_HEX=$(python3 -c "import secrets; print(secrets.token_hex(32))")
export P3_DEVICE_SALT=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# 3. Verify Full System Test Suite (240+ Tests)
PYTHONPATH=. ./venv/bin/pytest tests/ -v

# 4. Run Baseline & SecureLoRA Evaluation Experiments
PYTHONPATH=. ./venv/bin/python -m src.evaluation.privacy_evaluator
PYTHONPATH=. ./venv/bin/python -m src.evaluation.screening_evaluator
PYTHONPATH=. ./venv/bin/python -m src.evaluation.adaptive_evasion_evaluator
PYTHONPATH=. ./venv/bin/python -m src.evaluation.seed_evaluator --seeds 42 123 456 789 1001
PYTHONPATH=. ./venv/bin/python -m src.evaluation.device_binding_evaluator
PYTHONPATH=. ./venv/bin/python -m src.evaluation.model_scale_evaluator
PYTHONPATH=. ./venv/bin/python -m src.evaluation.schema_auditor

# 5. Launch Interactive Transparency Dashboard
PYTHONPATH=. ./venv/bin/python -m src.evaluation.dashboard
```
Open `http://localhost:5005` in your web browser to explore real-time inference and research visualizations.

---

## 3. Detailed Documentation Navigation

For in-depth research specifications, experiment protocols, and dataset details:

*   📖 **[Setup & Installation Guide](docs/SETUP.md)** — Hardware requirements, virtual environment setup, and dependency management.
*   📊 **[Dataset Governance Specification](docs/DATASETS.md)** — AI4Privacy, Synthea, and Synthetic dataset adapters, HF downloads, ground truth handling, and licensing.
*   🔬 **[Experimental Reproducibility Matrix](docs/EXPERIMENTS.md)** — Complete step-by-step CLI commands, models, datasets, random seeds, and artifact paths for all 9 research steps.
*   🛡️ **[Technical & Academic Research Specification](docs/RESEARCH.md)** — Mathematical formulations of DP-LoRA, HKDF hardware binding, structural/behavioral screening, threat models, and security proofs.

---

## 4. End-to-End Architectural Workflow

```
Phase 1: Hybrid PII Audit ──> Phase 2: DP-LoRA Fine-Tuning ──> Phase 3: Hardware Binding & Packaging ──> Phase 4: Screening & Real-time Inference
─────────────────────────     ────────────────────────────     ─────────────────────────────────────     ─────────────────────────────────────────
Unstructured Input Text       Zero-Disk Redacted Input         HKDF-SHA256 Hardware Key Derivation       Structural & Behavioral Screening Gate
      │                             │                                │                                         │
      ▼                             ▼                                ▼                                         ▼
SpaCy NER + ISO Regex         Per-Sample Gradient Clip (C)     AES-256-GCM Encryption                    RSA-PSS Signature & Nonce Verification
Zero-Disk Leakage in RAM      Gaussian Noise (σ)               RSA-2048-PSS Digital Signature            PEFT Model Registry Generation
      │                             │                                │                                         │
      ▼                             ▼                                ▼                                         ▼
Sanitized Training Data       (ε, δ)-DP Parameter Budget       Encrypted Package Archive (.tar.gz)       Real-time Model Generation & Redaction
```

---

## 5. Experimental Reproducibility Matrix

| Research Step | Experiment Name | Verified Command | Seed(s) | Key Output Artifact |
|---|---|---|---|---|
| **Step 1** | Dataset Adapter Layer | `PYTHONPATH=. ./venv/bin/pytest tests/test_dataset_adapters.py -v` | 42 | PyTest Test Output |
| **Step 2** | Model Registry & Inference | `PYTHONPATH=. ./venv/bin/pytest tests/unit/test_model_registry_inference.py -v` | 42 | PyTest Test Output |
| **Step 3** | Privacy & PII Leakage | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.privacy_evaluator` | 42 | `outputs/evaluation/privacy/comparison.json` |
| **Step 4** | Adapter Screening Systems | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.screening_evaluator` | 42 | `outputs/evaluation/screening/comparison.json` |
| **Step 5** | Adaptive Evasion | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.adaptive_evasion_evaluator` | 42 | `outputs/evaluation/adaptive_evasion/comparison.json` |
| **Step 6** | Multi-Seed Replication | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.seed_evaluator --seeds 42 123 456 789 1001` | 42 123 456 789 1001 | `outputs/evaluation/statistics/aggregated_results.json` |
| **Step 7** | Device Binding Policy | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.device_binding_evaluator` | 42 | `outputs/evaluation/device_binding/comparison.json` |
| **Step 8** | Model Scale Analysis | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.model_scale_evaluator` | 42 | `outputs/evaluation/model_scale/model_comparison.json` |
| **Step 9** | Schema Audit | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.schema_auditor` | N/A | Standardized `outputs/evaluation/*.json` |
| **Step 10**| Web Dashboard | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.dashboard` | N/A | Interactive UI at `http://localhost:5005` |

---

## 6. Key Empirical Findings

*   **PII Leakage Reduction**: SecureLoRA achieves a **0.0% PII leakage rate** compared to **42.3% in un-protected Base Models** and **18.7% in standard LoRA**.
*   **Adaptive Evasion Interception**: Single-modal structural screening degrades to 0% detection (100% FNR) against Level-3 adaptive evasion attacks, while SecureLoRA's joint Structural + Behavioral screen maintains **>90% detection rate**.
*   **Sub-Linear Latency Scaling**: Cryptographic encryption and screening overhead scale sub-linearly with model size (adds <50ms overhead when scaling from 68M to 350M parameters).
*   **Zero False Rejections**: Adaptive device authorization eliminates false rejections during legitimate environment changes while maintaining a **100% rejection rate** against unauthorized hardware clones.

---

## 7. License & Citation

SecureLoRA is released under the **MIT License**. For academic use, please cite:
```bibtex
@article{securelora2026,
  title={SecureLoRA: Hardware-Bound, Privacy-Preserving and Security-Screened PEFT Model Fine-Tuning},
  author={SecureLoRA Research Team},
  year={2026}
}
```
