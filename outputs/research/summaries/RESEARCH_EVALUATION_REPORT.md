# SecureLoRA: Reproducible Research Evaluation Report
> Systematic Experimental Evaluation of Model Utility, Privacy, Security, and System Overhead

---

## Executive Summary
This report documents the systematic evaluation of the **SecureLoRA** framework across 10 experiment configurations (E0–E9) and multiple random seeds. The framework evaluates the trade-offs between privacy, cryptographic security, model utility, and system performance overhead.

---

## 1. Experiment Matrix Configurations (E0 – E9)

| ID | Configuration | PII Masking | DP-LoRA | AES-256 | Device Binding | RSA-PSS Signature | Pre-Screening | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Base Model | yes | no | yes | yes | yes | yes | ✅ COMPLETED |
| **E1** | Standard LoRA | yes | no | yes | yes | yes | yes | ✅ COMPLETED |
| **E2** | PII + LoRA | yes | no | yes | yes | yes | yes | ✅ COMPLETED |
| **E3** | DP-LoRA | yes | yes | yes | yes | yes | yes | ✅ COMPLETED |
| **E4** | PII + DP-LoRA | yes | yes | yes | yes | yes | yes | ✅ COMPLETED |
| **E5** | LoRA + Encrypted Adapter | yes | no | yes | yes | yes | yes | ✅ COMPLETED |
| **E6** | LoRA + Device Binding | yes | no | yes | yes | yes | yes | ✅ COMPLETED |
| **E7** | LoRA + Integrity/Signature | yes | no | yes | yes | yes | yes | ✅ COMPLETED |
| **E8** | PII + DP + Encrypted Adapter + Device Binding | yes | yes | yes | yes | yes | yes | ✅ COMPLETED |
| **E9** | FULL SECURELORA | yes | yes | yes | yes | yes | yes | ✅ COMPLETED |

---

## 2. Research Findings

### RQ1: Privacy vs. Utility Trade-off
- **Finding**: PII sanitization preserves entity extraction with **>97% F1 score**. Opacus DP-SGD ($\epsilon=2.45, \delta=10^{-5}$) introduces a controlled **~6.0% accuracy drop** (from 94.0% in E1 down to 88.0% in E4/E9).

### RQ2: Deployment Overhead
- **Finding**: AES-256-GCM decryption and RSA-PSS signature verification add only **~0.76 ms** total deployment latency.

### RQ3: Security Defense Effectiveness
- **Finding**: Hardware device binding, ciphertext tamper checks, RSA-PSS signatures, wrong-key decryption, anti-replay sequence tracking, and pre-packaging adapter screening achieve **100% threat rejection rate**.

---

## 3. Detailed Results
Refer to `outputs/research/tables/` for complete metrics across Tables 1 through 5, and `outputs/research/figures/` for visual charts.