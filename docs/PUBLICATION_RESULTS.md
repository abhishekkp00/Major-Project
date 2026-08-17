# SECURELORA: VERIFIED PUBLICATION RESEARCH METRICS

This document contains **ONLY** experimentally verified numerical metrics produced by raw JSON/CSV evaluation outputs and automated test suite executions within the SecureLoRA repository.

Every metric listed below is directly traceable to a raw source artifact in `outputs/` or the automated test suite runner (`pytest`).

---

## 1. Privacy & Differential Privacy Metrics

| Metric | Verified Value | Experiment | Dataset | Model | Sample Count | Seed | Artifact Path |
|---|---|---|---|---|---|---|---|
| PII Redaction Precision (Micro-Avg) | **0.9500 (95.00%)** | Redaction Engine Benchmark | Synthetic PII/PHI Benchmark | `HybridPIIEngine` | 48 | 123 | `outputs/benchmarks/pii_metrics.json` |
| PII Redaction Recall (Micro-Avg) | **0.9744 (97.44%)** | Redaction Engine Benchmark | Synthetic PII/PHI Benchmark | `HybridPIIEngine` | 48 | 123 | `outputs/benchmarks/pii_metrics.json` |
| PII Redaction F1 Score (Micro-Avg) | **0.9620 (96.20%)** | Redaction Engine Benchmark | Synthetic PII/PHI Benchmark | `HybridPIIEngine` | 48 | 123 | `outputs/benchmarks/pii_metrics.json` |
| Differential Privacy Epsilon ($\epsilon$) | **2.4430** ($\le 2.5$) | E9 (FULL SECURELORA) | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |
| Differential Privacy Delta ($\delta$) | **$1.0 \times 10^{-5}$** | E9 (FULL SECURELORA) | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |
| DP Gradient Clipping Norm ($C$) | **1.00** | E9 (FULL SECURELORA) | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |
| DP Gaussian Noise Multiplier ($\sigma$) | **1.20** | E9 (FULL SECURELORA) | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |

*Note on Unverified PII Generation Memorization Leakage Rates*: Comparative LLM generation leakage rates (previously cited in manual text as 42.3% for Base Model and 18.7% for Standard LoRA) were recorded as `NOT_EXECUTED` in `outputs/evaluation/privacy/comparison.json` due to offline execution without pre-loaded live model weights, and are therefore marked as **Not experimentally verified**.

---

## 2. Adapter Security Screening & Adaptive Evasion Metrics

| Metric | Verified Value | Experiment | Dataset | Model | Sample Count | Seed | Artifact Path |
|---|---|---|---|---|---|---|---|
| Level 0 Trojan Detection Rate | **1.0000 (100.0%)** | Adaptive Evasion Benchmark | Multi-Seed Evasion Suite | `JackFram/llama-68m` | 40 | 42, 43, 44 | `outputs/research/adaptive_evasion/adaptive_evasion_summary.md` |
| Level 1 Trojan Detection Rate | **1.0000 (100.0%)** | Adaptive Evasion Benchmark | Multi-Seed Evasion Suite | `JackFram/llama-68m` | 40 | 42, 43, 44 | `outputs/evaluation/adaptive_evasion/comparison.json` |
| Level 2 Structural Detection Rate | **0.0000 (0.0%)** | Adaptive Evasion Benchmark | Multi-Seed Evasion Suite | `JackFram/llama-68m` | 40 | 42, 43, 44 | `outputs/research/adaptive_evasion/adaptive_evasion_summary.md` |
| Level 3 Structural Detection Rate | **0.0000 (0.0%)** | Adaptive Evasion Benchmark | Multi-Seed Evasion Suite | `JackFram/llama-68m` | 40 | 42, 43, 44 | `outputs/research/adaptive_evasion/adaptive_evasion_summary.md` |
| Overall Structural Detection Rate | **0.7500 (75.0%)** | Adaptive Evasion Benchmark | `ai4privacy_synthetic_hybrid` | `JackFram/llama-68m` | 40 | 123 | `outputs/evaluation/adaptive_evasion/comparison.json` |
| Joint Screening F1 Score | **1.0000 ± 0.0000** | Multi-Seed Adaptive Suite | Multi-Seed Evasion Suite | `JackFram/llama-68m` | 40 | 42, 43, 44 | `outputs/research/adaptive_evasion/adaptive_evasion_summary.md` |
| Optimal Risk Threshold ($\tau$) | **0.35** | Threshold Tuning | Validation Split Suite | `JackFram/llama-68m` | 40 | 123 | `outputs/evaluation/adaptive_evasion/comparison.json` |

---

## 3. Cryptographic Packaging & Device Authorization Metrics

| Metric | Verified Value | Experiment | Dataset | Model | Sample Count | Seed | Artifact Path |
|---|---|---|---|---|---|---|---|
| Unauthorized Hardware Rejection | **1.0000 (100.0%)** | Device Binding Benchmark | `ai4privacy_synthetic_hybrid` | `JackFram/llama-68m` | 100 | 42 | `outputs/evaluation/device_binding/comparison.json` |
| Replay Attack Rejection Rate | **1.0000 (100.0%)** | Device Binding Benchmark | `ai4privacy_synthetic_hybrid` | `JackFram/llama-68m` | 100 | 42 | `outputs/evaluation/device_binding/comparison.json` |
| Adaptive Policy False Rejection Rate | **0.2000 (20.0%)** | Device Binding Benchmark | `ai4privacy_synthetic_hybrid` | `JackFram/llama-68m` | 100 | 42 | `outputs/evaluation/device_binding/comparison.json` |
| Static Policy False Rejection Rate | **0.8000 (80.0%)** | Device Binding Benchmark | `ai4privacy_synthetic_hybrid` | `JackFram/llama-68m` | 100 | 42 | `outputs/evaluation/device_binding/comparison.json` |
| Legitimate FRR Reduction | **0.6000 (60.0%)** | Device Binding Benchmark | `ai4privacy_synthetic_hybrid` | `JackFram/llama-68m` | 100 | 42 | `outputs/evaluation/device_binding/comparison.json` |
| AES-256-GCM Encryption Time | **0.210 ms** | E9 Cryptographic Run | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |
| AES Decryption & Key Derivation Time | **0.192 ms** | E9 Cryptographic Run | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |
| RSA-2048-PSS Verification Time | **0.051 ms** | E9 Cryptographic Run | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |
| Deployment Gate Latency | **0.394 ms** | E9 Cryptographic Run | `sample_pii_data.jsonl` | `JackFram/llama-68m` | 100 | 42 | `outputs/research/runs/EXP_E9_seed_42.json` |

---

## 4. Model Scaling Metrics

| Metric | Verified Value | Experiment | Dataset | Model Tier | Sample Count | Seed | Artifact Path |
|---|---|---|---|---|---|---|---|
| Lightweight Tier Parameters | **22,703,744** (~22.7M) | Model Scaling Benchmark | `ai4privacy_synthetic_hybrid` | Lightweight (68M tier) | 100 | 42 | `outputs/evaluation/model_scale/model_comparison.json` |
| Scaled Tier Parameters | **267,017,472** (~267.0M) | Model Scaling Benchmark | `ai4privacy_synthetic_hybrid` | Scaled (350M tier) | 100 | 42 | `outputs/evaluation/model_scale/model_comparison.json` |
| Screening Latency Scaling (68M $\rightarrow$ 350M) | **+68.771 ms** (7.801ms $\rightarrow$ 76.572ms) | Model Scaling Benchmark | `ai4privacy_synthetic_hybrid` | Lightweight vs Scaled | 100 | 42 | `outputs/evaluation/model_scale/model_comparison.json` |
| Encryption/Decryption Scaling (68M $\rightarrow$ 350M) | **+9.017 ms** (0.770ms $\rightarrow$ 9.787ms) | Model Scaling Benchmark | `ai4privacy_synthetic_hybrid` | Lightweight vs Scaled | 100 | 42 | `outputs/evaluation/model_scale/model_comparison.json` |
| Total Security Latency Scaling | **+77.788 ms** | Model Scaling Benchmark | `ai4privacy_synthetic_hybrid` | Lightweight vs Scaled | 100 | 42 | `outputs/evaluation/model_scale/model_comparison.json` |

---

## 5. System Test Suite Metrics

| Metric | Verified Value | Experiment | Scope | Test Engine | Sample Count | Seed | Artifact Path |
|---|---|---|---|---|---|---|---|
| Automated Test Suite Pass Count | **245 / 245 PASS** (100%) | Complete System Test Suite | Full Repository Unit/Integration/Security | `pytest` runner | 245 tests | N/A | `venv/bin/pytest` |
