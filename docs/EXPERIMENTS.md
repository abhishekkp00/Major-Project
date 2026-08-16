# SecureLoRA Experimental Reproducibility Guide & Command Reference

This document defines the exact commands, datasets, target models, random seeds, and artifact paths required to reproduce all evaluation experiments in the SecureLoRA paper and framework.

---

## 1. Reproducibility Guarantee Matrix

| Step | Experiment Name | CLI Command | Dataset | Model | Seed(s) | Key Output Artifact(s) |
|---|---|---|---|---|---|---|
| **Step 1** | Dataset Adapter Verification | `PYTHONPATH=. ./venv/bin/pytest tests/test_dataset_adapters.py -v` | AI4Privacy, Synthea, Synthetic | N/A | 42 | PyTest Test Output |
| **Step 2** | Model Registry & Inference Verification | `PYTHONPATH=. ./venv/bin/pytest tests/unit/test_model_registry_inference.py -v` | Synthetic Benchmark | JackFram/llama-68m | 42 | PyTest Test Output |
| **Step 3** | Privacy & PII Leakage Evaluation | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.privacy_evaluator` | AI4Privacy / Synthetic | JackFram/llama-68m | 42 | `outputs/evaluation/privacy/{base_model,lora,dp_lora,securelora,comparison}.json` |
| **Step 4** | Adapter Screening Comparison | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.screening_evaluator` | SecureLoRA Benchmark | JackFram/llama-68m | 42 | `outputs/evaluation/screening/{structural_only,behavioral_only,combined,comparison}.json` |
| **Step 5** | Adaptive Evasion & Robustness | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.adaptive_evasion_evaluator` | SecureLoRA Benchmark | JackFram/llama-68m | 42 | `outputs/evaluation/adaptive_evasion/{baseline_attack,nonadaptive_attack,adaptive_attack,comparison}.json` |
| **Step 6** | Multi-Seed Statistical Replication | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.seed_evaluator --seeds 42 123 456 789 1001` | Synthetic Benchmark | JackFram/llama-68m | 42, 123, 456, 789, 1001 | `outputs/evaluation/statistics/{seed_results,aggregated_results}.json`, `comparison.csv` |
| **Step 7** | Device Binding & Policy Eval | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.device_binding_evaluator` | Simulated Environment | JackFram/llama-68m | 42 | `outputs/evaluation/device_binding/{static_policy,adaptive_policy,comparison}.json` |
| **Step 8** | Model Scale Evaluation | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.model_scale_evaluator` | Synthetic Benchmark | 68M (llama-68m) vs 350M (llama-350m) | 42 | `outputs/evaluation/model_scale/model_comparison.json`, `model_comparison.csv` |
| **Step 9** | Schema Audit & Standardization | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.schema_auditor` | All Outputs | N/A | N/A | Standardized JSON files in `outputs/evaluation/` |
| **Step 10**| Dashboard Server Launch | `PYTHONPATH=. ./venv/bin/python -m src.evaluation.dashboard` | Real Standardized Artifacts | PEFT Model Registry | 42 | Web Dashboard running on `http://localhost:5005` |

---

## 2. Detailed Command Protocol & Verification

### Step 1: Dataset Adapter Verification
Verifies data loading, schema integrity, and split generation across AI4Privacy, Synthea, and Synthetic sources.
```bash
PYTHONPATH=. ./venv/bin/pytest tests/test_dataset_adapters.py -v
```

### Step 2: PEFT Registry & Thread-Safe Model Inference
Verifies model registry creation, base LLM + PEFT adapter attachment, and deterministic text generation.
```bash
PYTHONPATH=. ./venv/bin/pytest tests/unit/test_model_registry_inference.py -v
```

### Step 3: Privacy & PII Leakage Benchmarking
Executes evaluation across Base Model, Standard LoRA, DP-LoRA, and SecureLoRA.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.privacy_evaluator
```
*Outputs*: `outputs/evaluation/privacy/comparison.json`

### Step 4: Adapter Screening Systems
Evaluates Structural-only, Behavioral-only, and Combined (SecureLoRA) adapter screening performance.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.screening_evaluator
```
*Outputs*: `outputs/evaluation/screening/comparison.json`

### Step 5: Adaptive Evasion & Adversarial Attack Evaluation
Evaluates detector robustness when an attacker iteratively optimizes adapter parameters to evade screening.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.adaptive_evasion_evaluator
```
*Outputs*: `outputs/evaluation/adaptive_evasion/comparison.json`

### Step 6: Multi-Seed Statistical Replication
Executes full experimental replication across 5 random seeds (`42 123 456 789 1001`) and computes mean $\pm$ standard deviation.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.seed_evaluator --seeds 42 123 456 789 1001
```
*Outputs*: `outputs/evaluation/statistics/aggregated_results.json`, `comparison.csv`

### Step 7: Device Binding & Authorization Policy Evaluation
Evaluates security (unauthorized rejection, replay rejection) and availability (false rejection, recovery time) across 8 deployment scenarios.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.device_binding_evaluator
```
*Outputs*: `outputs/evaluation/device_binding/comparison.json`

### Step 8: Model Scale Evaluation
Measures latency, memory usage, encryption/decryption overhead, and screening behavior across 68M and 350M model scales.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.model_scale_evaluator
```
*Outputs*: `outputs/evaluation/model_scale/model_comparison.json`

### Step 9: Research Artifact Schema Audit
Audits and standardizes all evaluation outputs to conform strictly to `UnifiedExperimentResult`.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.schema_auditor
```
*Outputs*: Updates all JSON outputs in `outputs/evaluation/` with valid status (`EXECUTED`, `FAILED`, `NOT_EXECUTED`).

### Step 10: Dashboard Web Application
Launches the web-based interactive transparency dashboard connected to real research artifacts.
```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.dashboard
```
*Access*: Open `http://localhost:5005` in your web browser.
