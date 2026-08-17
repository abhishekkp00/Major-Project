# SECURELORA: FORMAL REPRODUCIBILITY AUDIT REPORT (PHASE 3)

**Date of Audit**: August 16, 2026  
**Repository**: `https://github.com/abhishekkp00/Major-Project`  
**Audit Scope**: Complete Automated Test Suite Execution, Research Experiment Reproducibility, Multi-Seed Verification, and Documentation Alignment.

---

## 1. Complete Test Suite Execution Results

The repository test suite was executed in full via `pytest`:

- **Execution Command**: `PYTHONPATH=. ./venv/bin/pytest tests/ -v`
- **Total Tests Collected**: **245**
- **Passed**: **245**
- **Failed**: **0**
- **Skipped**: **0**
- **Errors**: **0**
- **Execution Wall Clock Time**: 147.87 seconds (~2 min 27 sec)
- **Status**: **100% PASS**

### Test Category Breakdown

| Test Category | File Path / Directory | Tests Collected | Passed | Failed | Skipped | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Integration Suite** | `tests/integration/` | 8 | 8 | 0 | 0 | PASSED |
| **Security & Cryptography** | `tests/security/` | 51 | 51 | 0 | 0 | PASSED |
| **Evaluator Suite** | `tests/test_*.py` | 18 | 18 | 0 | 0 | PASSED |
| **Unit & Subsystem Suite** | `tests/unit/` | 168 | 168 | 0 | 0 | PASSED |
| **Total System Test Suite** | `tests/` | **245** | **245** | **0** | **0** | **100% PASS** |

---

## 2. Research Experiment Reproducibility Matrix

Every major research experiment pipeline in the repository was audited against seven reproducibility criteria:
1. CLI Command existence & validity
2. Source script existence
3. Dataset availability
4. Model configuration definition
5. Output artifact location & schema compliance
6. Seed specification
7. Metric calculation implementation

### Summary Table

| Step | Research Experiment | Script Path | Dataset | Model Tier | Seed | Output Artifact | Classification |
|---|---|---|---|---|---|---|:---:|
| **Step 1** | Dataset Adapter Layer | `tests/test_dataset_adapters.py` | AI4Privacy, Synthea, Synthetic | N/A | 42 | PyTest Output | **VERIFIED** |
| **Step 2** | PEFT Model Registry & Inference | `tests/unit/test_model_registry_inference.py` | Synthetic Benchmark | JackFram/llama-68m | 42 | PyTest Output | **VERIFIED** |
| **Step 3** | Privacy & PII Leakage | `src/evaluation/privacy_evaluator.py` | Synthetic / AI4Privacy | JackFram/llama-68m | 42 | `outputs/evaluation/privacy/comparison.json` | **PARTIALLY VERIFIED** |
| **Step 4** | Adapter Screening Systems | `src/evaluation/screening_evaluator.py` | SecureLoRA Benchmark | JackFram/llama-68m | 42 | `outputs/evaluation/screening/comparison.json` | **VERIFIED** |
| **Step 5** | Adaptive Adversarial Evasion | `src/evaluation/adaptive_evasion_evaluator.py` | Multi-Seed Evasion Suite | JackFram/llama-68m | 42 | `outputs/evaluation/adaptive_evasion/comparison.json` | **VERIFIED** |
| **Step 6** | Multi-Seed Replication | `src/evaluation/seed_evaluator.py` | Synthetic Benchmark | JackFram/llama-68m | 42, 43, 44 | `outputs/evaluation/statistics/aggregated_results.json` | **PARTIALLY VERIFIED** |
| **Step 7** | Device Binding & Policy | `src/evaluation/device_binding_evaluator.py` | Synthetic / Simulated Env | JackFram/llama-68m | 42 | `outputs/evaluation/device_binding/comparison.json` | **VERIFIED** |
| **Step 8** | Model Scale Analysis | `src/evaluation/model_scale_evaluator.py` | Synthetic Benchmark | 68M vs 350M | 42 | `outputs/evaluation/model_scale/model_comparison.json` | **VERIFIED** |
| **Step 9** | Schema Audit & Standardization | `src/evaluation/schema_auditor.py` | Output JSON Artifacts | N/A | N/A | `outputs/evaluation/**/*.json` | **VERIFIED** |
| **Step 10**| Web Dashboard & Research API | `src/evaluation/dashboard.py` | Real Output Artifacts | PEFT Model Registry | 42 | `http://localhost:5005` | **VERIFIED** |

---

## 3. Detailed Classification & Audit Findings

### Step 1: Dataset Adapter Verification — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/pytest tests/test_dataset_adapters.py -v`
- **Script**: `tests/test_dataset_adapters.py` (and `src/data_sources/*.py`)
- **Dataset**: `ai4privacy/pii-masking-300k`, `synthea`, `synthetic`
- **Model**: N/A
- **Seed**: 42
- **Artifact**: PyTest output log
- **Findings**: Data loading, schema normalization, span extraction, and sampling logic are fully implemented and verified by 8 dedicated unit/integration tests.

### Step 2: PEFT Registry & Model Inference — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/pytest tests/unit/test_model_registry_inference.py -v`
- **Script**: `src/orchestrator/model_registry.py` & `src/orchestrator/inference_service.py`
- **Dataset**: Synthetic PII Benchmark
- **Model**: `JackFram/llama-68m` + LoRA adapter
- **Seed**: 42
- **Artifact**: PyTest output log
- **Findings**: Model registration, dynamic adapter attachment, tokenizer alignment, and side-by-side inference generation are fully reproducible and verified.

### Step 3: Privacy & PII Leakage Evaluation — **PARTIALLY VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.privacy_evaluator`
- **Script**: `src/evaluation/privacy_evaluator.py`
- **Dataset**: `outputs/benchmarks/pii_metrics.json` & Synthetic Benchmark
- **Model**: `JackFram/llama-68m`
- **Seed**: 42
- **Artifact**: `outputs/evaluation/privacy/comparison.json`
- **Findings**: 
  - The hybrid PII redaction engine metrics (Micro F1 = 0.9620, Precision = 0.9500, Recall = 0.9744) and DP parameters ($\epsilon=2.4430, \delta=10^{-5}$) are **fully verified**.
  - Generation-level LLM memorization rates across Base, LoRA, and DP-LoRA models are recorded as `"status": "NOT_EXECUTED"` when run without active weight downloads. Thus, generation-level leakage rates are classified as **Partially Verified / Not Executed**.

### Step 4: Adapter Screening Systems — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.screening_evaluator`
- **Script**: `src/evaluation/screening_evaluator.py`
- **Dataset**: `outputs/evaluation/screening/comparison.json`
- **Model**: `JackFram/llama-68m`
- **Seed**: 42 (and 123 in comparison split)
- **Artifact**: `outputs/evaluation/screening/comparison.json`
- **Findings**: SVD rank anomaly detection, spectral norm calculation, threshold sweeps ($\tau=0.10$ to $\tau=0.60$), and confusion matrix calculations are fully reproducible.

### Step 5: Adaptive Adversarial Evasion — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.adaptive_evasion_evaluator`
- **Script**: `src/evaluation/adaptive_evasion_evaluator.py`
- **Dataset**: Multi-seed adaptive evasion attack suite
- **Model**: `JackFram/llama-68m`
- **Seed**: 42
- **Artifact**: `outputs/evaluation/adaptive_evasion/comparison.json`
- **Findings**: Robustness testing across Level 0, Level 1, Level 2, and Level 3 evasion attacks is fully functional. The drop of structural-only screening to 0.0% detection at Level 2/3 and the 1.0000 F1 performance of joint screening are verified.

### Step 6: Multi-Seed Statistical Replication — **PARTIALLY VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.seed_evaluator --seeds 42 123 456 789 1001`
- **Script**: `src/evaluation/seed_evaluator.py`
- **Dataset**: Synthetic Benchmark
- **Model**: `JackFram/llama-68m`
- **Seed Coverage**: 
  - Raw research output logs (`outputs/research/runs/*.json` and `outputs/research/metrics/*.json`) contain **3 fully executed seeds**: `42`, `43`, and `44`.
  - The CLI script `seed_evaluator.py` supports multi-seed execution, but `outputs/evaluation/statistics/aggregated_results.json` reflects single-run defaults unless explicitly passed all 5 seeds.
- **Findings**: Multi-seed statistical calculation logic is fully functional. Raw outputs support 3-seed evaluation (Seeds 42, 43, 44). Classified as **Partially Verified** for 5-seed claims.

### Step 7: Device Binding & Policy Evaluation — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.device_binding_evaluator`
- **Script**: `src/evaluation/device_binding_evaluator.py`
- **Dataset**: Simulated environment threat matrix (8 scenarios)
- **Model**: `JackFram/llama-68m`
- **Seed**: 42
- **Artifact**: `outputs/evaluation/device_binding/comparison.json`
- **Findings**: Hardware fingerprinting, HKDF-SHA256 key derivation, static policy (80% FRR), and adaptive policy (20% FRR, 100% unauthorized clone rejection) are fully verified.

### Step 8: Model Scale Evaluation — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.model_scale_evaluator`
- **Script**: `src/evaluation/model_scale_evaluator.py`
- **Dataset**: Synthetic Benchmark
- **Model**: Lightweight (68M / 22.7M params) vs. Scaled (350M / 267.0M params)
- **Seed**: 42
- **Artifact**: `outputs/evaluation/model_scale/model_comparison.json`
- **Findings**: Parameter counting, screening latency scaling (+68.77 ms), and crypto latency scaling (+9.02 ms) are fully verified.

### Step 9: Schema Audit & Standardization — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.schema_auditor`
- **Script**: `src/evaluation/schema_auditor.py`
- **Artifact**: `outputs/evaluation/**/*.json`
- **Findings**: Schema auditor successfully validates all evaluation output files against the unified schema (`UnifiedExperimentResult`).

### Step 10: Web Dashboard & Research API — **VERIFIED**
- **CLI Command**: `PYTHONPATH=. ./venv/bin/python -m src.evaluation.dashboard`
- **Script**: `src/evaluation/dashboard.py` & `src/evaluation/research_api.py`
- **Findings**: Server initializes on `http://localhost:5005`, exposes read-only research APIs (`/api/research/*`), serves side-by-side inference, and visualizes security demonstration events.

---

## 4. Multi-Seed Execution Statistics Verification

Across the raw 3-seed research runs (`outputs/research/runs/` and `outputs/research/metrics/E9_summary.json` for Seeds `42`, `43`, and `44`), the empirical statistical metrics are verified as follows:

| Metric | Sample Count ($n$) | Mean ($\mu$) | Std Dev ($\sigma$) | Min Value | Max Value |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training Loss** | 3 | 0.8909 | 0.0113 | 0.8809 | 0.9032 |
| **Validation Loss** | 3 | 0.8301 | 0.0164 | 0.8124 | 0.8449 |
| **Perplexity** | 3 | 2.2938 | 0.0376 | 2.2534 | 2.3278 |
| **Task Accuracy** | 3 | 0.8780 | 0.0159 | 0.8640 | 0.8952 |
| **F1 Score** | 3 | 0.8672 | 0.0117 | 0.8553 | 0.8786 |
| **Training Time (s)** | 3 | 14.5003 | 0.1984 | 14.2750 | 14.6490 |
| **Encryption Time (ms)** | 3 | 0.2463 | 0.0332 | 0.2100 | 0.2750 |
| **Decryption Time (ms)** | 3 | 0.2117 | 0.0358 | 0.1900 | 0.2530 |
| **Deployment Gate Latency (ms)** | 3 | 0.4583 | 0.0719 | 0.3940 | 0.5360 |
| **Inference Latency (ms)** | 3 | 12.3123 | 0.1747 | 12.1120 | 12.4330 |

---

## 5. Summary of Reproducibility Audit Verdicts

- **Total Experiments Evaluated**: 10 Steps
- **Verified**: **8 Steps** (Steps 1, 2, 4, 5, 7, 8, 9, 10)
- **Partially Verified**: **2 Steps** (Step 3 Privacy generation-level memorization & Step 6 5-seed evaluation vs 3-seed raw runs)
- **Not Reproducible**: **0 Steps**

All core security mechanisms, cryptographic packaging routines, screening gates, dataset adapters, schema auditors, and test suites are **100% reproducible and operational**.
