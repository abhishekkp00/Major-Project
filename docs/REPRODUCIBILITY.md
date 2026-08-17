# SecureLoRA Experimental Reproducibility Guide

This document provides explicit, reproducible CLI commands and configuration protocols for reproducing every evaluation step and research experiment in the SecureLoRA framework.

---

## Prerequisites & Environment Setup

Ensure the Python environment is activated and `PYTHONPATH` includes the repository root:

```bash
cd /home/abhishek/Projects/MAJOR_PROJECT
source venv/bin/activate
export PYTHONPATH=.
```

---

## Reproduction Commands by Research Step

### Step 1: Dataset Adapter Layer Verification
Validates reproducible loading and normalization across AI4Privacy, Synthea, and Synthetic datasets.

```bash
PYTHONPATH=. ./venv/bin/pytest tests/test_dataset_adapters.py -v
```

---

### Step 2: PEFT Model Registry & Inference Pipeline
Verifies thread-safe registration of base LLM + trained PEFT adapter for dynamic generation.

```bash
PYTHONPATH=. ./venv/bin/pytest tests/unit/test_model_registry_inference.py -v
```

---

### Step 3: Privacy & PII Leakage Benchmarking
Runs evaluation comparing Base Model vs. Standard LoRA vs. DP-LoRA vs. Full SecureLoRA.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.privacy_evaluator
```

**Artifacts Generated**: `outputs/evaluation/privacy/{base_model,lora,dp_lora,securelora,comparison}.json`

---

### Step 4: Comparative Adapter Screening Systems
Evaluates Structural-only vs. Behavioral-only vs. Combined (SecureLoRA) screening systems.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.screening_evaluator
```

**Artifacts Generated**: `outputs/evaluation/screening/{structural_only,behavioral_only,combined,comparison}.json`

---

### Step 5: Adaptive Evasion & Adversarial Robustness
Evaluates detector robustness against non-adaptive vs. adaptive gradient/pertubation attacks.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.adaptive_evasion_evaluator
```

**Artifacts Generated**: `outputs/evaluation/adaptive_evasion/{baseline_attack,nonadaptive_attack,adaptive_attack,comparison}.json`

---

### Step 6: Multi-Seed Statistical Replication Engine
Executes full experimental replication across 5 seeds (`42, 123, 456, 789, 1001`) with mean ± stdev reporting.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.seed_evaluator --seeds 42,123,456,789,1001
```

**Artifacts Generated**: `outputs/evaluation/statistics/{seed_results,aggregated_results}.json`, `comparison.csv`

---

### Step 7: Device Binding & Authorization Policy Evaluation
Compares Static Fingerprint Policy vs. Adaptive Device Authorization Policy across 8 operational/threat scenarios.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.device_binding_evaluator
```

**Artifacts Generated**: `outputs/evaluation/device_binding/{static_policy,adaptive_policy,comparison}.json`

---

### Step 8: Model Scale Evaluation Pipeline
Measures computational overhead and security screening behavior across Lightweight (68M) and Scaled (350M) models.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.model_scale_evaluator
```

**Artifacts Generated**: `outputs/evaluation/model_scale/model_comparison.json`, `model_comparison.csv`

---

### Step 9: Research Artifact Audit & Schema Standardization
Audits and standardizes all JSON outputs in `outputs/evaluation/` to conform strictly to `UnifiedExperimentResult`.

```bash
PYTHONPATH=. ./venv/bin/python -m src.evaluation.schema_auditor
```

---

## Full Test Suite Execution

To run the complete automated test suite (245 unit & integration tests):

```bash
PYTHONPATH=. ./venv/bin/pytest tests/ -v
```
