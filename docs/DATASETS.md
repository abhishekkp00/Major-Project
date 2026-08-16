# SecureLoRA Dataset Governance & Adapter Specification

## 1. Executive Summary

SecureLoRA implements a standardized **Dataset Adapter Architecture** (`src/data_sources/`) designed to evaluate privacy-preserving LoRA fine-tuning and PII screening on diverse, research-grade datasets. The architecture enforces zero-disk-leakage processing, strict data governance, reproducible subsampling, and transparent licensing attribution.

---

## 2. Dataset Overview & Classification

| Dataset ID | Name | Source | License | Ground Truth PII Labels | Redistribution | Domain | Default Subset |
|---|---|---|---|---|---|---|---|
| `ai4privacy` | AI4Privacy PII-Masking Benchmark (300k) | Hugging Face (`ai4privacy/pii-masking-300k`) | Apache-2.0 / CC-BY-4.0 | **Yes** (Span Annotations) | **No** (Fetch via API) | Open Web / Enterprise PII | 10,000 |
| `synthea` | Synthea / SyntheticMass Clinical Records | MITRE Synthea™ Patient Generator | Apache License 2.0 | **No** (EHR Clinical Notes) | **Yes** (Synthetic EHR) | Healthcare EHR / Clinical | 5,000 |
| `synthetic` | SecureLoRA Synthetic Benchmark | Internal Synthetic Generator | MIT License | **Yes** (Exact Generated Spans) | **Yes** (Repository Sample) | Enterprise PII / PHI | 10,000 |

---

## 3. Detailed Dataset Specifications

### 3.1 AI4Privacy (`ai4privacy/pii-masking-300k`)
- **Primary Source**: Hugging Face Hub (`ai4privacy/pii-masking-300k`)
- **License & Attribution**: Apache-2.0 / CC-BY-4.0. Attribution to AI4Privacy project (https://huggingface.co/datasets/ai4privacy/pii-masking-300k).
- **Data Governance Policy**: Raw dataset files are **NEVER committed** to this GitHub repository. Loaded on-demand into RAM via the Hugging Face `datasets` API or deterministic adapter cache.
- **Ground Truth PII Annotations**: Includes explicit `privacy_mask` annotations with entity types (`SSN`, `EMAIL`, `PHONE`, `PERSON`, `LOCATION`, `IP_ADDRESS`, etc.) and start/end character offsets.
- **Evaluation Scope**: Used to compute true positives, false positives, false negatives, precision, recall, and F1-score for PII detection.

### 3.2 Synthea / SyntheticMass Healthcare Records
- **Primary Source**: Synthea™ Synthetic Patient Generator, MITRE Corporation (https://github.com/synthetichealth/synthea).
- **License & Attribution**: Apache License 2.0. Attribution to MITRE Corporation and SyntheticMass.
- **Clinical Schema Extraction**: Clinical narrative text synthesized from patient encounters, conditions/diagnoses, active medication regimens, procedures performed, and care events.
- **Ground Truth Note**: Synthea dataset records represent synthetic EHR histories, but **do not provide ground-truth PII entity span annotations**.
- **Compliance & Metric Rules**: To prevent fabricated research results, Synthea evaluations **DO NOT claim PII detection F1-scores**. Instead, evaluation measures:
  1. Sanitization coverage percentage.
  2. Total detected and masked entities per clinical category.
  3. Qualitative & structural statistics (avg narrative length, clinical event distribution).

### 3.3 SecureLoRA Synthetic Benchmark
- **Primary Source**: Internal SecureLoRA synthetic dataset generator and sample benchmarks (`samples/sample_pii_data.jsonl`, `samples/sample_medical_phi.jsonl`, `real_world_pii.jsonl`).
- **License**: MIT License.
- **Ground Truth Annotations**: Includes exact ground-truth entity spans generated alongside prompt templates.
- **Evaluation Scope**: Unit testing, adversarial security verification, reproducible regression tests, and exact PII detection metric computation.

---

## 4. Standardized Internal JSONL Schema

All dataset adapters convert raw inputs into SecureLoRA's canonical internal record format:

```json
{
  "record_id": "ai4privacy_train_00001",
  "domain": "Diverse Open Web PII / Multilingual Text",
  "instruction": "Redact Personally Identifiable Information (PII) from this text.",
  "input": "Contact Alice Smith at alice@company.org or 415-555-0192.",
  "output": "Contact [NAME] at [EMAIL] or [PHONE].",
  "source_dataset": "ai4privacy/pii-masking-300k",
  "synthetic_source": false,
  "pii_entities": [
    {"type": "NAME", "start": 8, "end": 19, "text": "Alice Smith"},
    {"type": "EMAIL", "start": 23, "end": 40, "text": "alice@company.org"},
    {"type": "PHONE", "start": 44, "end": 56, "text": "415-555-0192"}
  ]
}
```

---

## 5. Subsampling & Reproducibility

Subsampling is deterministic and reproducible using a fixed random seed:
```python
from src.data_sources import get_dataset_adapter

# Load reproducible 10,000 record subset of AI4Privacy
adapter = get_dataset_adapter("ai4privacy")
records = adapter.load_dataset(subset_size=10000, split="train", seed=42)

# Get complete metadata capturing revision, timestamp, and seed
metadata = adapter.get_metadata()
```

Supported dashboard subset sizes: `1000`, `5000`, `10000`, `25000`.

---

## 6. Automated Research Outputs

Running the benchmark evaluator generates standardized summary and metrics JSON files:
- `outputs/evaluation/datasets/ai4privacy_summary.json`
- `outputs/evaluation/datasets/synthea_summary.json`
- `outputs/evaluation/datasets/synthetic_summary.json`
- `outputs/evaluation/pii_benchmark/ai4privacy_metrics.json`
- `outputs/evaluation/pii_benchmark/synthea_metrics.json`
- `outputs/evaluation/pii_benchmark/synthetic_metrics.json`

Each file records the dataset ID, version, revision hash, license, subset size, seed, download timestamp, and evaluation metrics without storing raw sensitive record content.
