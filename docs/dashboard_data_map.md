# SecureLoRA Dashboard — Data Source Map

**Generated**: 2026-08-16 (Step 1 Audit)  
**Purpose**: Authoritative mapping from every dashboard metric to its exact source file, result file, API route, live vs. historical classification, and current availability.  
**Rule**: If a metric does not exist in a file or live service, the API returns `{ "available": false, "reason": "..." }`. No fabricated zeroes.

---

## Data Classification

| Class | Meaning |
|---|---|
| **LIVE** | Derived from the running Flask process or on-disk state that changes during pipeline execution |
| **HISTORICAL** | Read from a pre-written JSON/MD result file in `outputs/research/` |
| **COMPUTED-ON-READ** | Aggregated from multiple raw result files at request time |

---

## Metric Group 1 — Live Pipeline State

### 1a. Phase 4 Deployment / Device Status

| Metric | Source Module | Source File / API | Type | Available |
|---|---|---|---|---|
| Adapter loaded (bool) | `src/phase4/adapter_loader.py` | Flask global `adapter_loaded` / `GET /api/phase4/status` | LIVE | ✅ |
| Device fingerprint prefix (masked) | `src/phase4/device_auth.py` | `get_fingerprint_hash()` → `GET /api/phase4/status` | LIVE | ✅ |
| Verification step results (Steps 1–8) | `src/phase4/package_validator.py` | `GET /api/phase4/status` → `last_verification_steps` | LIVE | ✅ |
| Validation report (Steps 1–8, full) | `outputs/deployment_validation/validation_report.json` | `GET /api/phase4/status` (reads from disk if cache empty) | LIVE | ✅ |
| Anti-replay state / sequence number | `outputs/.deployment_state.json` | Read-only disk state | LIVE | ✅ |
| Adapter active in inference | Flask global `peft_model` | `POST /api/phase4/verify` response | LIVE | ✅ |

### 1b. Orchestrator / Job State

| Metric | Source Module | Source File / API | Type | Available |
|---|---|---|---|---|
| Active / completed jobs list | `src/orchestrator/service.py` | `GET /api/orchestrator/jobs` | LIVE | ✅ |
| Job status (PENDING/RUNNING/COMPLETED) | `src/orchestrator/service.py` | `GET /api/orchestrator/jobs/<job_id>` | LIVE | ✅ |
| Job logs | `src/orchestrator/service.py` | `GET /api/orchestrator/jobs/<job_id>/logs` | LIVE | ✅ |
| Job metrics | `src/orchestrator/service.py` | `GET /api/orchestrator/jobs/<job_id>/metrics` | LIVE | ✅ |

---

## Metric Group 2 — Research Results (Historical)

### 2a. ML Utility Metrics

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| Validation loss (mean ± std) | `outputs/research/metrics/B8_summary.json` | `utility_summary.val_loss` | ✅ |
| Perplexity (mean ± std) | `outputs/research/metrics/B8_summary.json` | `utility_summary.perplexity` | ✅ |
| Task accuracy (mean ± std) | `outputs/research/metrics/B8_summary.json` | `utility_summary.task_accuracy` | ✅ |
| F1 score (mean ± std) | `outputs/research/metrics/B8_summary.json` | `utility_summary.f1_score` | ✅ |
| Per-baseline utility comparison | `outputs/research/metrics/summary_metrics.json` | `E0..E9.utility_summary` | ✅ |
| Per-seed raw utility | `outputs/research/runs/B8_seed_42.json` etc. | `utility.*` | ✅ |

### 2b. PII / Privacy Metrics

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| PII Precision (mean ± std) | `outputs/research/metrics/B8_summary.json` | `privacy_summary.pii_precision` | ✅ |
| PII Recall (mean ± std) | `outputs/research/metrics/B8_summary.json` | `privacy_summary.pii_recall` | ✅ |
| PII F1 (mean ± std) | `outputs/research/metrics/B8_summary.json` | `privacy_summary.pii_f1` | ✅ |
| DP Epsilon | `outputs/research/metrics/B8_summary.json` | `privacy_summary.epsilon` | ✅ |
| DP Delta | `outputs/research/metrics/B8_summary.json` | `privacy_summary.delta` | ✅ |
| Clipping norm | `outputs/research/metrics/B8_summary.json` | `privacy_summary.clipping_norm` | ✅ |
| Noise multiplier | `outputs/research/metrics/B8_summary.json` | `privacy_summary.noise_multiplier` | ✅ |

### 2c. Security Metrics

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| Cross-device rejection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.cross_device_rejection_rate` | ✅ |
| Tamper rejection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.tamper_rejection_rate` | ✅ |
| Signature rejection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.signature_rejection_rate` | ✅ |
| Replay rejection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.replay_rejection_rate` | ✅ |
| Malicious adapter detection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.malicious_adapter_detection_rate` | ✅ |
| Unauthorized deployment rejection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.unauthorized_deployment_rejection_rate` | ✅ |

### 2d. Overhead / Latency Metrics

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| Training time (s) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.training_time_s` | ✅ |
| Encryption time (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.encryption_time_ms` | ✅ |
| Decryption time (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.decryption_time_ms` | ✅ |
| Signing time (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.signing_time_ms` | ✅ |
| Verification time (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.verification_time_ms` | ✅ |
| Packaging time (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.packaging_time_ms` | ✅ |
| Deployment latency (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.deployment_latency_ms` | ✅ |
| Inference latency (ms) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.inference_latency_ms` | ✅ |
| Memory usage (MB) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.memory_usage_mb` | ✅ |
| Storage overhead (bytes) | `outputs/research/metrics/B8_summary.json` | `overhead_summary.storage_overhead_bytes` | ✅ |

### 2e. Adapter Security Screening Results

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| True positives / negatives / FP / FN | `outputs/research/adapter_screening/metrics.json` | `true_positives`, `false_positives`, etc. | ✅ |
| Precision / Recall / F1 | `outputs/research/adapter_screening/metrics.json` | `precision`, `recall`, `f1_score` | ✅ |
| False positive rate | `outputs/research/adapter_screening/metrics.json` | `false_positive_rate` | ✅ |
| False negative rate | `outputs/research/adapter_screening/metrics.json` | `false_negative_rate` | ✅ |
| Mean screening latency (ms) | `outputs/research/adapter_screening/metrics.json` | `mean_latency_ms` | ✅ |
| ROC-AUC | `outputs/research/adapter_screening/metrics.json` | `roc_auc` | ✅ |

### 2f. Adaptive Evasion Results

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| Category summary (CLEAN/BASIC/ADAPTIVE) | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `category_summary.*` | ✅ |
| Level-wise structural distance | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `level_summary.level_N.mean_overall_structural_dist` | ✅ |
| Level-wise structural-only detection rate | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `level_summary.level_N.struct_only_detection_rate` | ✅ |
| Level-wise combined detection rate | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `level_summary.level_N.combined_detection_rate` | ✅ |
| Level-wise combined FNR | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `level_summary.level_N.combined_fnr` | ✅ |
| Ablation config (S0–S4) precision/recall/F1 | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `ablations.*` | ✅ |
| Threshold grid (precision/recall/F1 per τ) | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `test_threshold_grid` | ✅ |
| Multi-seed stats (mean ± std) | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `seed_stats.*` | ✅ |
| Hypothesis test verdicts (H1–H5) | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | `hypotheses.*` | ✅ |

### 2g. Ablation Study Results

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| Component-wise ablation (B0–B8 / E0–E9) | `outputs/research/metrics/ablation_study_summary.json` | array of `{baseline_id, utility_delta_accuracy, pii_f1, security_score, overhead_latency_ms}` | ✅ |
| Per-experiment summary metrics | `outputs/research/metrics/summary_metrics.json` | `E0..E9.{utility_summary, privacy_summary, security_summary, overhead_summary}` | ✅ |
| Per-baseline summaries | `outputs/research/metrics/B0_summary.json` … `B8_summary.json` | full nested structure | ✅ |

### 2h. Provenance / Anti-Replay (Research Experiments)

| Metric | Source File | JSON Key Path | Available |
|---|---|---|---|
| Replay rejection rate | `outputs/research/metrics/B8_summary.json` | `security_summary.replay_rejection_rate` | ✅ |
| Sequence number (deployment state) | `outputs/.deployment_state.json` | `adapters.*.last_sequence` | ✅ (LIVE) |
| Package IDs processed | `outputs/.deployment_state.json` | `adapters.*.processed_packages` | ✅ (LIVE, count only — do not expose IDs) |

---

## Metric Group 3 — Currently Unavailable

| Metric | Reason |
|---|---|
| PII entity-level breakdown (PERSON/ORG/DATE counts) | Not written to any result file; only computed live during masking in RAM |
| Device authorization policy state machine transitions | Not exported to JSON; internal state in `src/security/device_auth_policy.py` |
| DP-LoRA real Opacus training epsilon trace | Not written to experiment output; only final epsilon value captured |
| Per-record PII audit log | Intentionally not persisted (zero-disk-leakage requirement) |
| Real model output divergence (actual forward pass) | Behavioral screening uses simulation only; no real LLM weights used in benchmark |
| Threat model validation experiment results | `src/evaluation/threat_model.py` exists but no JSON output files found in `outputs/` |
| Baseline comparison raw data | `src/evaluation/baseline_comparison.py` exists but no corresponding output files found |
| Crypto benchmark raw data | `src/evaluation/crypto_benchmark.py` exists but no corresponding output files found |

---

## API Route Map (Step 1 — New Endpoints Only)

> **Rule**: Reuse every existing API. Only create new READ-ONLY research endpoints where no equivalent exists.

### Existing APIs (Reused — No Changes)

| Route | Blueprint | Purpose |
|---|---|---|
| `GET /api/phase4/status` | `dashboard.py` | Live deployment status, steps 1–8, fingerprint prefix |
| `POST /api/phase4/verify` | `dashboard.py` | Trigger Phase 4 full verification |
| `POST /api/phase4/generate` | `dashboard.py` | Run PII-masked inference |
| `POST /api/transparency/inspect` | `dashboard.py` | Transparency trace with hash chain |
| `POST /api/tamper/simulate` | `dashboard.py` | Attack simulation (Stages 1–4) |
| `POST /api/chat` | `dashboard.py` | Privacy-preserving Q&A |
| `GET /api/orchestrator/jobs` | `routes.py` | List all jobs |
| `GET /api/orchestrator/jobs/<id>` | `routes.py` | Get single job status |
| `GET /api/orchestrator/jobs/<id>/metrics` | `routes.py` | Get job-level training metrics |
| `GET /api/orchestrator/jobs/<id>/logs` | `routes.py` | Get job logs |

### New READ-ONLY Research APIs (Created in Step 1)

| Route | Data Source | Notes |
|---|---|---|
| `GET /api/research/summary` | `outputs/research/metrics/B8_summary.json` | Full pipeline utility + privacy + security + overhead summary |
| `GET /api/research/ablation` | `outputs/research/metrics/ablation_study_summary.json` + `summary_metrics.json` | Ablation matrix, component contribution |
| `GET /api/research/privacy` | `outputs/research/metrics/B8_summary.json` + `E3..E4` | DP epsilon/delta, PII precision/recall/F1 across configs |
| `GET /api/research/screening` | `outputs/research/adapter_screening/metrics.json` | Adapter screening TP/FP/TN/FN, latency, AUC |
| `GET /api/research/adaptive-evasion` | `outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json` | Full evasion benchmark, hypothesis verdicts |
| `GET /api/research/overhead` | `outputs/research/metrics/B8_summary.json` | All cryptographic and training latency metrics |

---

## Security Constraints

The following fields are **never** exposed by any API:

- Private keys, HKDF salts, AES keys
- Full device fingerprint hash (only first 16 chars + `...`)
- Raw device identifiers (machine-id, MAC, disk UUID)
- Per-record PII audit logs
- Plaintext adapter weights
- Package IDs list (only count is safe to expose)
- Internal exception stack traces
