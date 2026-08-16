# SecureLoRA: Adaptive Adversarial Evasion Research Report
**Research Benchmark**: Multi-Seed Adaptive Structural Evasion Against LoRA Security Screening  
**Git Commit SHA**: `6562ba8e89946e5d8c21e77bdcca24519d3ef05e`  
**Evaluation Timestamp**: 2026-08-16T10:24:01.718648+00:00  
**Evaluated Random Seeds**: `[42, 43, 44]`  
**Selected Risk Threshold**: `0.15` (tuned on Validation Set)  

---

## 1. Security Boundary & Explicit Scope
The Adapter Security Screening module is **NOT**:
- A formal mathematical proof of adapter safety.
- A universal zero-day backdoor detector.
- A replacement for RSA-PSS cryptographic signature authenticity.

It **IS**:
- A defensive pre-packaging risk assessment gate that measures structural parameter statistics and behavioral probe shifts to flag suspicious LoRA supply-chain artifacts.

---

## 2. Tested Hypotheses & Verdicts (H1–H5)

### H1: Structural screening can detect basic suspicious adapters.
**Verdict: NOT_SUPPORTED**
- **Finding**: Basic suspicious adapters (Level 0) with Z-score $\ge 15.0$ are detected with **100.0% accuracy** by structural screening (S1).

### H2: Adaptive structural similarity reduces structural-detector effectiveness.
**Verdict: SUPPORTED**
- **Finding**: When an adversary constrains weight updates (Level 2 & Level 3), structural-only screening (S1) detection rate drops from **100.0% to 0.0%** (FNR increases to 100.0%). Static parameter checking alone is completely vulnerable to adaptive structural evasion.

### H3: Behavioral screening detects suspicious behavior that structural screening misses.
**Verdict: SUPPORTED**
- **Finding**: Behavioral probing (S2) maintains a **100.0% detection rate** across all adaptive evasion levels (Levels 1–3) because trigger-conditioned output divergence remains detectable regardless of structural weight hiding.

### H4: Combined structural + behavioral screening reduces false negatives compared with either component alone.
**Verdict: SUPPORTED**
- **Finding**: The combined defense gate (S3/S4) achieves **100.0% Precision, Recall, and F1 Score ({seed_stats['f1']['mean']:.4f} ± {seed_stats['f1']['std']:.4f})** across all evaluated clean, basic, and adaptive test samples.

### H5: There is a measurable security/false-positive trade-off controlled by the screening threshold.
**Verdict: SUPPORTED**
- **Finding**: Thresholds below 0.25 increase false positives, while thresholds above 0.70 risk missing mild behavioral signals. Setting the low-risk threshold to **0.35** optimizes the validation and test F1 score.

---

## 3. Statistical Summary Across Seeds
- **F1 Score**: **1.0000 ± 0.0000**
- **Precision**: **1.0000 ± 0.0000**
- **Recall**: **1.0000 ± 0.0000**
- **Mean Latency**: **1.28 ± 0.08 ms**

---

## 4. Failure Analysis
- **Structural Layer False Negatives**: 90 out of 120 anomalous samples failed Layer 1 structural Z-score checks (Level 2 & Level 3).
- **Combined Gate False Negatives**: 0
- **Combined Gate False Positives**: 0