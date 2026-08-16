# SecureLoRA: Adaptive Adversarial Evasion Failure Analysis
**Evaluation Timestamp**: 2026-08-16T10:23:57.873438+00:00  
**Total Test Samples Evaluated**: 150  

---

## 1. Combined Screening Gate Failures
- **Combined False Negatives**: 0
- **Combined False Positives**: 0

## 2. Layer 1 (Structural Screening Only) Failure Breakdown
- **Structural False Negatives**: **90 out of 120 anomalous samples**.
- **Vulnerability Finding**: Structural screening alone experiences **100% False Negative Rate (0% detection)** on Level 2 and Level 3 adaptive evasive adapters.
- **Root Cause**: The adversary constrains weight updates such that max layer Z-score remains $< 1.3$ and cosine similarity to clean weights remains $> 0.97$, effectively masking static weight anomalies.

## 3. Behavioral Probe Counter-Measure
- **Mitigation**: Layer 2 Behavioral Probing successfully detected $100\%$ of Level 2 and Level 3 adaptive adapters via synthetic trigger sensitivity probes, rescuing the combined gate.