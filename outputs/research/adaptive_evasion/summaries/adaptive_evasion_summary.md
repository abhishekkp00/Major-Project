# SecureLoRA: Adaptive Adversarial Evasion Research Report
**Research Benchmark**: Adaptive Structural Evasion Against LoRA Security Screening  
**Git Commit SHA**: `4800d19370168fbe6198c5742231f7c18b40fb6a`  
**Evaluation Timestamp**: 2026-08-16T10:03:31.164433+00:00  

---

## 1. Security Boundary & Explicit Scope
The Adapter Security Screening module is **NOT**:
- A formal mathematical proof of adapter safety.
- A universal zero-day backdoor detector.
- A replacement for RSA-PSS cryptographic signature authenticity.

It **IS**:
- A defensive pre-packaging risk assessment gate that measures structural parameter statistics and behavioral probe shifts to flag suspicious LoRA supply-chain artifacts.

---

## 2. Research Answers to Core Questions (RQs 1–5)

### RQ1: Can basic suspicious adapters be detected?
**Verdict: SUPPORTED**
- **Finding**: Basic suspicious adapters (Level 0) exhibiting parameter norm anomalies (Z-score $\ge 15.0$) or extreme trigger sensitivity are detected with **100.0% accuracy** by both structural (S1) and behavioral (S2) screening layers.

### RQ2: Does adaptive structural evasion reduce structural detection?
**Verdict: SUPPORTED**
- **Finding**: When an adversary constrains weight perturbations (Level 2 & Level 3 adaptive adapters), structural-only screening (S1) detection rate drops from **100.0% to 0.0%** (FNR increases to 100.0%). Static weight analysis alone is completely vulnerable to adaptive structural evasion.

### RQ3: Does behavioral screening improve robustness?
**Verdict: SUPPORTED**
- **Finding**: Controlled behavioral probing (S2) maintains **100.0% detection rate** across all adaptive evasion levels (Levels 1–3) because the trigger-conditioned output divergence remains detectable regardless of structural weight hiding.

### RQ4: Does combining structural and behavioral screening reduce false negatives?
**Verdict: SUPPORTED**
- **Finding**: The combined defense gate (S3/S4) achieves **100.0% Precision, Recall, and F1 Score (0.0% FNR, 0.0% FPR)** across all evaluated clean, basic, and adaptive samples, outperforming any single screening dimension.

### RQ5: What is the security/availability trade-off caused by threshold selection?
**Verdict: SUPPORTED**
- **Finding**: Thresholds below 0.25 increase false positives (FPR up to 20%), while thresholds above 0.70 risk missing mild behavioral trigger signals. Setting the default low-risk threshold to **0.35** optimizes the F1 score at **1.0000**.

---

## 3. False Positive & False Negative Failure Analysis
- **Structural False Negatives (Level 2 & 3)**: Structural screening failed on Level 2/3 adaptive adapters because the adversary's weight perturbations fell within standard Gaussian distribution bounds (Z-score $< 1.3$, cosine similarity $> 0.97$).
- **Behavioral Robustness**: Behavioral probing succeeded on all trigger-conditioned samples because synthetic trigger probes directly test model output divergence rather than weight distributions.
- **Zero False Positives**: Clean adapters (Baseline A & B) maintained mean risk scores of **0.0103 - 0.0121**, well below the 0.35 risk gate.

---

## 4. Research Implications
1. **Multi-Layer Defense Necessity**: Static parameter screening is necessary for catching unconstrained errors or raw weight poisoning, but dynamic behavioral probing is mandatory to withstand adaptive adversaries.
2. **Negligible Latency**: Combining structural and behavioral screening adds only **~1.13 ms** execution latency per adapter.