# Table 3: Detector Configuration vs Precision / Recall / F1

| Code | Configuration Name | Precision | Recall | F1 Score | FPR | FNR | Screening Latency (ms) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** | S0: No Screening | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.00 |
| **S1** | S1: Structural Screening Only | 1.0000 | 0.2500 | 0.4000 | 0.0000 | 0.7500 | 0.58 |
| **S2** | S2: Behavioral Screening Only | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.71 |
| **S3** | S3: Structural + Behavioral Screening | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.28 |
| **S4** | S4: Structural + Behavioral + Adaptive-Evasion Evaluation | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.41 |