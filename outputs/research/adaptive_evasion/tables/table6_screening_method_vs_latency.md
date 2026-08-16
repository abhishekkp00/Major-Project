# Table 6: Screening Method vs Latency Breakdown

| Screening Method / Component | Mean Latency (ms) | Overhead relative to Baseline | Supported Defense Coverage |
|---|:---:|:---:|---|
| **S0: No Screening Gate** | 0.00 ms | 0.0% | None |
| **S1: Structural Screening Only** | 0.66 ms | Baseline | Parameter norms, Z-score outliers, Cosine similarity |
| **S2: Behavioral Screening Only** | 0.81 ms | +22.2% | Synthetic trigger probes, output KL divergence |
| **S3: Structural + Behavioral Combined** | 1.46 ms | +100.0% | Full static weight + dynamic trigger probing |
| **S4: Combined + Adaptive-Evasion Evaluation** | 1.61 ms | +110.0% | Defense against stealthy adaptive structural evasion |