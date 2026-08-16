# SecureLoRA: Ablation Study Summary

| Component / Step | Baseline ID | Utility Delta (Acc) | Perplexity Delta | DP Epsilon | PII F1 | Security Score | Deployment Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base Model | E0 | -0.2139 | +4.6303 | N/A | 0.0000 | 0.0000 | 0.00 |
| Standard LoRA | E1 | +0.0000 | +0.0000 | N/A | 0.0000 | 0.0000 | 0.00 |
| PII + LoRA | E2 | +0.0000 | +0.0000 | N/A | 0.9620 | 0.0000 | 0.00 |
| DP-LoRA | E3 | -0.0610 | +0.5517 | 2.443 | 0.0000 | 0.0000 | 0.00 |
| PII + DP-LoRA | E4 | -0.0610 | +0.5517 | 2.443 | 0.9620 | 0.0000 | 0.00 |
| LoRA + Encrypted Adapter | E5 | +0.0000 | +0.0000 | N/A | 0.0000 | 0.2500 | 0.49 |
| LoRA + Device Binding | E6 | +0.0000 | +0.0000 | N/A | 0.0000 | 0.6250 | 0.40 |
| LoRA + Integrity/Signature | E7 | +0.0000 | +0.0000 | N/A | 0.0000 | 0.5000 | 0.41 |
| PII + DP + Enc + Binding | E8 | -0.0610 | +0.5517 | 2.443 | 0.9620 | 0.6250 | 0.39 |
| FULL SECURELORA | E9 | -0.0610 | +0.5517 | 2.443 | 0.9620 | 1.0000 | 0.46 |