# Table 5: Complete Ablation Matrix (E0 - E9)

| Experiment ID | Configuration | Val Loss | Perplexity | Accuracy | PII F1 | DP Epsilon | Rejection Rate | Deployment (ms) | Package Size (bytes) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E0 | Base Model | 1.8518 | 6.3724 | 0.7251 | 0.0000 | N/A | 0.00% | 0.00 | 524288 |
| E1 | Standard LoRA | 0.5551 | 1.7421 | 0.9390 | 0.0000 | N/A | 0.00% | 0.00 | 524288 |
| E2 | PII + LoRA | 0.5551 | 1.7421 | 0.9390 | 0.9620 | N/A | 0.00% | 0.00 | 524288 |
| E3 | DP-LoRA | 0.8301 | 2.2938 | 0.8780 | 0.0000 | 2.44 | 0.00% | 0.00 | 524288 |
| E4 | PII + DP-LoRA | 0.8301 | 2.2938 | 0.8780 | 0.9620 | 2.44 | 0.00% | 0.00 | 524288 |
| E5 | LoRA + Encrypted Adapter | 0.5551 | 1.7421 | 0.9390 | 0.0000 | N/A | 25.00% | 0.49 | 524560 |
| E6 | LoRA + Device Binding | 0.5551 | 1.7421 | 0.9390 | 0.0000 | N/A | 62.50% | 0.40 | 524560 |
| E7 | LoRA + Integrity/Signature | 0.5551 | 1.7421 | 0.9390 | 0.0000 | N/A | 50.00% | 0.41 | 525072 |
| E8 | PII + DP + Encrypted Adapter + Device Binding | 0.8301 | 2.2938 | 0.8780 | 0.9620 | 2.44 | 62.50% | 0.39 | 524560 |
| E9 | FULL SECURELORA | 0.8301 | 2.2938 | 0.8780 | 0.9620 | 2.44 | 100.00% | 0.46 | 525072 |