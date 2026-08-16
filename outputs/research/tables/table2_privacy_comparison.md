# Table 2: Privacy Comparison

| Experiment ID | Configuration | PII Precision | PII Recall | PII F1 | DP Enabled | Epsilon (ε) | Delta (δ) | Clipping Norm | Noise Multiplier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E0 | Base Model | 0.0000 | 0.0000 | 0.0000 | No | N/A | N/A | N/A | N/A |
| E1 | Standard LoRA | 0.0000 | 0.0000 | 0.0000 | No | N/A | N/A | N/A | N/A |
| E2 | PII + LoRA | 0.9500 | 0.9744 | 0.9620 | No | N/A | N/A | N/A | N/A |
| E3 | DP-LoRA | 0.0000 | 0.0000 | 0.0000 | Yes | 2.4430 | 1.0e-05 | 1.00 | 1.20 |
| E4 | PII + DP-LoRA | 0.9500 | 0.9744 | 0.9620 | Yes | 2.4430 | 1.0e-05 | 1.00 | 1.20 |
| E5 | LoRA + Encrypted Adapter | 0.0000 | 0.0000 | 0.0000 | No | N/A | N/A | N/A | N/A |
| E6 | LoRA + Device Binding | 0.0000 | 0.0000 | 0.0000 | No | N/A | N/A | N/A | N/A |
| E7 | LoRA + Integrity/Signature | 0.0000 | 0.0000 | 0.0000 | No | N/A | N/A | N/A | N/A |
| E8 | PII + DP + Encrypted Adapter + Device Binding | 0.9500 | 0.9744 | 0.9620 | Yes | 2.4430 | 1.0e-05 | 1.00 | 1.20 |
| E9 | FULL SECURELORA | 0.9500 | 0.9744 | 0.9620 | Yes | 2.4430 | 1.0e-05 | 1.00 | 1.20 |