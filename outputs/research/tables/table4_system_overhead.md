# Table 4: System Overhead Comparison

| Experiment ID | Configuration | Training (s) | Encryption (ms) | Signing (ms) | Verification (ms) | Decryption (ms) | Deployment Time (ms) | Inference Latency (ms) | Peak Memory (MB) | Package Size (bytes) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E0 | Base Model | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12.29 | 120.0 | 524288 |
| E1 | Standard LoRA | 1.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12.24 | 125.0 | 524288 |
| E2 | PII + LoRA | 1.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12.24 | 125.0 | 524288 |
| E3 | DP-LoRA | 2.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12.31 | 155.0 | 524288 |
| E4 | PII + DP-LoRA | 2.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12.31 | 155.0 | 524288 |
| E5 | LoRA + Encrypted Adapter | 1.80 | 3.39 | 0.00 | 0.00 | 0.53 | 0.53 | 12.24 | 125.0 | 524560 |
| E6 | LoRA + Device Binding | 1.80 | 0.28 | 0.00 | 0.00 | 0.25 | 0.40 | 12.24 | 125.0 | 524560 |
| E7 | LoRA + Integrity/Signature | 1.80 | 0.25 | 2.05 | 0.18 | 0.23 | 0.42 | 12.24 | 125.0 | 525072 |
| E8 | PII + DP + Encrypted Adapter + Device Binding | 2.50 | 0.29 | 0.00 | 0.00 | 0.25 | 0.40 | 12.31 | 155.0 | 524560 |
| E9 | FULL SECURELORA | 2.50 | 1.01 | 2.79 | 0.21 | 0.40 | 0.76 | 12.31 | 155.0 | 525072 |