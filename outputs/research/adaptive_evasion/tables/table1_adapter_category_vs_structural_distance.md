# Table 1: Adapter Category vs Structural Distance

| Category | Evasion Level | Overall Distance | Norm Distance | Sparsity Distance | Max Z-Score | Cosine Sim to Trusted |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CLEAN** | 0 | 0.0035 ± 0.0002 | 0.0002 | 0.0000 | 1.08 | 0.9904 |
| **BASIC_SUSPICIOUS** | 0 (Unconstrained) | 2.2993 ± 0.0446 | 9.1158 | 0.0000 | 2.64 | 1.0000 |
| **ADAPTIVE_SUSPICIOUS** | 1 (Light Constraint) | 0.2766 ± 0.0067 | 1.1350 | 0.0000 | 2.42 | 1.0000 |
| **ADAPTIVE_SUSPICIOUS** | 2 (Moderate Constraint) | 0.1927 ± 0.0042 | 0.2850 | 0.0000 | 1.77 | 0.6531 |
| **ADAPTIVE_SUSPICIOUS** | 3 (Strong Constraint) | 0.0000 ± 0.0000 | 0.0420 | 0.0000 | 1.08 | 1.0000 |