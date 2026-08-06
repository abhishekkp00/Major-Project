# outputs/

This directory contains all runtime-generated files from the SecureLoRA pipeline.

> ⚠️ **Most subdirectories are gitignored** because they contain sensitive keys,
> encrypted artefacts, or large binary model files. See the table below.

## Directory Map

| Directory | Committed | Contents |
|-----------|-----------|----------|
| `paper_results/` | ✅ Yes | Genuine evaluation outputs for academic publication |
| `evaluation/` | ✅ Yes | `eval_report.json` from LoRA training |
| `deployment_validation/` | ✅ Yes | Phase 4 gate validation reports |
| `final_adapter/` | ❌ No | Trained LoRA adapter weights (large binary) |
| `protected_adapter/` | ❌ No | Encrypted adapter + private key (sensitive) |
| `jobs/` | ❌ No | Per-job workspaces with ephemeral secrets |

## Reproducing paper_results/

```bash
# From the project root:
python run_paper_evaluation.py

# Results written to:
#   outputs/paper_results/paper_evaluation_results.json  — full data
#   outputs/paper_results/paper_evaluation_summary.md    — Markdown tables
#   outputs/paper_results/benchmarks/*.json              — per-module data
```
