# Helper Scripts

This directory contains standalone execution and evaluation utilities for SecureLoRA:

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_paper_evaluation.py` | Runs all 4 evaluation modules end-to-end (PII metrics, crypto timing, baseline benchmarks, threat model simulations) | `python scripts/run_paper_evaluation.py` |
| `generate_paper_figures.py` | Generates high-resolution IEEE-compliant monochrome publication charts & diagrams | `python scripts/generate_paper_figures.py` |
| `download_real_pii.py` | Fetches real-world sanitized PII dataset samples from Hugging Face for testing | `python scripts/download_real_pii.py` |
| `download_training_data.py` | Utilities for fetching larger domain datasets | `python scripts/download_training_data.py` |
