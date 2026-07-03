"""
phase4 — Secure Deployment, Verification, and Inference Validation

Implements the complete deployment workflow:
  1. Receive and load a protected LoRA adapter package (Phase 3 output)
  2. Validate completeness, integrity, and authenticity
  3. Authorise the current device via fingerprint binding
  4. Decrypt only on the authorised device
  5. Load the adapter into the base model via PEFT
  6. Execute secure inference
  7. Generate machine-readable and human-readable validation reports
  8. Clean up all temporary plaintext material
"""

__version__ = "4.0.0"
