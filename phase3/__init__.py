"""
Phase 3: Secure Adapter Protection and Device Binding
======================================================
Encrypts, signs, and device-binds LoRA adapter artifacts produced by Phase 2.
Each sub-module handles one layer of the security stack:

  device_fingerprint  → stable, reproducible hardware identity
  key_derivation      → device-bound AES-256 key derivation
  adapter_encryptor   → AES-256-GCM authenticated encryption
  integrity           → SHA-256 hash verification (constant-time)
  signature_utils     → RSA-PSS digital signature / verification
  package_builder     → bundles all outputs into a deployable package
  verifier            → authorised-device verification + controlled decryption
  config              → centralised, env-driven configuration
  main                → CLI entry-point
"""

__version__ = "3.0.0"
__all__ = [
    "device_fingerprint",
    "key_derivation",
    "adapter_encryptor",
    "integrity",
    "signature_utils",
    "package_builder",
    "verifier",
    "config",
]
