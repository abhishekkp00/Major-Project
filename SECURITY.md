# Security Policy

## Supported Versions

This is a research prototype. Security fixes are applied to the `main` branch.

| Version | Supported |
|---------|-----------|
| main    | ✅ Yes    |

## Reporting a Vulnerability

**Do NOT open a public GitHub Issue for security vulnerabilities.**

Email the maintainer directly with:
1. A clear description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Any suggested mitigations

You will receive a response within 72 hours.

## Security Design Principles

SecureLoRA is built around these security invariants:

1. **No plaintext adapter weights ever touch non-volatile storage** — decryption happens exclusively in volatile RAM buffers.
2. **Decryption keys are never stored anywhere** — they are derived ephemerally at runtime from hardware identifiers via HKDF.
3. **Every adapter package is cryptographically signed** — unsigned or tampered packages are rejected before decryption is attempted.
4. **PII is scrubbed before the model sees it** — the pipeline enforces data sanitization before tokenization.

## Scope of Security Guarantees

✅ **In scope:**
- Unauthorized adapter use on a different device
- Bit-level tamper detection of the encrypted package
- Supply chain integrity via RSA-PSS signing
- PII leakage into model weights during training

❌ **Out of scope (future work):**
- Cold-boot RAM extraction attacks
- Side-channel attacks (timing, power analysis)
- Root-level compromise of the authorized machine at decryption time
- Attacks on the CSPRNG (`os.urandom`)
