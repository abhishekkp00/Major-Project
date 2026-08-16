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
2. **Decryption keys are never stored anywhere** — derived transiently via HKDF-SHA256 from salt and software-derived device identity.
3. **Every adapter package is cryptographically signed** — unsigned or tampered packages are rejected before decryption is attempted.
4. **PII is scrubbed before the model sees it** — the pipeline enforces data sanitization before tokenization.

## Device Binding Classification & Threat Boundaries

> [!IMPORTANT]
> SecureLoRA uses **software-derived device identity with policy-controlled authorization**. It does **NOT** claim a hardware root of trust, TPM PCR sealing, or unbreakable hardware binding.

### Scope of Security Guarantees

✅ **In Scope:**
- Unauthorized adapter deployment on a different target machine (software identity mismatch)
- Bit-level tamper detection of the encrypted package (RSA-PSS manifest signature validation)
- Supply chain integrity and replay rejection (Monotonic sequence tracking)
- PII leakage into model weights during training (PII engine + Opacus DP-SGD)

❌ **Out of Scope & Explicit Limitations:**
- **Root Compromise**: An attacker with root privileges on the authorized host can inspect `/etc/machine-id` or extract ephemeral RAM keys.
- **Spoofable Identifiers**: Software attributes (`machine-id`, MAC address, CPU model) can be spoofed by root users or modified via kernel hooks.
- **VM Cloning**: Cloning an authorized hypervisor image preserves `/etc/machine-id` and hypervisor-emulated hardware UUIDs.
- **Hardware Replacement**: Legitimate hardware replacement (e.g. CPU or primary disk swap) changes the device identity, requiring administrator re-authorization.
- **Identifier Manipulation**: Environment variables or sysfs overrides in unisolated environments can manipulate reported system attributes.

