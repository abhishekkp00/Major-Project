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

1. **Ephemeral Plaintext with Bounded Lifecycle** — Decrypted adapter weights exist temporarily on disk within an access-controlled directory (mode `0700`) solely for the duration of PEFT model loading, followed by immediate overwrite-and-unlink cleanup on context exit.
2. **Decryption keys are never stored anywhere** — derived transiently via HKDF-SHA256 from salt and software-derived device identity.
3. **Every adapter package is cryptographically signed** — unsigned or tampered packages are rejected before decryption is attempted.
4. **PII is scrubbed before the model sees it** — the pipeline enforces data sanitization before tokenization.

## Plaintext Lifecycle & Data-at-Rest Boundaries

> [!WARNING]
> **Filesystem Requirement Notice:** Because PEFT/PyTorch adapter loading (`peft.PeftModel.from_pretrained`) requires a local filesystem path, decrypted adapter weights are written to a temporary directory on disk for the duration of model loading. SecureLoRA does **NOT** claim RAM-only model loading or zero plaintext on disk.

### Ephemeral Storage and Cleanup Guarantees
- **Access Restrictions:** Temporary directories are created with mode `0700` on POSIX platforms; decrypted files inside are restricted to mode `0600`.
- **Bounded Lifetime:** Plaintext weights exist on disk strictly within the active `DecryptedAdapterContext` block. Upon context exit (both on success and on any exception), all temporary files and directories are shredded and deleted.
- **Overwrite-Before-Unlink:** Temporary files are overwritten with random bytes (3 passes) before unlinking.
- **Cleanup Failure Surfacing:** Cleanup failures are surfaced when the operation succeeds, or logged without masking the primary exception if an error occurred during model loading.
- **Physical Erasure Limitations:** Overwriting before unlinking is a defense-in-depth measure and does **not** guarantee physical media sanitization on solid-state drives (SSDs with flash translation layers/wear leveling), copy-on-write filesystems (Btrfs, ZFS, APFS), volume snapshots, journaling filesystems, or OS swap space.
- **Configurable Storage Policy:** For deployments requiring strict memory-backed execution or encrypted storage, operators can set `SECURELORA_TEMP_DIR` to point to a RAM-backed filesystem (`tmpfs` / `/dev/shm`) or an encrypted LUKS volume, and ensure OS swap is encrypted or disabled.

## Device Binding Classification & Threat Boundaries

> [!IMPORTANT]
> SecureLoRA uses **software-derived device identity with policy-controlled authorization**. It does **NOT** claim a hardware root of trust, TPM PCR sealing, or unbreakable hardware binding.

### Scope of Security Guarantees

✅ **In Scope:**
- Unauthorized adapter deployment on a different target machine (software identity mismatch)
- Bit-level tamper detection of the encrypted package (RSA-PSS manifest signature validation)
- Supply chain integrity and replay rejection (Monotonic sequence tracking)
- PII leakage into model weights during training (PII engine + Opacus DP-SGD)
- Strict bounded lifetime of temporary decrypted files on disk

❌ **Out of Scope & Explicit Limitations:**
- **Physical SSD Erasure**: Flash translation layer wear-leveling remapping out-of-place writes.
- **Root Compromise**: An attacker with root privileges on the authorized host can inspect `/etc/machine-id` or extract ephemeral RAM keys.
- **Spoofable Identifiers**: Software attributes (`machine-id`, MAC address, CPU model) can be spoofed by root users or modified via kernel hooks.
- **VM Cloning**: Cloning an authorized hypervisor image preserves `/etc/machine-id` and hypervisor-emulated hardware UUIDs.
- **Hardware Replacement**: Legitimate hardware replacement (e.g. CPU or primary disk swap) changes the device identity, requiring administrator re-authorization.
- **Identifier Manipulation**: Environment variables or sysfs overrides in unisolated environments can manipulate reported system attributes.

