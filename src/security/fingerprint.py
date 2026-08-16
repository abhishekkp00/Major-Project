"""
fingerprint.py
==============
Software-derived device identity for SecureLoRA hardware binding.

IMPORTANT — ACCURATE CHARACTERISATION
======================================
The fingerprint produced by this module is a *software-derived device identity*
based on selected OS and hardware attributes.  It is NOT a cryptographic
hardware root of trust.

Entropy sources (in order of priority):
  1. /etc/machine-id  — systemd-generated 128-bit UUID, written once at
                         OS install time.  Stable across reboots on the same
                         installation.
  2. /proc/cpuinfo model name — CPU model string (e.g. "Intel Core i7-12700").
                         Stable unless the CPU is physically replaced.
  3. /dev/disk/by-uuid/ (first UUID) or blkid output — Block device UUID of
                         the first partition, in alphabetical UUID order.
                         Changes if the disk is replaced or re-formatted.

Stability limitations:
  - Replacing the operating system re-generates /etc/machine-id and breaks binding.
  - Replacing the CPU (different model string) breaks binding.
  - Replacing or re-partitioning the primary disk breaks binding.
  - Re-installing the OS on the same hardware breaks binding.

Spoofing limitations:
  - An attacker with root access can trivially read and copy all three sources.
  - A compromised authorised device can be used to derive the correct key.
  - The deployment salt (P3_DEVICE_SALT) is the only true secret; the
    fingerprint provides device identity, not a secret.

Virtualisation limitations:
  - VM hypervisors typically expose a configurable machine-id and can emulate
    arbitrary CPU model strings.
  - An attacker who clones the authorised VM image obtains an identical
    fingerprint and can therefore derive the correct decryption key.

Hardware replacement behaviour:
  - Any of the three source replacements causes derive_key() to produce a
    different key, resulting in AES-GCM authentication tag failure.
  - No manual re-binding step is currently implemented; re-packaging on the
    new machine is required.

Design rationale:
  This fingerprint raises the cost of unauthorised relocation of a stolen
  adapter package: the attacker must either reproduce the exact OS/hardware
  environment or obtain the deployment salt.  It does not provide strong
  hardware attestation in the sense of TPM-based schemes.
"""

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

from src.common.exceptions import DeviceFingerprintError

logger = logging.getLogger("secure_lora.security.fingerprint")

_SEP = "||SECLORA||"
_SENSITIVE_KEYS = {"machine_id", "cpu_model", "disk_uuid"}


def _read_machine_id() -> Optional[str]:
    """
    Returns the systemd machine-id from /etc/machine-id, or None if unavailable.

    Stability: stable across reboots; changes on OS reinstall.
    Spoofing: readable/writable by root.
    """
    try:
        content = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
        return content if content else None
    except (OSError, PermissionError):
        return None


def _read_cpu_model() -> Optional[str]:
    """
    Extracts the 'model name' field from /proc/cpuinfo.

    Stability: stable unless CPU is physically replaced.
    Spoofing: readable by any process; can be spoofed in VMs.
    """
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip()
    except (OSError, PermissionError):
        pass
    return None


def _read_first_disk_uuid() -> Optional[str]:
    """
    Returns the alphabetically first UUID from /dev/disk/by-uuid/ (symlinks).
    Falls back to blkid output if the path is unavailable.

    Stability: stable unless disk is replaced or re-partitioned.
    Spoofing: readable by root; blkid may require elevated privileges.
    Virtualisation: UUID is hypervisor-controlled in VMs.
    """
    uuid_dir = Path("/dev/disk/by-uuid")
    if uuid_dir.exists():
        try:
            uuids = sorted(p.name for p in uuid_dir.iterdir() if p.is_symlink())
            if uuids:
                return uuids[0]
        except (OSError, PermissionError):
            pass

    # Fallback: parse blkid output
    try:
        out = subprocess.check_output(
            ["blkid", "-s", "UUID", "-o", "value"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.decode().splitlines() if ln.strip()]
        if lines:
            return sorted(lines)[0]
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    return None


def collect_identifiers() -> dict[str, str]:
    """
    Returns a dict of available OS/hardware identifiers used for fingerprinting.

    Keys: machine_id, cpu_model, disk_uuid
    Values: the collected string, or "UNAVAILABLE" if the source could not be read.
    """
    return {
        "machine_id": _read_machine_id() or "UNAVAILABLE",
        "cpu_model":  _read_cpu_model()   or "UNAVAILABLE",
        "disk_uuid":  _read_first_disk_uuid() or "UNAVAILABLE",
    }


def build_canonical_string(identifiers: dict[str, str]) -> str:
    """
    Produces a deterministic, normalised fingerprint string from a dict of identifiers.

    Keys are sorted lexicographically to ensure the canonical form is
    independent of dict insertion order.
    """
    parts = [f"{k}={v}" for k, v in sorted(identifiers.items())]
    return _SEP.join(parts)


def compute_fingerprint_hash(canonical: str) -> str:
    """
    Hashes the canonical fingerprint string with SHA-256 and returns hex digest.

    This hash serves as the input key material (IKM) for HKDF; it is NOT
    a secret — it is a device identifier.
    """
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_fingerprint_hash() -> str:
    """
    High-level entry point: collects OS/hardware identifiers, builds the
    canonical string, and returns the SHA-256 hex digest.

    Raises DeviceFingerprintError if all sources are unavailable (extremely rare).
    """
    ids = collect_identifiers()

    # Log which sources contributed, but never log their raw values.
    availability = {k: (v != "UNAVAILABLE") for k, v in ids.items()}
    logger.debug("Fingerprint source availability: %s", availability)

    if not any(availability.values()):
        raise DeviceFingerprintError(
            "All fingerprint sources are UNAVAILABLE. "
            "Cannot derive a device identity on this machine."
        )

    canonical = build_canonical_string(ids)
    fp_hash = compute_fingerprint_hash(canonical)

    logger.info("Device fingerprint computed. hash_prefix=%s…", fp_hash[:8])
    return fp_hash
