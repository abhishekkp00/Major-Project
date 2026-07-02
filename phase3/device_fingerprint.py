"""
phase3/device_fingerprint.py  —  Step 2
-----------------------------------------
Generates a stable, reproducible device fingerprint from hardware and OS
identifiers that are unlikely to change between reboots.

Security design
~~~~~~~~~~~~~~~
* Raw hardware strings are **never** written to disk or emitted to logs.
  Only the final SHA-256 hash of the canonical fingerprint is persisted.
* The function collects multiple sources and concatenates them with a fixed
  separator so that partial matches across machines are impossible.
* A fallback mechanism is included for sources that are unavailable (e.g.
  containers, VMs) so the fingerprint degrades gracefully rather than
  crashing — but it will still differ from a fully-populated machine.

Collected identifiers (Linux-first)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ``/etc/machine-id``     — systemd machine UUID (stable across reboots)
2. ``/proc/cpuinfo``       — CPU model/family string
3. First disk UUID from    ``/dev/disk/by-uuid/``  (block device layer)

Adding more sources
~~~~~~~~~~~~~~~~~~~
Extend ``_collect_identifiers()`` and add the key to ``SOURCES_USED``.
The hash changes automatically; the caller must re-derive keys for migrated
devices.
"""

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Separator used to join identifier segments — must not appear naturally in any
# hardware string to prevent trivial collisions between components.
_SEP = "||SECLORA||"

# Keys whose raw values are never logged, only their presence/absence.
_SENSITIVE_KEYS = {"machine_id", "cpu_model", "disk_uuid"}


# ---------------------------------------------------------------------------
# Private collectors
# ---------------------------------------------------------------------------

def _read_machine_id() -> Optional[str]:
    """Returns the systemd machine-id or None if unavailable."""
    try:
        content = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
        return content if content else None
    except (OSError, PermissionError):
        return None


def _read_cpu_model() -> Optional[str]:
    """Extracts the 'model name' field from /proc/cpuinfo."""
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
    Falls back to running ``blkid`` if the sysfs path is empty (e.g. containers).
    """
    uuid_dir = Path("/dev/disk/by-uuid")
    if uuid_dir.exists():
        uuids = sorted(p.name for p in uuid_dir.iterdir() if p.is_symlink())
        if uuids:
            return uuids[0]

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_identifiers() -> dict[str, str]:
    """
    Returns a dict of available hardware/OS identifiers.

    Missing sources are represented as the literal string ``"UNAVAILABLE"``
    so the fingerprint still differs from a machine where they are present.
    """
    return {
        "machine_id": _read_machine_id() or "UNAVAILABLE",
        "cpu_model":  _read_cpu_model()   or "UNAVAILABLE",
        "disk_uuid":  _read_first_disk_uuid() or "UNAVAILABLE",
    }


def build_canonical_string(identifiers: dict[str, str]) -> str:
    """
    Produces a deterministic, normalised fingerprint string from a dict of
    identifiers.  Keys are sorted so insertion order does not matter.
    """
    parts = [f"{k}={v}" for k, v in sorted(identifiers.items())]
    return _SEP.join(parts)


def compute_fingerprint_hash(canonical: str) -> str:
    """
    Hashes the canonical fingerprint string with SHA-256 and returns the
    lowercase hex digest.  This is the only value that is safe to persist.
    """
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_fingerprint_hash() -> str:
    """
    High-level entry point: collect identifiers, build canonical string, hash
    it, log only the hash (masked to first 8 chars for readability), and
    return the full hex digest.

    The raw canonical string lives only in local scope and is never stored.
    """
    ids = collect_identifiers()

    # Log which sources contributed, but not their raw values.
    availability = {k: (v != "UNAVAILABLE") for k, v in ids.items()}
    logger.debug("Fingerprint source availability: %s", availability)

    canonical = build_canonical_string(ids)
    fp_hash = compute_fingerprint_hash(canonical)

    logger.info("Device fingerprint computed. hash_prefix=%s…", fp_hash[:8])
    return fp_hash
