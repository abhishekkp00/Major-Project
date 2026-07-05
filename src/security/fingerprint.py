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
    """Returns the alphabetically first UUID from /dev/disk/by-uuid/ (symlinks)."""
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
    """Returns a dict of available hardware/OS identifiers."""
    return {
        "machine_id": _read_machine_id() or "UNAVAILABLE",
        "cpu_model":  _read_cpu_model()   or "UNAVAILABLE",
        "disk_uuid":  _read_first_disk_uuid() or "UNAVAILABLE",
    }


def build_canonical_string(identifiers: dict[str, str]) -> str:
    """Produces a deterministic, normalised fingerprint string from a dict of identifiers."""
    parts = [f"{k}={v}" for k, v in sorted(identifiers.items())]
    return _SEP.join(parts)


def compute_fingerprint_hash(canonical: str) -> str:
    """Hashes the canonical fingerprint string with SHA-256 and returns hex digest."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_fingerprint_hash() -> str:
    """High-level entry point: collects and hashes fingerprint."""
    ids = collect_identifiers()

    # Log which sources contributed, but not their raw values.
    availability = {k: (v != "UNAVAILABLE") for k, v in ids.items()}
    logger.debug("Fingerprint source availability: %s", availability)

    canonical = build_canonical_string(ids)
    fp_hash = compute_fingerprint_hash(canonical)

    logger.info("Device fingerprint computed. hash_prefix=%s…", fp_hash[:8])
    return fp_hash
