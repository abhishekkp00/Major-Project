"""
device_secret.py
================
Manages the device-bound cryptographic secret for SecureLoRA key derivation.

DESIGN
------
A 32-byte secret is generated ONCE using the OS CSPRNG (secrets.token_bytes)
and stored in a protected file outside the repository:

    {XDG_DATA_HOME or ~/.local/share}/securelora/device.secret

The file is created with mode 0600 on Linux so only the owning user can read it.
If the file already exists, it is loaded; the secret is NEVER regenerated
on a normal run.  Regeneration happens only through an explicit initialization
call (``initialize_device_secret``).

SECURITY NOTES
--------------
* The raw secret is NEVER logged, printed, returned in API responses, or
  included in any package artifact or package manifest.
* If the secret file is missing during an operation that requires it, the
  operation FAILS CLOSED (raises DeviceSecretError).  There is no silent
  fallback, no demo default, and no random regeneration on missing secret.
* The secret is the IKM (Input Key Material) for HKDF-SHA256.  The device
  fingerprint hash is bound into the HKDF ``info`` field, not used as IKM.
* On Linux/POSIX the file permissions are set to 0600 immediately after
  creation.  If the chmod fails, the file is deleted and initialization
  fails with an error.
"""

import logging
import os
import stat
from pathlib import Path
from typing import Optional

from src.common.exceptions import SecureLoraError

logger = logging.getLogger("secure_lora.security.device_secret")

# ── Public exception ──────────────────────────────────────────────────────────

class DeviceSecretError(SecureLoraError, OSError):
    """
    Raised when the device secret cannot be loaded, is missing, or when
    initialization fails.  Never silently recovered from.
    """


# ── Constants ─────────────────────────────────────────────────────────────────

_SECRET_LENGTH: int = 32  # bytes — matches AES-256 key size for symmetric use
_SECRET_FILENAME: str = "device.secret"
_DIR_NAME: str = "securelora"


# ── Storage location ──────────────────────────────────────────────────────────

def _secret_dir() -> Path:
    """
    Returns the application-specific directory for storing the device secret.

    Respects XDG_DATA_HOME if set; otherwise falls back to ~/.local/share.
    The directory is OUTSIDE the repository.
    """
    xdg = os.environ.get("XDG_DATA_HOME", "")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".local" / "share"
    return base / _DIR_NAME


def _secret_path() -> Path:
    override = os.environ.get("SECURELORA_SECRET_PATH", "")
    if override:
        return Path(override)
    return _secret_dir() / _SECRET_FILENAME


# ── Low-level I/O helpers ─────────────────────────────────────────────────────

def _enforce_permissions(path: Path) -> None:
    """
    Sets mode 0600 on *path*.  If this fails (e.g. on non-POSIX systems),
    raises DeviceSecretError — we do not silently continue.
    """
    if os.name == "posix":
        try:
            path.chmod(0o600)
        except OSError as exc:
            # Best effort: try to remove the partially written file.
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise DeviceSecretError(
                f"Failed to set restrictive permissions (0600) on device secret "
                f"file '{path}': {exc}. "
                "The secret file has been removed for safety. "
                "Run initialization again."
            ) from exc


def _verify_permissions(path: Path) -> None:
    """
    Verifies that *path* has at most 0600 permissions on POSIX.
    Raises DeviceSecretError if other users can read the file.
    """
    if os.name != "posix":
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        raise DeviceSecretError(
            f"Device secret file '{path}' has insecure permissions "
            f"({oct(mode & 0o777)}). Expected at most 0600. "
            "Fix or remove the file and re-initialize."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def secret_file_path() -> Path:
    """Returns the canonical path to the device secret file (for diagnostics only)."""
    return _secret_path()


def initialize_device_secret(force: bool = False) -> Path:
    """
    Generates a new 32-byte CSPRNG secret and writes it to the protected file.

    Parameters
    ----------
    force : bool
        If False (default) and the secret already exists, raises DeviceSecretError
        instead of overwriting.  Pass ``force=True`` only for explicit
        re-enrollment operations.

    Returns
    -------
    Path
        The path to the newly created secret file.

    Raises
    ------
    DeviceSecretError
        If the secret already exists (and force=False), if the directory
        cannot be created, or if permissions cannot be enforced.
    """
    import secrets as _secrets  # stdlib — OS CSPRNG

    secret_path = _secret_path()

    if secret_path.exists() and not force:
        raise DeviceSecretError(
            f"Device secret already exists at '{secret_path}'. "
            "Use force=True only for explicit re-enrollment."
        )

    # Create directory with restrictive permissions
    secret_dir = secret_path.parent
    try:
        secret_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            secret_dir.chmod(0o700)
    except OSError as exc:
        raise DeviceSecretError(
            f"Failed to create device secret directory '{secret_dir}': {exc}"
        ) from exc

    # Generate secret
    raw_secret = _secrets.token_bytes(_SECRET_LENGTH)

    # Write atomically: write to temp then rename
    tmp_path = secret_path.with_suffix(".tmp")
    try:
        tmp_path.write_bytes(raw_secret)
        _enforce_permissions(tmp_path)
        tmp_path.rename(secret_path)
    except DeviceSecretError:
        raise
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise DeviceSecretError(
            f"Failed to write device secret to '{secret_path}': {exc}"
        ) from exc

    logger.info(
        "Device secret initialized at '%s' (mode=0600, length=%d bytes).",
        secret_path,
        _SECRET_LENGTH,
    )
    return secret_path


def load_device_secret() -> bytes:
    """
    Loads the 32-byte device secret from the protected file.

    FAIL-CLOSED: If the file is missing, unreadable, has wrong permissions,
    or has the wrong length, raises DeviceSecretError.  There is NO fallback,
    NO silent regeneration, and NO demo default.

    Returns
    -------
    bytes
        The 32-byte device secret.

    Raises
    ------
    DeviceSecretError
        If the secret cannot be loaded for any reason.
    """
    secret_path = _secret_path()

    if not secret_path.exists():
        raise DeviceSecretError(
            f"Device secret file not found at '{secret_path}'. "
            "Run 'initialize_device_secret()' (or the CLI enrollment command) "
            "to generate the secret for this device before packaging or deploying."
        )

    _verify_permissions(secret_path)

    try:
        raw = secret_path.read_bytes()
    except OSError as exc:
        raise DeviceSecretError(
            f"Failed to read device secret from '{secret_path}': {exc}"
        ) from exc

    if len(raw) != _SECRET_LENGTH:
        raise DeviceSecretError(
            f"Device secret at '{secret_path}' has wrong length "
            f"{len(raw)} bytes (expected {_SECRET_LENGTH}). "
            "The file may be corrupted. Re-initialize with care."
        )

    logger.debug("Device secret loaded from '%s' (%d bytes).", secret_path, len(raw))
    return raw


def device_secret_exists() -> bool:
    """Returns True if the device secret file exists (without loading it)."""
    return _secret_path().exists()


def get_device_secret(auto_initialize: bool = False) -> bytes:
    """
    High-level entry point: loads and returns the 32-byte device secret.
    If auto_initialize is True and the secret does not exist, initializes it.
    Defaults to auto_initialize=False so operations fail closed if the device secret is missing.
    """
    if not device_secret_exists():
        if auto_initialize:
            initialize_device_secret()
        else:
            return load_device_secret()
    return load_device_secret()


