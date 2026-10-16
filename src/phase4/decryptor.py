"""
decryptor.py
============
Phase 4 — Secure Adapter Decryption Context Manager.

PLAINTEXT-ON-DISK NOTICE & THREAT BOUNDARIES
============================================
The PEFT library (peft.PeftModel.from_pretrained) requires a filesystem path
to load adapter weights. Consequently, decrypted adapter bytes are written
to a temporary directory on disk for the duration of PEFT model loading.

What we guarantee:
  - The temporary directory is created with restrictive mode (0o700 on POSIX).
  - All decrypted files within the directory are restricted to mode 0o600 on POSIX.
  - The temporary storage location is configurable via the `temp_storage_dir`
    argument or the `SECURELORA_TEMP_DIR` environment variable (e.g. to point
    to a volatile tmpfs ramdisk such as /dev/shm or an encrypted LUKS volume).
  - The decrypted tar.gz is shredded immediately after extraction.
  - The entire temporary directory is shredded upon context exit on both
    success and on any exception (guaranteed by try/finally).
  - Cleanup failures are surfaced when the block succeeds, and logged
    without hiding primary exceptions when an error occurs during model load.

What we do NOT claim (Explicit Erasure Limitations):
  - RAM-only adapter loading: Weights exist temporarily on disk during model loading.
  - Zero plaintext on disk: Plaintext exists in the temporary directory until __exit__.
  - Physical erasure on SSDs: Flash Translation Layers (FTL) perform wear-leveling
    and out-of-place writes; file overwrite cannot guarantee physical cell clearing.
  - Copy-on-Write (CoW) filesystems: Btrfs, ZFS, APFS, and snapshots write to new blocks.
  - OS Swap: Unencrypted swap partitions may page memory to disk.

For deployments requiring strict in-memory execution, configure SECURELORA_TEMP_DIR
to a RAM-backed filesystem (tmpfs / /dev/shm) and ensure OS swap is encrypted or disabled.
"""

import os
import stat
import tarfile
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, List

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.security import (
    shred_file,
    shred_directory,
    safe_extract_tar,
    DEFAULT_MAX_ARCHIVE_SIZE,
    DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE,
    DEFAULT_MAX_MEMBER_SIZE,
    DEFAULT_MAX_MEMBER_COUNT,
)
from src.common.exceptions import SecurityError

logger = logging.getLogger("secure_lora.phase4.decryptor")

NONCE_BYTES = 12


class DecryptedAdapterContext:
    """
    Context manager that decrypts the adapter to an ephemeral temporary folder on disk
    and shreds all plaintext files on exit.

    Plaintext lifecycle:
      __enter__:
        1. Reads encrypted bytes from enc_path into memory.
        2. Decrypts in-memory via AES-256-GCM (raises on auth tag failure).
        3. Creates a restricted temporary directory (mode 0700 on POSIX).
        4. Writes decrypted tar.gz to the temporary directory with mode 0600.
        5. Safely extracts the tar.gz to the temporary directory.
        6. Enforces restrictive file permissions (0600 files, 0700 dirs) on extracted contents.
        7. Shreds the tar.gz immediately after extraction.
        8. Returns the path of the extracted adapter directory.

      context body:
        PEFT loads adapter weights from the returned filesystem path.
        Plaintext files exist on disk for the duration of this block.

      __exit__:
        Shreds every file in the temporary directory, then removes it.
        Called on both success and exception.
    """

    def __init__(
        self,
        enc_path: Path,
        key: bytes,
        max_archive_size: int = DEFAULT_MAX_ARCHIVE_SIZE,
        max_total_uncompressed_size: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE,
        max_member_size: int = DEFAULT_MAX_MEMBER_SIZE,
        max_member_count: int = DEFAULT_MAX_MEMBER_COUNT,
        temp_storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.enc_path = Path(enc_path)
        self.key = key
        self.max_archive_size = max_archive_size
        self.max_total_uncompressed_size = max_total_uncompressed_size
        self.max_member_size = max_member_size
        self.max_member_count = max_member_count
        self.temp_storage_dir = Path(temp_storage_dir) if temp_storage_dir else None
        self.temp_dir: Optional[Path] = None
        self.tar_path: Optional[Path] = None

    def _resolve_base_temp_dir(self) -> Optional[Path]:
        """Resolves and validates the base temporary storage directory."""
        configured = self.temp_storage_dir
        if configured is None:
            env_override = os.environ.get("SECURELORA_TEMP_DIR", "").strip()
            if env_override:
                configured = Path(env_override)

        if configured is None:
            return None

        base = configured.resolve()
        if not base.exists():
            raise SecurityError(
                f"Configured temporary storage directory '{base}' does not exist. "
                "Create the directory or ensure the RAM disk is mounted."
            )
        if not base.is_dir():
            raise SecurityError(
                f"Configured temporary storage path '{base}' is not a directory."
            )

        if os.name == "posix":
            mode = base.stat().st_mode
            # Disallow world-writable directories unless the sticky bit is set (like /tmp)
            if (mode & stat.S_IWOTH) and not (mode & stat.S_ISVTX):
                raise SecurityError(
                    f"Configured temporary storage directory '{base}' has insecure permissions "
                    f"({oct(mode & 0o777)}): world-writable without sticky bit."
                )

        return base

    def __enter__(self) -> Path:
        if not self.enc_path.exists():
            raise FileNotFoundError(f"Encrypted adapter file not found: {self.enc_path}")
        if len(self.key) != 32:
            raise ValueError("AES key must be exactly 32 bytes.")

        logger.info("Initializing secure decryption block.")
        logger.info(
            "NOTE: Decrypted adapter bytes will be written to a temporary directory on disk. "
            "All temporary files are shredded on context exit."
        )

        base_dir = self._resolve_base_temp_dir()

        # Create temporary directory with restrictive permissions (0700)
        mkdtemp_kwargs = {"prefix": "secure_lora_decrypted_"}
        if base_dir is not None:
            mkdtemp_kwargs["dir"] = str(base_dir)

        self.temp_dir = Path(tempfile.mkdtemp(**mkdtemp_kwargs)).resolve()
        if os.name == "posix":
            try:
                os.chmod(self.temp_dir, 0o700)
            except OSError as exc:
                logger.warning("Could not set 0700 permissions on %s: %s", self.temp_dir, exc)

        self.tar_path = self.temp_dir / "adapter.tar.gz"

        try:
            # Step 1: Decrypt in memory — AES-GCM authenticates before any plaintext is produced.
            raw_bytes = self.enc_path.read_bytes()
            nonce = raw_bytes[:NONCE_BYTES]
            ciphertext = raw_bytes[NONCE_BYTES:]

            aesgcm = AESGCM(self.key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

            # Enforce decrypted archive size quota before disk write
            if len(plaintext) > self.max_archive_size:
                raise SecurityError(
                    f"Decrypted archive size ({len(plaintext)} bytes) exceeds limit of "
                    f"{self.max_archive_size} bytes."
                )

            # Step 2: Write decrypted tarball to disk with restricted permissions (0600)
            self.tar_path.write_bytes(plaintext)
            del plaintext  # Release in-memory plaintext reference as early as possible.
            if os.name == "posix":
                try:
                    os.chmod(self.tar_path, 0o600)
                except OSError:
                    pass

            # Step 3: Extract — validate paths and resource limits before extracting.
            with tarfile.open(self.tar_path, "r:gz") as tar:
                safe_extract_tar(
                    tar=tar,
                    destination_dir=self.temp_dir,
                    max_total_uncompressed_size=self.max_total_uncompressed_size,
                    max_member_size=self.max_member_size,
                    max_member_count=self.max_member_count,
                )

            # Step 4: Restrict permissions on extracted files (0600) and dirs (0700)
            if os.name == "posix":
                for item in self.temp_dir.rglob("*"):
                    try:
                        if item.is_file():
                            os.chmod(item, 0o600)
                        elif item.is_dir():
                            os.chmod(item, 0o700)
                    except OSError:
                        pass

            # Step 5: Shred the tar.gz immediately after extraction.
            shred_file(self.tar_path)

            # Step 6: Locate the adapter directory (contains adapter_config.json).
            config_paths = list(self.temp_dir.rglob("adapter_config.json"))
            if not config_paths:
                raise FileNotFoundError("Decrypted adapter does not contain adapter_config.json")

            adapter_dir = config_paths[0].parent
            logger.info("Adapter extracted to temporary directory: %s", adapter_dir)
            logger.info(
                "Plaintext adapter files exist on disk until context exit. "
                "Do not persist this path beyond the with-block."
            )
            return adapter_dir

        except Exception as e:
            self._cleanup_internal()
            if isinstance(e, (SecurityError, FileNotFoundError)):
                raise
            raise ValueError(
                f"Decryption or extraction failed (wrong key or tampered ciphertext): {e}"
            ) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_errors = self._cleanup_internal()
        if exc_val is not None:
            # Primary exception occurred in the with-block.
            # Log cleanup failures without masking the primary failure.
            if cleanup_errors:
                logger.error(
                    "Primary exception '%s' occurred; cleanup also encountered errors: %s",
                    exc_val,
                    cleanup_errors,
                )
            return False

        # No exception in with-block: if cleanup failed, surface it!
        if cleanup_errors:
            raise SecurityError(
                f"Decrypted plaintext cleanup failed: temporary directory '{self.temp_dir}' "
                f"could not be completely removed: {cleanup_errors[0]}"
            ) from cleanup_errors[0]

        return False

    def _cleanup_internal(self) -> List[Exception]:
        """Shred all temporary files and remove the temporary directory."""
        errors: List[Exception] = []
        if self.tar_path and self.tar_path.exists():
            try:
                shred_file(self.tar_path)
            except Exception as e:
                logger.warning("Failed to shred temporary tar file %s: %s", self.tar_path, e)
                errors.append(e)

        if self.temp_dir and self.temp_dir.exists():
            logger.info("Shredding temporary decrypted adapter files from disk...")
            try:
                shred_directory(self.temp_dir)
            except Exception as e:
                logger.warning("Failed to shred temporary directory %s: %s", self.temp_dir, e)
                errors.append(e)

            if self.temp_dir.exists():
                err = SecurityError(
                    f"Temporary directory '{self.temp_dir}' still exists after cleanup attempt."
                )
                logger.error("%s", err)
                errors.append(err)

            self.temp_dir = None

        return errors
