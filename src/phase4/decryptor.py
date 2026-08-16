"""
decryptor.py
============
Phase 4 — Secure Adapter Decryption Context Manager.

PLAINTEXT-ON-DISK NOTICE
=========================
The PEFT library (peft.PeftModel.from_pretrained) requires a filesystem path
to load adapter weights.  Therefore this module DOES write decrypted adapter
bytes to a temporary directory on disk before PEFT loads them.

What we guarantee:
  - The temporary directory is created with mkdtemp (mode 0o700 on POSIX).
  - The decrypted tar.gz is shredded immediately after extraction.
  - The entire temporary directory is shredded on context-manager exit,
    both on success and on any exception (guaranteed by try/finally).
  - "Shredding" means overwrite-then-unlink (see src/security/shred.py).

What we do NOT claim:
  - RAM-only adapter loading.
  - Zero plaintext on disk.
  - Guaranteed erasure on kernel-managed swap or copy-on-write filesystems.

The plaintext adapter files exist on disk for the duration of the PEFT
model-loading call inside the context block.  After __exit__ runs they
are securely overwritten and removed.
"""

import tarfile
import logging
import tempfile
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.security import shred_file, shred_directory
from src.common.exceptions import SecurityError

logger = logging.getLogger("secure_lora.phase4.decryptor")

NONCE_BYTES = 12


class DecryptedAdapterContext:
    """
    Context manager that decrypts the adapter to a temporary folder on disk
    and shreds all plaintext files on exit.

    Plaintext lifecycle:
      __enter__:
        1. Reads encrypted bytes from enc_path entirely into memory.
        2. Decrypts in-memory via AES-256-GCM (raises on auth tag failure).
        3. Writes decrypted tar.gz to a temporary directory (disk write).
        4. Extracts the tar.gz to the same temporary directory (disk write).
        5. Shreds the tar.gz immediately after extraction.
        6. Returns the path of the extracted adapter directory.

      context body:
        PEFT loads adapter weights from the returned filesystem path.
        Plaintext files exist on disk during this period.

      __exit__:
        Shreds every file in the temporary directory, then removes it.
        Called on both success and exception.
    """

    def __init__(self, enc_path: Path, key: bytes):
        self.enc_path = Path(enc_path)
        self.key = key
        self.temp_dir: Optional[Path] = None
        self.tar_path: Optional[Path] = None

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

        # Create temp folder for decrypted adapter files (mode 0o700 on POSIX)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="secure_lora_decrypted_"))
        self.tar_path = self.temp_dir / "adapter.tar.gz"

        try:
            # Step 1: Decrypt in memory — AES-GCM authenticates before any plaintext is produced.
            raw_bytes = self.enc_path.read_bytes()
            nonce = raw_bytes[:NONCE_BYTES]
            ciphertext = raw_bytes[NONCE_BYTES:]

            aesgcm = AESGCM(self.key)
            # decrypt() raises cryptography.exceptions.InvalidTag if auth check fails.
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

            # Step 2: Write decrypted tarball to disk (required for tarfile.extractall).
            self.tar_path.write_bytes(plaintext)
            del plaintext  # Release in-memory plaintext reference as early as possible.

            # Step 3: Extract — validate paths to prevent directory traversal.
            with tarfile.open(self.tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    target_path = (self.temp_dir / member.name).resolve()
                    if not str(target_path).startswith(str(self.temp_dir)):
                        raise SecurityError(f"Directory traversal detected: {member.name}")
                tar.extractall(path=self.temp_dir)

            # Step 4: Shred the tar.gz immediately after extraction.
            shred_file(self.tar_path)

            # Step 5: Locate the adapter directory (contains adapter_config.json).
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
            if isinstance(e, SecurityError):
                raise
            raise ValueError(
                f"Decryption or extraction failed (wrong key or tampered ciphertext): {e}"
            ) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_internal()
        # Do not suppress exceptions.
        return False

    def _cleanup_internal(self):
        """Shred all temporary files and remove the temporary directory."""
        if self.tar_path and self.tar_path.exists():
            shred_file(self.tar_path)
        if self.temp_dir and self.temp_dir.exists():
            logger.info("Shredding temporary decrypted adapter files from disk...")
            shred_directory(self.temp_dir)
            self.temp_dir = None
