"""
phase4/decryptor.py
--------------------
Handles secure decryption of the adapter package using the derived key.
Extracts the resulting plaintext adapter tarball into a temporary directory
and implements cryptographic shredding to ensure no plaintext is left behind.
"""

import os
import tarfile
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

NONCE_BYTES = 12

def secure_shred_file(file_path: Path, passes: int = 3) -> None:
    """Overwrites a file with random bytes and flushes before unlinking."""
    if not file_path.exists():
        return
    try:
        size = file_path.stat().st_size
        # Overwrite file multiple times with random bytes
        with open(file_path, "r+b", buffering=0) as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(size))
                os.fsync(f.fileno())
    except OSError as e:
        logger.warning("OS error during secure shredding of %s: %s", file_path.name, e)
    finally:
        file_path.unlink(missing_ok=True)

def secure_shred_directory(dir_path: Path) -> None:
    """Recursively shreds all files in a directory and deletes the directory."""
    if not dir_path.exists():
        return
    for item in dir_path.rglob("*"):
        if item.is_file():
            secure_shred_file(item)
    try:
        shutil.rmtree(dir_path)
        logger.debug("Successfully removed decrypted directory structure: %s", dir_path.name)
    except Exception as e:
        logger.warning("Failed to remove directory %s: %s", dir_path, e)

class DecryptedAdapterContext:
    """
    Context manager that decrypts the adapter to a temp folder and shreds it on exit.
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
        
        # Create temp folder for decrypted adapter files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="secure_lora_decrypted_"))
        self.tar_path = self.temp_dir / "adapter.tar.gz"

        try:
            # 1. Decrypt adapter.enc to adapter.tar.gz
            raw_bytes = self.enc_path.read_bytes()
            nonce = raw_bytes[:NONCE_BYTES]
            ciphertext = raw_bytes[NONCE_BYTES:]

            aesgcm = AESGCM(self.key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
            
            # Write decrypted tarball
            self.tar_path.write_bytes(plaintext)

            # 2. Extract adapter.tar.gz to directory
            with tarfile.open(self.tar_path, "r:gz") as tar:
                # Security: prevent path traversal attacks
                for member in tar.getmembers():
                    target_path = (self.temp_dir / member.name).resolve()
                    if not str(target_path).startswith(str(self.temp_dir)):
                        raise SecurityError(f"Directory traversal detected: {member.name}")
                tar.extractall(path=self.temp_dir)

            # Shred the temporary tar.gz file now that it is extracted
            secure_shred_file(self.tar_path)
            
            # Locate folder structure inside temp_dir
            # Look for folder containing adapter_config.json
            config_paths = list(self.temp_dir.rglob("adapter_config.json"))
            if not config_paths:
                raise FileNotFoundError("Decrypted adapter does not contain adapter_config.json")
            
            adapter_dir = config_paths[0].parent
            logger.info("Adapter decrypted and extracted successfully to: %s", adapter_dir)
            return adapter_dir

        except Exception as e:
            self._cleanup_internal()
            raise ValueError(f"Decryption or extraction failed (wrong key or tampered ciphertext): {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_internal()

    def _cleanup_internal(self):
        if self.tar_path and self.tar_path.exists():
            secure_shred_file(self.tar_path)
        if self.temp_dir and self.temp_dir.exists():
            logger.info("Shredding temporary decrypted adapter files from disk...")
            secure_shred_directory(self.temp_dir)
            self.temp_dir = None
