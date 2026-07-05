import os
import struct
import hashlib
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Union, Generator, Iterator, Optional
from contextlib import contextmanager

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.common.exceptions import CryptoError
from src.security.shred import shred_file

logger = logging.getLogger("secure_lora.security.crypto")

MAGIC_BYTES = b"SECLORA"
VERSION = 1
CHUNK_SIZE = 64 * 1024  # 64 KB chunk size for memory-safe processing
_NONCE_BYTES = 12


def generate_key() -> bytes:
    """Generates a secure 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


# ── Streaming AES-GCM (Phase 1 & 2 Datasets) ─────────────────────────────────

def encrypt_stream(instream, outstream, key: bytes) -> None:
    """
    Encrypts data from instream to outstream chunk-by-chunk using AES-256-GCM.
    """
    if len(key) != 32:
        raise CryptoError(f"Key must be exactly 32 bytes; got {len(key)}.")

    aesgcm = AESGCM(key)
    outstream.write(MAGIC_BYTES)
    outstream.write(struct.pack(">B", VERSION))

    chunk_idx = 0
    buffer = instream.read(CHUNK_SIZE)

    while buffer:
        next_buffer = instream.read(CHUNK_SIZE)
        is_final = 1 if len(next_buffer) == 0 else 0

        # Associated data format: chunk_idx (8 bytes big-endian) + is_final (1 byte)
        ad = struct.pack(">QB", chunk_idx, is_final)
        nonce = os.urandom(_NONCE_BYTES)

        encrypted_data = aesgcm.encrypt(nonce, buffer, ad)

        # Output layout: nonce (12B) + payload_len (4B) + is_final (1B) + payload (varB)
        outstream.write(nonce)
        outstream.write(struct.pack(">I", len(encrypted_data)))
        outstream.write(struct.pack(">B", is_final))
        outstream.write(encrypted_data)

        buffer = next_buffer
        chunk_idx += 1


def decrypt_stream(instream, outstream, key: bytes) -> None:
    """
    Decrypts data from instream to outstream chunk-by-chunk using AES-256-GCM.
    """
    if len(key) != 32:
        raise CryptoError(f"Key must be exactly 32 bytes; got {len(key)}.")

    aesgcm = AESGCM(key)

    magic = instream.read(len(MAGIC_BYTES))
    if magic != MAGIC_BYTES:
        raise CryptoError("Invalid magic bytes. Not a secure dataset or file is corrupted.")

    version_bytes = instream.read(1)
    if not version_bytes:
        raise CryptoError("Corrupted file header: version byte missing.")
    version = struct.unpack(">B", version_bytes)[0]
    if version != VERSION:
        raise CryptoError(f"Unsupported file version: {version}")

    chunk_idx = 0
    has_final_chunk = False

    while True:
        nonce = instream.read(12)
        if not nonce:
            break

        len_bytes = instream.read(4)
        if len(len_bytes) < 4:
            raise CryptoError("Corrupted chunk header: length field incomplete.")
        chunk_len = struct.unpack(">I", len_bytes)[0]

        is_final_bytes = instream.read(1)
        if not is_final_bytes:
            raise CryptoError("Corrupted chunk header: is_final field missing.")
        is_final = struct.unpack(">B", is_final_bytes)[0]

        encrypted_data = instream.read(chunk_len)
        if len(encrypted_data) < chunk_len:
            raise CryptoError("Corrupted chunk data: truncated block.")

        ad = struct.pack(">QB", chunk_idx, is_final)
        try:
            decrypted_data = aesgcm.decrypt(nonce, encrypted_data, ad)
        except Exception as e:
            raise CryptoError(
                f"Decryption failed at chunk {chunk_idx}. The file might have been tampered with."
            ) from e

        outstream.write(decrypted_data)

        if is_final == 1:
            has_final_chunk = True
            break

        chunk_idx += 1

    if not has_final_chunk:
        raise CryptoError("File was truncated. The final block was not reached.")


def decrypt_generator(encrypted_file_path: Union[str, Path], key: bytes) -> Generator[dict, None, None]:
    """
    Decrypts an encrypted dataset streamingly and yields individual preprocessed JSON dicts.
    """
    if len(key) != 32:
        raise CryptoError(f"Key must be exactly 32 bytes; got {len(key)}.")

    aesgcm = AESGCM(key)

    with open(encrypted_file_path, "rb") as instream:
        magic = instream.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise CryptoError("Invalid magic bytes. Not a secure dataset or file is corrupted.")

        version_bytes = instream.read(1)
        if not version_bytes:
            raise CryptoError("Corrupted file header: version byte missing.")
        version = struct.unpack(">B", version_bytes)[0]
        if version != VERSION:
            raise CryptoError(f"Unsupported file version: {version}")

        chunk_idx = 0
        has_final_chunk = False
        line_buffer = bytearray()

        while True:
            nonce = instream.read(12)
            if not nonce:
                break

            len_bytes = instream.read(4)
            if len(len_bytes) < 4:
                raise CryptoError("Corrupted chunk header: length field incomplete.")
            chunk_len = struct.unpack(">I", len_bytes)[0]

            is_final_bytes = instream.read(1)
            if not is_final_bytes:
                raise CryptoError("Corrupted chunk header: is_final field missing.")
            is_final = struct.unpack(">B", is_final_bytes)[0]

            encrypted_data = instream.read(chunk_len)
            if len(encrypted_data) < chunk_len:
                raise CryptoError("Corrupted chunk data: truncated block.")

            ad = struct.pack(">QB", chunk_idx, is_final)
            try:
                decrypted_data = aesgcm.decrypt(nonce, encrypted_data, ad)
            except Exception as e:
                raise CryptoError(
                    f"Decryption failed at chunk {chunk_idx}. The file might have been tampered with."
                ) from e

            line_buffer.extend(decrypted_data)

            while b"\n" in line_buffer:
                idx = line_buffer.index(b"\n")
                line = line_buffer[:idx]
                del line_buffer[:idx + 1]

                cleaned_line = line.strip()
                if cleaned_line:
                    yield json.loads(cleaned_line.decode('utf-8'))

            if is_final == 1:
                has_final_chunk = True
                break

            chunk_idx += 1

        if not has_final_chunk:
            raise CryptoError("File was truncated. The final block was not reached.")

        if line_buffer:
            cleaned_line = line_buffer.strip()
            if cleaned_line:
                yield json.loads(cleaned_line.decode('utf-8'))


@contextmanager
def decrypted_temporary_file(encrypted_file_path: Union[str, Path], key: bytes) -> Iterator[Path]:
    """
    Context manager that decrypts an encrypted dataset to a temporary plaintext file
    on disk, yields its path, and guarantees secure shredding and removal upon exit.
    """
    # Create a secure temporary file in the OS default temp directory
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".jsonl")
    os.close(temp_fd)
    temp_path = Path(temp_path_str)

    try:
        with open(encrypted_file_path, "rb") as fin, open(temp_path, "wb") as fout:
            decrypt_stream(fin, fout, key)
        yield temp_path
    finally:
        shred_file(temp_path)


# ── Block-level AES-GCM (Phase 3 & 4 Adapter Weights) ─────────────────────────

def _archive_directory(source_dir: Path, dest_tar: Path) -> None:
    """Creates a deterministic, reproducible tar.gz from source_dir."""
    with tarfile.open(dest_tar, "w:gz") as tar:
        for entry in sorted(source_dir.rglob("*")):
            tar.add(entry, arcname=entry.relative_to(source_dir.parent))
    logger.debug("Archived directory → %s (%d bytes)", dest_tar.name, dest_tar.stat().st_size)


def _encrypt_file(plaintext_path: Path, output_path: Path, key: bytes) -> tuple[str, str]:
    """Encrypts plaintext_path into output_path with AES-256-GCM."""
    nonce = os.urandom(_NONCE_BYTES)
    aesgcm = AESGCM(key)

    plaintext = plaintext_path.read_bytes()
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    plaintext_sha256 = hashlib.sha256(plaintext).hexdigest()

    # Atomic write
    tmp_path = output_path.with_suffix(".tmp")
    try:
        tmp_path.write_bytes(nonce + ciphertext)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.debug("Encrypted file → %s", output_path.name)
    return nonce.hex(), plaintext_sha256


def encrypt_adapter(
    adapter_input: Path,
    output_enc_path: Path,
    key: bytes,
    fingerprint_hash: str,
    metadata_path: Optional[Path] = None,
) -> dict:
    """Encrypts the LoRA adapter (file or directory) and writes ciphertext to output_enc_path."""
    if not adapter_input.exists():
        raise FileNotFoundError(f"Adapter input not found: {adapter_input}")
    if len(key) != 32:
        raise CryptoError(f"Key must be 32 bytes for AES-256; got {len(key)}.")

    output_enc_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_archive: Optional[Path] = None
    plaintext_target: Path

    try:
        if adapter_input.is_dir():
            tmp_fd, tmp_str = tempfile.mkstemp(suffix=".tar.gz", prefix="p3_adapter_")
            os.close(tmp_fd)
            tmp_archive = Path(tmp_str)
            _archive_directory(adapter_input, tmp_archive)
            plaintext_target = tmp_archive
            adapter_format = "directory:tar.gz"
        else:
            plaintext_target = adapter_input
            adapter_format = f"file:{adapter_input.suffix or 'bin'}"

        nonce_hex, plaintext_sha256 = _encrypt_file(plaintext_target, output_enc_path, key)
    finally:
        if tmp_archive is not None:
            shred_file(tmp_archive)

    metadata = {
        "algorithm": "AES-256-GCM",
        "nonce_hex": nonce_hex,
        "adapter_format": adapter_format,
        "plaintext_sha256_ref": plaintext_sha256,
        "fingerprint_hash_ref": fingerprint_hash,
        "version": "3.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if metadata_path is not None:
        tmp_meta = metadata_path.with_suffix(".tmp")
        try:
            tmp_meta.write_text(json.dumps(metadata, indent=2))
            os.replace(tmp_meta, metadata_path)
        except Exception:
            tmp_meta.unlink(missing_ok=True)
            raise
        logger.info("Encryption metadata saved → %s", metadata_path.name)

    return metadata


def decrypt_adapter(
    enc_path: Path,
    output_path: Path,
    key: bytes,
) -> None:
    """Decrypts enc_path into output_path using the provided AES-256-GCM key."""
    if not enc_path.exists():
        raise FileNotFoundError(f"Encrypted adapter not found: {enc_path}")
    if len(key) != 32:
        raise CryptoError(f"Key must be 32 bytes; got {len(key)}.")

    raw = enc_path.read_bytes()
    nonce = raw[:_NONCE_BYTES]
    ciphertext = raw[_NONCE_BYTES:]

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except Exception as exc:
        raise CryptoError(
            "AES-GCM decryption failed. The key is wrong, or the ciphertext has been tampered with."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(".tmp")
    try:
        tmp_out.write_bytes(plaintext)
        os.replace(tmp_out, output_path)
    except Exception:
        tmp_out.unlink(missing_ok=True)
        raise

    logger.info("Adapter decrypted successfully → %s", output_path.name)


def verify_integrity(enc_path: Path, hash_path: Path) -> None:
    """
    Recomputes the SHA-256 hash of enc_path and compares it against the
    stored digest in hash_path using constant-time comparison.
    """
    import hmac
    if not enc_path.exists():
        raise FileNotFoundError(f"Encrypted adapter not found: {enc_path}")
    if not hash_path.exists():
        raise FileNotFoundError(f"Hash file not found: {hash_path}")

    stored = hash_path.read_text(encoding="utf-8").strip()
    if len(stored) != 64 or not all(c in "0123456789abcdef" for c in stored):
        raise ValueError(
            f"'{hash_path.name}' does not contain a valid SHA-256 hex digest."
        )

    actual = compute_sha256(enc_path)

    if not hmac.compare_digest(stored, actual):
        raise ValueError(
            "Integrity check FAILED. "
            f"Stored hash: {stored[:12]}… | Computed hash: {actual[:12]}… "
            "The encrypted adapter has been tampered with or replaced."
        )

    logger.info("Integrity check PASSED for %s.", enc_path.name)


def save_hash(digest: str, hash_path: Path) -> None:
    """Writes the hex digest to hash_path using an atomic write."""
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = hash_path.with_suffix(".tmp")
    try:
        tmp.write_text(digest + "\n", encoding="utf-8")
        os.replace(tmp, hash_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Integrity hash saved → %s", hash_path.name)


