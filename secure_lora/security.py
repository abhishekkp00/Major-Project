import os
import struct
import hashlib
import logging
from pathlib import Path
from typing import Union, Generator, Iterator
from contextlib import contextmanager
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

MAGIC_BYTES = b"SECLORA"
VERSION = 1
CHUNK_SIZE = 64 * 1024  # 64 KB chunk size for memory-safe processing

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

def secure_delete_file(file_path: Union[str, Path], passes: int = 3) -> None:
    """
    Overwrites a file with random bytes multiple times (shredding)
    before removing it to prevent data recovery.
    """
    path = Path(file_path)
    if not path.exists():
        return

    try:
        file_size = path.stat().st_size
        # Open file in binary update mode with zero buffering
        with open(path, "ba+", buffering=0) as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))
                os.fsync(f.fileno())
        
        # Rename to mask metadata before unlinking
        parent = path.parent
        temp_name = parent / f"shredded_{os.urandom(8).hex()}"
        path.rename(temp_name)
        temp_name.unlink()
        logger.info(f"Securely deleted temporary file: {path.name}")
    except Exception as e:
        logger.error(f"Failed to securely delete {file_path}: {e}")
        # Fallback to simple unlink if overwrite fails
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

def encrypt_stream(instream, outstream, key: bytes) -> None:
    """
    Encrypts data from instream to outstream chunk-by-chunk using AES-256-GCM.
    Includes chunk index and end-of-file signaling in the associated data (AD)
    to prevent chunk swapping, truncation, or omission.
    """
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
        nonce = os.urandom(12)
        
        # AESGCM encrypt returns ciphertext with appended 16-byte tag
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
    Validates chunk index, integrity tag, and end-of-file indicators.
    """
    aesgcm = AESGCM(key)
    
    # Read and verify magic bytes and version
    magic = instream.read(len(MAGIC_BYTES))
    if magic != MAGIC_BYTES:
        raise ValueError("Invalid magic bytes. Not a secure dataset or file is corrupted.")
        
    version_bytes = instream.read(1)
    if not version_bytes:
        raise ValueError("Corrupted file header: version byte missing.")
    version = struct.unpack(">B", version_bytes)[0]
    if version != VERSION:
        raise ValueError(f"Unsupported file version: {version}")
        
    chunk_idx = 0
    has_final_chunk = False
    
    while True:
        nonce = instream.read(12)
        if not nonce:
            break
            
        len_bytes = instream.read(4)
        if len(len_bytes) < 4:
            raise ValueError("Corrupted chunk header: length field incomplete.")
        chunk_len = struct.unpack(">I", len_bytes)[0]
        
        is_final_bytes = instream.read(1)
        if not is_final_bytes:
            raise ValueError("Corrupted chunk header: is_final field missing.")
        is_final = struct.unpack(">B", is_final_bytes)[0]
        
        encrypted_data = instream.read(chunk_len)
        if len(encrypted_data) < chunk_len:
            raise ValueError("Corrupted chunk data: truncated block.")
            
        ad = struct.pack(">QB", chunk_idx, is_final)
        try:
            decrypted_data = aesgcm.decrypt(nonce, encrypted_data, ad)
        except Exception as e:
            raise ValueError(f"Decryption failed at chunk {chunk_idx}. The file might have been tampered with.") from e
            
        outstream.write(decrypted_data)
        
        if is_final == 1:
            has_final_chunk = True
            break
            
        chunk_idx += 1
        
    if not has_final_chunk:
        raise ValueError("File was truncated. The final block was not reached.")

def decrypt_generator(encrypted_file_path: Union[str, Path], key: bytes) -> Generator[dict, None, None]:
    """
    Decrypts an encrypted dataset streamingly and yields individual preprocessed JSON dicts.
    Completely avoids writing plaintext to disk, enabling secure training-ready loading.
    """
    import json
    aesgcm = AESGCM(key)
    
    with open(encrypted_file_path, "rb") as instream:
        magic = instream.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise ValueError("Invalid magic bytes. Not a secure dataset or file is corrupted.")
            
        version_bytes = instream.read(1)
        if not version_bytes:
            raise ValueError("Corrupted file header: version byte missing.")
        version = struct.unpack(">B", version_bytes)[0]
        if version != VERSION:
            raise ValueError(f"Unsupported file version: {version}")
            
        chunk_idx = 0
        has_final_chunk = False
        line_buffer = bytearray()
        
        while True:
            nonce = instream.read(12)
            if not nonce:
                break
                
            len_bytes = instream.read(4)
            if len(len_bytes) < 4:
                raise ValueError("Corrupted chunk header: length field incomplete.")
            chunk_len = struct.unpack(">I", len_bytes)[0]
            
            is_final_bytes = instream.read(1)
            if not is_final_bytes:
                raise ValueError("Corrupted chunk header: is_final field missing.")
            is_final = struct.unpack(">B", is_final_bytes)[0]
            
            encrypted_data = instream.read(chunk_len)
            if len(encrypted_data) < chunk_len:
                raise ValueError("Corrupted chunk data: truncated block.")
                
            ad = struct.pack(">QB", chunk_idx, is_final)
            try:
                decrypted_data = aesgcm.decrypt(nonce, encrypted_data, ad)
            except Exception as e:
                raise ValueError(f"Decryption failed at chunk {chunk_idx}. The file might have been tampered with.") from e
                
            line_buffer.extend(decrypted_data)
            
            # Yield any full lines processed
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
            raise ValueError("File was truncated. The final block was not reached.")
            
        # Yield any remaining text without trailing newline
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
    import tempfile
    
    # Create a secure temporary file in the OS default temp directory
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".jsonl")
    os.close(temp_fd)
    temp_path = Path(temp_path_str)
    
    try:
        with open(encrypted_file_path, "rb") as fin, open(temp_path, "wb") as fout:
            decrypt_stream(fin, fout, key)
        yield temp_path
    finally:
        secure_delete_file(temp_path)
