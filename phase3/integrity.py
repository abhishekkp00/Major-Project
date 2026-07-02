"""
phase3/integrity.py  —  Step 5
---------------------------------
Computes and verifies the SHA-256 hash of the encrypted adapter artefact.

Why hash the *encrypted* file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We sign and hash ``adapter.enc``, not the plaintext.  This means:
* Verification can happen **without decrypting** — tampering is detectable
  before any key material is used.
* The hash only leaks information about the ciphertext structure, not the
  adapter weights.

Security note
~~~~~~~~~~~~~
``hmac.compare_digest`` is used for the final comparison, which runs in
constant time regardless of where the strings first differ.  This prevents
timing oracle attacks that could otherwise be used to forge a valid hash
byte-by-byte.
"""

import hashlib
import hmac
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Read in 64 KiB blocks to keep memory usage bounded for large adapters.
_CHUNK = 64 * 1024


def compute_file_hash(file_path: Path) -> str:
    """
    Streams ``file_path`` through SHA-256 and returns the lowercase hex digest.

    Parameters
    ----------
    file_path : Path
        Path to the file to hash.  Typically ``adapter.enc``.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            sha256.update(chunk)

    digest = sha256.hexdigest()
    logger.debug("SHA-256 computed for %s: %s…", file_path.name, digest[:12])
    return digest


def save_hash(digest: str, hash_path: Path) -> None:
    """
    Writes the hex digest to ``hash_path`` using an atomic write.

    Parameters
    ----------
    digest : str
        Hex SHA-256 digest to persist.
    hash_path : Path
        Destination path (``adapter.hash``).
    """
    import os
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = hash_path.with_suffix(".tmp")
    try:
        tmp.write_text(digest + "\n", encoding="utf-8")
        os.replace(tmp, hash_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Integrity hash saved → %s", hash_path.name)


def load_hash(hash_path: Path) -> str:
    """
    Loads and strips the hex digest from ``hash_path``.

    Raises
    ------
    FileNotFoundError
        If ``hash_path`` does not exist.
    ValueError
        If the file does not contain a valid 64-character hex string.
    """
    if not hash_path.exists():
        raise FileNotFoundError(f"Hash file not found: {hash_path}")

    stored = hash_path.read_text(encoding="utf-8").strip()
    if len(stored) != 64 or not all(c in "0123456789abcdef" for c in stored):
        raise ValueError(
            f"'{hash_path.name}' does not contain a valid SHA-256 hex digest."
        )
    return stored


def verify_integrity(enc_path: Path, hash_path: Path) -> None:
    """
    Recomputes the SHA-256 hash of ``enc_path`` and compares it against the
    stored digest in ``hash_path`` using constant-time comparison.

    Parameters
    ----------
    enc_path : Path
        The encrypted adapter file to verify (``adapter.enc``).
    hash_path : Path
        The stored hash file (``adapter.hash``).

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    ValueError
        If the hashes do not match (tampering detected).
    """
    stored = load_hash(hash_path)
    actual = compute_file_hash(enc_path)

    if not hmac.compare_digest(stored, actual):
        raise ValueError(
            "Integrity check FAILED. "
            f"Stored hash: {stored[:12]}… | Computed hash: {actual[:12]}… "
            "The encrypted adapter has been tampered with or replaced."
        )

    logger.info("Integrity check PASSED for %s.", enc_path.name)
