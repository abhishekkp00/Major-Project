"""
phase3/signature_utils.py  —  Step 6
---------------------------------------
RSA-PSS digital signature generation and verification for the encrypted adapter.

What is signed
~~~~~~~~~~~~~~
The SHA-256 hex digest of ``adapter.enc`` (the string stored in
``adapter.hash``) is signed, not the raw encrypted bytes.  This keeps the
signing operation fast and independent of adapter size, while still
cryptographically binding the signature to the exact ciphertext.

Key management design
~~~~~~~~~~~~~~~~~~~~~
* ``generate_dev_keypair()``  — creates a fresh RSA-2048 (default) keypair for
  development and testing.  **Never use in production without replacing keys.**
* Private key path is kept separate from the package output.  The package
  bundles only the **public** key.
* Production deployments should load the private key from a HSM, Vault, or CI
  secret — not from disk.

Signature scheme
~~~~~~~~~~~~~~~~
RSA-PSS with SHA-256 hash and maximum salt length (``PSS.MAX_LENGTH``).
PSS is probabilistic (different signature each run) and is the current
cryptographic best practice over PKCS#1 v1.5 for RSA signing.

Verification
~~~~~~~~~~~~
Verification raises ``InvalidSignature`` (re-raised as ``ValueError``) on any
mismatch — wrong key, wrong hash, or tampered ciphertext.  It happens before
decryption so the private key is never exercised on suspect data.
"""

import logging
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key generation (development / testing only)
# ---------------------------------------------------------------------------

def generate_dev_keypair(
    private_key_path: Path,
    public_key_path: Path,
    key_size: int = 2048,
) -> None:
    """
    Generates a fresh RSA keypair and writes PEM files to the given paths.

    This is intended **only** for local development and automated tests.
    In production, substitute a key obtained from a secure key-management
    system and never call this function during normal runtime.

    Parameters
    ----------
    private_key_path : Path
        Where to write the PKCS#8 PEM-encoded private key (no passphrase in
        dev mode — add one for staging environments).
    public_key_path : Path
        Where to write the SubjectPublicKeyInfo PEM-encoded public key.
    key_size : int
        RSA modulus size in bits (default 2048; use 4096 for higher security).
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(private_key_path, private_pem)
    _atomic_write_bytes(public_key_path, public_pem)

    # Restrict private key permissions to owner-read-only.
    os.chmod(private_key_path, 0o600)

    logger.info(
        "Dev RSA keypair generated. key_size=%d bits | priv=%s | pub=%s",
        key_size,
        private_key_path.name,
        public_key_path.name,
    )


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def sign_digest(digest_hex: str, private_key_path: Path) -> bytes:
    """
    Signs ``digest_hex`` (the SHA-256 hex digest of ``adapter.enc``) using the
    RSA private key stored at ``private_key_path``.

    Parameters
    ----------
    digest_hex : str
        64-character hex string (output of ``integrity.compute_file_hash``).
    private_key_path : Path
        PEM file containing the RSA private key.

    Returns
    -------
    bytes
        Raw RSA-PSS signature bytes.

    Raises
    ------
    FileNotFoundError
        If the private key file does not exist.
    """
    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    private_key = _load_private_key(private_key_path)
    message = digest_hex.encode("utf-8")

    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    logger.info(
        "Adapter digest signed. sig_len=%d bytes | key=%s",
        len(signature),
        private_key_path.name,
    )
    return signature


def save_signature(signature: bytes, sig_path: Path) -> None:
    """Writes raw signature bytes to ``sig_path`` atomically."""
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(sig_path, signature)
    logger.info("Signature saved → %s", sig_path.name)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_signature(
    digest_hex: str,
    sig_path: Path,
    public_key_path: Path,
) -> None:
    """
    Verifies the RSA-PSS signature of ``digest_hex`` against the signature
    stored in ``sig_path`` using the public key at ``public_key_path``.

    Parameters
    ----------
    digest_hex : str
        The SHA-256 digest string that was signed (re-computed from
        ``adapter.enc`` by the verifier, not taken from the package).
    sig_path : Path
        Path to ``adapter.sig``.
    public_key_path : Path
        Path to the PEM public key.

    Raises
    ------
    FileNotFoundError
        If either ``sig_path`` or ``public_key_path`` is missing.
    ValueError
        If the signature is invalid (wraps ``InvalidSignature``).
    """
    if not sig_path.exists():
        raise FileNotFoundError(f"Signature file not found: {sig_path}")
    if not public_key_path.exists():
        raise FileNotFoundError(f"Public key not found: {public_key_path}")

    public_key = _load_public_key(public_key_path)
    signature = sig_path.read_bytes()
    message = digest_hex.encode("utf-8")

    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise ValueError(
            "Signature verification FAILED. "
            "The adapter may have been tampered with, or the wrong public key was used."
        ) from exc

    logger.info("Signature verification PASSED.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_private_key(path: Path):
    """Loads an RSA private key from a PEM file (no passphrase)."""
    return serialization.load_pem_private_key(
        path.read_bytes(),
        password=None,
    )


def _load_public_key(path: Path):
    """Loads an RSA public key from a PEM file."""
    return serialization.load_pem_public_key(path.read_bytes())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Writes ``data`` to ``path`` via a sibling .tmp file, then renames."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
