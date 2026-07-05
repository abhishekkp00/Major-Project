import logging
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.common.exceptions import CryptoError

logger = logging.getLogger("secure_lora.security.signature")


def generate_dev_keypair(
    private_key_path: Path,
    public_key_path: Path,
    key_size: int = 2048,
) -> None:
    """Generates a fresh RSA keypair and writes PEM files to the given paths."""
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


def sign_digest(digest_hex: str, private_key_path: Path) -> bytes:
    """Signs digest_hex using the RSA private key."""
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
    """Writes raw signature bytes to sig_path atomically."""
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(sig_path, signature)
    logger.info("Signature saved → %s", sig_path.name)


def verify_signature(
    digest_hex: str,
    sig_path: Path,
    public_key_path: Path,
) -> None:
    """Verifies the RSA-PSS signature of digest_hex."""
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


def _load_private_key(path: Path):
    """Loads an RSA private key from a PEM file."""
    return serialization.load_pem_private_key(
        path.read_bytes(),
        password=None,
    )


def _load_public_key(path: Path):
    """Loads an RSA public key from a PEM file."""
    return serialization.load_pem_public_key(path.read_bytes())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Writes data to path via a sibling .tmp file, then renames."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
