import hashlib
import pytest
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.phase4.package_validator import validate_package_integrity
from src.security import compute_sha256
from src.common.exceptions import IntegrityValidationError, SignatureValidationError


@pytest.fixture
def keys_and_paths(tmp_path: Path):
    pkg_dir = tmp_path / "test_pkg"
    pkg_dir.mkdir()

    # Generate RSA keypair
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    priv_key_path = tmp_path / "private.pem"
    pub_key_path = pkg_dir / "public.pem"
    priv_key_path.write_bytes(private_pem)
    pub_key_path.write_bytes(public_pem)

    # Generate dummy adapter.enc
    enc_path = pkg_dir / "adapter.enc"
    enc_content = b"encrypted-lora-adapter-payload" * 10
    enc_path.write_bytes(enc_content)

    # Compute hash
    computed_hash = hashlib.sha256(enc_content).hexdigest()
    hash_path = pkg_dir / "adapter.hash"
    hash_path.write_text(computed_hash)

    # Sign the hash
    signature = private_key.sign(
        computed_hash.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    sig_path = pkg_dir / "adapter.sig"
    sig_path.write_bytes(signature)

    return {
        "pkg_dir": pkg_dir,
        "priv_key_path": priv_key_path,
        "pub_key_path": pub_key_path,
        "enc_path": enc_path,
        "hash_path": hash_path,
        "sig_path": sig_path,
        "private_key": private_key
    }


def test_validator_success(keys_and_paths):
    pkg_dir = keys_and_paths["pkg_dir"]
    verified_hash = validate_package_integrity(pkg_dir)
    assert verified_hash == compute_sha256(keys_and_paths["enc_path"])


def test_validator_tampered_ciphertext(keys_and_paths):
    pkg_dir = keys_and_paths["pkg_dir"]
    enc_path = keys_and_paths["enc_path"]

    # Tamper with one byte of the encrypted file
    data = bytearray(enc_path.read_bytes())
    data[0] ^= 0xFF
    enc_path.write_bytes(bytes(data))

    with pytest.raises(IntegrityValidationError):
        validate_package_integrity(pkg_dir)


def test_validator_wrong_signature(keys_and_paths):
    pkg_dir = keys_and_paths["pkg_dir"]
    sig_path = keys_and_paths["sig_path"]

    # Modify the signature file (wrong signature)
    sig_path.write_bytes(b"invalid-signature-bytes" * 10)

    with pytest.raises(SignatureValidationError):
        validate_package_integrity(pkg_dir)


def test_validator_wrong_public_key(keys_and_paths, tmp_path):
    pkg_dir = keys_and_paths["pkg_dir"]

    # Generate a different keypair and overwrite public.pem
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_public_pem = other_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    (pkg_dir / "public.pem").write_bytes(other_public_pem)

    with pytest.raises(SignatureValidationError):
        validate_package_integrity(pkg_dir)
