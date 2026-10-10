import hashlib
import json
import pytest
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.phase4.package_validator import (
    validate_package_integrity,
    validate_package_provenance,
    SUPPORTED_CURRENT_VERSIONS,
    SUPPORTED_LEGACY_VERSIONS,
)
from src.security import compute_sha256, compute_canonical_manifest_digest
from src.common.exceptions import (
    IntegrityValidationError,
    SignatureValidationError,
    ManifestSchemaError,
)


@pytest.fixture
def keys_and_paths(tmp_path: Path):
    """Legacy un-manifested package fixture."""
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

    # Sign the raw ciphertext hash (legacy behavior)
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
        "private_key": private_key,
        "computed_hash": computed_hash,
    }


@pytest.fixture
def current_package_bundle(tmp_path: Path):
    """Fixture generating a valid current-format (canonical manifest signed) package."""
    pkg_dir = tmp_path / "current_pkg"
    pkg_dir.mkdir()

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

    priv_key_path = tmp_path / "private_curr.pem"
    pub_key_path = pkg_dir / "public.pem"
    priv_key_path.write_bytes(private_pem)
    pub_key_path.write_bytes(public_pem)

    enc_path = pkg_dir / "adapter.enc"
    enc_content = b"encrypted-lora-current-payload-bytes" * 8
    enc_path.write_bytes(enc_content)

    actual_hash = hashlib.sha256(enc_content).hexdigest()
    hash_path = pkg_dir / "adapter.hash"
    hash_path.write_text(actual_hash)

    manifest = {
        "schema_version": "1.0.0",
        "package_id": "11112222-3333-4444-5555-666677778888",
        "adapter_id": "medical-lora-v1",
        "base_model_id": "distilbert-base-uncased",
        "model_revision": "main",
        "adapter_revision": "v1.0.0",
        "package_version": "1.0.0",
        "creation_timestamp": "2026-08-16T12:00:00+00:00",
        "expiration_timestamp": "2030-01-01T00:00:00+00:00",
        "binding_policy_version": "1.0.0",
        "kdf_version": "hkdf-sha256-v1",
        "encryption_version": "aes-256-gcm-v1",
        "signature_algorithm": "rsa-pss-2048-sha256",
        "digest_algorithm": "sha256",
        "nonce_metadata": {"iv_bytes": 12, "tag_bytes": 16},
        "deployment_policy": {"strictness": "high"},
        "sequence_number": 42,
        "device_fingerprint_hash_ref": "3926c635fa8a12607cf843d884442ae151b5253f54529dc053cd6f0cebddfb93",
        "encrypted_adapter_digest": actual_hash,
    }
    manifest_path = pkg_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Compute canonical digest and sign it
    canonical_digest = compute_canonical_manifest_digest(manifest, actual_hash)
    signature = private_key.sign(
        canonical_digest.encode("utf-8"),
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
        "manifest": manifest,
        "manifest_path": manifest_path,
        "actual_hash": actual_hash,
        "canonical_digest": canonical_digest,
        "priv_key_path": priv_key_path,
        "pub_key_path": pub_key_path,
        "private_key": private_key,
        "enc_path": enc_path,
        "sig_path": sig_path,
        "hash_path": hash_path,
    }


# =============================================================================
# Legacy Validator Tests
# =============================================================================

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


# =============================================================================
# Current-Format Package Cryptographic Verification Tests
# =============================================================================

def test_valid_current_format_package_passes(current_package_bundle):
    pkg_dir = current_package_bundle["pkg_dir"]
    manifest, actual_hash = validate_package_provenance(pkg_dir)
    assert manifest["package_id"] == "11112222-3333-4444-5555-666677778888"
    assert actual_hash == current_package_bundle["actual_hash"]
    assert validate_package_integrity(pkg_dir) == actual_hash


@pytest.mark.parametrize("field,tampered_val", [
    ("base_model_id", "malicious-injected-base-model"),
    ("adapter_id", "unauthorized-adapter-id"),
    ("expiration_timestamp", "2099-12-31T23:59:59+00:00"),
    ("sequence_number", 99999),
    ("deployment_policy", {"strictness": "permissive", "bypass": True}),
    ("package_id", "99999999-9999-9999-9999-999999999999"),
    ("model_revision", "compromised_rev"),
    ("adapter_revision", "tampered_adapter_rev"),
    ("device_fingerprint_hash_ref", "00" * 32),
])
def test_modified_security_field_fails_verification(current_package_bundle, field, tampered_val):
    """Any modification to a canonical security-critical field must cause signature validation to fail."""
    pkg_dir = current_package_bundle["pkg_dir"]
    manifest_path = current_package_bundle["manifest_path"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = tampered_val
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(SignatureValidationError):
        validate_package_provenance(pkg_dir)


def test_modified_ciphertext_fails_integrity(current_package_bundle):
    """Modified ciphertext must fail integrity verification."""
    pkg_dir = current_package_bundle["pkg_dir"]
    enc_path = current_package_bundle["enc_path"]

    data = bytearray(enc_path.read_bytes())
    data[0] ^= 0x55
    enc_path.write_bytes(bytes(data))

    with pytest.raises(IntegrityValidationError):
        validate_package_provenance(pkg_dir)


def test_raw_ciphertext_signature_cannot_authenticate_current_manifest(current_package_bundle):
    """
    CRITICAL SECURITY TEST (Defect Remediation):
    A signature generated over the raw ciphertext digest MUST NOT be accepted
    for a current-format manifest. The validator must NOT fall back to raw digest
    verification when canonical manifest verification fails.
    """
    pkg_dir = current_package_bundle["pkg_dir"]
    private_key = current_package_bundle["private_key"]
    actual_hash = current_package_bundle["actual_hash"]
    sig_path = current_package_bundle["sig_path"]

    # Sign the raw ciphertext digest directly (the old/flawed fallback signature)
    raw_signature = private_key.sign(
        actual_hash.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    sig_path.write_bytes(raw_signature)

    # Must be strictly rejected with SignatureValidationError, not accepted via fallback
    with pytest.raises(SignatureValidationError) as exc_info:
        validate_package_provenance(pkg_dir)
    assert "Cryptographic signature check failed" in str(exc_info.value) or "Signature verification FAILED" in str(exc_info.value)


def test_legacy_package_explicit_version_handling(current_package_bundle):
    """
    Legacy packages pass only through their explicitly supported legacy version.
    """
    pkg_dir = current_package_bundle["pkg_dir"]
    private_key = current_package_bundle["private_key"]
    actual_hash = current_package_bundle["actual_hash"]
    sig_path = current_package_bundle["sig_path"]
    manifest_path = current_package_bundle["manifest_path"]

    # Generate a raw ciphertext signature
    raw_signature = private_key.sign(
        actual_hash.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    sig_path.write_bytes(raw_signature)

    # 1. Manifest with explicitly supported legacy version ("0.1.0") -> passes legacy path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = "0.1.0"
    manifest["schema_version"] = "0.1.0"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    m, h = validate_package_provenance(pkg_dir, allow_legacy=True)
    assert h == actual_hash
    assert m["package_version"] == "0.1.0"

    # 2. When allow_legacy=False, legacy package must be rejected
    with pytest.raises(ManifestSchemaError):
        validate_package_provenance(pkg_dir, allow_legacy=False)

    # 3. Manifest with an unsupported legacy/unknown version -> rejected
    manifest["package_version"] = "0.9.9-unsupported"
    manifest["schema_version"] = "0.9.9-unsupported"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ManifestSchemaError) as exc_err:
        validate_package_provenance(pkg_dir, allow_legacy=True)
    assert "Unsupported package version" in str(exc_err.value)


def test_inconsistent_manifest_digest_rejected(current_package_bundle):
    """
    If manifest['encrypted_adapter_digest'] does not match actual ciphertext digest,
    the package must be rejected with IntegrityValidationError.
    """
    pkg_dir = current_package_bundle["pkg_dir"]
    manifest_path = current_package_bundle["manifest_path"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Inconsistent digest
    manifest["encrypted_adapter_digest"] = "00" * 32
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(IntegrityValidationError):
        validate_package_provenance(pkg_dir)


def test_missing_public_key_fails_closed(current_package_bundle):
    """Missing public key must fail closed with SignatureValidationError."""
    pkg_dir = current_package_bundle["pkg_dir"]
    pub_key_path = current_package_bundle["pub_key_path"]
    pub_key_path.unlink()

    with pytest.raises(SignatureValidationError) as exc_info:
        validate_package_provenance(pkg_dir)
    assert "Public key not found" in str(exc_info.value)


def test_missing_signature_fails_closed(current_package_bundle):
    """Missing signature file must fail closed with SignatureValidationError."""
    pkg_dir = current_package_bundle["pkg_dir"]
    sig_path = current_package_bundle["sig_path"]
    sig_path.unlink()

    with pytest.raises(SignatureValidationError) as exc_info:
        validate_package_provenance(pkg_dir)
    assert "Signature file not found" in str(exc_info.value)


def test_corrupted_public_key_fails_closed(current_package_bundle):
    """Corrupted/malformed public key must fail closed with SignatureValidationError."""
    pkg_dir = current_package_bundle["pkg_dir"]
    pub_key_path = current_package_bundle["pub_key_path"]
    pub_key_path.write_bytes(b"-----BEGIN PUBLIC KEY-----\ncorrupted-junk\n-----END PUBLIC KEY-----\n")

    with pytest.raises(SignatureValidationError):
        validate_package_provenance(pkg_dir)


def test_malformed_manifest_json_fails_closed(current_package_bundle):
    """Malformed JSON in manifest must raise ManifestSchemaError."""
    pkg_dir = current_package_bundle["pkg_dir"]
    manifest_path = current_package_bundle["manifest_path"]
    manifest_path.write_text("{malformed: json: invalid", encoding="utf-8")

    with pytest.raises(ManifestSchemaError):
        validate_package_provenance(pkg_dir)


def test_manifest_missing_required_field_fails_closed(current_package_bundle):
    """Missing required security field in manifest must raise ManifestSchemaError."""
    pkg_dir = current_package_bundle["pkg_dir"]
    manifest_path = current_package_bundle["manifest_path"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["package_id"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ManifestSchemaError):
        validate_package_provenance(pkg_dir)


def test_manifest_unsupported_signature_algorithm_fails_closed(current_package_bundle):
    """Unsupported signature algorithm must raise ManifestSchemaError."""
    pkg_dir = current_package_bundle["pkg_dir"]
    manifest_path = current_package_bundle["manifest_path"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["signature_algorithm"] = "rsa-pkcs1v15-unsupported"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ManifestSchemaError):
        validate_package_provenance(pkg_dir)
