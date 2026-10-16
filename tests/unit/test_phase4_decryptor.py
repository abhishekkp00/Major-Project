import io
import os
import tarfile
import pytest
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.phase4.decryptor import DecryptedAdapterContext
from src.common.exceptions import SecurityError


@pytest.fixture
def keys_and_payloads(tmp_path: Path):
    key = AESGCM.generate_key(bit_length=256)

    # Create fake adapter directory
    adapter_src = tmp_path / "adapter_src"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text('{"r": 8, "peft_type": "LORA"}')
    (adapter_src / "adapter_model.safetensors").write_bytes(b"model-weights-bytes")

    # Tar it
    tar_path = tmp_path / "adapter.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in adapter_src.iterdir():
            tar.add(p, arcname=p.name)

    # Encrypt
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, tar_path.read_bytes(), associated_data=None)

    enc_path = tmp_path / "adapter.enc"
    enc_path.write_bytes(nonce + ciphertext)

    # Clean up intermediate tar
    tar_path.unlink()

    return {
        "key": key,
        "enc_path": enc_path,
        "config_content": '{"r": 8, "peft_type": "LORA"}',
        "weights_content": b"model-weights-bytes"
    }


def test_decryption_success_and_cleanup(keys_and_payloads):
    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]

    temp_adapter_dir = None
    with DecryptedAdapterContext(enc_path, key) as decrypted_dir:
        temp_adapter_dir = decrypted_dir
        assert decrypted_dir.exists()
        assert (decrypted_dir / "adapter_config.json").exists()
        assert (decrypted_dir / "adapter_model.safetensors").exists()
        assert (decrypted_dir / "adapter_config.json").read_text() == keys_and_payloads["config_content"]
        assert (decrypted_dir / "adapter_model.safetensors").read_bytes() == keys_and_payloads["weights_content"]

    # Outside the context block, files must be shredded and directory deleted
    assert temp_adapter_dir is not None
    assert not temp_adapter_dir.exists()


def test_decryption_wrong_key_fails(keys_and_payloads):
    enc_path = keys_and_payloads["enc_path"]
    wrong_key = os.urandom(32)

    with pytest.raises(ValueError, match="Decryption or extraction failed"):
        with DecryptedAdapterContext(enc_path, wrong_key):
            pass


def _encrypt_tar_payload(tar_bytes: bytes, key: bytes, enc_path: Path):
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, tar_bytes, associated_data=None)
    enc_path.write_bytes(nonce + ciphertext)


def test_decryptor_rejects_oversized_archive_and_cleans_up(tmp_path: Path):
    key = AESGCM.generate_key(bit_length=256)
    enc_path = tmp_path / "oversized.enc"

    # Tarball containing adapter_config.json
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        ti = tarfile.TarInfo(name="adapter_config.json")
        content = b'{"r": 8}' * 20
        ti.size = len(content)
        tar.addfile(ti, io.BytesIO(content))

    tar_bytes = tar_buf.getvalue()
    _encrypt_tar_payload(tar_bytes, key, enc_path)

    # Set max_archive_size below the tar size
    context = DecryptedAdapterContext(enc_path, key, max_archive_size=len(tar_bytes) - 1)
    with pytest.raises(SecurityError, match="exceeds limit"):
        with context:
            pass

    assert context.temp_dir is None


def test_decryptor_rejects_traversal_and_cleans_up(tmp_path: Path):
    key = AESGCM.generate_key(bit_length=256)
    enc_path = tmp_path / "traversal.enc"

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        ti1 = tarfile.TarInfo(name="adapter_config.json")
        ti1.size = 8
        tar.addfile(ti1, io.BytesIO(b'{"r": 8}'))

        ti2 = tarfile.TarInfo(name="../escape.txt")
        ti2.size = 5
        tar.addfile(ti2, io.BytesIO(b'evil!'))

    _encrypt_tar_payload(tar_buf.getvalue(), key, enc_path)

    context = DecryptedAdapterContext(enc_path, key)
    with pytest.raises(SecurityError, match="Directory traversal"):
        with context:
            pass

    assert context.temp_dir is None
    assert not (tmp_path / "escape.txt").exists()


def test_decryptor_rejects_symlink_escape_and_cleans_up(tmp_path: Path):
    key = AESGCM.generate_key(bit_length=256)
    enc_path = tmp_path / "symlink.enc"

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        ti1 = tarfile.TarInfo(name="adapter_config.json")
        ti1.size = 8
        tar.addfile(ti1, io.BytesIO(b'{"r": 8}'))

        ti2 = tarfile.TarInfo(name="evil_symlink")
        ti2.type = tarfile.SYMTYPE
        ti2.linkname = "/etc/passwd"
        tar.addfile(ti2)

    _encrypt_tar_payload(tar_buf.getvalue(), key, enc_path)

    context = DecryptedAdapterContext(enc_path, key)
    with pytest.raises(SecurityError, match="Absolute symlink target"):
        with context:
            pass

    assert context.temp_dir is None


def test_decryptor_rejects_special_file_and_cleans_up(tmp_path: Path):
    key = AESGCM.generate_key(bit_length=256)
    enc_path = tmp_path / "special.enc"

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        ti1 = tarfile.TarInfo(name="adapter_config.json")
        ti1.size = 8
        tar.addfile(ti1, io.BytesIO(b'{"r": 8}'))

        ti2 = tarfile.TarInfo(name="evil_fifo")
        ti2.type = tarfile.FIFOTYPE
        tar.addfile(ti2)

    _encrypt_tar_payload(tar_buf.getvalue(), key, enc_path)

    context = DecryptedAdapterContext(enc_path, key)
    with pytest.raises(SecurityError, match="Unsupported special"):
        with context:
            pass

    assert context.temp_dir is None


def test_decryptor_permissions_restrictive_on_posix(keys_and_payloads):
    """Verifies that temporary directory is 0700 and extracted files are 0600 on POSIX."""
    if os.name != "posix":
        pytest.skip("POSIX permissions test only applies to POSIX platforms")

    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]

    with DecryptedAdapterContext(enc_path, key) as decrypted_dir:
        dir_mode = decrypted_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700, f"Expected 0700 on decrypted adapter dir, got {oct(dir_mode)}"

        config_file = decrypted_dir / "adapter_config.json"
        assert config_file.exists()
        file_mode = config_file.stat().st_mode & 0o777
        assert file_mode == 0o600, f"Expected 0600 on extracted file, got {oct(file_mode)}"

    assert not decrypted_dir.exists()


def test_decryptor_cleans_up_on_model_load_exception(keys_and_payloads):
    """Cleanup executes when model loading raises an exception inside the context."""
    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]
    recorded_dir = None

    with pytest.raises(RuntimeError, match="Model loading crashed"):
        with DecryptedAdapterContext(enc_path, key) as decrypted_dir:
            recorded_dir = decrypted_dir
            assert decrypted_dir.exists()
            raise RuntimeError("Model loading crashed")

    assert recorded_dir is not None
    assert not recorded_dir.exists()


def test_decryptor_cleans_up_on_inference_exception(keys_and_payloads):
    """Cleanup executes when an inference execution error occurs inside the context."""
    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]
    recorded_dir = None

    with pytest.raises(ValueError, match="Inference execution aborted"):
        with DecryptedAdapterContext(enc_path, key) as decrypted_dir:
            recorded_dir = decrypted_dir
            assert decrypted_dir.exists()
            raise ValueError("Inference execution aborted")

    assert recorded_dir is not None
    assert not recorded_dir.exists()


def test_decryptor_configurable_temp_storage(keys_and_payloads, tmp_path):
    """DecryptedAdapterContext respects configurable temp_storage_dir / SECURELORA_TEMP_DIR."""
    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]

    custom_storage = tmp_path / "custom_ramdisk"
    custom_storage.mkdir()
    if os.name == "posix":
        custom_storage.chmod(0o700)

    recorded_dir = None
    with DecryptedAdapterContext(enc_path, key, temp_storage_dir=custom_storage) as decrypted_dir:
        recorded_dir = decrypted_dir
        assert custom_storage in decrypted_dir.parents
        assert decrypted_dir.exists()

    assert recorded_dir is not None
    assert not recorded_dir.exists()


def test_decryptor_rejects_insecure_temp_storage_dir(keys_and_payloads, tmp_path):
    """DecryptedAdapterContext rejects world-writable temp storage directories without sticky bit."""
    if os.name != "posix":
        pytest.skip("POSIX permissions test")

    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]

    insecure_dir = tmp_path / "insecure_world_writable"
    insecure_dir.mkdir()
    insecure_dir.chmod(0o777)  # World-writable without sticky bit (0o1777)

    with pytest.raises(SecurityError, match="world-writable without sticky bit"):
        with DecryptedAdapterContext(enc_path, key, temp_storage_dir=insecure_dir):
            pass


def test_decryptor_cleanup_preserves_primary_exception(keys_and_payloads, tmp_path, monkeypatch):
    """Primary exception in with-block is NOT masked by cleanup failures."""
    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]

    mock_storage = tmp_path / "failing_cleanup_dir"
    mock_storage.mkdir()
    if os.name == "posix":
        mock_storage.chmod(0o700)

    # Mock shred_directory to simulate a cleanup failure
    def mock_failing_shred_dir(path):
        raise OSError("Simulated cleanup disk failure")

    monkeypatch.setattr("src.phase4.decryptor.shred_directory", mock_failing_shred_dir)

    with pytest.raises(RuntimeError, match="Primary model failure"):
        with DecryptedAdapterContext(enc_path, key, temp_storage_dir=mock_storage) as decrypted_dir:
            raise RuntimeError("Primary model failure")


def test_decryptor_surfaces_cleanup_failure_on_clean_exit(keys_and_payloads, tmp_path, monkeypatch):
    """Cleanup failure is surfaced when with-block finishes normally without exception."""
    enc_path = keys_and_payloads["enc_path"]
    key = keys_and_payloads["key"]

    mock_storage = tmp_path / "failing_clean_dir"
    mock_storage.mkdir()
    if os.name == "posix":
        mock_storage.chmod(0o700)

    # Mock shred_directory to simulate a failure and leave directory existing
    def mock_failing_shred_dir(path):
        raise OSError("Disk hardware failure during shred")

    monkeypatch.setattr("src.phase4.decryptor.shred_directory", mock_failing_shred_dir)

    with pytest.raises(SecurityError, match="Decrypted plaintext cleanup failed"):
        with DecryptedAdapterContext(enc_path, key, temp_storage_dir=mock_storage):
            pass

