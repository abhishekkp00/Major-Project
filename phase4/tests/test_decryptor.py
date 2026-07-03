import os
import tarfile
import pytest
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from phase4.decryptor import DecryptedAdapterContext, secure_shred_file

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
