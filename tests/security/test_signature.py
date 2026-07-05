import pytest
from pathlib import Path

from src.security import (
    generate_dev_keypair,
    sign_digest,
    save_signature,
    verify_signature,
)

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path

@pytest.fixture(autouse=True)
def keypair(tmp_dir):
    self = TestSignature
    self.priv_path = tmp_dir / "dev_private.pem"
    self.pub_path  = tmp_dir / "public.pem"
    generate_dev_keypair(self.priv_path, self.pub_path, key_size=2048)


class TestSignature:
    def test_keypair_files_created(self):
        assert self.priv_path.exists()
        assert self.pub_path.exists()

    def test_private_key_permissions_are_600(self):
        mode = oct(self.priv_path.stat().st_mode)[-3:]
        assert mode == "600", f"Private key permissions should be 600, got {mode}"

    def test_sign_and_verify_succeeds(self):
        digest = "a" * 64
        sig = sign_digest(digest, self.priv_path)
        sig_path = self.priv_path.parent / "adapter.sig"
        save_signature(sig, sig_path)
        verify_signature(digest, sig_path, self.pub_path)  # must not raise

    def test_verify_fails_on_modified_digest(self, tmp_dir):
        digest_original = "a" * 64
        digest_modified = "b" * 64
        sig = sign_digest(digest_original, self.priv_path)
        sig_path = tmp_dir / "adapter.sig"
        save_signature(sig, sig_path)
        with pytest.raises(ValueError, match="Signature verification FAILED"):
            verify_signature(digest_modified, sig_path, self.pub_path)

    def test_verify_fails_with_wrong_public_key(self, tmp_dir):
        # Generate a second keypair
        priv2 = tmp_dir / "other_private.pem"
        pub2  = tmp_dir / "other_public.pem"
        generate_dev_keypair(priv2, pub2, key_size=2048)

        digest = "a" * 64
        sig = sign_digest(digest, self.priv_path)  # sign with key 1
        sig_path = tmp_dir / "adapter.sig"
        save_signature(sig, sig_path)

        with pytest.raises(ValueError, match="Signature verification FAILED"):
            verify_signature(digest, sig_path, pub2)  # verify with key 2
