import pytest
from pathlib import Path

from src.security import shred_file, shred_directory

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestShredding:
    def test_shred_file_removes_file(self, tmp_dir):
        f = tmp_dir / "secret.txt"
        f.write_text("my super secret payload", encoding="utf-8")
        assert f.exists()
        shred_file(f)
        assert not f.exists()

    def test_shred_directory_removes_all(self, tmp_dir):
        d = tmp_dir / "secrets_dir"
        d.mkdir()
        f1 = d / "secret1.txt"
        f1.write_text("data 1", encoding="utf-8")
        f2 = d / "secret2.txt"
        f2.write_text("data 2", encoding="utf-8")

        assert d.exists()
        shred_directory(d)
        assert not d.exists()
