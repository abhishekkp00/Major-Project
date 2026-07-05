import tarfile
import pytest
from pathlib import Path

from src.phase4.package_loader import PackageLoader
from src.common.exceptions import IncompletePackageError, InvalidArchiveError


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def valid_package_files(tmp_dir: Path) -> Path:
    pkg = tmp_dir / "valid_pkg"
    pkg.mkdir()
    for fname in ["adapter.enc", "adapter.hash", "adapter.sig", "metadata.json", "package_manifest.json"]:
        (pkg / fname).write_text(f"dummy-{fname}")
    return pkg


@pytest.fixture
def invalid_package_files(tmp_dir: Path) -> Path:
    pkg = tmp_dir / "invalid_pkg"
    pkg.mkdir()
    for fname in ["adapter.enc", "adapter.hash"]:
        (pkg / fname).write_text(f"dummy-{fname}")
    return pkg


def test_package_loader_directory_success(valid_package_files):
    with PackageLoader(valid_package_files) as extracted_path:
        assert extracted_path == valid_package_files
        assert (extracted_path / "adapter.enc").exists()


def test_package_loader_directory_missing_file(invalid_package_files):
    with pytest.raises(IncompletePackageError) as exc_info:
        with PackageLoader(invalid_package_files):
            pass
    assert "Missing files" in str(exc_info.value)


def test_package_loader_tar_gz_success(valid_package_files, tmp_dir):
    archive_path = tmp_dir / "package.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in valid_package_files.iterdir():
            tar.add(p, arcname=p.name)

    with PackageLoader(archive_path) as extracted_path:
        assert extracted_path != archive_path
        assert (extracted_path / "adapter.enc").exists()
        assert (extracted_path / "package_manifest.json").exists()


def test_package_loader_corrupted_archive(tmp_dir):
    corrupt_archive = tmp_dir / "corrupt.tar.gz"
    corrupt_archive.write_bytes(b"not a valid tar.gz file")

    with pytest.raises(InvalidArchiveError):
        with PackageLoader(corrupt_archive):
            pass
