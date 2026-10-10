import tarfile
import pytest
from pathlib import Path

from src.phase4.package_loader import PackageLoader
from src.common.exceptions import (
    IncompletePackageError,
    InvalidArchiveError,
    SecurityError,
)


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

    loader = PackageLoader(archive_path)
    with loader as extracted_path:
        assert extracted_path != archive_path
        assert (extracted_path / "adapter.enc").exists()
        assert (extracted_path / "package_manifest.json").exists()
        temp_dir_path = Path(loader.temp_dir.name) if loader.temp_dir else None
        assert temp_dir_path is not None
        assert temp_dir_path.exists()

    # Verify deterministic cleanup on success
    assert loader.temp_dir is None
    assert not temp_dir_path.exists()


def test_package_loader_corrupted_archive(tmp_dir):
    corrupt_archive = tmp_dir / "corrupt.tar.gz"
    corrupt_archive.write_bytes(b"not a valid tar.gz file")

    with pytest.raises(InvalidArchiveError):
        with PackageLoader(corrupt_archive):
            pass


def test_package_loader_rejects_oversized_archive(valid_package_files, tmp_dir):
    archive_path = tmp_dir / "oversized.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in valid_package_files.iterdir():
            tar.add(p, arcname=p.name)

    file_size = archive_path.stat().st_size
    # Set limit below actual archive size
    loader = PackageLoader(archive_path, max_bytes=file_size - 1)
    with pytest.raises(SecurityError, match="exceeds safety limit"):
        with loader:
            pass

    assert loader.temp_dir is None


def test_package_loader_rejects_traversal_and_cleans_up(valid_package_files, tmp_dir):
    archive_path = tmp_dir / "traversal_pkg.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in valid_package_files.iterdir():
            tar.add(p, arcname=p.name)
        # Add evil traversal member
        ti = tarfile.TarInfo(name="../evil.txt")
        ti.size = 5
        import io
        tar.addfile(ti, io.BytesIO(b"evil!"))

    loader = PackageLoader(archive_path)
    with pytest.raises(SecurityError, match="Directory traversal"):
        with loader:
            pass

    # Verify cleanup on failure
    assert loader.temp_dir is None
    assert not (tmp_dir / "evil.txt").exists()


def test_package_loader_rejects_absolute_path_and_cleans_up(valid_package_files, tmp_dir):
    archive_path = tmp_dir / "absolute_pkg.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in valid_package_files.iterdir():
            tar.add(p, arcname=p.name)
        ti = tarfile.TarInfo(name="/tmp/evil_pkg.txt")
        ti.size = 5
        import io
        tar.addfile(ti, io.BytesIO(b"evil!"))

    loader = PackageLoader(archive_path)
    with pytest.raises(SecurityError, match="Absolute path member"):
        with loader:
            pass

    assert loader.temp_dir is None


def test_package_loader_rejects_symlink_escape_and_cleans_up(valid_package_files, tmp_dir):
    archive_path = tmp_dir / "symlink_pkg.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in valid_package_files.iterdir():
            tar.add(p, arcname=p.name)
        ti = tarfile.TarInfo(name="evil_link")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tar.addfile(ti)

    loader = PackageLoader(archive_path)
    with pytest.raises(SecurityError, match="Absolute symlink target"):
        with loader:
            pass

    assert loader.temp_dir is None
