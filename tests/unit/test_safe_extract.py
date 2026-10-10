"""
test_safe_extract.py
====================
Comprehensive tests for SecureLoRA hardened tar archive validation and extraction.

Tests cover:
  - Normal valid archive extraction.
  - Directory traversal ('..') and absolute paths rejection.
  - Sibling-prefix paths rejection (/tmp/root-evil vs /tmp/root).
  - Symlink and hard-link escape rejection.
  - Special files (FIFOs, device files) rejection.
  - Resource quota violations (archive size, total uncompressed size, member count, member size).
  - Pre-extraction member validation (no extraction before validation passes).
"""

import io
import os
import tarfile
from pathlib import Path

import pytest

from src.common.exceptions import SecurityError, InvalidArchiveError
from src.security.safe_extract import (
    safe_extract_tar,
    validate_archive_member,
    DEFAULT_MAX_ARCHIVE_SIZE,
    DEFAULT_MAX_MEMBER_SIZE,
    DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE,
    DEFAULT_MAX_MEMBER_COUNT,
)


def _create_tar_with_members(tar_path: Path, members_data: list[tuple[str, bytes, dict | None]]):
    """
    Helper to create tar archives with custom member properties (e.g. types, linknames).
    members_data is a list of (name, content_bytes, optional_attrs_dict)
    """
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in members_data:
            name, content, attrs = item
            ti = tarfile.TarInfo(name=name)
            ti.size = len(content)
            ti.mtime = 1700000000
            if attrs:
                for k, v in attrs.items():
                    setattr(ti, k, v)
            if ti.isreg():
                tar.addfile(ti, io.BytesIO(content))
            else:
                tar.addfile(ti)


def test_safe_extract_valid_archive(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "valid.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        f1 = tmp_path / "file1.txt"
        f1.write_text("hello file 1")
        tar.add(f1, arcname="file1.txt")

        sub = tmp_path / "subdir"
        sub.mkdir()
        f2 = sub / "file2.txt"
        f2.write_text("hello file 2")
        tar.add(f2, arcname="subdir/file2.txt")

    extracted_root = safe_extract_tar(tarfile.open(archive_path, "r:gz"), dest)
    assert extracted_root == dest
    assert (dest / "file1.txt").read_text() == "hello file 1"
    assert (dest / "subdir" / "file2.txt").read_text() == "hello file 2"


def test_safe_extract_valid_relative_symlink(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "symlink_valid.tar.gz"

    # Symlink within dest root: link.txt -> target.txt
    _create_tar_with_members(
        archive_path,
        [
            ("target.txt", b"target content", {"type": tarfile.REGTYPE}),
            ("link.txt", b"", {"type": tarfile.SYMTYPE, "linkname": "target.txt"}),
        ],
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract_tar(tar, dest)

    assert (dest / "target.txt").read_text() == "target content"
    assert (dest / "link.txt").is_symlink()


@pytest.mark.parametrize(
    "bad_name",
    [
        "../evil.txt",
        "../../etc/passwd",
        "sub/../../evil.txt",
        "sub/../evil.txt",
        "foo/..\\bar.txt",
        "..",
    ],
)
def test_safe_extract_rejects_dot_dot_traversal(tmp_path: Path, bad_name: str):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "traversal.tar.gz"

    _create_tar_with_members(archive_path, [(bad_name, b"pwned", {"type": tarfile.REGTYPE})])

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError, match="Directory traversal"):
            safe_extract_tar(tar, dest)

    # Ensure evil file was never created
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize(
    "bad_name",
    [
        "/tmp/evil.txt",
        "/etc/shadow",
        "\\Windows\\System32\\cmd.exe",
        "C:evil.txt",
        "D:\\data.txt",
    ],
)
def test_safe_extract_rejects_absolute_and_drive_paths(tmp_path: Path, bad_name: str):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "absolute.tar.gz"

    _create_tar_with_members(archive_path, [(bad_name, b"pwned", {"type": tarfile.REGTYPE})])

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError, match="Absolute path member"):
            safe_extract_tar(tar, dest)


def test_safe_extract_rejects_sibling_prefix(tmp_path: Path):
    dest = tmp_path / "root"
    dest.mkdir()
    sibling_evil = tmp_path / "root-evil"

    # Craft TarInfo directly that resolves to sibling prefix /root-evil/attack.txt
    ti = tarfile.TarInfo(name=f"../root-evil/attack.txt")
    ti.size = 10
    ti.type = tarfile.REGTYPE

    # validate_archive_member must reject this via relative_to containment check
    with pytest.raises(SecurityError) as exc_info:
        validate_archive_member(ti, root=dest)

    assert "Directory traversal" in str(exc_info.value) or "escapes" in str(exc_info.value) or "Sibling prefix" in str(exc_info.value)


def test_sibling_prefix_relative_to_guard(tmp_path: Path):
    """
    Demonstrate that string.startswith() falsely accepts sibling prefix,
    while validate_archive_member and Path.relative_to() strictly reject it.
    """
    root = (tmp_path / "sandbox").resolve()
    root.mkdir()
    sibling = (tmp_path / "sandbox-evil" / "payload.txt").resolve()

    # Vulnerability demonstration: string prefix falsely succeeds
    assert str(sibling).startswith(str(root))

    # Security fix: relative_to strictly fails
    with pytest.raises(ValueError):
        sibling.relative_to(root)

    # Security engine: validate_archive_member strictly rejects
    ti = tarfile.TarInfo(name="../sandbox-evil/payload.txt")
    ti.size = 5
    with pytest.raises(SecurityError):
        validate_archive_member(ti, root=root)


@pytest.mark.parametrize(
    "bad_link",
    [
        "/etc/passwd",
        "../../../../etc/passwd",
        "../dest-evil/file.txt",
        ".",
        "..",
    ],
)
def test_safe_extract_rejects_symlink_escapes(tmp_path: Path, bad_link: str):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "symlink_escape.tar.gz"

    _create_tar_with_members(
        archive_path,
        [("link_to_escape", b"", {"type": tarfile.SYMTYPE, "linkname": bad_link})],
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError):
            safe_extract_tar(tar, dest)


@pytest.mark.parametrize(
    "bad_link",
    [
        "/etc/passwd",
        "../../../../etc/passwd",
        "../dest-evil/file.txt",
        ".",
    ],
)
def test_safe_extract_rejects_hardlink_escapes(tmp_path: Path, bad_link: str):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "hardlink_escape.tar.gz"

    _create_tar_with_members(
        archive_path,
        [("link_to_escape", b"", {"type": tarfile.LNKTYPE, "linkname": bad_link})],
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError):
            safe_extract_tar(tar, dest)


@pytest.mark.parametrize(
    "special_type",
    [
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
    ],
)
def test_safe_extract_rejects_special_files(tmp_path: Path, special_type: bytes):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "special.tar.gz"

    _create_tar_with_members(
        archive_path,
        [("special_dev", b"", {"type": special_type})],
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError, match="Unsupported special"):
            safe_extract_tar(tar, dest)


def test_safe_extract_rejects_null_byte_in_name(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    ti = tarfile.TarInfo(name="file\0evil.txt")
    ti.size = 10
    with pytest.raises(SecurityError, match="Null byte"):
        validate_archive_member(ti, root=dest)


def test_safe_extract_rejects_empty_name(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    ti = tarfile.TarInfo(name="   ")
    ti.size = 10
    with pytest.raises(SecurityError, match="Empty member name"):
        validate_archive_member(ti, root=dest)


def test_safe_extract_rejects_oversized_member(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "large_member.tar.gz"

    _create_tar_with_members(
        archive_path,
        [("large_file.bin", b"x" * 100, {"type": tarfile.REGTYPE})],
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError, match="exceeds safety limit"):
            safe_extract_tar(tar, dest, max_member_size=50)


def test_safe_extract_rejects_total_uncompressed_limit(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "total_large.tar.gz"

    _create_tar_with_members(
        archive_path,
        [
            ("f1.bin", b"x" * 60, {"type": tarfile.REGTYPE}),
            ("f2.bin", b"y" * 60, {"type": tarfile.REGTYPE}),
        ],
    )

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError, match="total uncompressed size"):
            safe_extract_tar(tar, dest, max_total_uncompressed_size=100)


def test_safe_extract_rejects_member_count_limit(tmp_path: Path):
    dest = tmp_path / "dest"
    dest.mkdir()
    archive_path = tmp_path / "many_members.tar.gz"

    members = [(f"f_{i}.txt", b"a", {"type": tarfile.REGTYPE}) for i in range(15)]
    _create_tar_with_members(archive_path, members)

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(SecurityError, match="maximum member count"):
            safe_extract_tar(tar, dest, max_member_count=10)
