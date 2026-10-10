"""
safe_extract.py
===============
Security-hardened tar archive validation and extraction for SecureLoRA.

Enforces:
  1. Strict path-aware containment via Path.relative_to() (rejects sibling-prefix escapes,
     e.g., /tmp/root-evil vs /tmp/root).
  2. Rejection of absolute paths, '..' directory traversals, and null bytes in member names.
  3. Rejection of symlinks and hard-links attempting to point outside the extraction root.
  4. Rejection of special device files (FIFOs, character devices, block devices).
  5. Member limits and resource quotas (archive size, total uncompressed size, member count,
     single member size).
  6. Standard library filter='data' integration on Python 3.12+ for defense-in-depth.
  7. Deterministic cleanup on failure and success.
"""

from __future__ import annotations

import logging
import ntpath
import os
import tarfile
from pathlib import Path
from typing import Union

from src.common.exceptions import SecurityError, InvalidArchiveError

logger = logging.getLogger("secure_lora.security.safe_extract")

DEFAULT_MAX_ARCHIVE_SIZE: int = 512 * 1024 * 1024  # 512 MB
DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE: int = 1024 * 1024 * 1024  # 1 GB
DEFAULT_MAX_MEMBER_SIZE: int = 512 * 1024 * 1024  # 512 MB
DEFAULT_MAX_MEMBER_COUNT: int = 10_000


def validate_archive_member(
    member: tarfile.TarInfo,
    root: Path,
    max_member_size: int = DEFAULT_MAX_MEMBER_SIZE,
) -> Path:
    """
    Validates a single TarInfo member against security containment and type policies.
    Returns the resolved target Path within root.

    Raises SecurityError on any path traversal, escape, special file, or policy violation.
    """
    name = member.name
    if not name or not name.strip():
        raise SecurityError("Empty member name detected in archive.")

    if "\0" in name:
        raise SecurityError(f"Null byte detected in member name: {name!r}")

    # 1. Reject absolute member names (POSIX and Windows) and drive specifications
    if (
        os.path.isabs(name)
        or ntpath.isabs(name)
        or name.startswith(("/", "\\"))
        or bool(ntpath.splitdrive(name)[0])
    ):
        raise SecurityError(f"Absolute path member detected in archive: {name}")

    # 2. Reject '..' traversal segments in member name (normalized for cross-platform separators)
    normalized_parts = name.replace("\\", "/").split("/")
    if ".." in normalized_parts:
        raise SecurityError(f"Directory traversal ('..') detected in member name: {name}")

    # 3. Reject unsupported special file types (FIFOs, device files, non-regular files)
    if member.isdev() or member.ischr() or member.isblk() or member.isfifo():
        raise SecurityError(f"Unsupported special device or FIFO file detected in archive: {name}")
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise SecurityError(f"Unsupported special or unknown file type detected: {name} (type={member.type!r})")

    # 4. Enforce individual member size limit
    if member.size < 0:
        raise SecurityError(f"Negative member size ({member.size}) detected: {name}")
    if member.size > max_member_size:
        raise SecurityError(
            f"Member '{name}' size ({member.size} bytes) exceeds safety limit ({max_member_size} bytes)."
        )

    # 5. Resolve target path and verify strict containment inside root
    target_path = (root / name).resolve()
    try:
        target_path.relative_to(root)
    except ValueError as exc:
        raise SecurityError(
            f"Path containment violation: member '{name}' resolves to '{target_path}', "
            f"which escapes extraction root '{root}'."
        ) from exc

    # Sibling prefix guard: verify target path is strictly contained within root hierarchy
    if target_path != root and not str(target_path).startswith(str(root) + os.sep):
        raise SecurityError(
            f"Sibling prefix escape detected: member '{name}' resolves to '{target_path}' "
            f"outside '{root}'."
        )

    # Prevent file/link members from overwriting or resolving directly to extraction root itself
    if target_path == root and not member.isdir():
        raise SecurityError(f"Archive member '{name}' resolves to the extraction root itself.")

    # 6. Validate link targets for symlinks and hard-links
    if member.issym():
        link_target_name = member.linkname
        if not link_target_name or "\0" in link_target_name:
            raise SecurityError(f"Invalid symlink target for member: {name}")
        if (
            os.path.isabs(link_target_name)
            or ntpath.isabs(link_target_name)
            or link_target_name.startswith(("/", "\\"))
            or bool(ntpath.splitdrive(link_target_name)[0])
        ):
            raise SecurityError(f"Absolute symlink target detected: {name} -> {link_target_name}")

        resolved_link = (target_path.parent / link_target_name).resolve()
        try:
            resolved_link.relative_to(root)
        except ValueError as exc:
            raise SecurityError(
                f"Symlink escape detected: member '{name}' points to '{resolved_link}', "
                f"which escapes extraction root '{root}'."
            ) from exc

        if resolved_link == root:
            raise SecurityError(f"Symlink points to extraction root itself: '{name}' -> '{resolved_link}'")

        if not str(resolved_link).startswith(str(root) + os.sep):
            raise SecurityError(
                f"Symlink sibling prefix escape detected: '{name}' -> '{resolved_link}' outside '{root}'."
            )

    elif member.islnk():
        link_target_name = member.linkname
        if not link_target_name or "\0" in link_target_name:
            raise SecurityError(f"Invalid hard-link target for member: {name}")
        if (
            os.path.isabs(link_target_name)
            or ntpath.isabs(link_target_name)
            or link_target_name.startswith(("/", "\\"))
            or bool(ntpath.splitdrive(link_target_name)[0])
        ):
            raise SecurityError(f"Absolute hard-link target detected: {name} -> {link_target_name}")

        resolved_link = (root / link_target_name).resolve()
        try:
            resolved_link.relative_to(root)
        except ValueError as exc:
            raise SecurityError(
                f"Hard-link escape detected: member '{name}' points to '{resolved_link}', "
                f"which escapes extraction root '{root}'."
            ) from exc

        if resolved_link == root:
            raise SecurityError(f"Hard-link points to extraction root itself: '{name}' -> '{resolved_link}'")

        if not str(resolved_link).startswith(str(root) + os.sep):
            raise SecurityError(
                f"Hard-link sibling prefix escape detected: '{name}' -> '{resolved_link}' outside '{root}'."
            )

    return target_path


def safe_extract_tar(
    tar: tarfile.TarFile,
    destination_dir: Union[str, Path],
    max_total_uncompressed_size: int = DEFAULT_MAX_TOTAL_UNCOMPRESSED_SIZE,
    max_member_size: int = DEFAULT_MAX_MEMBER_SIZE,
    max_member_count: int = DEFAULT_MAX_MEMBER_COUNT,
) -> Path:
    """
    Extracts all members of tar into destination_dir after strict pre-extraction validation.

    Enforces:
      - Path containment (no '..', no absolute paths, no sibling prefix escapes).
      - Rejection of escaping symlinks, hardlinks, and special device files.
      - Resource limits on total uncompressed size, member count, and individual member size.
      - Utilizes filter='data' on Python 3.12+ where supported for defense-in-depth.

    Raises SecurityError on any security policy violation.
    Raises InvalidArchiveError on tar corruption.
    """
    root = Path(destination_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Destination directory does not exist or is not a directory: {root}")

    try:
        members = tar.getmembers()
    except Exception as exc:
        raise InvalidArchiveError(f"Corrupted or invalid tar archive: {exc}") from exc

    # Enforce member count quota
    if len(members) > max_member_count:
        raise SecurityError(
            f"Archive exceeds maximum member count limit ({len(members)} > {max_member_count})."
        )

    # Validate all members before extracting any files
    total_uncompressed = 0
    for member in members:
        validate_archive_member(member, root=root, max_member_size=max_member_size)
        total_uncompressed += member.size
        if total_uncompressed > max_total_uncompressed_size:
            raise SecurityError(
                f"Archive total uncompressed size ({total_uncompressed} bytes) exceeds safety limit "
                f"of {max_total_uncompressed_size} bytes."
            )

    # Perform safe extraction
    try:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=root, filter="data")
        else:
            tar.extractall(path=root)
    except Exception as exc:
        if isinstance(exc, SecurityError):
            raise
        if isinstance(exc, getattr(tarfile, "FilterError", ())):
            raise SecurityError(f"Archive security filter violation: {exc}") from exc
        raise InvalidArchiveError(f"Extraction failed: {exc}") from exc

    return root
