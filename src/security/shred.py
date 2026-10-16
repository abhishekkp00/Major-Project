import os
import stat
import shutil
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger("secure_lora.security.shred")


def shred_file(file_path: Union[str, Path], passes: int = 3) -> None:
    """
    Overwrites a file with random bytes before unlinking.

    THREAT MODEL & LIMITATION NOTICE:
    ---------------------------------
    This function provides a best-effort defense-in-depth mitigation against
    casual block inspection on traditional magnetic block media.

    LIMITATIONS (NO PHYSICAL ERASURE GUARANTEE):
    * Flash memory / Solid State Drives (SSDs): Flash Translation Layers (FTL)
      perform out-of-place writes and wear-leveling. Overwriting a file in-place
      allocates new flash blocks rather than physically overwriting existing cells.
    * Copy-on-Write (CoW) filesystems: Systems like Btrfs, ZFS, APFS, and bcachefs
      allocate new disk blocks for writes, preserving original data blocks.
    * Filesystem journaling & snapshots: Filesystem journals (ext4, XFS) and
      volume snapshots retain historic block data.
    * Swap memory: Any memory paged to unencrypted OS swap partitions can persist.

    For deployments requiring strict zero-persistence guarantees, administrators
    must utilize volatile RAM-backed storage (tmpfs / /dev/shm) via SECURELORA_TEMP_DIR
    and ensure swap is disabled or encrypted (dm-crypt).
    """
    path = Path(file_path)
    if not path.exists():
        return

    try:
        if path.is_file():
            # Ensure file is writable on POSIX if it was set to read-only
            if os.name == "posix":
                try:
                    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass

            file_size = path.stat().st_size
            if file_size > 0:
                # Open file in binary update mode with zero buffering
                with open(path, "ba+", buffering=0) as f:
                    for _ in range(passes):
                        f.seek(0)
                        f.write(os.urandom(file_size))
                        os.fsync(f.fileno())

            # Rename to mask metadata before unlinking
            parent = path.parent
            temp_name = parent / f"shredded_{os.urandom(8).hex()}"
            try:
                path.rename(temp_name)
                temp_name.unlink()
            except OSError:
                path.unlink(missing_ok=True)
            logger.debug("Overwritten and unlinked file: %s", path.name)
    except Exception as e:
        logger.error("Failed to overwrite file %s before removal: %s", path.name, e)
        # Fallback to simple unlink if overwrite fails
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _on_rm_error(func, path_str, exc_info):
    """Error handler for shutil.rmtree to change read-only permissions and retry."""
    try:
        os.chmod(path_str, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        func(path_str)
    except Exception:
        pass


def shred_directory(dir_path: Union[str, Path], passes: int = 3) -> None:
    """
    Recursively overwrites all files in a directory and removes the directory tree.
    See shred_file for physical erasure limitations on SSDs and CoW filesystems.
    """
    path = Path(dir_path)
    if not path.exists():
        return

    try:
        # Shred all files recursively
        for item in sorted(path.rglob("*"), reverse=True):
            if item.is_file() or item.is_symlink():
                shred_file(item, passes=passes)
            elif item.is_dir() and os.name == "posix":
                try:
                    os.chmod(item, stat.S_IRWXU)
                except OSError:
                    pass

        # Delete the directory tree
        shutil.rmtree(path, onexc=_on_rm_error)
        logger.debug("Removed temporary directory tree: %s", path.name)
    except Exception as e:
        logger.error("Failed to cleanly remove directory %s: %s", path, e)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
