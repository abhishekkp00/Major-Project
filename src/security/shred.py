import os
import shutil
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger("secure_lora.security.shred")


def shred_file(file_path: Union[str, Path], passes: int = 3) -> None:
    """
    Overwrites a file with random bytes multiple times (shredding)
    before removing it to prevent data recovery.
    """
    path = Path(file_path)
    if not path.exists():
        return

    try:
        if path.is_file():
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
            path.rename(temp_name)
            temp_name.unlink()
            logger.info("Securely shredded and deleted file: %s", path.name)
    except Exception as e:
        logger.error("Failed to securely shred file %s: %s", path.name, e)
        # Fallback to simple unlink if overwrite fails
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def shred_directory(dir_path: Union[str, Path], passes: int = 3) -> None:
    """
    Recursively shreds all files in a directory, then deletes the directory structure.
    """
    path = Path(dir_path)
    if not path.exists():
        return

    try:
        # Shred all files recursively
        for item in path.rglob("*"):
            if item.is_file():
                shred_file(item, passes=passes)

        # Delete the empty directory tree
        shutil.rmtree(path)
        logger.info("Successfully shredded and removed directory: %s", path.name)
    except Exception as e:
        logger.error("Failed to remove directory %s: %s", path, e)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
