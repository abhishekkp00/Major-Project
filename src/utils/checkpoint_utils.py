import re
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("secure_lora.utils.checkpoint")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """
    Scans the checkpoint directory and returns the path to the latest
    checkpoint folder (e.g. checkpoint-150) or None if no checkpoints exist.
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = []
    # Match folders named checkpoint-<digits>
    pattern = re.compile(r"^checkpoint-(\d+)$")

    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            match = pattern.match(subdir.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, str(subdir)))

    if not checkpoints:
        return None

    # Sort by step number descending and return the highest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_path = checkpoints[0]
    logger.info("Detected latest training checkpoint: %s (Step %d)", Path(latest_path).name, latest_step)
    return latest_path


def rotate_checkpoints(checkpoint_dir: Path, max_to_keep: int = 2) -> None:
    """
    Ensures only the latest N checkpoints are preserved on disk.
    Wipes older checkpoints securely.
    """
    if not checkpoint_dir.exists():
        return

    checkpoints = []
    pattern = re.compile(r"^checkpoint-(\d+)$")

    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            match = pattern.match(subdir.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, subdir))

    if len(checkpoints) <= max_to_keep:
        return

    # Sort checkpoints ascending (oldest first)
    checkpoints.sort(key=lambda x: x[0])

    # Identify folders to remove
    to_remove = checkpoints[:-max_to_keep]
    for step, path in to_remove:
        try:
            logger.info("Rotating/removing old checkpoint directory: %s", path.name)
            shutil.rmtree(path)
        except Exception as e:
            logger.error("Error rotating checkpoint %s: %s", path.name, e)
