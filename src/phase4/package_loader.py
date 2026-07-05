import os
import tarfile
import logging
import tempfile
from pathlib import Path
from typing import Optional

from src.common.exceptions import (
    IncompletePackageError,
    InvalidArchiveError,
    SecurityError
)

logger = logging.getLogger("secure_lora.phase4.package_loader")

REQUIRED_FILES = {
    "adapter.enc",
    "adapter.hash",
    "adapter.sig",
    "metadata.json",
    "package_manifest.json",
}


class PackageLoader:
    """
    Context manager to load and prepare a protected adapter package.
    Detects directory vs tar.gz, extracts if needed, and validates completeness.
    """
    def __init__(self, package_path: Path, max_bytes: int = 512 * 1024 * 1024):
        self.package_path = Path(package_path)
        self.max_bytes = max_bytes
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.extracted_path: Optional[Path] = None

    def __enter__(self) -> Path:
        if not self.package_path.exists():
            raise FileNotFoundError(f"Package path does not exist: {self.package_path}")

        if self.package_path.is_dir():
            logger.info("Loading package directly from directory: %s", self.package_path)
            self.extracted_path = self.package_path
        elif self.package_path.suffix == ".gz" or self.package_path.name.endswith(".tar.gz"):
            logger.info("Extracting package archive: %s", self.package_path)
            self._extract_archive()
        else:
            raise ValueError(f"Unsupported package format: {self.package_path.name}")

        self.verify_completeness()
        return self.extracted_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def _extract_archive(self):
        archive_size = self.package_path.stat().st_size
        if archive_size > self.max_bytes:
            raise InvalidArchiveError(
                f"Archive size ({archive_size} bytes) exceeds safety limit of {self.max_bytes} bytes."
            )

        self.temp_dir = tempfile.TemporaryDirectory(prefix="secure_lora_deploy_")
        self.extracted_path = Path(self.temp_dir.name)

        try:
            with tarfile.open(self.package_path, "r:gz") as tar:
                # Security: inspect members to prevent path traversal (Slip vulnerabilities)
                for member in tar.getmembers():
                    target_path = (self.extracted_path / member.name).resolve()
                    if not str(target_path).startswith(str(self.extracted_path)):
                        raise SecurityError(f"Directory traversal attempt detected in archive member: {member.name}")

                tar.extractall(path=self.extracted_path)

            subdirs = list(self.extracted_path.iterdir())
            if len(subdirs) == 1 and subdirs[0].is_dir():
                self.extracted_path = subdirs[0]

        except Exception as e:
            self.cleanup()
            if isinstance(e, SecurityError):
                raise
            raise InvalidArchiveError(f"Failed to extract package archive: {e}") from e

    def verify_completeness(self):
        """Checks if all required package files are present."""
        if not self.extracted_path:
            raise ValueError("No extracted package path available.")

        existing_files = {p.name for p in self.extracted_path.iterdir() if p.is_file()}
        missing_files = REQUIRED_FILES - existing_files

        if missing_files:
            raise IncompletePackageError(
                f"Package is incomplete. Missing files: {sorted(list(missing_files))}"
            )
        logger.info("Package completeness verified. All required files are present.")

    def cleanup(self):
        """Cleans up any temporary directories created."""
        if self.temp_dir:
            logger.info("Cleaning up temporary package extraction directory.")
            try:
                self.temp_dir.cleanup()
            except Exception as e:
                logger.warning("Failed to clean up temporary directory %s: %s", self.temp_dir.name, e)
            self.temp_dir = None
            self.extracted_path = None
