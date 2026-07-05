import pytest
from pathlib import Path
import os
import json

from src.phase3.package_builder import (
    verify_package_completeness,
    build_manifest,
    build_package,
)

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _build_minimal_package(pkg_dir: Path):
    (pkg_dir / "adapter.enc").write_bytes(os.urandom(128))
    (pkg_dir / "adapter.hash").write_text("a" * 64)
    (pkg_dir / "adapter.sig").write_bytes(os.urandom(256))
    (pkg_dir / "metadata.json").write_text("{}")
    (pkg_dir / "public.pem").write_text("---fake pem---")


def test_completeness_check_passes(tmp_dir):
    _build_minimal_package(tmp_dir)
    verify_package_completeness(tmp_dir)


def test_completeness_check_fails_on_missing_file(tmp_dir):
    _build_minimal_package(tmp_dir)
    (tmp_dir / "adapter.sig").unlink()
    with pytest.raises(FileNotFoundError, match="adapter.sig"):
        verify_package_completeness(tmp_dir)


def test_manifest_contains_required_fields(tmp_dir):
    _build_minimal_package(tmp_dir)
    manifest = build_manifest(
        package_dir=tmp_dir,
        adapter_id="test-adapter",
        model_reference="test-model",
        fingerprint_hash="fp" * 32,
        package_version="3.0.0",
        enc_metadata={"algorithm": "AES-256-GCM", "adapter_format": "file:.bin"},
    )
    for field in [
        "schema_version", "adapter_id", "model_reference",
        "device_fingerprint_hash_ref", "created_at_utc",
        "artefact_hashes", "verification_instructions", "security_notes",
    ]:
        assert field in manifest
