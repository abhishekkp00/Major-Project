"""
tests/unit/test_phase3_screening_integration.py
================================================
Regression tests for the Phase 3 pre-packaging adapter screening integration.

Verifies that:
  A. Valid adapter_model.safetensors -> real weights are screened (not synthetic).
  B. Valid adapter_model.bin -> real weights are screened (not synthetic).
  C. Missing weight file -> package build fails hard (FileNotFoundError).
  D. Directory passed as weight path -> screening gate fails hard.
  E. Research fallback (synthetic weights) cannot execute from the production
     packaging path — i.e. _resolve_weights in production mode asserts/raises.
  F. Behavioral screening without a real inference callable in production mode
     does NOT raise but marks real_callable_used=False in the report.
  G. Screening report records the actual adapter artifact path (not package dir,
     not "mock", not a random-weight sentinel).

All tests use torch + numpy real tensors to write genuine .bin/.safetensors
files where applicable, or rely on FileNotFoundError for missing-file scenarios.
"""

import os
import struct
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _write_torch_bin(path: Path, weights: Dict[str, np.ndarray]) -> None:
    """Write a minimal PyTorch state-dict .bin using torch.save."""
    import torch
    state = {k: torch.tensor(v) for k, v in weights.items()}
    torch.save(state, path)


def _write_safetensors(path: Path, weights: Dict[str, np.ndarray]) -> None:
    """Write a minimal safetensors file using the safetensors library if available,
    otherwise fall back to writing a torch state-dict and rename so the
    *file-existence* path is exercised for .safetensors format tests."""
    try:
        from safetensors.torch import save_file
        import torch
        tensors = {k: torch.tensor(v) for k, v in weights.items()}
        save_file(tensors, str(path))
    except ImportError:
        # safetensors not installed — write torch .bin and pretend it's safetensors
        # so file-existence tests still work; load will fall through to torch.
        import torch
        state = {k: torch.tensor(v) for k, v in weights.items()}
        torch.save(state, path)


_REAL_WEIGHTS = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight":
        np.random.randn(8, 64).astype(np.float32) * 0.01,
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight":
        np.random.randn(64, 8).astype(np.float32) * 0.01,
}


def _minimal_package_dir(tmp_path: Path, *, weight_file: str = None) -> Path:
    """Creates the minimal artefact set required by verify_package_completeness."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "adapter.enc").write_bytes(os.urandom(128))
    (pkg / "adapter.hash").write_text("a" * 64)
    (pkg / "adapter.sig").write_bytes(os.urandom(256))
    (pkg / "metadata.json").write_text("{}")
    (pkg / "public.pem").write_text("---fake pem---")
    if weight_file:
        dst = pkg / weight_file
        if weight_file.endswith(".safetensors"):
            _write_safetensors(dst, _REAL_WEIGHTS)
        else:
            _write_torch_bin(dst, _REAL_WEIGHTS)
    return pkg


# ─────────────────────────────────────────────────────────────────────────────
# A. safetensors weight file → real weights screened
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetensorsWeightFile:

    def test_resolve_adapter_weight_file_finds_safetensors(self, tmp_path):
        pkg = _minimal_package_dir(tmp_path, weight_file="adapter_model.safetensors")
        from src.phase3.package_builder import _resolve_adapter_weight_file
        resolved = _resolve_adapter_weight_file(pkg)
        assert resolved.name == "adapter_model.safetensors"
        assert resolved.is_file()

    def test_screening_uses_real_weights_for_safetensors(self, tmp_path):
        weight_path = tmp_path / "adapter_model.safetensors"
        _write_safetensors(weight_path, _REAL_WEIGHTS)

        from src.security.adapter_screening import ScreeningPipeline
        pipe = ScreeningPipeline()
        report = pipe.screen_adapter(
            adapter_source=weight_path,
            adapter_id="test-safetensors",
            mode="production",
        )
        assert report.real_weights_used is True, (
            f"Expected real_weights_used=True for safetensors file, got: {report.real_weights_used}"
        )
        assert report.adapter_weight_path is not None
        assert "adapter_model.safetensors" in report.adapter_weight_path
        assert report.screening_mode == "production"


# ─────────────────────────────────────────────────────────────────────────────
# B. .bin weight file → real weights screened
# ─────────────────────────────────────────────────────────────────────────────

class TestBinWeightFile:

    def test_resolve_adapter_weight_file_finds_bin_fallback(self, tmp_path):
        pkg = _minimal_package_dir(tmp_path, weight_file="adapter_model.bin")
        from src.phase3.package_builder import _resolve_adapter_weight_file
        resolved = _resolve_adapter_weight_file(pkg)
        assert resolved.name == "adapter_model.bin"
        assert resolved.is_file()

    def test_screening_uses_real_weights_for_bin(self, tmp_path):
        weight_path = tmp_path / "adapter_model.bin"
        _write_torch_bin(weight_path, _REAL_WEIGHTS)

        from src.security.adapter_screening import ScreeningPipeline
        pipe = ScreeningPipeline()
        report = pipe.screen_adapter(
            adapter_source=weight_path,
            adapter_id="test-bin",
            mode="production",
        )
        assert report.real_weights_used is True, (
            f"Expected real_weights_used=True for .bin file, got: {report.real_weights_used}"
        )
        assert "adapter_model.bin" in report.adapter_weight_path
        assert report.screening_mode == "production"

    def test_safetensors_preferred_over_bin(self, tmp_path):
        """When both formats exist safetensors is resolved first."""
        pkg = _minimal_package_dir(tmp_path, weight_file="adapter_model.safetensors")
        _write_torch_bin(pkg / "adapter_model.bin", _REAL_WEIGHTS)

        from src.phase3.package_builder import _resolve_adapter_weight_file
        resolved = _resolve_adapter_weight_file(pkg)
        assert resolved.name == "adapter_model.safetensors"


# ─────────────────────────────────────────────────────────────────────────────
# C. Missing weight file → build fails hard
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingWeightFile:

    def test_resolve_raises_file_not_found_when_no_weight(self, tmp_path):
        pkg = _minimal_package_dir(tmp_path)  # no weight file
        from src.phase3.package_builder import _resolve_adapter_weight_file
        with pytest.raises(FileNotFoundError, match="adapter_model"):
            _resolve_adapter_weight_file(pkg)

    def test_build_package_fails_when_weight_missing(self, tmp_path):
        """build_package() must raise FileNotFoundError when no weight file exists."""
        pkg = _minimal_package_dir(tmp_path)  # no weight file written
        pub = tmp_path / "public.pem"
        pub.write_text("---fake pem---")
        from src.phase3.package_builder import build_package
        with pytest.raises(FileNotFoundError, match="adapter_model"):
            build_package(
                package_dir=pkg,
                public_key_src=pub,
                enable_screening=True,
            )

    def test_screening_gate_production_raises_on_missing_file(self, tmp_path):
        """pre_packaging_screening_gate in production mode raises on missing path."""
        missing = tmp_path / "does_not_exist.safetensors"
        from src.security.adapter_screening import (
            pre_packaging_screening_gate,
            SecurityScreeningError,
        )
        with pytest.raises(SecurityScreeningError, match="does not exist"):
            pre_packaging_screening_gate(
                adapter_source=missing,
                adapter_id="ghost-adapter",
                mode="production",
            )


# ─────────────────────────────────────────────────────────────────────────────
# D. Directory passed as weight source → gate fails hard
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectoryAsWeightSource:

    def test_resolve_raises_value_error_for_directory_weight(self, tmp_path):
        """_resolve_adapter_weight_file raises ValueError if candidate is a dir."""
        pkg = _minimal_package_dir(tmp_path)
        # Create a directory named adapter_model.safetensors (malformed package)
        dir_path = pkg / "adapter_model.safetensors"
        dir_path.mkdir()
        from src.phase3.package_builder import _resolve_adapter_weight_file
        with pytest.raises(ValueError, match="directory"):
            _resolve_adapter_weight_file(pkg)

    def test_screening_pipeline_production_rejects_directory_source(self, tmp_path):
        """ScreeningPipeline._resolve_weights raises for a directory in production mode."""
        from src.security.adapter_screening import (
            ScreeningPipeline,
            SecurityScreeningError,
        )
        pipe = ScreeningPipeline()
        with pytest.raises(SecurityScreeningError, match="directory"):
            pipe.screen_adapter(
                adapter_source=tmp_path,      # directory, not a file
                adapter_id="dir-adapter",
                mode="production",
            )


# ─────────────────────────────────────────────────────────────────────────────
# E. Research fallback cannot execute from production packaging path
# ─────────────────────────────────────────────────────────────────────────────

class TestProductionFallbackPrevention:

    def test_resolve_weights_production_raises_instead_of_synthetic(self, tmp_path):
        """In production mode, passing a non-existent path must raise, never
        fall through to synthetic Gaussian weights."""
        from src.security.adapter_screening.screening_pipeline import (
            ScreeningPipeline,
            SecurityScreeningError,
        )
        pipe = ScreeningPipeline()
        missing = tmp_path / "nonexistent.bin"
        with pytest.raises(SecurityScreeningError):
            pipe._resolve_weights(missing, mode="production")

    def test_resolve_weights_production_raises_for_directory(self, tmp_path):
        """Directory source in production mode must raise SecurityScreeningError."""
        from src.security.adapter_screening.screening_pipeline import (
            ScreeningPipeline,
            SecurityScreeningError,
        )
        pipe = ScreeningPipeline()
        with pytest.raises(SecurityScreeningError, match="directory"):
            pipe._resolve_weights(tmp_path, mode="production")

    def test_research_mode_still_returns_synthetic_for_missing(self, tmp_path):
        """Research mode preserves the synthetic fallback for non-existent paths
        (backward compatibility for existing research/eval tests)."""
        from src.security.adapter_screening.screening_pipeline import ScreeningPipeline
        pipe = ScreeningPipeline()
        missing = tmp_path / "nonexistent.bin"
        weights = pipe._resolve_weights(missing, mode="research")
        assert isinstance(weights, dict) and len(weights) > 0, (
            "Research mode must still return synthetic weights for unresolvable sources."
        )


# ─────────────────────────────────────────────────────────────────────────────
# F. Behavioral screening without real callable in production → fails closed
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralScreeningProductionMode:

    def test_behavioral_screening_production_no_callable_marks_report(self, tmp_path):
        """When no callable is provided in production mode, behavioral screening
        falls back to research mode synthetic baseline and marks
        real_callable_used=False in the report (does NOT raise)."""
        weight_path = tmp_path / "adapter_model.bin"
        _write_torch_bin(weight_path, _REAL_WEIGHTS)

        from src.security.adapter_screening import ScreeningPipeline
        pipe = ScreeningPipeline()
        report = pipe.screen_adapter(
            adapter_source=weight_path,
            adapter_id="no-callable-adapter",
            candidate_model_fn=None,    # no live inference callable
            mode="production",
        )
        assert report.real_callable_used is False, (
            "Without a real callable, real_callable_used must be False."
        )
        assert report.real_weights_used is True, (
            "Weight loading must still be real even when callable is missing."
        )

    def test_behavioral_screening_raises_in_production_when_callable_errors(self, tmp_path):
        """If a callable is provided but it raises, production mode propagates the error."""
        weight_path = tmp_path / "adapter_model.bin"
        _write_torch_bin(weight_path, _REAL_WEIGHTS)

        def broken_callable(prompt, probe_type):
            raise RuntimeError("Simulated inference failure")

        from src.security.adapter_screening import (
            ScreeningPipeline,
        )
        from src.security.adapter_screening.behavioral_analysis import BehavioralScreeningError
        pipe = ScreeningPipeline()
        with pytest.raises(BehavioralScreeningError):
            pipe.screen_adapter(
                adapter_source=weight_path,
                adapter_id="broken-callable-adapter",
                candidate_model_fn=broken_callable,
                mode="production",
            )


# ─────────────────────────────────────────────────────────────────────────────
# G. Screening report records actual adapter artifact path
# ─────────────────────────────────────────────────────────────────────────────

class TestScreeningReportProvenance:

    def test_report_records_actual_weight_path(self, tmp_path):
        weight_path = tmp_path / "adapter_model.bin"
        _write_torch_bin(weight_path, _REAL_WEIGHTS)

        from src.security.adapter_screening import ScreeningPipeline
        pipe = ScreeningPipeline()
        report = pipe.screen_adapter(
            adapter_source=weight_path,
            adapter_id="provenance-test",
            mode="production",
        )
        assert report.adapter_weight_path is not None, "adapter_weight_path must be set"
        assert "adapter_model.bin" in report.adapter_weight_path, (
            f"Report must record the actual weight file, got: {report.adapter_weight_path!r}"
        )
        # The report must NOT contain any mock/random/synthetic sentinel
        assert "mock" not in (report.adapter_weight_path or "").lower()
        assert "random" not in (report.adapter_weight_path or "").lower()
        assert "synthetic" not in (report.adapter_weight_path or "").lower()

    def test_report_records_screening_mode(self, tmp_path):
        weight_path = tmp_path / "adapter_model.bin"
        _write_torch_bin(weight_path, _REAL_WEIGHTS)

        from src.security.adapter_screening import ScreeningPipeline
        pipe = ScreeningPipeline()
        report = pipe.screen_adapter(
            adapter_source=weight_path,
            adapter_id="mode-test",
            mode="production",
        )
        assert report.screening_mode == "production"

    def test_report_to_dict_contains_all_provenance_fields(self, tmp_path):
        weight_path = tmp_path / "adapter_model.bin"
        _write_torch_bin(weight_path, _REAL_WEIGHTS)

        from src.security.adapter_screening import ScreeningPipeline
        pipe = ScreeningPipeline()
        report = pipe.screen_adapter(
            adapter_source=weight_path,
            adapter_id="dict-test",
            mode="production",
        )
        d = report.to_dict()
        for field in (
            "adapter_weight_path",
            "screening_mode",
            "real_weights_used",
            "real_callable_used",
            "risk_score",
            "decision",
        ):
            assert field in d, f"to_dict() must contain '{field}'"

    def test_no_random_mock_path_in_production_report(self, tmp_path):
        """Confirm the full path chain from build_package does not inject any
        synthetic/random weight sentinel into the screening report."""
        pkg = _minimal_package_dir(tmp_path, weight_file="adapter_model.bin")
        pub = tmp_path / "test.pem"
        pub.write_text("---fake pem---")

        # build_package will call _resolve_adapter_weight_file then
        # pre_packaging_screening_gate(mode="production").
        # The manifest's screening_report.adapter_weight_path must be the
        # actual file, never a synthetic placeholder.
        from src.phase3.package_builder import build_package
        manifest = build_package(
            package_dir=pkg,
            public_key_src=pub,
            enable_screening=True,
        )
        assert "screening_report" in manifest, "Manifest must include screening_report"
        sr = manifest["screening_report"]
        assert sr.get("screening_mode") == "production"
        assert sr.get("real_weights_used") is True
        assert "adapter_model.bin" in (sr.get("adapter_weight_path") or ""), (
            f"screening_report.adapter_weight_path must point to the actual weight file, "
            f"got: {sr.get('adapter_weight_path')!r}"
        )
        # No synthetic/mock/random anywhere in the path
        # Only inspect the filename component, not the full absolute path (which may
        # contain pytest temp dir names that incidentally include our sentinel words).
        reported_path = sr.get("adapter_weight_path") or ""
        path_str = Path(reported_path).name.lower() if reported_path else ""
        for sentinel in ("mock", "random", "synthetic", "fallback"):
            assert sentinel not in path_str, (
                f"Found sentinel '{sentinel}' in adapter_weight_path: {path_str!r}"
            )
