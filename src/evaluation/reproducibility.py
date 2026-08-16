"""
reproducibility.py
==================
Reproducibility Metadata and Environment Provenance Collector for SecureLoRA.

Captures hardware, OS, Python runtime, package versions, git commit hash,
dataset/model IDs, seeds, and execution timestamps for research experiments.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit_sha(repo_path: Optional[Path] = None) -> str:
    """Returns the current git commit SHA or UNKNOWN_GIT_SHA if unavailable."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = str(repo_path) if repo_path else os.getcwd()
        res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_SHA"


def get_hardware_info() -> Dict[str, Any]:
    """Collects hardware info (CPU, RAM, GPU availability)."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "architecture": platform.architecture()[0],
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    
    # Try reading CPU model name on Linux
    try:
        if Path("/proc/cpuinfo").exists():
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    # Try checking RAM
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        info["ram_total_gb"] = "UNKNOWN"

    # Try checking PyTorch CUDA GPU
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = False

    return info


def get_package_versions() -> Dict[str, str]:
    """Collects version strings for key machine learning and security dependencies."""
    packages = ["torch", "transformers", "opacus", "cryptography", "numpy", "spacy", "presidio_analyzer", "scikit-learn", "scipy"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "NOT_INSTALLED"
    return versions


@dataclass
class ReproducibilityMetadata:
    experiment_id: str
    git_commit_sha: str
    timestamp_utc: str
    seed: int
    model_identifier: str
    dataset_identifier: str
    dataset_split: str
    python_version: str
    os_info: str
    hardware: Dict[str, Any]
    package_versions: Dict[str, str]
    configuration_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_reproducibility_metadata(
    experiment_id: str,
    seed: int,
    model_identifier: str = "distilbert-base-uncased",
    dataset_identifier: str = "sample_medical_phi.jsonl",
    dataset_split: str = "test",
    configuration_snapshot: Optional[Dict[str, Any]] = None,
) -> ReproducibilityMetadata:
    """Creates a comprehensive ReproducibilityMetadata instance."""
    return ReproducibilityMetadata(
        experiment_id=experiment_id,
        git_commit_sha=get_git_commit_sha(),
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=seed,
        model_identifier=model_identifier,
        dataset_identifier=dataset_identifier,
        dataset_split=dataset_split,
        python_version=platform.python_version(),
        os_info=f"{platform.system()} {platform.release()}",
        hardware=get_hardware_info(),
        package_versions=get_package_versions(),
        configuration_snapshot=configuration_snapshot or {},
    )
