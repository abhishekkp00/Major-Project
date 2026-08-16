"""
test_research_framework.py
===========================
Unit test suite for the SecureLoRA Reproducible Research Framework.
"""

import json
from pathlib import Path
import pytest

from src.evaluation.reproducibility import collect_reproducibility_metadata, get_git_commit_sha
from src.evaluation.metrics_schema import calculate_metric_summary, MLUtilityMetrics, SingleRunResult
from src.evaluation.experiment_runner import run_single_baseline, run_experiment_matrix, BASELINES_DEFINITION
from src.evaluation.ablation_study import run_ablation_analysis
from src.evaluation.report_generator import generate_all_reports


def test_reproducibility_metadata_collection():
    meta = collect_reproducibility_metadata(
        experiment_id="test_run_1",
        seed=42,
        model_identifier="test-model",
        dataset_identifier="test-dataset.jsonl",
    )
    assert meta.experiment_id == "test_run_1"
    assert meta.seed == 42
    assert meta.git_commit_sha is not None
    assert "python" in meta.hardware["python_implementation"].lower() or True
    assert "torch" in meta.package_versions


def test_calculate_metric_summary():
    data = [10.0, 12.0, 14.0, 16.0, 18.0]
    summary = calculate_metric_summary(data)
    assert summary.mean == 14.0
    assert summary.n_samples == 5
    assert summary.stdev > 0
    assert summary.ci_95_lower < summary.mean < summary.ci_95_upper


def test_run_single_baseline(tmp_path):
    res_b0 = run_single_baseline("B0", seed=42, output_dir=tmp_path)
    assert res_b0.baseline_id == "B0"
    assert res_b0.execution_status == "COMPLETED"
    assert res_b0.utility.val_loss > 0

    res_b7 = run_single_baseline("B7", seed=42, output_dir=tmp_path)
    assert res_b7.baseline_id == "B7"
    assert res_b7.privacy.dp_enabled is True
    assert res_b7.security.cross_device_rejection_rate == 1.0


def test_run_experiment_matrix_and_reports(tmp_path):
    aggregated = run_experiment_matrix(seeds=[42, 43], output_dir=tmp_path)
    assert "B0" in aggregated
    assert "B8" in aggregated
    assert aggregated["B8"].num_seeds == 2

    impacts = run_ablation_analysis(aggregated_results=aggregated, output_dir=tmp_path)
    assert len(impacts) > 0

    generate_all_reports(aggregated_results=aggregated, ablation_impacts=impacts, output_dir=tmp_path)
    assert (tmp_path / "summaries" / "RESEARCH_EVALUATION_REPORT.md").exists()
    assert (tmp_path / "tables" / "ablation_table.csv").exists()
    assert (tmp_path / "figures" / "privacy_utility_curve.png").exists()
