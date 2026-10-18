"""
test_utility_evaluation.py
==========================
Comprehensive tests for ML evaluation improvements:
  1. Split integrity and reproducibility (train/test disjointness, hash stability).
  2. Data leakage detection (detection of overlapping IDs or duplicate inputs).
  3. Metric correctness on small known fixtures (token precision, recall, F1, accuracy, retention).
  4. Missing model or data produces explicit failure / NOT_EXECUTED status without inventing numbers.
  5. Baseline comparison uses identical evaluation protocol and held-out test split.
  6. Results can be reloaded and verified by the research API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
import torch

from src.evaluation.utility_evaluator import (
    prepare_task_dataset,
    compute_token_f1,
    compute_placeholder_retention,
    compute_sequence_accuracy,
    format_instruction_prompt,
    UtilityEvaluationError,
    DatasetSplits,
)
from src.evaluation.experiment_runner import run_single_baseline
from src.evaluation.research_api import research_api_bp
from flask import Flask


@pytest.fixture
def test_app():
    app = Flask("test_utility_research_api")
    app.config["TESTING"] = True
    app.register_blueprint(research_api_bp)
    return app.test_client()


# ---------------------------------------------------------------------------
# 1. Split Integrity & Reproducibility Checks
# ---------------------------------------------------------------------------

def test_split_integrity_and_reproducibility():
    """Verify train and held-out test sets are strictly disjoint and reproducible by seed."""
    splits_1 = prepare_task_dataset(dataset_name="synthetic", subset_size=12, train_ratio=0.7, seed=42)
    splits_2 = prepare_task_dataset(dataset_name="synthetic", subset_size=12, train_ratio=0.7, seed=42)

    # Reproducibility check: same seed produces identical hashes and counts
    assert splits_1.train_hash == splits_2.train_hash
    assert splits_1.test_hash == splits_2.test_hash
    assert splits_1.n_train == splits_2.n_train
    assert splits_1.n_test == splits_2.n_test

    # Disjointness check
    train_ids = {r["record_id"] for r in splits_1.train_records}
    test_ids = {r["record_id"] for r in splits_1.test_records}
    assert train_ids.isdisjoint(test_ids), f"Overlapping IDs found: {train_ids & test_ids}"

    train_inputs = {r["input"].strip() for r in splits_1.train_records}
    test_inputs = {r["input"].strip() for r in splits_1.test_records}
    assert train_inputs.isdisjoint(test_inputs), f"Overlapping inputs found: {train_inputs & test_inputs}"


def test_split_leakage_detection():
    """Verify that artificial train/test overlap triggers UtilityEvaluationError."""
    # Create splits with an intentional overlapping record
    overlapping_record = {"record_id": "leak_001", "input": "Shared secret text", "output": "Masked text"}
    bad_splits = DatasetSplits(
        train_records=[overlapping_record, {"record_id": "rec_002", "input": "Train text", "output": "Masked"}],
        test_records=[overlapping_record, {"record_id": "rec_003", "input": "Test text", "output": "Masked"}],
        train_hash="hash_tr",
        test_hash="hash_te",
        n_train=2,
        n_test=2,
        dataset_name="synthetic",
        seed=42,
    )

    with pytest.raises(UtilityEvaluationError, match="Split leakage detected"):
        bad_splits.verify_split_integrity()


# ---------------------------------------------------------------------------
# 2. Metric Correctness on Known Fixtures
# ---------------------------------------------------------------------------

def test_token_f1_known_fixture():
    """Verify token-level precision, recall, and F1 calculations on deterministic test pairs."""
    # Perfect match
    p, r, f1 = compute_token_f1("Patient [NAME] admitted on [DATE]", "Patient [NAME] admitted on [DATE]")
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    # Completely disjoint
    p, r, f1 = compute_token_f1("Alpha Beta Gamma", "Delta Epsilon Zeta")
    assert p == 0.0 and r == 0.0 and f1 == 0.0

    # Partial match:
    # Pred: "Alpha Beta Gamma" (3 tokens: alpha, beta, gamma)
    # Gold: "Beta Gamma Delta Epsilon" (4 tokens: beta, gamma, delta, epsilon)
    # Shared: 2 tokens (beta, gamma)
    # Expected: P = 2/3, R = 2/4 = 0.5, F1 = 2*(2/3*1/2)/(2/3+1/2) = (2/3)/(7/6) = 4/7 ≈ 0.5714
    p, r, f1 = compute_token_f1("Alpha Beta Gamma", "Beta Gamma Delta Epsilon")
    assert pytest.approx(p, 0.001) == 2.0 / 3.0
    assert pytest.approx(r, 0.001) == 0.5
    assert pytest.approx(f1, 0.001) == 4.0 / 7.0

    # Empty string edge cases
    assert compute_token_f1("", "") == (1.0, 1.0, 1.0)
    assert compute_token_f1("Something", "") == (0.0, 0.0, 0.0)


def test_placeholder_retention_known_fixture():
    """Verify entity placeholder preservation calculation."""
    gold = "Send [EMAIL] to Dr. [PHYSICIAN] on [DATE]."
    
    # All 3 placeholders preserved
    pred_all = "Contact [EMAIL] and [PHYSICIAN] by [DATE]."
    assert compute_placeholder_retention(pred_all, gold) == 1.0

    # 1 of 3 preserved ([EMAIL])
    pred_one = "Contact [EMAIL] and Doctor Smith on Monday."
    assert pytest.approx(compute_placeholder_retention(pred_one, gold), 0.001) == 1.0 / 3.0

    # None preserved (unredacted leak)
    pred_none = "Contact test@example.com and Dr. Smith on Monday."
    assert compute_placeholder_retention(pred_none, gold) == 0.0

    # Gold with no placeholders returns 1.0
    assert compute_placeholder_retention("Any text", "No placeholders here") == 1.0


def test_sequence_accuracy_known_fixture():
    """Verify normalized sequence accuracy matches shared / max_length."""
    # 2 matching tokens out of max length 4: Acc = 2/4 = 0.5
    acc = compute_sequence_accuracy("apple banana orange", "banana orange grape pear")
    assert pytest.approx(acc, 0.001) == 2.0 / 4.0

    # Identical: Acc = 1.0
    assert compute_sequence_accuracy("same words", "same words") == 1.0


# ---------------------------------------------------------------------------
# 3. Missing Model / Data Produces Explicit Failure (NOT_EXECUTED)
# ---------------------------------------------------------------------------

def test_missing_model_produces_not_executed(tmp_path, monkeypatch):
    """Verify that a model loading failure returns NOT_EXECUTED without fabricating numbers."""
    from transformers import AutoModelForCausalLM

    def mock_fail(*args, **kwargs):
        raise RuntimeError("Local checkpoint weights corrupted or unavailable.")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", mock_fail)

    res = run_single_baseline("E0", seed=42, output_dir=tmp_path, quick_mode=True)
    assert res.execution_status == "NOT_EXECUTED"
    assert res.not_executed_reason is not None
    assert "corrupted" in res.not_executed_reason.lower() or "unavailable" in res.not_executed_reason.lower()


def test_missing_dataset_produces_not_executed(tmp_path, monkeypatch):
    """Verify missing/empty dataset causes explicit NOT_EXECUTED status."""
    import src.evaluation.utility_evaluator as u_eval

    def mock_prepare_fail(*args, **kwargs):
        raise UtilityEvaluationError("Dataset source partition unavailable.")

    monkeypatch.setattr(u_eval, "prepare_task_dataset", mock_prepare_fail)

    res = run_single_baseline("E0", seed=42, output_dir=tmp_path, quick_mode=True)
    assert res.execution_status == "NOT_EXECUTED"
    assert "unavailable" in res.not_executed_reason.lower() or "dataset" in res.not_executed_reason.lower()


# ---------------------------------------------------------------------------
# 4. Identical Protocol Across Baselines
# ---------------------------------------------------------------------------

def test_identical_protocol_across_baselines(tmp_path):
    """Verify E0 (zero-shot base model) and E2 (LoRA fine-tuned) run under identical test splits."""
    res_e0 = run_single_baseline("E0", seed=42, output_dir=tmp_path, quick_mode=True)
    res_e2 = run_single_baseline("E2", seed=42, output_dir=tmp_path, quick_mode=True)

    assert res_e0.execution_status == "COMPLETED"
    assert res_e2.execution_status == "COMPLETED"

    # Both must report real validation loss and perplexity on response tokens
    assert res_e0.utility.val_loss > 0
    assert res_e2.utility.val_loss > 0
    assert res_e0.utility.perplexity >= 1.0
    assert res_e2.utility.perplexity >= 1.0

    # Verify both have identical test split hashes
    assert res_e0.metadata.hardware.get("split_hash_test") == res_e2.metadata.hardware.get("split_hash_test")
    assert res_e0.metadata.hardware.get("num_test_samples") == res_e2.metadata.hardware.get("num_test_samples")

    # Verify scoring protocol is documented and identical
    assert res_e0.utility.scoring_protocol is not None
    assert "target-masked" in res_e0.utility.scoring_protocol.lower()
    assert res_e0.utility.scoring_protocol == res_e2.utility.scoring_protocol


# ---------------------------------------------------------------------------
# 5. Research API Reloadability
# ---------------------------------------------------------------------------

def test_research_api_reloads_rigorous_utility_results(test_app):
    """Verify the research API loads and exposes real utility metrics with full provenance."""
    resp = test_app.get("/api/research/summary")
    assert resp.status_code == 200
    data = resp.get_json()

    prov = data.get("provenance", {})
    assert "utility.val_loss" in prov
    assert "utility.perplexity" in prov
    assert "utility.f1" in prov

    m_loss = prov["utility.val_loss"]
    assert m_loss["semantic_category"] == "utility"
    assert m_loss["unit"] == "loss"
    assert m_loss["source_artifact"] is not None
