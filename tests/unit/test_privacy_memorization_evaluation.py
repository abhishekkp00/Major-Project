"""
test_privacy_memorization_evaluation.py
=======================================
Unit tests for SecureLoRA generation-level memorization & canary extraction evaluation:
  1. Score extraction verifies exact and normalized secret matches.
  2. Invariant verification: evaluation prompts strictly exclude the target secret (held-out extraction).
  3. Status truthfulness: unconstructed variants (DP-LoRA, SecureLoRA) remain strictly NOT_EXECUTED.
  4. Separation of theoretical DP guarantees (epsilon, delta) from empirical extraction rates.
  5. Raw output probe preservation and metadata traceability.
  6. Distinction between input PII redaction and generation-level memorization leakage.
"""

import json
from pathlib import Path
import pytest

from src.evaluation.canary_benchmark import (
    get_canary_records,
    score_extraction,
    normalize_secret,
    CANARY_RECORDS
)
from src.evaluation.privacy_evaluator import (
    calculate_memorization_extraction_metrics,
    calculate_privacy_metrics,
    evaluate_privacy_pipeline
)
from src.orchestrator.model_registry import model_registry


class TestCanaryBenchmarkIntegrity:
    """Validates structural integrity and prompt/secret separation in canary benchmark."""

    def test_secrets_strictly_excluded_from_prompts(self):
        """CRITICAL INVARIANT: The evaluation prompt prefix MUST NOT contain the secret canary."""
        canaries = get_canary_records()
        assert len(canaries) >= 20, "Benchmark must contain at least 20 controlled canaries."

        for c in canaries:
            prompt = c["prompt_prefix"]
            secret = c["ground_truth_secret"]
            assert secret not in prompt, (
                f"Data leakage in prompt! Secret '{secret}' found in prompt prefix '{prompt}'"
            )
            norm_secret = normalize_secret(secret)
            norm_prompt = normalize_secret(prompt)
            assert norm_secret not in norm_prompt, (
                f"Normalized secret '{norm_secret}' found in normalized prompt '{norm_prompt}'"
            )

    def test_canary_domains_and_types_diversity(self):
        """Verifies canary records span diverse sensitive domains and secret categories."""
        canaries = get_canary_records()
        domains = {c["domain"] for c in canaries}
        secret_types = {c["secret_type"] for c in canaries}

        assert "healthcare" in domains
        assert "enterprise" in domains
        assert "finance" in domains
        assert "hr" in domains

        assert "SSN" in secret_types
        assert "API_KEY" in secret_types


class TestExtractionScoring:
    """Tests exact, normalized, and negative match classification."""

    def test_exact_match_detection(self):
        res = score_extraction("481-92-3019", "The patient SSN is 481-92-3019 and was verified.")
        assert res["exact_match"] is True
        assert res["normalized_match"] is True
        assert res["match_type"] == "EXACT"

    def test_normalized_match_detection(self):
        # Different punctuation/formatting: spaces instead of hyphens
        res = score_extraction("481-92-3019", "Patient SSN is 481 92 3019 on file.")
        assert res["exact_match"] is False
        assert res["normalized_match"] is True
        assert res["match_type"] == "NORMALIZED"

    def test_negative_control_no_match(self):
        res = score_extraction("481-92-3019", "Patient is in stable condition with normal vital signs.")
        assert res["exact_match"] is False
        assert res["normalized_match"] is False
        assert res["match_type"] == "NONE"
        assert res["normalized_similarity"] < 0.50

    def test_empty_outputs_handling(self):
        res = score_extraction("481-92-3019", "")
        assert res["exact_match"] is False
        assert res["normalized_match"] is False
        assert res["match_type"] == "NONE"


class TestMemorizationMetricsComputation:
    """Tests aggregation of extraction metrics across a batch of probes."""

    def test_calculate_memorization_extraction_metrics(self):
        canaries = [
            {"ground_truth_secret": "SEC-001", "secret_type": "KEY"},
            {"ground_truth_secret": "SEC-002", "secret_type": "KEY"},
            {"ground_truth_secret": "SEC-003", "secret_type": "SSN"},
            {"ground_truth_secret": "SEC-004", "secret_type": "SSN"},
        ]
        # 1 exact, 1 normalized, 2 non-matches
        raw_outputs = [
            "Output has SEC-001 here.",
            "Output has sec002 here.",
            "Completely unrelated output 1.",
            "Completely unrelated output 2.",
        ]

        metrics = calculate_memorization_extraction_metrics(raw_outputs, canaries)
        assert metrics["total_canaries_tested"] == 4
        assert metrics["exact_matches_count"] == 1
        assert metrics["partial_matches_count"] == 2
        assert metrics["exact_extraction_rate"] == 0.25
        assert metrics["normalized_extraction_rate"] == 0.50
        assert "threat_model" in metrics
        assert "limitations" in metrics["threat_model"]


class TestSeparationAndTruthfulness:
    """Verifies unexecuted status preservation, DP separation, and artifact schema."""

    def test_unconstructed_variants_strictly_not_executed(self, tmp_path):
        """Unconstructed variants (DP-LoRA, SecureLoRA) must be marked NOT_EXECUTED with metrics=None."""
        model_registry.clear()
        res = evaluate_privacy_pipeline(
            samples=5,
            seed=42,
            output_dir=tmp_path,
            allow_direct_loading=False
        )

        variants = res["variants"]
        for v in ["base_model", "lora", "dp_lora", "securelora"]:
            assert variants[v]["status"] == "NOT_EXECUTED"
            assert variants[v]["metrics"] is None
            assert variants[v]["observed_extraction_metrics"] is None
            assert "reason" in variants[v]

    def test_dp_guarantee_separation(self, tmp_path):
        """DP parameters must be kept distinct in differential_privacy_guarantee and not confused with extraction."""
        res = evaluate_privacy_pipeline(
            samples=5,
            seed=42,
            output_dir=tmp_path,
            allow_direct_loading=False
        )

        dp_lora = res["variants"]["dp_lora"]
        dp_info = dp_lora["differential_privacy_guarantee"]
        assert dp_info["dp_enabled"] is True
        assert dp_info["epsilon"] == 2.443
        assert dp_info["delta"] == 1e-05
        assert "formal_guarantee_note" in dp_info
        # Observed extraction metrics must be None when not executed, never 0.0 or copied
        assert dp_lora["observed_extraction_metrics"] is None

    def test_canonical_artifact_schema(self, tmp_path):
        """Verifies comparison.json and canary_benchmark.json artifacts are generated with full metadata."""
        res = evaluate_privacy_pipeline(
            samples=5,
            seed=42,
            output_dir=tmp_path,
            allow_direct_loading=False
        )

        comp_path = tmp_path / "comparison.json"
        canary_path = tmp_path / "canary_benchmark.json"

        assert comp_path.exists()
        assert canary_path.exists()

        comp_data = json.loads(comp_path.read_text(encoding="utf-8"))
        assert "threat_model" in comp_data
        assert "distinction" in comp_data["threat_model"]
        assert "dp_separation_note" in comp_data["threat_model"]
        assert "variants" in comp_data
