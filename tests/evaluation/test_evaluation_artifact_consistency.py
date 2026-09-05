"""
test_evaluation_artifact_consistency.py
==========================================
Unit test for evaluation artifact consistency and metric audit validation.
"""

from scripts.validate_evaluation_artifacts import run_full_evaluation_audit, main


def test_evaluation_artifacts_consistency():
    """Verifies that all research evaluation JSON artifacts match reported metrics and schemas."""
    audits = run_full_evaluation_audit()
    assert len(audits) >= 5
    for item in audits:
        assert item.get("verified") is True
        assert "artifact" in item
        assert "status" in item


def test_validation_script_main_exit_code():
    """Verifies that main() returns exit code 0 when all artifacts are consistent."""
    assert main() == 0
