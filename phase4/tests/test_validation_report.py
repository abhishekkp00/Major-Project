import json
import pytest
from pathlib import Path

from phase4.validation_report import generate_validation_reports

def test_generate_validation_reports(tmp_path: Path):
    output_dir = tmp_path / "reports"
    manifest = {
        "adapter_id": "test-adapter-v1",
        "model_reference": "llama-test",
        "schema_version": "3.0.0"
    }
    fingerprint_hash = "3926c635fa8a12607cf843d884442ae151b5253f54529dc053cd6f0cebddfb93"
    steps_status = {
        "Step 1: Package Completeness": "PASSED",
        "Step 2: Integrity Check": "PASSED",
        "Step 3: Signature Verification": "PASSED",
        "Step 4: Device Authorization": "PASSED",
        "Step 5: Key Derivation": "PASSED",
        "Step 6: Decryption": "PASSED"
    }
    inference_result = {
        "prompt": "Test prompt",
        "base_output": "Base response",
        "peft_output": "PEFT response",
        "adapter_active": True
    }
    
    json_path, md_path = generate_validation_reports(
        output_dir=output_dir,
        manifest=manifest,
        fingerprint_hash=fingerprint_hash,
        steps_status=steps_status,
        verification_success=True,
        inference_result=inference_result
    )
    
    assert json_path.exists()
    assert md_path.exists()
    
    # Verify JSON content
    data = json.loads(json_path.read_text())
    assert data["report_type"] == "Phase 4 Secure Deployment & Inference Validation Report"
    assert data["target_device"]["fingerprint_hash_prefix"] == "3926c635fa8a1260..."
    assert data["verification_pipeline"]["success"] is True
    assert data["inference_validation"]["adapter_active"] is True
    
    # Verify MD content
    md_text = md_path.read_text()
    assert "# Secure Device-Bound LoRA" in md_text
    assert "3926c635fa8a1260..." in md_text
    assert "PASSED" in md_text
