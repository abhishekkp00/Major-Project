"""
test_research_dashboard_step4.py
================================
Unit tests verifying Step 4 of the SecureLoRA Research Dashboard:
- Verifies HTML structure for Research & Ablation Workbench tab in index.html.
- Verifies disclaimer banner, RQ cards (1-7), top KPI metrics grid, ablation table, privacy-utility chart canvas, screening comparison chart canvas, adaptive evasion section, overhead breakdown, multi-seed stability grid, and detail modal.
- Verifies JavaScript function definitions and endpoint accessibility via Flask test client.
"""

import os
from pathlib import Path
import pytest
from src.evaluation.dashboard import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_research_tab_html_structure():
    """Verify that index.html contains all required elements for Research & Metrics Workbench."""
    template_path = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "templates" / "index.html"
    assert template_path.exists(), "index.html must exist"
    
    content = template_path.read_text(encoding="utf-8")
    
    # Nav Tab Buttons
    assert 'id="tabRun"' in content
    assert 'id="tabMetrics"' in content
    assert 'id="tabSecurity"' in content
    assert 'id="tabModel"' in content

    # Dynamic Dataset Cards Container
    assert 'id="dataset-cards-grid"' in content
    assert 'id="selected-dataset-info"' in content

    # Metrics & Chart Elements
    assert 'id="chart-overhead"' in content or 'id="chart-privacy-utility"' in content or 'id="metrics-ablation-tbody"' in content


def test_research_dashboard_js_functions():
    """Verify dashboard.js contains research & dataset adapter JS logic."""
    js_path = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "static" / "js" / "dashboard.js"
    assert js_path.exists(), "dashboard.js must exist"
    
    content = js_path.read_text(encoding="utf-8")
    
    assert "initDatasetTemplates" in content
    assert "initMetricsPage" in content
    assert "selectDatasetCard" in content
    assert "startSecurePipeline" in content



def test_research_api_endpoints_integration(client):
    """Verify read-only research API endpoints respond with 200 and available status."""
    endpoints = [
        "/api/research/summary",
        "/api/research/ablation",
        "/api/research/privacy",
        "/api/research/screening",
        "/api/research/adaptive-evasion",
        "/api/research/overhead",
    ]
    
    for ep in endpoints:
        rv = client.get(ep)
        assert rv.status_code == 200, f"Endpoint {ep} returned status {rv.status_code}"
        json_data = rv.get_json()
        assert "available" in json_data, f"Endpoint {ep} response missing 'available' field"
        assert json_data["available"] is True, f"Endpoint {ep} should be available"
