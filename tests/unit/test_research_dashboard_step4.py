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
    """Verify that index.html contains all required elements for Step 4 Research Workbench."""
    template_path = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "templates" / "index.html"
    assert template_path.exists(), "index.html must exist"
    
    content = template_path.read_text(encoding="utf-8")
    
    # 1. Nav Tab Button
    assert 'id="tabResearch"' in content
    assert "switchTab(this,'research')" in content
    assert "Research &amp; Ablation" in content

    # 2. Tab Content Container
    assert 'id="tab-research"' in content
    assert 'class="tab-content"' in content

    # 3. Disclaimer Banner
    assert 'class="res-disclaimer-banner"' in content
    assert "Results represent the experiments currently executed in this repository." in content

    # 4. Core Research Questions (RQ1 - RQ7)
    for rq_num in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7"]:
        assert rq_num in content, f"{rq_num} card missing from research questions grid"

    # 5. Top KPI Card Elements
    assert 'id="res-kpi-pii-f1"' in content
    assert 'id="res-kpi-utility"' in content
    assert 'id="res-kpi-epsilon"' in content
    assert 'id="res-kpi-screening-f1"' in content
    assert 'id="res-kpi-fnr"' in content
    assert 'id="res-kpi-evasion-rate"' in content
    assert 'id="res-kpi-overhead"' in content

    # 6. Ablation Table
    assert 'id="res-ablation-table"' in content
    assert 'id="res-ablation-tbody"' in content

    # 7. Privacy vs Utility Canvas
    assert 'id="chart-privacy-utility"' in content

    # 8. Screening Comparison Canvas
    assert 'id="chart-screening-comparison"' in content

    # 9. Adaptive Evasion Benchmark Section & Notice
    assert 'id="res-adaptive-evasion-section"' in content
    assert 'id="chart-evasion-detection"' in content
    assert 'id="chart-evasion-fnr"' in content
    assert 'id="evasion-missing-notice"' in content
    assert "Adaptive-evasion experiment not executed." in content

    # 10. Overhead Breakdown
    assert 'id="chart-overhead-breakdown"' in content
    assert 'id="res-overhead-metrics-list"' in content

    # 11. Multi-Seed Stability Grid
    assert 'id="res-multiseed-grid"' in content
    assert "Mean ± Std Dev" in content

    # 12. Research Detail Modal
    assert 'id="research-detail-modal"' in content
    assert 'id="res-modal-title"' in content
    assert 'id="res-modal-body"' in content


def test_research_dashboard_js_functions():
    """Verify dashboard.js contains Step 4 JS logic and functions."""
    js_path = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "static" / "js" / "dashboard.js"
    assert js_path.exists(), "dashboard.js must exist"
    
    content = js_path.read_text(encoding="utf-8")
    
    assert "initResearchTab" in content
    assert "fmtMeanStd" in content
    assert "openResearchDetailModal" in content
    assert "renderAblationTable" in content
    assert "renderPrivacyUtilityChart" in content
    assert "renderScreeningComparisonChart" in content
    assert "renderAdaptiveEvasionSection" in content
    assert "renderOverheadBreakdown" in content
    assert "renderMultiSeedStability" in content


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
