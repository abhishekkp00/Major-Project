"""
research_api.py
===============
READ-ONLY research results API for the SecureLoRA dashboard.

Exposes HISTORICAL experiment outputs from outputs/research/ to the
dashboard UI layer. Strictly separates research data from live pipeline
state (which lives in /api/phase4/* and /api/orchestrator/*).

Rules enforced here:
  - Never expose: private keys, salts, raw device IDs, weights, credentials.
  - Never fabricate: if a result file is missing, return {"available": false}.
  - Never mix live state and historical results in the same response.
  - All endpoints are GET, read-only.

Data sources (all pre-written JSON files):
  outputs/research/metrics/B8_summary.json         — full pipeline summary
  outputs/research/metrics/summary_metrics.json     — per-experiment summaries (E0-E9)
  outputs/research/metrics/ablation_study_summary.json
  outputs/research/adapter_screening/metrics.json   — screening confusion matrix
  outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify

logger = logging.getLogger("secure_lora.research_api")

research_api_bp = Blueprint("research_api", __name__)

# ---------------------------------------------------------------------------
# Path constants — all relative to project root; resolved at import time.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_DIR = _PROJECT_ROOT / "outputs" / "research"

_PATHS = {
    "b8_summary":         _RESEARCH_DIR / "metrics" / "B8_summary.json",
    "summary_metrics":    _RESEARCH_DIR / "metrics" / "summary_metrics.json",
    "ablation_summary":   _RESEARCH_DIR / "metrics" / "ablation_study_summary.json",
    "screening_metrics":  _RESEARCH_DIR / "adapter_screening" / "metrics.json",
    "evasion_metrics":    _RESEARCH_DIR / "adaptive_evasion" / "metrics" / "adaptive_evasion_metrics.json",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safely load a JSON result file. Returns (data, None) or (None, error_reason)."""
    path = _PATHS.get(key)
    if path is None:
        return None, f"Unknown result key: {key}"
    if not path.exists():
        return None, f"Experiment not executed — result file not found: {path.name}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, None
    except json.JSONDecodeError as e:
        logger.error("Malformed JSON in %s: %s", path, e)
        return None, f"Result file is malformed (JSON decode error): {path.name}"
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return None, f"Could not read result file: {path.name}"


def _unavailable(reason: str):
    return jsonify({"available": False, "reason": reason})


# ---------------------------------------------------------------------------
# GET /api/research/summary
# ---------------------------------------------------------------------------

@research_api_bp.route("/api/research/summary", methods=["GET"])
def research_summary():
    """
    Full SecureLoRA pipeline research summary.

    Returns utility (val_loss, perplexity, accuracy, F1), privacy (DP epsilon/delta,
    PII precision/recall/F1), security (rejection rates), and overhead metrics
    from the B8 (Full SecureLoRA + Screening) multi-seed experiment.

    Source: outputs/research/metrics/B8_summary.json
    Classification: HISTORICAL
    """
    data, err = _load_json("b8_summary")
    if err:
        return _unavailable(err)

    # Only expose safe summary fields — no weights, keys, or credentials.
    utility = data.get("utility_summary", {})
    privacy = data.get("privacy_summary", {})
    security = data.get("security_summary", {})
    overhead = data.get("overhead_summary", {})

    return jsonify({
        "available": True,
        "classification": "HISTORICAL",
        "source": "outputs/research/metrics/B8_summary.json",
        "experiment": {
            "baseline_id": data.get("baseline_id"),
            "baseline_name": data.get("baseline_name"),
            "execution_status": data.get("execution_status"),
            "num_seeds": data.get("num_seeds"),
        },
        "utility": utility,
        "privacy": privacy,
        "security": security,
        "overhead": overhead,
    })


# ---------------------------------------------------------------------------
# GET /api/research/ablation
# ---------------------------------------------------------------------------

@research_api_bp.route("/api/research/ablation", methods=["GET"])
def research_ablation():
    """
    Ablation study — component contribution matrix.

    Returns:
      - ablation_rows: per-component delta (B0–B8 / E0–E9) from ablation_study_summary.json
      - experiment_summaries: per-experiment aggregated metrics from summary_metrics.json

    Source: outputs/research/metrics/ablation_study_summary.json
            outputs/research/metrics/summary_metrics.json
    Classification: HISTORICAL
    """
    ablation_data, err1 = _load_json("ablation_summary")
    summary_data, err2 = _load_json("summary_metrics")

    if err1 and err2:
        return _unavailable(f"Both ablation files unavailable — {err1}; {err2}")

    return jsonify({
        "available": True,
        "classification": "HISTORICAL",
        "ablation_rows": ablation_data if ablation_data is not None else [],
        "ablation_available": ablation_data is not None,
        "ablation_error": err1,
        "experiment_summaries": summary_data if summary_data is not None else {},
        "experiment_summaries_available": summary_data is not None,
        "experiment_summaries_error": err2,
    })


# ---------------------------------------------------------------------------
# GET /api/research/privacy
# ---------------------------------------------------------------------------

@research_api_bp.route("/api/research/privacy", methods=["GET"])
def research_privacy():
    """
    Privacy metrics — DP epsilon/delta, PII detection performance.

    Extracts privacy fields from the B8 full pipeline summary and also
    provides a cross-configuration comparison using summary_metrics (E0-E9).

    Source: outputs/research/metrics/B8_summary.json
            outputs/research/metrics/summary_metrics.json
    Classification: HISTORICAL
    """
    b8, err1 = _load_json("b8_summary")
    summary, err2 = _load_json("summary_metrics")

    if err1:
        return _unavailable(err1)

    # Build cross-config DP/PII comparison from experiment summaries
    pii_comparison = []
    if summary:
        for exp_id, exp_data in summary.items():
            ps = exp_data.get("privacy_summary", {})
            pii_f1 = ps.get("pii_f1")
            eps = ps.get("epsilon")
            # Only include rows where PII or DP data is present
            if pii_f1 is not None or eps is not None:
                pii_comparison.append({
                    "baseline_id": exp_id,
                    "baseline_name": exp_data.get("baseline_name"),
                    "epsilon": eps,
                    "delta": ps.get("delta"),
                    "pii_precision": ps.get("pii_precision"),
                    "pii_recall": ps.get("pii_recall"),
                    "pii_f1": pii_f1,
                })

    return jsonify({
        "available": True,
        "classification": "HISTORICAL",
        "full_pipeline_privacy": b8.get("privacy_summary", {}),
        "cross_configuration_comparison": pii_comparison,
        "cross_comparison_available": len(pii_comparison) > 0,
    })


# ---------------------------------------------------------------------------
# GET /api/research/screening
# ---------------------------------------------------------------------------

@research_api_bp.route("/api/research/screening", methods=["GET"])
def research_screening():
    """
    Adapter security screening research results.

    Returns: TP/FP/TN/FN confusion matrix, precision, recall, F1, FPR, FNR,
             mean latency per adapter, and ROC-AUC.

    Source: outputs/research/adapter_screening/metrics.json
    Classification: HISTORICAL
    """
    data, err = _load_json("screening_metrics")
    if err:
        return _unavailable(err)

    return jsonify({
        "available": True,
        "classification": "HISTORICAL",
        "source": "outputs/research/adapter_screening/metrics.json",
        "confusion_matrix": {
            "true_positives":  data.get("true_positives"),
            "false_positives": data.get("false_positives"),
            "true_negatives":  data.get("true_negatives"),
            "false_negatives": data.get("false_negatives"),
            "total_test_samples": data.get("total_test_samples"),
        },
        "detection_metrics": {
            "precision":           data.get("precision"),
            "recall":              data.get("recall"),
            "f1_score":            data.get("f1_score"),
            "false_positive_rate": data.get("false_positive_rate"),
            "false_negative_rate": data.get("false_negative_rate"),
            "roc_auc":             data.get("roc_auc"),
        },
        "overhead": {
            "mean_latency_ms": data.get("mean_latency_ms"),
        },
    })


# ---------------------------------------------------------------------------
# GET /api/research/adaptive-evasion
# ---------------------------------------------------------------------------

@research_api_bp.route("/api/research/adaptive-evasion", methods=["GET"])
def research_adaptive_evasion():
    """
    Multi-seed adaptive adversarial evasion benchmark results.

    Returns:
      - metadata (git SHA, seeds, selected threshold)
      - category_summary (CLEAN / BASIC_SUSPICIOUS / ADAPTIVE_SUSPICIOUS)
      - level_summary (Level 0–3 structural distance, detection rate, FNR)
      - ablations (S0–S4 precision/recall/F1/latency)
      - test_threshold_grid (threshold vs precision/recall/F1)
      - seed_stats (mean ± std across seeds)
      - hypotheses (H1–H5 verdicts)

    Source: outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json
    Classification: HISTORICAL
    """
    data, err = _load_json("evasion_metrics")
    if err:
        return _unavailable(err)

    # Strip environment block (may contain system-level info we don't need to expose)
    metadata = data.get("metadata", {})
    safe_metadata = {
        "experiment_id":        metadata.get("experiment_id"),
        "git_commit_sha":       metadata.get("git_commit_sha"),
        "timestamp_utc":        metadata.get("timestamp_utc"),
        "seeds_evaluated":      metadata.get("seeds_evaluated"),
        "selected_threshold_val": metadata.get("selected_threshold_val"),
        "sample_counts":        metadata.get("sample_counts"),
    }

    return jsonify({
        "available": True,
        "classification": "HISTORICAL",
        "source": "outputs/research/adaptive_evasion/metrics/adaptive_evasion_metrics.json",
        "metadata": safe_metadata,
        "category_summary": data.get("category_summary", {}),
        "level_summary": data.get("level_summary", {}),
        "ablations": data.get("ablations", {}),
        "test_threshold_grid": data.get("test_threshold_grid", []),
        "seed_stats": data.get("seed_stats", {}),
        "hypotheses": data.get("hypotheses", {}),
    })


# ---------------------------------------------------------------------------
# GET /api/research/overhead
# ---------------------------------------------------------------------------

@research_api_bp.route("/api/research/overhead", methods=["GET"])
def research_overhead():
    """
    Cryptographic and system overhead breakdown.

    Returns all timing and storage metrics from the full pipeline (B8) experiment
    plus a cross-configuration comparison from experiment summaries.

    Source: outputs/research/metrics/B8_summary.json
            outputs/research/metrics/summary_metrics.json
    Classification: HISTORICAL
    """
    b8, err1 = _load_json("b8_summary")
    summary, err2 = _load_json("summary_metrics")

    if err1:
        return _unavailable(err1)

    # Cross-config overhead comparison
    overhead_comparison = []
    if summary:
        for exp_id, exp_data in summary.items():
            oh = exp_data.get("overhead_summary", {})
            if oh:
                overhead_comparison.append({
                    "baseline_id":       exp_id,
                    "baseline_name":     exp_data.get("baseline_name"),
                    "overhead_summary":  oh,
                })

    return jsonify({
        "available": True,
        "classification": "HISTORICAL",
        "full_pipeline_overhead": b8.get("overhead_summary", {}),
        "cross_configuration_comparison": overhead_comparison,
        "cross_comparison_available": len(overhead_comparison) > 0,
    })
