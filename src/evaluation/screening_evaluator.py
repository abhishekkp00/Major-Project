"""
screening_evaluator.py
======================
Comparative Evaluation Pipeline for LoRA Adapter Security Screening (STEP 4).

Compares three experimentally aligned screening systems:
  System A: Structural-Only
  System B: Behavioral-Only
  System C: Structural + Behavioral (Combined SecureLoRA)

Evaluation Protocol:
  1. Identical benchmark dataset (clean vs malicious/adaptive adapters) generated via AdaptiveAdapterFactory.
  2. Identical train/val/test splits, random seeds (default 42), and evaluation protocol.
  3. Validation split used for threshold parameter selection (sweeping threshold tau in [0.10, 0.90]).
  4. Selected threshold applied to test split to compute unbiased test set performance metrics.

Output Directory:
  outputs/evaluation/screening/
    ├── structural_only.json
    ├── behavioral_only.json
    ├── combined.json
    ├── comparison.json
    └── screening_comparison.csv
"""

import os
import sys
import csv
import json
import time
import argparse
import logging
import statistics
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.security.adapter_screening.structural_analysis import StructuralAnalyzer
from src.security.adapter_screening.behavioral_analysis import BehavioralAnalyzer
from src.security.adapter_screening.risk_scoring import RiskScorer
from src.security.adapter_screening.adaptive_evasion import AdaptiveAdapterFactory, AdaptiveAdapterSample

logger = logging.getLogger("secure_lora.evaluation.screening_evaluator")
SCREENING_OUT_DIR = _PROJECT_ROOT / "outputs" / "evaluation" / "screening"


def _calculate_classification_metrics(
    y_true: List[bool],
    y_scores: List[float],
    threshold: float
) -> Dict[str, Any]:
    """Calculates TP, TN, FP, FN, precision, recall, F1, accuracy, FPR, FNR for a given score threshold."""
    tp = sum(1 for gt, s in zip(y_true, y_scores) if gt and s >= threshold)
    tn = sum(1 for gt, s in zip(y_true, y_scores) if not gt and s < threshold)
    fp = sum(1 for gt, s in zip(y_true, y_scores) if not gt and s >= threshold)
    fn = sum(1 for gt, s in zip(y_true, y_scores) if gt and s < threshold)

    total = len(y_true)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr
    }


def evaluate_threshold_sweep(
    y_true: List[bool],
    y_scores: List[float],
    thresholds: List[float]
) -> List[Dict[str, Any]]:
    """Evaluates metrics across a grid of candidate threshold values."""
    sweep = []
    for tau in thresholds:
        m = _calculate_classification_metrics(y_true, y_scores, tau)
        sweep.append({
            "threshold": m["threshold"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "false_positive_rate": m["false_positive_rate"],
            "false_negative_rate": m["false_negative_rate"]
        })
    return sweep


def run_screening_evaluation(
    num_samples_per_cat: int = 15,
    seed: int = 42,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes fair, comparative screening evaluation across Systems A, B, and C.
    """
    out_dir = Path(output_dir) if output_dir else SCREENING_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = AdaptiveAdapterFactory()
    trusted = factory.generate_clean_adapter(seed=seed)

    # 1. Build validation and test suites
    val_samples = factory.build_benchmark_suite(num_samples_per_cat=max(5, num_samples_per_cat // 2), seed=seed, split="val")
    test_samples = factory.build_benchmark_suite(num_samples_per_cat=num_samples_per_cat, seed=seed, split="test")

    struct_analyzer = StructuralAnalyzer()
    beh_analyzer = BehavioralAnalyzer()
    risk_scorer = RiskScorer()

    systems = ["structural_only", "behavioral_only", "combined"]
    system_data: Dict[str, Dict[str, Any]] = {}

    # Candidate thresholds to evaluate on validation set
    candidate_thresholds = [round(t * 0.05, 2) for t in range(2, 19)]  # 0.10 to 0.90 in steps of 0.05

    for sys_name in systems:
        # --- Validation Pass ---
        val_y_true = [s.ground_truth_anomalous for s in val_samples]
        val_scores = []

        for sample in val_samples:
            w = sample.weights
            s_ev = struct_analyzer.analyze(weights=w, trusted_weights=trusted)
            b_ev = beh_analyzer.evaluate(candidate_model_or_fn=w, seed=seed)
            r_ev = risk_scorer.evaluate(s_ev, b_ev)

            if sys_name == "structural_only":
                val_scores.append(r_ev.structural_score)
            elif sys_name == "behavioral_only":
                val_scores.append(r_ev.behavioral_score)
            else:  # combined
                val_scores.append(r_ev.adapter_risk_score)

        # Threshold analysis on validation data
        val_sweep = evaluate_threshold_sweep(val_y_true, val_scores, candidate_thresholds)

        # Select optimal threshold on validation data (maximizing F1, minimizing FPR)
        best_val = max(val_sweep, key=lambda x: (x["f1"], -x["false_positive_rate"]))
        selected_threshold = best_val["threshold"]

        # --- Test Pass ---
        test_y_true = [s.ground_truth_anomalous for s in test_samples]
        test_scores = []
        latencies = []

        for sample in test_samples:
            w = sample.weights
            t0 = time.perf_counter()

            s_ev = struct_analyzer.analyze(weights=w, trusted_weights=trusted)
            b_ev = beh_analyzer.evaluate(candidate_model_or_fn=w, seed=seed)
            r_ev = risk_scorer.evaluate(s_ev, b_ev)

            if sys_name == "structural_only":
                score = r_ev.structural_score
            elif sys_name == "behavioral_only":
                score = r_ev.behavioral_score
            else:  # combined
                score = r_ev.adapter_risk_score

            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)
            test_scores.append(score)

        # Test metrics using optimal validation threshold
        test_metrics = _calculate_classification_metrics(test_y_true, test_scores, selected_threshold)
        test_metrics["mean_latency_ms"] = round(statistics.mean(latencies), 3)
        test_metrics["stdev_latency_ms"] = round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0

        now_iso = datetime.now(timezone.utc).isoformat()

        report_data = {
            "system": sys_name,
            "description": f"LoRA adapter security screening detector: {sys_name}",
            "sample_count": len(test_samples),
            "seed": seed,
            "timestamp": now_iso,
            "validation_threshold_sweep": val_sweep,
            "selected_threshold": selected_threshold,
            "threshold_selection_criterion": "Maximum F1 score on validation split",
            "test_metrics": test_metrics
        }

        # Write individual json
        sys_file = out_dir / f"{sys_name}.json"
        with open(sys_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        system_data[sys_name] = report_data

    # --- Write comparison.json ---
    comparison_file = out_dir / "comparison.json"
    comp_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(test_samples),
        "seed": seed,
        "systems": system_data,
        "summary": {
            "best_f1_system": max(system_data.keys(), key=lambda k: system_data[k]["test_metrics"]["f1"]),
            "best_fpr_system": min(system_data.keys(), key=lambda k: system_data[k]["test_metrics"]["false_positive_rate"]),
        }
    }
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)

    # --- Write CSV file (screening_comparison.csv) ---
    csv_file = out_dir / "screening_comparison.csv"
    csv_headers = [
        "system",
        "selected_threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "mean_latency_ms",
        "tp",
        "tn",
        "fp",
        "fn"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for sys_name, data in system_data.items():
            tm = data["test_metrics"]
            writer.writerow([
                sys_name,
                data["selected_threshold"],
                tm["precision"],
                tm["recall"],
                tm["f1"],
                tm["accuracy"],
                tm["false_positive_rate"],
                tm["false_negative_rate"],
                tm["mean_latency_ms"],
                tm["tp"],
                tm["tn"],
                tm["fp"],
                tm["fn"]
            ])

    logger.info("Saved screening comparison files to %s", out_dir)
    return comp_data


def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter Security Screening Comparison Pipeline (STEP 4)")
    parser.add_argument("--samples-per-cat", type=int, default=15, help="Number of benchmark samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=str(SCREENING_OUT_DIR), help="Output directory")

    args = parser.parse_args()

    res = run_screening_evaluation(
        num_samples_per_cat=args.samples_per_cat,
        seed=args.seed,
        output_dir=Path(args.output_dir)
    )
    print(f"\n✅ Security screening comparison completed. Output generated at -> {args.output_dir}")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
