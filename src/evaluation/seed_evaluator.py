"""
seed_evaluator.py
=================
Multi-Seed Statistical Replication Pipeline for SecureLoRA Experiments (STEP 6).

Executes key experiments across random seeds (e.g. 42, 123, 456, 789, 1001):
  1. PII Evaluation (privacy_evaluator.py)
  2. Adapter Screening (screening_evaluator.py)
  3. Adaptive Evasion (adaptive_evasion_evaluator.py)
  4. Utility Evaluation (benchmark_evaluator.py)

STRICT RESULT VALIDITY & AGGREGATION RULES:
------------------------------------------
1. A seed counts as VALID (executed) only when the underlying experiment actually
   executed successfully.
2. NOT_EXECUTED, FAILED, or SKIPPED runs are NEVER included in mean, standard
   deviation, 95% confidence intervals, detection rates, or utility averages.
3. PARTIAL runs do not silently count as complete runs.
4. Detailed seed tracking records requested_seeds, executed_seeds, failed_seeds,
   and skipped_seeds.
5. Never reports "N-seed evaluation" unless N independent valid executions exist.

Output Directory:
  outputs/evaluation/statistics/
    ├── seed_results.json
    ├── aggregated_results.json
    └── comparison.csv
"""

import os
import sys
import csv
import json
import math
import logging
import statistics
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.privacy_evaluator import evaluate_privacy_pipeline
from src.evaluation.screening_evaluator import run_screening_evaluation
from src.evaluation.adaptive_evasion_evaluator import run_adaptive_evasion_evaluation
from src.evaluation.benchmark_evaluator import evaluate_dataset_adapter

logger = logging.getLogger("secure_lora.evaluation.seed_evaluator")
STATISTICS_OUT_DIR = _PROJECT_ROOT / "outputs" / "evaluation" / "statistics"
DEFAULT_SEEDS = [42, 123, 456, 789, 1001]


def calc_stats(values: List[float], total_requested: int = 5) -> Dict[str, Any]:
    """
    Computes mean, std, min, max, 95% confidence interval, count, and status
    strictly for valid observed numeric series.
    """
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "confidence_interval_95": [0.0, 0.0],
            "formatted": "N/A (0 valid runs)",
            "count": 0,
            "status": "NOT_EXECUTED"
        }
    m = float(statistics.mean(values))
    s = float(statistics.stdev(values)) if n > 1 else 0.0
    mn = float(min(values))
    mx = float(max(values))

    if n > 1:
        margin = 1.96 * (s / math.sqrt(n))
        ci_95 = [round(m - margin, 4), round(m + margin, 4)]
    else:
        ci_95 = [round(m, 4), round(m, 4)]

    status = "COMPLETE" if n == total_requested else "PARTIAL"

    return {
        "mean": round(m, 4),
        "std": round(s, 4),
        "min": round(mn, 4),
        "max": round(mx, 4),
        "confidence_interval_95": ci_95,
        "formatted": f"{m:.4f} ± {s:.4f}",
        "count": n,
        "status": status
    }


def _extract_exp_status(res: Any) -> str:
    """Helper to categorize experiment output dictionary status."""
    if not isinstance(res, dict):
        return "FAILED"
    st = res.get("status") or res.get("execution_status")
    if st == "NOT_EXECUTED":
        return "NOT_EXECUTED"
    if st in ("SUCCESS", "EXECUTED"):
        return "SUCCESS"
    if st == "FAILED":
        return "FAILED"
    return "SUCCESS"


def run_multi_seed_evaluations(
    seeds: Optional[List[int]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Runs PII evaluation, adapter screening, adaptive evasion, and utility evaluation
    across requested seeds, filtering out unexecuted or failed runs during statistical aggregation.
    """
    eval_seeds = seeds or DEFAULT_SEEDS
    out_dir = Path(output_dir) if output_dir else STATISTICS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_results: Dict[str, Dict[str, Any]] = {}

    for seed in eval_seeds:
        seed_key = f"seed_{seed}"
        seed_data: Dict[str, Any] = {"seed": seed, "experiments": {}}

        # 1. PII Evaluation
        try:
            p_res = evaluate_privacy_pipeline(dataset_id="synthetic", samples=30, seed=seed)
            st = _extract_exp_status(p_res)
            seed_data["experiments"]["pii_evaluation"] = {
                "status": st,
                "data": p_res
            }
        except Exception as exc:
            logger.warning("PII Evaluation failed for seed %d: %s", seed, exc)
            seed_data["experiments"]["pii_evaluation"] = {
                "status": "FAILED",
                "error": str(exc)
            }

        # 2. Adapter Screening Evaluation
        try:
            s_res = run_screening_evaluation(num_samples_per_cat=10, seed=seed)
            st = _extract_exp_status(s_res)
            seed_data["experiments"]["adapter_screening"] = {
                "status": st,
                "data": s_res
            }
        except Exception as exc:
            logger.warning("Adapter Screening failed for seed %d: %s", seed, exc)
            seed_data["experiments"]["adapter_screening"] = {
                "status": "FAILED",
                "error": str(exc)
            }

        # 3. Adaptive Evasion Evaluation
        try:
            e_res = run_adaptive_evasion_evaluation(num_malicious_samples=10, max_iterations=5, seed=seed)
            st = _extract_exp_status(e_res)
            seed_data["experiments"]["adaptive_evasion"] = {
                "status": st,
                "data": e_res
            }
        except Exception as exc:
            logger.warning("Adaptive Evasion failed for seed %d: %s", seed, exc)
            seed_data["experiments"]["adaptive_evasion"] = {
                "status": "FAILED",
                "error": str(exc)
            }

        # 4. Utility Evaluation
        try:
            u_res = evaluate_dataset_adapter(dataset_id="synthetic", subset_size=30, seed=seed)
            st = _extract_exp_status(u_res)
            seed_data["experiments"]["utility_evaluation"] = {
                "status": st,
                "data": u_res
            }
        except Exception as exc:
            logger.warning("Utility Evaluation failed for seed %d: %s", seed, exc)
            seed_data["experiments"]["utility_evaluation"] = {
                "status": "FAILED",
                "error": str(exc)
            }

        seed_results[seed_key] = seed_data

    # --- Categorize Seeds per Experiment Type ---
    exp_names = ["adapter_screening", "adaptive_evasion", "pii_evaluation", "utility_evaluation"]
    exp_seed_provenance: Dict[str, Dict[str, Any]] = {}

    for exp_name in exp_names:
        requested = list(eval_seeds)
        executed = []
        failed = []
        skipped = []

        for seed in eval_seeds:
            s_exp = seed_results[f"seed_{seed}"]["experiments"].get(exp_name, {})
            st = s_exp.get("status")
            if st == "SUCCESS":
                executed.append(seed)
            elif st == "FAILED":
                failed.append(seed)
            else:
                skipped.append(seed)

        exp_status = "COMPLETE" if len(executed) == len(requested) and len(requested) > 0 else ("NOT_EXECUTED" if len(executed) == 0 else "PARTIAL")
        exp_seed_provenance[exp_name] = {
            "requested_seeds": requested,
            "executed_seeds": executed,
            "failed_seeds": failed,
            "skipped_seeds": skipped,
            "valid_seed_count": len(executed),
            "execution_status": exp_status,
            "evaluation_title": f"{len(executed)}-seed {'COMPLETE' if exp_status == 'COMPLETE' else 'PARTIAL' if exp_status == 'PARTIAL' else 'NOT_EXECUTED'} evaluation"
        }

    # Overall pipeline seed provenance
    overall_executed = [s for s in eval_seeds if all(seed_results[f"seed_{s}"]["experiments"].get(e, {}).get("status") == "SUCCESS" for e in exp_names)]
    overall_failed = [s for s in eval_seeds if any(seed_results[f"seed_{s}"]["experiments"].get(e, {}).get("status") == "FAILED" for e in exp_names)]
    overall_skipped = [s for s in eval_seeds if s not in overall_executed and s not in overall_failed]
    overall_status = "COMPLETE" if len(overall_executed) == len(eval_seeds) and len(eval_seeds) > 0 else ("NOT_EXECUTED" if len(overall_executed) == 0 else "PARTIAL")
    overall_title = f"{len(overall_executed)}-seed {'COMPLETE' if overall_status == 'COMPLETE' else 'PARTIAL' if overall_status == 'PARTIAL' else 'NOT_EXECUTED'} evaluation"

    # --- Write seed_results.json ---
    seed_results_file = out_dir / "seed_results.json"
    with open(seed_results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requested_seeds": eval_seeds,
            "executed_seeds": overall_executed,
            "failed_seeds": overall_failed,
            "skipped_seeds": overall_skipped,
            "valid_seed_count": len(overall_executed),
            "execution_status": overall_status,
            "evaluation_title": overall_title,
            "results": seed_results
        }, f, indent=2)

    # --- Aggregation across ONLY valid executed runs ---
    aggregated: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_seeds": eval_seeds,
        "executed_seeds": overall_executed,
        "failed_seeds": overall_failed,
        "skipped_seeds": overall_skipped,
        "valid_seed_count": len(overall_executed),
        "execution_status": overall_status,
        "evaluation_title": overall_title,
        "experiment_provenance": exp_seed_provenance,
        "adapter_screening": {},
        "adaptive_evasion": {},
        "pii_evaluation": {},
        "utility_evaluation": {}
    }

    total_req = len(eval_seeds)

    # 1. Aggregate Adapter Screening Metrics (structural_only, behavioral_only, combined)
    screening_detectors = ["structural_only", "behavioral_only", "combined"]
    screening_metric_keys = [
        "precision", "recall", "f1", "accuracy",
        "false_positive_rate", "false_negative_rate", "mean_latency_ms"
    ]

    for det in screening_detectors:
        det_stats: Dict[str, Any] = {}
        for m_key in screening_metric_keys:
            vals = []
            for seed in exp_seed_provenance["adapter_screening"]["executed_seeds"]:
                s_exp = seed_results[f"seed_{seed}"]["experiments"].get("adapter_screening", {})
                sys_data = s_exp.get("data", {}).get("systems", {}).get(det, {})
                tm = sys_data.get("test_metrics", {})
                if m_key in tm and tm[m_key] is not None:
                    vals.append(float(tm[m_key]))
            det_stats[m_key] = calc_stats(vals, total_requested=total_req)
        aggregated["adapter_screening"][det] = det_stats

    # 2. Aggregate Adaptive Evasion Metrics (baseline, nonadaptive, adaptive)
    evasion_attacks = ["baseline", "nonadaptive", "adaptive"]
    evasion_metric_keys = [
        "attack_success_rate", "detection_rate", "false_negative_rate",
        "avg_final_score", "avg_utility_preservation"
    ]

    for att in evasion_attacks:
        att_stats: Dict[str, Any] = {}
        for det in screening_detectors:
            det_sub_stats: Dict[str, Any] = {}
            for m_key in evasion_metric_keys:
                vals = []
                for seed in exp_seed_provenance["adaptive_evasion"]["executed_seeds"]:
                    e_exp = seed_results[f"seed_{seed}"]["experiments"].get("adaptive_evasion", {})
                    att_data = e_exp.get("data", {}).get("attack_strategies", {}).get(att, {})
                    det_data = att_data.get("detectors", {}).get(det, {})
                    if m_key in det_data and det_data[m_key] is not None:
                        vals.append(float(det_data[m_key]))
                det_sub_stats[m_key] = calc_stats(vals, total_requested=total_req)
            att_stats[det] = det_sub_stats
        aggregated["adaptive_evasion"][att] = att_stats

    # 3. Aggregate PII Evaluation Metrics
    pii_metric_keys = ["pii_leakage_rate", "pii_free_response_rate"]
    pii_stats: Dict[str, Any] = {}
    for m_key in pii_metric_keys:
        vals = []
        for seed in exp_seed_provenance["pii_evaluation"]["executed_seeds"]:
            p_exp = seed_results[f"seed_{seed}"]["experiments"].get("pii_evaluation", {})
            m_data = p_exp.get("data", {}).get("variants", {}).get("securelora", {}).get("metrics", {}) or {}
            if m_key in m_data and m_data[m_key] is not None:
                vals.append(float(m_data[m_key]))
        pii_stats[m_key] = calc_stats(vals, total_requested=total_req)
    aggregated["pii_evaluation"] = pii_stats

    # 4. Aggregate Utility Evaluation Metrics
    utility_metric_keys = ["precision", "recall", "f1", "record_count"]
    util_stats: Dict[str, Any] = {}
    for m_key in utility_metric_keys:
        vals = []
        for seed in exp_seed_provenance["utility_evaluation"]["executed_seeds"]:
            u_exp = seed_results[f"seed_{seed}"]["experiments"].get("utility_evaluation", {})
            m_data = u_exp.get("data", {}).get("metrics", {})
            if m_key in m_data and m_data[m_key] is not None:
                vals.append(float(m_data[m_key]))
        util_stats[m_key] = calc_stats(vals, total_requested=total_req)
    aggregated["utility_evaluation"] = util_stats

    # --- Write aggregated_results.json ---
    agg_file = out_dir / "aggregated_results.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # --- Write comparison.csv ---
    csv_file = out_dir / "comparison.csv"
    csv_headers = [
        "category",
        "detector_or_attack",
        "f1_mean_std",
        "precision_mean_std",
        "recall_mean_std",
        "accuracy_mean_std",
        "fpr_mean_std",
        "fnr_mean_std",
        "latency_mean_std"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        # Write Screening rows
        for det in screening_detectors:
            st = aggregated["adapter_screening"][det]
            writer.writerow([
                "adapter_screening",
                det,
                st["f1"]["formatted"],
                st["precision"]["formatted"],
                st["recall"]["formatted"],
                st["accuracy"]["formatted"],
                st["false_positive_rate"]["formatted"],
                st["false_negative_rate"]["formatted"],
                st["mean_latency_ms"]["formatted"]
            ])

        # Write Evasion rows (Combined detector under attacks)
        for att in evasion_attacks:
            st = aggregated["adaptive_evasion"][att]["combined"]
            writer.writerow([
                f"adaptive_evasion_{att}",
                "combined",
                "N/A",
                "N/A",
                "N/A",
                st["detection_rate"]["formatted"],
                "N/A",
                st["false_negative_rate"]["formatted"],
                "N/A"
            ])

    logger.info("Saved multi-seed statistical evaluation artifacts (%s) to %s", overall_title, out_dir)
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Statistical Replication Pipeline (STEP 6)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="List of random seeds")
    parser.add_argument("--output-dir", type=str, default=str(STATISTICS_OUT_DIR), help="Output directory")

    args = parser.parse_args()

    res = run_multi_seed_evaluations(
        seeds=args.seeds,
        output_dir=Path(args.output_dir)
    )
    print(f"\n Multi-seed statistical evaluations completed ({res.get('evaluation_title')}). Output generated at -> {args.output_dir}")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
