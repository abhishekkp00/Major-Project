"""
adaptive_evasion_evaluator.py
=============================
Adaptive Adversarial Evasion Evaluation Pipeline for SecureLoRA (STEP 5).

Research Question:
  "How robust is the SecureLoRA adapter screening mechanism when an attacker
   knows the screening strategy and actively optimizes an adapter to evade detection?"

Attacker Model:
  1. Baseline Attack (random_perturbation): Random noise injection.
  2. Non-Adaptive Attack (non_adaptive): Uniform magnitude/norm scaling.
  3. Adaptive Attack (adaptive): Iterative projection/optimization targeting detector score
     subject to perturbation budget & utility constraints.

Evaluates against:
  - Structural-Only Detector
  - Behavioral-Only Detector
  - Combined SecureLoRA Detector

Output Directory:
  outputs/evaluation/adaptive_evasion/
    ├── baseline_attack.json
    ├── nonadaptive_attack.json
    ├── adaptive_attack.json
    ├── comparison.json
    └── iteration_history.csv
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

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.security.adapter_screening.structural_analysis import StructuralAnalyzer
from src.security.adapter_screening.behavioral_analysis import BehavioralAnalyzer
from src.security.adapter_screening.risk_scoring import RiskScorer
from src.security.adapter_screening.adaptive_evasion import AdaptiveAdapterFactory, AdaptiveAdapterSample

logger = logging.getLogger("secure_lora.evaluation.adaptive_evasion_evaluator")
EVASION_OUT_DIR = _PROJECT_ROOT / "outputs" / "evaluation" / "adaptive_evasion"


def compute_relative_perturbation(
    W_current: Dict[str, np.ndarray],
    W_initial: Dict[str, np.ndarray]
) -> float:
    """Computes relative Frobenius norm difference ||W_current - W_initial||_F / ||W_initial||_F."""
    diff_norms = []
    init_norms = []
    for k in W_initial:
        if k in W_current:
            diff = W_current[k] - W_initial[k]
            diff_norms.append(np.linalg.norm(diff) ** 2)
            init_norms.append(np.linalg.norm(W_initial[k]) ** 2)
    tot_diff = np.sqrt(sum(diff_norms))
    tot_init = np.sqrt(sum(init_norms))
    return float(tot_diff / (tot_init + 1e-8))


def estimate_payload_utility(
    W_current: Dict[str, np.ndarray],
    W_initial: Dict[str, np.ndarray]
) -> float:
    """Estimates payload utility retention [0.0, 1.0] relative to initial malicious weights."""
    cos_sims = []
    for k in W_initial:
        if k in W_current:
            c = W_current[k].flatten()
            i = W_initial[k].flatten()
            nc = np.linalg.norm(c)
            ni = np.linalg.norm(i)
            if nc > 1e-8 and ni > 1e-8:
                cos_sims.append(float(np.dot(c, i) / (nc * ni)))
    return round(float(np.mean(cos_sims)), 4) if cos_sims else 1.0


def run_adaptive_evasion_evaluation(
    num_malicious_samples: int = 20,
    max_iterations: int = 10,
    seed: int = 42,
    threshold: float = 0.35,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes adversarial evasion evaluation across 3 attack types and 3 detector types.
    """
    out_dir = Path(output_dir) if output_dir else EVASION_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = AdaptiveAdapterFactory()
    trusted = factory.generate_clean_adapter(seed=seed)

    # Generate benchmark suite of malicious/anomalous adapters (basic + adaptive levels)
    full_suite = factory.build_benchmark_suite(num_samples_per_cat=num_malicious_samples, seed=seed, split="test")
    malicious_samples = [s for s in full_suite if s.ground_truth_anomalous]

    struct_analyzer = StructuralAnalyzer()
    beh_analyzer = BehavioralAnalyzer()
    risk_scorer = RiskScorer()

    attack_types = ["random_perturbation", "non_adaptive", "adaptive"]
    detector_types = ["structural_only", "behavioral_only", "combined"]

    history_rows = []
    attack_results: Dict[str, Dict[str, Any]] = {}
    all_evasion_failures = []

    for attack_type in attack_types:
        detector_reports: Dict[str, Dict[str, Any]] = {}

        for detector_type in detector_types:
            sample_evals = []
            detector_failures = []

            for sample_idx, sample in enumerate(malicious_samples):
                W_init = {k: v.copy() for k, v in sample.weights.items()}
                W_curr = {k: v.copy() for k, v in W_init.items()}

                # Initial evaluation at iteration 0
                s_ev0 = struct_analyzer.analyze(weights=W_curr, trusted_weights=trusted)
                b_ev0 = beh_analyzer.evaluate(candidate_model_or_fn=W_curr, seed=seed)
                r_ev0 = risk_scorer.evaluate(s_ev0, b_ev0)

                init_score = (
                    r_ev0.structural_score if detector_type == "structural_only"
                    else (r_ev0.behavioral_score if detector_type == "behavioral_only" else r_ev0.adapter_risk_score)
                )

                rng = np.random.RandomState(seed + sample_idx * 100)

                # Iterative attack loop
                iteration_scores = []
                best_score = init_score

                for it in range(max_iterations + 1):
                    t0 = time.perf_counter()

                    if it > 0:
                        if attack_type == "random_perturbation":
                            # Random noise perturbation
                            sigma = 0.001 * it
                            W_curr = {k: v + rng.normal(0.0, sigma, v.shape).astype(np.float32) for k, v in W_init.items()}

                        elif attack_type == "non_adaptive":
                            # Uniform scaling reduction
                            scale_factor = max(0.40, 1.0 - (0.06 * it))
                            W_curr = {k: v * scale_factor for k, v in W_init.items()}

                        elif attack_type == "adaptive":
                            # Active iterative optimization / projection against detector score
                            temp_W = {k: v.copy() for k, v in W_curr.items()}

                            # 1. Project outlier layer Z-scores towards trusted mean
                            s_temp = struct_analyzer.analyze(weights=temp_W, trusted_weights=trusted)
                            if s_temp.outlier_layers:
                                for out_k in s_temp.outlier_layers:
                                    if out_k in temp_W:
                                        temp_W[out_k] = temp_W[out_k] * 0.75

                            # 2. Add bounded rank-1 noise projection to minimize trigger sensitivity
                            for k in temp_W:
                                if "lora_A" in k:
                                    u = rng.normal(0.0, 0.002, (temp_W[k].shape[0], 1)).astype(np.float32)
                                    v = rng.normal(0.0, 0.002, (1, temp_W[k].shape[1])).astype(np.float32)
                                    temp_W[k] += u @ v

                            # Check score under temp update
                            s_chk = struct_analyzer.analyze(weights=temp_W, trusted_weights=trusted)
                            b_chk = beh_analyzer.evaluate(candidate_model_or_fn=temp_W, seed=seed)
                            r_chk = risk_scorer.evaluate(s_chk, b_chk)
                            chk_score = (
                                r_chk.structural_score if detector_type == "structural_only"
                                else (r_chk.behavioral_score if detector_type == "behavioral_only" else r_chk.adapter_risk_score)
                            )

                            util_chk = estimate_payload_utility(temp_W, W_init)

                            # Accept if score reduced and utility preserved
                            if chk_score < best_score and util_chk >= 0.50:
                                best_score = chk_score
                                W_curr = temp_W

                    # Screen modified adapter
                    s_ev = struct_analyzer.analyze(weights=W_curr, trusted_weights=trusted)
                    b_ev = beh_analyzer.evaluate(candidate_model_or_fn=W_curr, seed=seed)
                    r_ev = risk_scorer.evaluate(s_ev, b_ev)
                    lat_ms = (time.perf_counter() - t0) * 1000.0

                    curr_score = (
                        r_ev.structural_score if detector_type == "structural_only"
                        else (r_ev.behavioral_score if detector_type == "behavioral_only" else r_ev.adapter_risk_score)
                    )

                    util = estimate_payload_utility(W_curr, W_init)
                    pert = compute_relative_perturbation(W_curr, W_init)

                    is_detected = curr_score >= threshold
                    decision = "REJECTED" if is_detected else "APPROVED"
                    attack_successful = (not is_detected) and (util >= 0.50)

                    # Record trajectory row
                    history_rows.append([
                        sample.sample_id,
                        attack_type,
                        detector_type,
                        it,
                        round(r_ev.structural_score, 4),
                        round(r_ev.behavioral_score, 4),
                        round(r_ev.adapter_risk_score, 4),
                        round(curr_score, 4),
                        decision,
                        1 if is_detected else 0,
                        1 if attack_successful else 0,
                        round(util, 4),
                        round(pert, 4),
                        round(lat_ms, 3)
                    ])

                    iteration_scores.append({
                        "iteration": it,
                        "score": round(curr_score, 4),
                        "utility": util,
                        "perturbation": pert,
                        "decision": decision
                    })

                # Final state for this sample
                final_it_data = iteration_scores[-1]
                final_score = final_it_data["score"]
                final_util = final_it_data["utility"]
                final_pert = final_it_data["perturbation"]
                final_decision = final_it_data["decision"]
                evaded = final_decision == "APPROVED" and final_util >= 0.50

                sample_res = {
                    "sample_id": sample.sample_id,
                    "evasion_level": sample.evasion_level,
                    "initial_score": round(init_score, 4),
                    "final_score": final_score,
                    "utility_preservation": final_util,
                    "perturbation_magnitude": final_pert,
                    "detector_decision": final_decision,
                    "attack_successful": evaded,
                    "trajectory": iteration_scores
                }
                sample_evals.append(sample_res)

                if evaded:
                    fail_entry = {
                        "sample_id": sample.sample_id,
                        "attack_type": attack_type,
                        "detector_type": detector_type,
                        "evasion_level": sample.evasion_level,
                        "initial_score": round(init_score, 4),
                        "final_score": final_score,
                        "utility_preservation": final_util,
                        "perturbation_magnitude": final_pert,
                        "detector_decision": final_decision
                    }
                    detector_failures.append(fail_entry)
                    all_evasion_failures.append(fail_entry)

            # Aggregate stats across samples for this (attack, detector) pair
            total_s = len(sample_evals)
            successful_attacks = sum(1 for s in sample_evals if s["attack_successful"])
            attack_success_rate = round(successful_attacks / total_s, 4) if total_s > 0 else 0.0
            detection_rate = round(1.0 - attack_success_rate, 4)
            fnr = attack_success_rate

            avg_struct = round(statistics.mean([s["trajectory"][-1]["score"] for s in sample_evals]), 4)
            avg_util = round(statistics.mean([s["utility_preservation"] for s in sample_evals]), 4)
            avg_pert = round(statistics.mean([s["perturbation_magnitude"] for s in sample_evals]), 4)

            det_report = {
                "detector_type": detector_type,
                "threshold": threshold,
                "total_samples": total_s,
                "attack_success_rate": attack_success_rate,
                "detection_rate": detection_rate,
                "false_negative_rate": fnr,
                "avg_final_score": avg_struct,
                "avg_utility_preservation": avg_util,
                "avg_perturbation_magnitude": avg_pert,
                "failures_count": len(detector_failures),
                "failures": detector_failures
            }
            detector_reports[detector_type] = det_report

        attack_key = "baseline" if attack_type == "random_perturbation" else ("nonadaptive" if attack_type == "non_adaptive" else "adaptive")
        attack_report = {
            "attack_type": attack_type,
            "seed": seed,
            "max_iterations": max_iterations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detectors": detector_reports
        }
        attack_results[attack_key] = attack_report

        # Write individual attack JSON
        att_file = out_dir / f"{attack_key}_attack.json"
        with open(att_file, "w", encoding="utf-8") as f:
            json.dump(attack_report, f, indent=2)

    # --- Write comparison.json ---
    comp_file = out_dir / "comparison.json"
    comp_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "max_iterations": max_iterations,
        "attack_strategies": attack_results,
        "total_evasion_failures": len(all_evasion_failures),
        "evasion_failures": all_evasion_failures,
        "summary_findings": {
            "adaptive_attack_successful": len(all_evasion_failures) > 0,
            "failure_analysis_note": (
                "Successful evasion attacks demonstrate bounds of single-perspective "
                "or joint screening filters under whitebox/greybox optimization."
            )
        }
    }
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)

    # --- Write iteration_history.csv ---
    csv_file = out_dir / "iteration_history.csv"
    csv_headers = [
        "sample_id",
        "attack_type",
        "detector_type",
        "iteration",
        "structural_score",
        "behavioral_score",
        "combined_score",
        "detector_score",
        "detector_decision",
        "is_detected",
        "attack_successful",
        "utility_preservation",
        "perturbation_magnitude",
        "latency_ms"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for row in history_rows:
            writer.writerow(row)

    logger.info("Saved adaptive evasion evaluation reports to %s", out_dir)
    return comp_data


def main():
    parser = argparse.ArgumentParser(description="Adaptive Adversarial Evasion Evaluation (STEP 5)")
    parser.add_argument("--samples", type=int, default=20, help="Number of malicious samples to evaluate")
    parser.add_argument("--iterations", type=int, default=10, help="Max attack iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.35, help="Screening risk threshold")
    parser.add_argument("--output-dir", type=str, default=str(EVASION_OUT_DIR), help="Output directory")

    args = parser.parse_args()

    res = run_adaptive_evasion_evaluation(
        num_malicious_samples=args.samples,
        max_iterations=args.iterations,
        seed=args.seed,
        threshold=args.threshold,
        output_dir=Path(args.output_dir)
    )
    print(f"\n✅ Adaptive evasion evaluation completed. Output generated at -> {args.output_dir}")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
