"""
run_research_ablations.py
==========================
Master Reproducible Research CLI Runner for the SecureLoRA Experimentation Framework.

Executes standardized experiment configurations E0 through E9 across multiple random seeds,
computes statistical metrics (mean, stdev, 95% CIs), computes ablation deltas, and generates:
  - Structured JSON runs in outputs/research/runs/
  - Raw aggregated data in outputs/research/raw/raw_experiments.json
  - Aggregated metric summaries in outputs/research/metrics/
  - 5 Standardized CSV & Markdown Tables in outputs/research/tables/
  - 5 Publication-grade PNG Figures in outputs/research/figures/
  - Research evaluation reports in outputs/research/summaries/

Usage Examples:
    # Run complete experiment suite (E0 to E9) across 3 seeds:
    python scripts/run_research_ablations.py --all

    # Run specific experiments (e.g. E1 and E4):
    python scripts/run_research_ablations.py --experiment E1 E4 --seeds 42 43 44

    # Fast verification / smoke test:
    python scripts/run_research_ablations.py --all --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment_runner import run_experiment_matrix, EXPERIMENTS_DEFINITION, normalize_experiment_id
from src.evaluation.ablation_study import run_ablation_analysis
from src.evaluation.report_generator import generate_all_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_research_ablations")


def main():
    parser = argparse.ArgumentParser(
        description="SecureLoRA Reproducible Research Ablation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiment configurations (E0 through E9)",
    )
    parser.add_argument(
        "--experiment",
        nargs="+",
        type=str,
        default=None,
        help="Specific experiment configuration(s) to execute (e.g. --experiment E1 E4 E9 or B1 B7)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds for statistical evaluation (default: 42 43 44)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/research",
        help="Root directory for research outputs (default: outputs/research)",
    )
    parser.add_argument(
        "--quick",
        "--smoke-test",
        action="store_true",
        dest="quick",
        help="Run in quick mode for rapid validation and testing",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine targeted experiment IDs
    if args.all or not args.experiment:
        target_experiments = list(EXPERIMENTS_DEFINITION.keys())
    else:
        target_experiments = [normalize_experiment_id(e) for e in args.experiment]

    print("\n" + "=" * 85)
    print("  SECURE LORA: REPRODUCIBLE RESEARCH ABLATION FRAMEWORK")
    print(f"  Target Experiments : {', '.join(target_experiments)}")
    print(f"  Random Seeds       : {args.seeds}")
    print(f"  Output Directory   : {out_dir}")
    print(f"  Quick Mode         : {args.quick}")
    print("=" * 85 + "\n")

    # 1. Execute Experiment Matrix
    aggregated_results = run_experiment_matrix(
        seeds=args.seeds,
        output_dir=out_dir,
        experiment_ids=target_experiments,
        quick_mode=args.quick,
    )

    # 2. Execute Ablation Analysis
    ablation_impacts = run_ablation_analysis(aggregated_results=aggregated_results, output_dir=out_dir)

    # 3. Generate Reports, CSVs, Markdown Tables, Figures
    generate_all_reports(aggregated_results=aggregated_results, ablation_impacts=ablation_impacts, output_dir=out_dir)

    print("\n" + "=" * 85)
    print("  ✅ REPRODUCIBLE RESEARCH ABLATION STUDY COMPLETE!")
    print(f"  📄 Research Report  : {out_dir / 'summaries' / 'RESEARCH_EVALUATION_REPORT.md'}")
    print(f"  📊 Markdown Tables   : {out_dir / 'tables'}")
    print(f"  🖼️  PNG Figures      : {out_dir / 'figures'}")
    print(f"  📁 Raw Runs Data    : {out_dir / 'runs'}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
