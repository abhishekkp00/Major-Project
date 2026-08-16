"""
run_reproducible_experiments.py
===============================
Master Research CLI Runner for the SecureLoRA Experimentation Framework.

Executes the full experiment matrix (B0 to B8) across multiple random seeds,
computes statistical metrics (mean, std, 95% CI), runs the ablation study,
generates publication-grade plots, CSV tables, JSON summaries, and answers RQ1-RQ6
in outputs/research/summaries/RESEARCH_EVALUATION_REPORT.md.

Usage:
    python scripts/run_reproducible_experiments.py
    PYTHONPATH=. python scripts/run_reproducible_experiments.py --seeds 42 43 44 --output-dir outputs/research
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

from src.evaluation.experiment_runner import run_experiment_matrix
from src.evaluation.ablation_study import run_ablation_analysis
from src.evaluation.report_generator import generate_all_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_reproducible_experiments")


def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Reproducible Research Experimentation Framework")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds for statistical evaluation")
    parser.add_argument("--output-dir", type=str, default="outputs/research", help="Root directory for research outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  SECURE LORA: REPRODUCIBLE RESEARCH EXPERIMENTATION FRAMEWORK")
    print("  Evaluating Baselines B0 - B8 Across Multiple Random Seeds...")
    print("=" * 80 + "\n")

    # 1. Execute Experiment Matrix
    aggregated_results = run_experiment_matrix(seeds=args.seeds, output_dir=out_dir)

    # 2. Execute Ablation Analysis
    ablation_impacts = run_ablation_analysis(aggregated_results=aggregated_results, output_dir=out_dir)

    # 3. Generate Reports, CSVs, Figures
    generate_all_reports(aggregated_results=aggregated_results, ablation_impacts=ablation_impacts, output_dir=out_dir)

    print("\n" + "=" * 80)
    print("  ✅ REPRODUCIBLE EXPERIMENTATION COMPLETE!")
    print(f"  📄 Research Report: {out_dir / 'summaries' / 'RESEARCH_EVALUATION_REPORT.md'}")
    print(f"  📊 CSV Tables:      {out_dir / 'tables'}")
    print(f"  🖼️  Figures:         {out_dir / 'figures'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
