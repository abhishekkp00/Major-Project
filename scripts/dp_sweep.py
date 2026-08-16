#!/usr/bin/env python3
"""
scripts/dp_sweep.py
===================
Privacy-utility trade-off experiment runner for SecureLoRA DP-LoRA.

Runs controlled experiments comparing:
  - Mode A: Standard LoRA (no DP, ε = ∞)
  - Mode B: DP-LoRA at multiple target ε values (actual ε computed by accountant)

The actual epsilon achieved for each DP run is computed by the Opacus
privacy accountant and may differ from the target.  We always report
the accountant's value, never a manually entered one.

Results are written to: outputs/evaluation/dp_sweep_results.json
A human-readable table is printed to stdout.

Usage
-----
  # Minimal sweep (standard LoRA + one DP setting):
  python scripts/dp_sweep.py --epochs 2 --batch-size 4

  # Full sweep:
  python scripts/dp_sweep.py --epochs 3 --batch-size 4 --full

  # DP-only with custom epsilon:
  python scripts/dp_sweep.py --dp-only --target-epsilon 4.0

Research context
----------------
DP-LoRA is an established research direction (Li et al. 2021, Yu et al. 2021).
This sweep supports the research question:

    "What is the measured privacy-utility trade-off of training-data privacy
     (DP-SGD) when combined with device-bound adapter encryption?"

Claims we do NOT make:
  - DP-LoRA is novel because we added Opacus.
  - DP protects the adapter from theft.
  - ε precisely equals the target (the accountant gives the actual value).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH when run directly.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("secure_lora.dp_sweep")


# ── Target epsilon grid ───────────────────────────────────────────────────────
# We ask the accountant for these targets; the actual ε achieved may differ.
_EPSILON_TARGETS = [16.0, 8.0, 4.0, 2.0]


def _set_dp_env(enabled: bool, epsilon: float = None, delta: float = 1e-5,
                max_grad_norm: float = 1.0, accountant: str = "rdp") -> None:
    """Configure DP parameters via environment variables."""
    os.environ["DP_ENABLED"] = "1" if enabled else "0"
    if enabled and epsilon is not None:
        os.environ["DP_TARGET_EPSILON"] = str(epsilon)
        os.environ["DP_TARGET_DELTA"] = str(delta)
        os.environ["DP_MAX_GRAD_NORM"] = str(max_grad_norm)
        os.environ["DP_ACCOUNTANT"] = accountant
    # Clear noise_multiplier override (use auto-computation).
    os.environ.pop("DP_NOISE_MULTIPLIER", None)


def _run_single(mode_label: str, dp_enabled: bool, target_epsilon: float = None,
                epochs: int = 2, batch_size: int = 4) -> dict:
    """Run a single training experiment and return the result dict."""
    # Override training parameters for sweep consistency.
    os.environ["SECURE_LORA_EPOCHS"] = str(epochs)
    os.environ["SECURE_LORA_BATCH_SIZE"] = str(batch_size)

    if dp_enabled and target_epsilon is not None:
        _set_dp_env(True, epsilon=target_epsilon)
    else:
        _set_dp_env(False)

    logger.info("=" * 60)
    logger.info("Starting experiment: %s", mode_label)
    logger.info("=" * 60)

    # Reload config to pick up new env vars.
    import importlib
    import src.common.config_loader as _cl
    importlib.reload(_cl)
    from src.common.config_loader import config as _config  # noqa: F401

    from src.phase2.train_lora import run_training
    try:
        result = run_training(dp_enabled=dp_enabled)
    except Exception as e:
        logger.error("Experiment %s failed: %s", mode_label, e)
        result = {
            "training_mode": "dp-lora" if dp_enabled else "lora",
            "status": "failed",
            "error": str(e),
            "epsilon": None,
            "delta": None,
            "val_loss": None,
            "perplexity": None,
            "training_duration_seconds": None,
            "peak_memory_mb": None,
            "noise_multiplier": None,
            "max_grad_norm": None,
        }

    result["experiment_label"] = mode_label
    return result


def _format_table(results: list) -> str:
    """Format results as the required comparison table."""
    header = (
        f"{'Method':<20} {'ε (actual)':>12} {'δ':>10} "
        f"{'Val Loss':>10} {'Perplexity':>12} {'Time(s)':>9} {'Memory(MB)':>11}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for r in results:
        method = r.get("training_mode", "?")
        eps = r.get("epsilon")
        delta = r.get("delta")
        val_loss = r.get("val_loss")
        ppl = r.get("perplexity")
        dur = r.get("training_duration_seconds")
        mem = r.get("peak_memory_mb")
        status = r.get("status", "completed")

        eps_str = f"{eps:.4f}" if eps is not None else "N/A"
        delta_str = f"{delta:.2e}" if delta is not None else "N/A"
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        ppl_str = f"{ppl:.2f}" if ppl is not None else "N/A"
        dur_str = f"{dur:.1f}" if dur is not None else "N/A"
        mem_str = f"{mem:.1f}" if mem is not None else "N/A"

        if status == "failed":
            rows.append(f"{method:<20} {'FAILED':<50}")
        else:
            rows.append(
                f"{method:<20} {eps_str:>12} {delta_str:>10} "
                f"{loss_str:>10} {ppl_str:>12} {dur_str:>9} {mem_str:>11}"
            )

    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SecureLoRA DP-LoRA Privacy-Utility Trade-off Sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs for each run.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for each run.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full epsilon grid (ε ≈ 16, 8, 4, 2).")
    parser.add_argument("--dp-only", action="store_true",
                        help="Skip standard LoRA baseline.")
    parser.add_argument("--target-epsilon", type=float, default=None,
                        help="Single custom target epsilon (skips grid).")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Target delta for DP runs.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Per-example gradient clipping norm C.")
    parser.add_argument("--accountant", choices=["rdp", "prv"], default="rdp",
                        help="Privacy accountant type.")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="Directory to write sweep results JSON.")

    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine epsilon targets.
    if args.target_epsilon is not None:
        epsilon_targets = [args.target_epsilon]
    elif args.full:
        epsilon_targets = _EPSILON_TARGETS
    else:
        epsilon_targets = [8.0]  # default: one DP run at ε ≈ 8

    results = []

    # ── Mode A: Standard LoRA baseline ───────────────────────────────────────
    if not args.dp_only:
        result_a = _run_single(
            "lora",
            dp_enabled=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(result_a)

    # ── Mode B: DP-LoRA at each epsilon target ────────────────────────────────
    for eps_target in epsilon_targets:
        label = f"dp-lora (ε_target={eps_target})"
        result_b = _run_single(
            label,
            dp_enabled=True,
            target_epsilon=eps_target,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(result_b)

    # ── Write JSON results ────────────────────────────────────────────────────
    sweep_output = {
        "sweep_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "delta": args.delta,
            "max_grad_norm": args.max_grad_norm,
            "accountant": args.accountant,
            "epsilon_targets": epsilon_targets,
            "disclaimer": (
                "DP-LoRA is an established research direction. "
                "Our research contribution is the measured interaction of "
                "training-data privacy, device-bound adapter protection, and deployment security."
            ),
        },
        "results": results,
    }

    sweep_path = output_dir / "dp_sweep_results.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_output, f, indent=4)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SecureLoRA — Privacy-Utility Trade-off Comparison")
    print("=" * 80)
    print(_format_table(results))
    print("=" * 80)
    print(f"\nDetailed results saved → {sweep_path}")
    print("\nNOTE: ε values are computed by the Opacus privacy accountant,")
    print("      not manually entered. Target ε ≠ actual ε in general.\n")
    print("NOTE: DP-LoRA addresses training-data privacy (membership inference).")
    print("      Adapter theft protection is provided by device-bound encryption (Phase 3/4).")
    print("      These are orthogonal mechanisms.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
