"""
run_paper_evaluation.py
=======================
Master runner: executes all four evaluation modules and produces a single
consolidated paper-ready JSON report + human-readable Markdown summary.

Modules run (in order):
  1. pii_metrics.py          — PII/PHI detection precision, recall, F1
  2. crypto_benchmark.py     — AES, HKDF, RSA, SHA-256 timing benchmarks
  3. baseline_comparison.py  — Security matrix + timing vs. 5 baselines
  4. threat_model.py         — Formal STRIDE threat model + attack simulations

Usage:
    python run_paper_evaluation.py
    python run_paper_evaluation.py --output-dir outputs/paper_results
"""

import sys
import json
import argparse
import platform
from pathlib import Path
from datetime import datetime, timezone

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _run_module(label: str, fn, *args, **kwargs):
    print(f"\n{'─'*60}")
    print(f"  Running: {label}")
    print(f"{'─'*60}")
    try:
        result = fn(*args, **kwargs)
        print(f"  ✅  {label} — DONE")
        return result
    except Exception as e:
        print(f"  ❌  {label} — FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def generate_markdown_summary(report: dict, out_path: Path):
    """Writes a human-readable Markdown summary for the paper appendix."""
    ts = report["metadata"]["timestamp_utc"]
    sys_info = report["metadata"]["system"]

    lines = [
        "# SecureLoRA — Paper Evaluation Summary",
        f"> Generated: {ts}",
        f"> System: {sys_info.get('cpu', 'N/A')} | {sys_info.get('os', 'N/A')} {sys_info.get('os_release', '')}",
        "",

        "---",
        "",
        "## 1. PII/PHI Detection Metrics",
        "",
        "| PII Type | Precision | Recall | F1-Score | TP | FP | FN |",
        "|----------|-----------|--------|----------|----|----|----|",
    ]

    pii = report.get("pii_metrics", {})
    for pii_type, m in pii.get("per_class_metrics", {}).items():
        lines.append(
            f"| {pii_type} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )

    micro = pii.get("micro_average", {})
    macro = pii.get("macro_average", {})
    sample_acc = pii.get("sample_accuracy", "N/A")

    lines += [
        f"| **Micro Avg** | **{micro.get('precision', 0):.4f}** | "
        f"**{micro.get('recall', 0):.4f}** | **{micro.get('f1', 0):.4f}** | — | — | — |",
        f"| **Macro Avg** | **{macro.get('precision', 0):.4f}** | "
        f"**{macro.get('recall', 0):.4f}** | **{macro.get('f1', 0):.4f}** | — | — | — |",
        "",
        f"**Sample-level Accuracy:** {sample_acc:.2%}" if isinstance(sample_acc, float) else f"**Sample-level Accuracy:** {sample_acc}",
        "",

        "---",
        "",
        "## 2. Cryptographic Performance Benchmarks",
        "",
        "### 2.1 AES-256-GCM Streaming Encryption",
        "",
        "| Payload (KB) | Mean (ms) | Stdev (ms) | Throughput (MB/s) |",
        "|-------------|-----------|------------|-------------------|",
    ]

    bench = report.get("crypto_benchmark", {}).get("results", {})
    for r in bench.get("aes_gcm_encryption", []):
        lines.append(
            f"| {r['payload_kb']:>12} | {r['mean_ms']:>9.3f} | {r.get('stdev_ms', 0):>10.3f} | "
            f"{r['throughput_mb_per_s']:>17.3f} |"
        )

    lines += [
        "",
        "### 2.2 AES-256-GCM Streaming Decryption",
        "",
        "| Payload (KB) | Mean (ms) | Stdev (ms) | Throughput (MB/s) |",
        "|-------------|-----------|------------|-------------------|",
    ]
    for r in bench.get("aes_gcm_decryption", []):
        lines.append(
            f"| {r['payload_kb']:>12} | {r['mean_ms']:>9.3f} | {r.get('stdev_ms', 0):>10.3f} | "
            f"{r['throughput_mb_per_s']:>17.3f} |"
        )

    hkdf = bench.get("hkdf_key_derivation", {})
    hw_fp = bench.get("hardware_fingerprint", {})
    rsa_sign = bench.get("rsa_pss_sign", {})
    rsa_verify = bench.get("rsa_pss_verify", {})
    e2e = bench.get("e2e_overhead", {})

    lines += [
        "",
        "### 2.3 Cryptographic Primitive Latency",
        "",
        "| Operation | Mean (ms) | Stdev (ms) |",
        "|-----------|-----------|------------|",
        f"| HKDF Key Derivation (SHA-256) | {hkdf.get('mean_ms', 0):.4f} | {hkdf.get('stdev_ms', 0):.4f} |",
        f"| Hardware Fingerprint Collection | {hw_fp.get('mean_ms', 0):.4f} | {hw_fp.get('stdev_ms', 0):.4f} |",
        f"| RSA-{report.get('crypto_benchmark', {}).get('metadata', {}).get('rsa_key_bits', 2048)}-PSS Sign | {rsa_sign.get('mean_ms', 0):.4f} | {rsa_sign.get('stdev_ms', 0):.4f} |",
        f"| RSA-{report.get('crypto_benchmark', {}).get('metadata', {}).get('rsa_key_bits', 2048)}-PSS Verify | {rsa_verify.get('mean_ms', 0):.4f} | {rsa_verify.get('stdev_ms', 0):.4f} |",
        "",
        f"**End-to-End Security Overhead:** {e2e.get('security_overhead_ms', 0):.2f} ms  "
        f"({e2e.get('overhead_factor_x', 0):.1f}× vs. plain file copy, {e2e.get('payload_kb', 0)} KB payload)",
        "",

        "---",
        "",
        "## 3. Baseline Security Comparison",
        "",
        "### Security Property Matrix (0.0 – 1.0 scale)",
        "",
    ]

    baselines_data = report.get("baseline_comparison", {}).get("baselines", {})
    if baselines_data:
        dims = list(list(baselines_data.values())[0].get("security_scores", {}).keys())
        col_names = list(baselines_data.keys())
        header = "| Dimension |" + "|".join(f" {n[:20]} " for n in col_names) + "|"
        sep = "|---|" + "|".join(["---|"] * len(col_names))
        lines += [header, sep]
        for dim in dims:
            row = f"| {dim} |"
            for name in col_names:
                val = baselines_data[name].get("security_scores", {}).get(dim, 0)
                row += f" {val:.2f} |"
            lines.append(row)

        agg_row = "| **Aggregate** |"
        for name in col_names:
            agg = baselines_data[name].get("aggregate_security_score", 0)
            agg_row += f" **{agg:.4f}** |"
        lines.append(agg_row)

    lines += [
        "",
        "### Timing Comparison",
        "",
        "| Method | Mean (ms) | Stdev (ms) |",
        "|--------|-----------|------------|",
    ]
    for name, data in baselines_data.items():
        t = data.get("timing", {})
        lines.append(f"| {name} | {t.get('mean_ms', 0):.2f} | {t.get('stdev_ms', 0):.2f} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Formal Threat Model & Attack Simulations",
        "",
    ]

    tm = report.get("threat_model", {})
    meta_tm = tm.get("metadata", {})
    lines += [
        f"**Threats Analyzed:** {meta_tm.get('total_threats_analyzed', 0)}  ",
        f"**Simulations Run:** {meta_tm.get('total_simulations', 0)}  ",
        f"**Simulations Passed:** {meta_tm.get('simulations_passed', 0)}/{meta_tm.get('total_simulations', 0)}  ",
        f"**Overall Security Result:** `{meta_tm.get('overall_security_result', 'N/A')}`",
        "",
        "| Sim ID | Attack Scenario | Result | Time (ms) |",
        "|--------|-----------------|--------|-----------|",
    ]
    for sim in tm.get("simulation_results", []):
        lines.append(
            f"| {sim['sim_id']} | {sim['name']} | "
            f"{'✅ PASS' if sim['result'] == 'PASS' else '❌ FAIL'} | {sim['duration_ms']:.1f} |"
        )

    summary = tm.get("security_summary", {})
    if summary:
        lines += ["", "### Security Summary", ""]
        for prop, desc in summary.items():
            lines.append(f"- **{prop.replace('_', ' ').title()}**: {desc}")

    lines += [
        "",
        "---",
        "",
        f"*Report generated by SecureLoRA Paper Evaluation Suite v1.0.0 | {ts}*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  📄  Markdown summary → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Full Paper Evaluation")
    parser.add_argument("--output-dir", type=str, default="outputs/paper_results",
                        help="Directory to write all evaluation outputs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = out_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    print("\n" + "=" * 60)
    print("  SecureLoRA — Full Paper Evaluation Suite")
    print("  Generating genuine, non-hardcoded results...")
    print("=" * 60)

    # ── 1. PII Metrics ──────────────────────────────────────────────
    from src.evaluation.pii_metrics import evaluate_pii_detection
    pii_results = _run_module("PII/PHI Detection Metrics", evaluate_pii_detection, verbose=verbose)

    # ── 2. Crypto Benchmarks ─────────────────────────────────────────
    from src.evaluation.crypto_benchmark import run_all_benchmarks
    crypto_results = _run_module("Cryptographic Performance Benchmarks", run_all_benchmarks, verbose=verbose)

    # ── 3. Baseline Comparison ───────────────────────────────────────
    from src.evaluation.baseline_comparison import run_baseline_comparison
    baseline_results = _run_module("Baseline Security Comparison", run_baseline_comparison, verbose=verbose)

    # ── 4. Threat Model ──────────────────────────────────────────────
    from src.evaluation.threat_model import run_threat_model_analysis
    threat_results = _run_module("Formal Threat Model & Attack Simulations", run_threat_model_analysis, verbose=verbose)

    # ── Combine into consolidated report ────────────────────────────
    consolidated = {
        "metadata": {
            "report_title": "SecureLoRA: Device-Bound LoRA Protection — Paper Evaluation Results",
            "evaluation_suite_version": "1.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "system": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "os_release": platform.release(),
                "cpu": platform.processor() or platform.machine(),
            },
        },
        "pii_metrics": pii_results,
        "crypto_benchmark": crypto_results,
        "baseline_comparison": baseline_results,
        "threat_model": threat_results,
    }

    # Save consolidated JSON
    consolidated_path = out_dir / "paper_evaluation_results.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n  💾  Consolidated report → {consolidated_path}")

    # Save individual module reports
    for name, data in [
        ("pii_metrics", pii_results),
        ("crypto_benchmark", crypto_results),
        ("baseline_comparison", baseline_results),
        ("threat_model", threat_results),
    ]:
        p = bench_dir / f"{name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Generate Markdown summary
    md_path = out_dir / "paper_evaluation_summary.md"
    generate_markdown_summary(consolidated, md_path)

    print("\n" + "=" * 60)
    print("  ✅  All evaluations complete!")
    print(f"  📁  Output directory: {out_dir.resolve()}")
    print("=" * 60 + "\n")

    return consolidated


if __name__ == "__main__":
    main()
