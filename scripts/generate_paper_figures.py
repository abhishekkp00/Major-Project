"""
generate_paper_figures.py

Generates publication-quality Black & White / Grayscale figures for the paper:
  "SecureLoRA: A Device-Bound Cryptographic and Privacy-Preserving Framework for Parameter-Efficient LLM Fine-Tuning"

All data is read directly from real job runs stored in outputs/jobs/jobs_db.json.
Figure numbering in filenames and plots matches 1-to-1 with IEEE paper text:
  - Fig 1: Architecture Diagram (fig1_architecture.pdf)
  - Fig 2: PII Entity Detection (fig2_pii_detection.pdf)
  - Fig 3: Loss Convergence Curve (fig3_loss_curve.pdf)
  - Fig 4: Parameter Efficiency (fig4_parameter_efficiency.pdf)
  - Fig 5: Perplexity & Generalization (fig5_perplexity.pdf)
  - Fig 6: Security Pipeline Overhead (fig6_security_timing.pdf)
  - Fig 7: Security Gate Pass/Fail (fig7_security_gates.pdf)

Output directory: paper_figures/
"""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config — Black & White / Grayscale IEEE Theme
# ──────────────────────────────────────────────────────────────────────────────
JOBS_DB   = Path("outputs/jobs/jobs_db.json")
OUT_DIR   = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

STYLE = {
    "bg":        "#ffffff",
    "panel":     "#ffffff",
    "grid":      "#d1d5db",
    "text":      "#000000",
    "subtext":   "#374151",
    "black":     "#000000",
    "dark_gray": "#4b5563",
    "mid_gray":  "#9ca3af",
    "light_gray":"#e5e7eb",
    "white":     "#ffffff",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor":    STYLE["panel"],
    "figure.facecolor":  STYLE["bg"],
    "axes.edgecolor":    STYLE["black"],
    "axes.linewidth":    1.0,
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["text"],
    "ytick.color":       STYLE["text"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.titlesize":    10.5,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   STYLE["text"],
    "legend.facecolor":  STYLE["white"],
    "legend.edgecolor":  STYLE["black"],
    "legend.labelcolor": STYLE["text"],
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
    "savefig.facecolor": STYLE["bg"],
})

# ──────────────────────────────────────────────────────────────────────────────
# Load real data
# ──────────────────────────────────────────────────────────────────────────────
with open(JOBS_DB, encoding="utf-8") as f:
    all_jobs = list(json.load(f).values())

completed = [j for j in all_jobs if j.get("status") == "COMPLETED" and j.get("eval_metrics")]
if not completed:
    raise RuntimeError("No completed jobs with eval_metrics found in jobs_db.json.")

best_job = sorted(
    completed,
    key=lambda j: (j.get("epochs", 0), -j.get("eval_metrics", {}).get("validation_loss", 999))
)[-1]

print(f"Using best job: {best_job['job_id']} | epochs={best_job['epochs']} | val_loss={best_job['eval_metrics'].get('validation_loss'):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — System Architecture Overview (Black & White Professional Generator)
# ──────────────────────────────────────────────────────────────────────────────
def fig1_architecture():
    from generate_architecture_professional import draw_professional_architecture
    draw_professional_architecture()

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — PII Entity Type Detection Summary
# ──────────────────────────────────────────────────────────────────────────────
def fig2_pii_detection():
    agg = {}
    for j in all_jobs:
        pii = j.get("pii_summary", {})
        if not pii:
            continue
        for entity, count in pii.items():
            agg[entity] = agg.get(entity, 0) + count

    if not agg:
        print("  [SKIP] No PII summary data.")
        return

    agg = {k: v for k, v in agg.items() if v > 0}
    if not agg:
        print("  [SKIP] All PII counts are zero.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    labels = list(agg.keys())
    values = list(agg.values())
    
    grays   = [STYLE["dark_gray"], STYLE["mid_gray"], STYLE["light_gray"], STYLE["white"]]
    hatches = ["///", "\\\\\\", "...", "xxx", "///", "\\\\\\"]
    
    bars = ax.bar(labels, values, color=[grays[i % len(grays)] for i in range(len(labels))],
                  edgecolor=STYLE["black"], linewidth=1.2, width=0.5)
    
    for i, bar in enumerate(bars):
        bar.set_hatch(hatches[i % len(hatches)])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
                ha="center", va="bottom", color=STYLE["black"], fontsize=9, fontweight="bold")
    
    ax.set_ylabel("Total Entities Detected Across All Jobs")
    ax.set_title("PII Entity Type Detection Summary")
    ax.grid(True, axis="y")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = OUT_DIR / "fig2_pii_detection.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Training Loss Convergence Curve
# ──────────────────────────────────────────────────────────────────────────────
def fig3_loss_curve():
    loss_data = [h for h in best_job.get("loss_history", []) if h.get("loss") is not None]
    eval_data = [h for h in best_job.get("loss_history", []) if h.get("eval_loss") is not None]

    if not loss_data:
        print("  [SKIP] No training loss data in best job.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    steps  = list(range(1, len(loss_data) + 1))
    losses = [h["loss"] for h in loss_data]

    ax.plot(steps, losses, color=STYLE["black"], linewidth=1.8,
            linestyle="-", marker="o", markersize=4, markerfacecolor=STYLE["black"],
            label="Training Loss")

    if eval_data:
        eval_steps  = [round(len(loss_data) * (i + 1) / len(eval_data)) for i in range(len(eval_data))]
        eval_losses = [h["eval_loss"] for h in eval_data]
        ax.plot(eval_steps, eval_losses, color=STYLE["black"], linewidth=1.8,
                linestyle="--", marker="s", markersize=5, markerfacecolor=STYLE["white"],
                markeredgecolor=STYLE["black"], markeredgewidth=1.2, label="Validation Loss")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("LoRA Fine-Tuning Loss Convergence")
    ax.legend(frameon=True)
    ax.grid(True)

    final_loss = losses[-1]
    ax.annotate(f"Final: {final_loss:.4f}", xy=(steps[-1], final_loss),
                xytext=(-50, 15), textcoords="offset points",
                color=STYLE["black"], fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=STYLE["black"], lw=1.2))

    path = OUT_DIR / "fig3_loss_curve.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Model Efficiency: Trainable vs Total Parameters
# ──────────────────────────────────────────────────────────────────────────────
def fig4_parameter_efficiency():
    metrics = best_job.get("eval_metrics", {})
    total      = metrics.get("all_parameters")
    trainable  = metrics.get("trainable_parameters")

    if not total or not trainable:
        print("  [SKIP] Parameter counts not in eval_metrics.")
        return

    frozen = total - trainable
    pct    = metrics.get("trainable_percent", (trainable / total) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Pie chart
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        [trainable, frozen],
        labels=["LoRA Adapters\n(Trainable)", "Base LLM\n(Frozen)"],
        colors=[STYLE["light_gray"], STYLE["white"]],
        autopct="%1.3f%%",
        startangle=90,
        wedgeprops={"edgecolor": STYLE["black"], "linewidth": 1.5},
        textprops={"color": STYLE["black"]},
    )
    wedges[0].set_hatch("///")
    wedges[1].set_hatch("...")
    for at in autotexts:
        at.set_color(STYLE["black"])
        at.set_fontweight("bold")
    ax.set_title("Parameter Breakdown")

    # Bar comparison
    ax2 = axes[1]
    categories = ["Base LLM\n(JackFram/llama-68m)", "LoRA Adapters"]
    values     = [total / 1e6, trainable / 1e6]
    bars = ax2.bar(categories, values, color=[STYLE["white"], STYLE["light_gray"]],
                   edgecolor=STYLE["black"], linewidth=1.2, width=0.5)
    bars[0].set_hatch("\\\\\\")
    bars[1].set_hatch("///")
    ax2.set_ylabel("Parameters (millions)")
    ax2.set_title("Absolute Parameter Count")
    ax2.grid(True, axis="y")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.2f}M", ha="center", va="bottom",
                 color=STYLE["black"], fontsize=8.5, fontweight="bold")

    fig.suptitle("LoRA Parameter Efficiency (PEFT Breakdown)", fontsize=11, fontweight="bold")
    path = OUT_DIR / "fig4_parameter_efficiency.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Perplexity Across Completed Jobs
# ──────────────────────────────────────────────────────────────────────────────
def fig5_perplexity():
    rows = []
    for j in completed:
        m = j.get("eval_metrics", {})
        ppl = m.get("perplexity")
        if ppl and ppl < 1000:
            rows.append({
                "label":   j["job_id"][-8:],
                "epochs":  j.get("epochs", 1),
                "ppl":     ppl,
                "val_loss": m.get("validation_loss", 0),
            })

    if len(rows) < 2:
        print("  [SKIP] Not enough perplexity data points.")
        return

    rows.sort(key=lambda r: r["epochs"])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    labels = [r["label"] for r in rows]
    ppls   = [r["ppl"]   for r in rows]
    vloss  = [r["val_loss"] for r in rows]
    x = np.arange(len(rows))

    ax = axes[0]
    colors = [STYLE["dark_gray"] if r["epochs"] >= 5 else STYLE["light_gray"] for r in rows]
    hatches = ["///" if r["epochs"] >= 5 else "\\\\\\" for r in rows]
    bars = ax.bar(x, ppls, color=colors, edgecolor=STYLE["black"], width=0.55)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Perplexity (↓ better)")
    ax.set_title("Post-Training Perplexity per Job")
    ax.grid(True, axis="y")
    for bar, v in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=STYLE["black"])

    patch_high = mpatches.Patch(facecolor=STYLE["dark_gray"], edgecolor=STYLE["black"], hatch="///", label="≥5 Epochs")
    patch_low  = mpatches.Patch(facecolor=STYLE["light_gray"], edgecolor=STYLE["black"], hatch="\\\\\\", label="<5 Epochs")
    ax.legend(handles=[patch_high, patch_low], frameon=True)

    ax2 = axes[1]
    ax2.scatter(vloss, ppls, color=STYLE["black"], facecolors=STYLE["white"], s=60, edgecolors=STYLE["black"], linewidth=1.5, zorder=3)
    for r, vl, pp in zip(rows, vloss, ppls):
        ax2.annotate(r["label"], (vl, pp), textcoords="offset points",
                     xytext=(5, 3), fontsize=7, color=STYLE["subtext"])
    m_coef = np.polyfit(vloss, ppls, 1)
    x_line = np.linspace(min(vloss), max(vloss), 100)
    ax2.plot(x_line, np.poly1d(m_coef)(x_line),
             color=STYLE["black"], linewidth=1.5, linestyle="--", label="Linear fit")
    ax2.set_xlabel("Validation Loss")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity vs Validation Loss")
    ax2.legend(frameon=True)
    ax2.grid(True)

    fig.suptitle("Perplexity and Generalization Metrics", fontsize=11, fontweight="bold")
    path = OUT_DIR / "fig5_perplexity.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Security Pipeline Timing Breakdown
# ──────────────────────────────────────────────────────────────────────────────
def fig6_security_timing():
    timing_rows = []
    for j in completed:
        sm = j.get("security_metrics", {})
        if sm.get("encryption_time_seconds") and sm.get("verification_time_seconds"):
            timing_rows.append({
                "job_id":        j["job_id"][-8:],
                "encrypt_s":     sm["encryption_time_seconds"],
                "verify_s":      sm["verification_time_seconds"],
                "adapter_kb":    sm.get("adapter_size_before_encryption_bytes", 0) / 1024,
                "protected_kb":  sm.get("protected_package_size_bytes", 0) / 1024,
            })

    if not timing_rows:
        print("  [SKIP] No security timing data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    labels       = [r["job_id"] for r in timing_rows]
    encrypt_vals = [r["encrypt_s"] for r in timing_rows]
    verify_vals  = [r["verify_s"]  for r in timing_rows]
    x = np.arange(len(labels))
    w = 0.35

    ax = axes[0]
    b1 = ax.bar(x - w/2, encrypt_vals, w, label="Encryption", color=STYLE["dark_gray"], edgecolor=STYLE["black"], hatch="///")
    b2 = ax.bar(x + w/2, verify_vals,  w, label="Verification", color=STYLE["light_gray"], edgecolor=STYLE["black"], hatch="\\\\\\")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Encryption & Verification Latency")
    ax.legend(frameon=True)
    ax.grid(True, axis="y")

    # Package size bar
    ax2 = axes[1]
    adapter_vals   = [r["adapter_kb"]   for r in timing_rows]
    protected_vals = [r["protected_kb"] for r in timing_rows]
    b3 = ax2.bar(x - w/2, adapter_vals,   w, label="Adapter (raw)", color=STYLE["light_gray"], edgecolor=STYLE["black"], hatch="...")
    b4 = ax2.bar(x + w/2, protected_vals, w, label="Package (enc)", color=STYLE["white"], edgecolor=STYLE["black"], hatch="xxx")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Size (KB)")
    ax2.set_title("Adapter vs Encrypted Package Size")
    ax2.legend(frameon=True)
    ax2.grid(True, axis="y")

    fig.suptitle("Phase 3 Security Pipeline Overhead", fontsize=11, fontweight="bold")
    path = OUT_DIR / "fig6_security_timing.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 — Security Gate Results (Pass/Fail across all jobs)
# ──────────────────────────────────────────────────────────────────────────────
def fig7_security_gates():
    gates = {
        "Authorized\nDeployment": 0,
        "Tamper\nDetection":      0,
        "Unauthorized\nDevice Block": 0,
    }
    totals = {k: 0 for k in gates}

    for j in completed:
        sm = j.get("security_metrics", {})
        if not sm:
            continue
        totals["Authorized\nDeployment"]    += 1
        totals["Tamper\nDetection"]          += 1
        totals["Unauthorized\nDevice Block"] += 1
        if sm.get("authorized_deployment") == "pass":
            gates["Authorized\nDeployment"] += 1
        if sm.get("tamper_simulation") == "pass":
            gates["Tamper\nDetection"] += 1
        if sm.get("unauthorized_device_simulation") == "pass":
            gates["Unauthorized\nDevice Block"] += 1

    if not any(totals.values()):
        print("  [SKIP] No security gate data.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    labels     = list(gates.keys())
    pass_vals  = [gates[k] for k in labels]
    fail_vals  = [totals[k] - gates[k] for k in labels]
    x = np.arange(len(labels))
    w = 0.45

    bars_pass = ax.bar(x, pass_vals, w, label="PASS", color=STYLE["dark_gray"], edgecolor=STYLE["black"], hatch="///")
    bars_fail = ax.bar(x, fail_vals, w, bottom=pass_vals, label="FAIL", color=STYLE["white"], edgecolor=STYLE["black"], hatch="xxx")

    for bar, val in zip(bars_pass, pass_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val/2, str(val),
                    ha="center", va="center", color=STYLE["white"], fontweight="bold")
    for bar, pv, fv in zip(bars_fail, pass_vals, fail_vals):
        if fv > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pv + fv/2, str(fv),
                    ha="center", va="center", color=STYLE["black"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Jobs")
    ax.set_title("Security Gate Pass/Fail Validation")
    ax.legend(frameon=True)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = OUT_DIR / "fig7_security_gates.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Generating B&W paper figures with matched numbers 1 to 7...")
    print(f"  Jobs DB: {JOBS_DB}  ({len(all_jobs)} total, {len(completed)} completed)")
    print(f"  Output : {OUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    fig1_architecture()
    fig2_pii_detection()
    fig3_loss_curve()
    fig4_parameter_efficiency()
    fig5_perplexity()
    fig6_security_timing()
    fig7_security_gates()

    print(f"\n{'='*60}")
    print("  All B&W figures saved to:", OUT_DIR.resolve())
    files = sorted(OUT_DIR.glob("*.png"))
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:40s}  {size_kb:4d} KB")
    print(f"{'='*60}\n")
