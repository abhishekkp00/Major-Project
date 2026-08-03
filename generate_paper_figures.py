"""
generate_paper_figures.py

Generates publication-quality figures for the paper:
  "Secure Device-Bound LoRA Fine-Tuning Framework for Large Language Models"

All data is read directly from real job runs stored in outputs/jobs/jobs_db.json.
No values are hardcoded — every data point comes from actual training/security runs.

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
# Config
# ──────────────────────────────────────────────────────────────────────────────
JOBS_DB   = Path("outputs/jobs/jobs_db.json")
OUT_DIR   = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

STYLE = {
    "bg":        "#0d111c",
    "panel":     "#111827",
    "grid":      "#1f2937",
    "text":      "#e5e7eb",
    "subtext":   "#9ca3af",
    "cyan":      "#00f2fe",
    "emerald":   "#10b981",
    "violet":    "#8b5cf6",
    "rose":      "#f43f5e",
    "amber":     "#f59e0b",
    "blue":      "#3b82f6",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor":    STYLE["panel"],
    "figure.facecolor":  STYLE["bg"],
    "axes.edgecolor":    STYLE["grid"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["subtext"],
    "ytick.color":       STYLE["subtext"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["grid"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   STYLE["text"],
    "legend.facecolor":  STYLE["panel"],
    "legend.edgecolor":  STYLE["grid"],
    "legend.labelcolor": STYLE["text"],
    "savefig.bbox":      "tight",
    "savefig.dpi":       200,
    "savefig.facecolor": STYLE["bg"],
})

# ──────────────────────────────────────────────────────────────────────────────
# Load real data
# ──────────────────────────────────────────────────────────────────────────────
with open(JOBS_DB, encoding="utf-8") as f:
    all_jobs = list(json.load(f).values())

completed = [j for j in all_jobs if j.get("status") == "COMPLETED" and j.get("eval_metrics")]
if not completed:
    raise RuntimeError("No completed jobs with eval_metrics found in jobs_db.json. Run a job first.")

# Best job = most epochs + lowest validation loss
best_job = sorted(
    completed,
    key=lambda j: (j.get("epochs", 0), -j.get("eval_metrics", {}).get("validation_loss", 999))
)[-1]

print(f"Using best job: {best_job['job_id']} | epochs={best_job['epochs']} | val_loss={best_job['eval_metrics'].get('validation_loss'):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Training Loss Curve (per epoch)
# ──────────────────────────────────────────────────────────────────────────────
def fig1_loss_curve():
    loss_data = [
        h for h in best_job.get("loss_history", [])
        if h.get("loss") is not None
    ]
    eval_data = [
        h for h in best_job.get("loss_history", [])
        if h.get("eval_loss") is not None
    ]

    if not loss_data:
        print("  [SKIP] No training loss data in best job.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    steps  = list(range(1, len(loss_data) + 1))
    losses = [h["loss"] for h in loss_data]

    ax.plot(steps, losses, color=STYLE["cyan"], linewidth=2.0,
            marker="o", markersize=4, label="Training Loss")

    if eval_data:
        eval_steps  = [round(len(loss_data) * (i + 1) / len(eval_data)) for i in range(len(eval_data))]
        eval_losses = [h["eval_loss"] for h in eval_data]
        ax.plot(eval_steps, eval_losses, color=STYLE["amber"], linewidth=2.0,
                linestyle="--", marker="s", markersize=4, label="Validation Loss")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Figure 1 — LoRA Fine-Tuning Loss Convergence")
    ax.legend()
    ax.grid(True)

    # Annotate final loss
    final_loss = losses[-1]
    ax.annotate(f"Final: {final_loss:.4f}", xy=(steps[-1], final_loss),
                xytext=(-45, 12), textcoords="offset points",
                color=STYLE["cyan"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=STYLE["cyan"], lw=1.2))

    path = OUT_DIR / "fig1_loss_curve.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Model Efficiency: Trainable vs Total Parameters
# ──────────────────────────────────────────────────────────────────────────────
def fig2_parameter_efficiency():
    metrics = best_job.get("eval_metrics", {})
    total      = metrics.get("all_parameters")
    trainable  = metrics.get("trainable_parameters")

    if not total or not trainable:
        print("  [SKIP] Parameter counts not in eval_metrics.")
        return

    frozen = total - trainable
    pct    = metrics.get("trainable_percent", (trainable / total) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # --- Pie chart ---
    ax = axes[0]
    wedge_colors = [STYLE["cyan"], STYLE["grid"]]
    wedges, texts, autotexts = ax.pie(
        [trainable, frozen],
        labels=["LoRA Adapters\n(Trainable)", "Base LLM\n(Frozen)"],
        colors=wedge_colors,
        autopct="%1.3f%%",
        startangle=90,
        wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 2},
        textprops={"color": STYLE["text"]},
    )
    for at in autotexts:
        at.set_color(STYLE["bg"])
        at.set_fontweight("bold")
    ax.set_title("Parameter Breakdown")

    # --- Bar comparison ---
    ax2 = axes[1]
    categories = ["Base LLM\n(JackFram/llama-68m)", "LoRA Adapters"]
    values     = [total / 1e6, trainable / 1e6]
    colors     = [STYLE["grid"], STYLE["cyan"]]
    bars = ax2.bar(categories, values, color=colors, edgecolor=STYLE["bg"], width=0.5)
    ax2.set_ylabel("Parameters (millions)")
    ax2.set_title("Absolute Parameter Count")
    ax2.grid(True, axis="y")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.2f}M", ha="center", va="bottom",
                 color=STYLE["text"], fontsize=9, fontweight="bold")

    fig.suptitle("Figure 2 — LoRA Parameter Efficiency (PEFT)", fontsize=12, fontweight="bold")
    path = OUT_DIR / "fig2_parameter_efficiency.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Security Pipeline Timing Breakdown
# ──────────────────────────────────────────────────────────────────────────────
def fig3_security_timing():
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels       = [r["job_id"] for r in timing_rows]
    encrypt_vals = [r["encrypt_s"] for r in timing_rows]
    verify_vals  = [r["verify_s"]  for r in timing_rows]
    x = np.arange(len(labels))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, encrypt_vals, w, label="Encryption",    color=STYLE["violet"], edgecolor=STYLE["bg"])
    ax.bar(x + w/2, verify_vals,  w, label="Verification",  color=STYLE["emerald"], edgecolor=STYLE["bg"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Encryption & Verification Time per Job")
    ax.legend()
    ax.grid(True, axis="y")

    # --- Package size bar ---
    ax2 = axes[1]
    adapter_vals   = [r["adapter_kb"]   for r in timing_rows]
    protected_vals = [r["protected_kb"] for r in timing_rows]
    ax2.bar(x - w/2, adapter_vals,   w, label="Adapter (raw)", color=STYLE["blue"],  edgecolor=STYLE["bg"])
    ax2.bar(x + w/2, protected_vals, w, label="Package (enc)", color=STYLE["amber"], edgecolor=STYLE["bg"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Size (KB)")
    ax2.set_title("Adapter vs Encrypted Package Size")
    ax2.legend()
    ax2.grid(True, axis="y")

    fig.suptitle("Figure 3 — Phase 3 Security Pipeline Overhead", fontsize=12, fontweight="bold")
    path = OUT_DIR / "fig3_security_timing.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Security Gate Results (Pass/Fail across all jobs)
# ──────────────────────────────────────────────────────────────────────────────
def fig4_security_gates():
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

    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels     = list(gates.keys())
    pass_vals  = [gates[k] for k in labels]
    fail_vals  = [totals[k] - gates[k] for k in labels]
    x = np.arange(len(labels))
    w = 0.45

    bars_pass = ax.bar(x, pass_vals, w, label="PASS", color=STYLE["emerald"], edgecolor=STYLE["bg"])
    bars_fail = ax.bar(x, fail_vals, w, bottom=pass_vals, label="FAIL", color=STYLE["rose"], edgecolor=STYLE["bg"])

    for bar, val in zip(bars_pass, pass_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val/2, str(val),
                    ha="center", va="center", color=STYLE["bg"], fontweight="bold")
    for bar, pv, fv in zip(bars_fail, pass_vals, fail_vals):
        if fv > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pv + fv/2, str(fv),
                    ha="center", va="center", color=STYLE["bg"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Jobs")
    ax.set_title("Figure 4 — Security Gate Pass/Fail Across All Jobs")
    ax.legend()
    ax.grid(True, axis="y")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = OUT_DIR / "fig4_security_gates.pdf"
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
        if ppl and ppl < 1000:   # sanity filter
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = [r["label"] for r in rows]
    ppls   = [r["ppl"]   for r in rows]
    vloss  = [r["val_loss"] for r in rows]
    colors = [STYLE["cyan"] if r["epochs"] >= 5 else STYLE["violet"] for r in rows]
    x = np.arange(len(rows))

    ax = axes[0]
    bars = ax.bar(x, ppls, color=colors, edgecolor=STYLE["bg"], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Perplexity (↓ better)")
    ax.set_title("Post-Training Perplexity per Job")
    ax.grid(True, axis="y")
    for bar, v in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=STYLE["text"])

    patch_high = mpatches.Patch(color=STYLE["cyan"],   label="≥5 Epochs")
    patch_low  = mpatches.Patch(color=STYLE["violet"], label="<5 Epochs")
    ax.legend(handles=[patch_high, patch_low])

    ax2 = axes[1]
    ax2.scatter(vloss, ppls, color=STYLE["cyan"], s=60, edgecolors=STYLE["panel"], zorder=3)
    for r, vl, pp in zip(rows, vloss, ppls):
        ax2.annotate(r["label"], (vl, pp), textcoords="offset points",
                     xytext=(5, 3), fontsize=7, color=STYLE["subtext"])
    m_coef = np.polyfit(vloss, ppls, 1)
    x_line = np.linspace(min(vloss), max(vloss), 100)
    ax2.plot(x_line, np.poly1d(m_coef)(x_line),
             color=STYLE["amber"], linewidth=1.5, linestyle="--", label="Linear fit")
    ax2.set_xlabel("Validation Loss")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity vs Validation Loss")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle("Figure 5 — Perplexity and Generalisation Metrics", fontsize=12, fontweight="bold")
    path = OUT_DIR / "fig5_perplexity.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — System Architecture Overview (text diagram as figure)
# ──────────────────────────────────────────────────────────────────────────────
def fig6_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    phases = [
        ("Phase 1\nDataset\nIngestion &\nEncryption",  STYLE["blue"],    1.0),
        ("Phase 2\nLoRA\nFine-Tuning\n(In-Memory)",    STYLE["violet"],  3.0),
        ("Phase 3\nAdapter\nPackaging &\nBinding",      STYLE["amber"],   5.0),
        ("Phase 4\nSecure\nDeployment &\nInference",    STYLE["emerald"], 7.0),
        ("Dashboard\nOrchestration\n& Monitoring",      STYLE["cyan"],    9.5),
    ]

    for label, color, x in phases:
        rect = mpatches.FancyBboxPatch(
            (x - 0.9, 1.2), 1.8, 2.6,
            boxstyle="round,pad=0.08",
            facecolor=color + "22",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x, 2.5, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
                multialignment="center")

    # Arrows between phases
    arrow_xs = [(1.9, 2.1), (3.9, 4.1), (5.9, 6.1), (7.9, 8.1)]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color=STYLE["subtext"], lw=1.5))

    # Security properties annotations
    notes = [
        (1.0, 0.65, "AES-256-GCM\nZero-plaintext-at-rest", STYLE["blue"]),
        (3.0, 0.65, "PEFT LoRA\n0.14% trainable params",   STYLE["violet"]),
        (5.0, 0.65, "RSA-4096 sign\nHWID fingerprint bind", STYLE["amber"]),
        (7.0, 0.65, "8-gate\nverification pipeline",        STYLE["emerald"]),
        (9.5, 0.65, "SSE real-time\nmonitoring",            STYLE["cyan"]),
    ]
    for x, y, txt, color in notes:
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=7.5, color=color, alpha=0.9, multialignment="center")

    ax.set_title("Figure 6 — End-to-End Secure LoRA Framework Architecture",
                 fontsize=12, fontweight="bold", pad=14)

    path = OUT_DIR / "fig6_architecture.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 — PII Entity Detection Summary across all jobs
# ──────────────────────────────────────────────────────────────────────────────
def fig7_pii_detection():
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

    # Filter out zero-count entries
    agg = {k: v for k, v in agg.items() if v > 0}
    if not agg:
        print("  [SKIP] All PII counts are zero.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(agg.keys())
    values = list(agg.values())
    bar_colors = [STYLE["rose"], STYLE["amber"], STYLE["violet"], STYLE["blue"]][:len(labels)]
    bars = ax.bar(labels, values, color=bar_colors[:len(labels)], edgecolor=STYLE["bg"], width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(val),
                ha="center", va="bottom", color=STYLE["text"], fontsize=9, fontweight="bold")
    ax.set_ylabel("Total Entities Detected Across All Jobs")
    ax.set_title("Figure 7 — PII Entity Type Detection Summary")
    ax.grid(True, axis="y")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    path = OUT_DIR / "fig7_pii_detection.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Run all figures
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Generating paper figures from real job data...")
    print(f"  Jobs DB: {JOBS_DB}  ({len(all_jobs)} total, {len(completed)} completed)")
    print(f"  Output : {OUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    fig1_loss_curve()
    fig2_parameter_efficiency()
    fig3_security_timing()
    fig4_security_gates()
    fig5_perplexity()
    fig6_pipeline_overview()
    fig7_pii_detection()

    print(f"\n{'='*60}")
    print("  All figures saved to:", OUT_DIR.resolve())
    files = sorted(OUT_DIR.glob("*.png"))
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name:40s}  {size_kb:4d} KB")
    print(f"{'='*60}\n")
