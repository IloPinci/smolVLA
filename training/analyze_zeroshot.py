"""
analyze_zeroshot.py
-------------------
Reads results/zeroshot.csv and produces:
  results/zeroshot_tsr.pdf          — TSR bar with 95% CI
  results/zeroshot_subgoals.pdf     — sub-goal completion stacked bar
  results/zeroshot_pss_dist.pdf     — PSS distribution histogram
  results/zeroshot_summary.txt      — paper-ready numbers

Usage:
    python analyze_zeroshot.py --csv results/zeroshot.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paper style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

BLUE   = "#6c9fff"
GREEN  = "#42d9a0"
AMBER  = "#ffb347"
PURPLE = "#c07dfa"
RED    = "#ff6b6b"


def load_csv(path):
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            records.append({
                "episode":  int(row["episode"]),
                "success":  row["success"].strip().lower() == "true",
                "pss":      float(row["pss"]),
                "steps":    int(row["steps"]),
                "reached":  row["reached"].strip().lower() == "true",
                "lifted":   row["lifted"].strip().lower()  == "true",
                "placed":   row["placed"].strip().lower()  == "true",
            })
    return records


def wilson_ci(p, n, z=1.96):
    """Wilson score 95% CI — more accurate than normal approx at extremes."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def plot_tsr(records, out_dir):
    n       = len(records)
    tsr     = sum(r["success"] for r in records) / n
    lo, hi  = wilson_ci(tsr, n)

    fig, ax = plt.subplots(figsize=(4, 4))
    bar = ax.bar(["Zero-shot"], [tsr], color=BLUE, alpha=0.85,
                 yerr=[[tsr - lo], [hi - tsr]], capsize=10,
                 error_kw={"elinewidth": 1.5, "ecolor": "white"})
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Task Success Rate (TSR)")
    ax.set_title("SmolVLA Base — Zero-shot Evaluation\n(N=25, nominal scene)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="50% reference")
    ax.legend(fontsize=9)

    # Annotate value
    ax.text(0, tsr + (hi - tsr) + 0.03, f"{tsr:.0%}", ha="center",
            fontsize=12, fontweight="bold", color=BLUE)

    # 95% CI note
    ax.text(0.97, 0.03, f"95% CI [{lo:.2f}, {hi:.2f}]",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")

    plt.tight_layout()
    path = out_dir / "zeroshot_tsr.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {path}")
    return tsr, lo, hi


def plot_subgoals(records, out_dir):
    n = len(records)
    rates = {
        "Reached\nblock":  sum(r["reached"] for r in records) / n,
        "Lifted\nblock":   sum(r["lifted"]  for r in records) / n,
        "Placed\non target": sum(r["placed"] for r in records) / n,
    }
    colors = [GREEN, AMBER, BLUE]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(rates.keys(), rates.values(), color=colors, alpha=0.85)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Sub-goal Completion — Zero-shot\n(N=25, nominal scene)")

    for bar, val in zip(bars, rates.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = out_dir / "zeroshot_subgoals.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {path}")
    return rates


def plot_pss_distribution(records, out_dir):
    pss_vals = [r["pss"] for r in records]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pss_vals, bins=np.arange(0, 1.65, 0.2) - 0.1,
            color=PURPLE, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(pss_vals), color=AMBER, lw=2,
               label=f"Mean PSS = {np.mean(pss_vals):.3f}")
    ax.set_xlabel("Partial Success Score (PSS)")
    ax.set_ylabel("Episode count")
    ax.set_title("PSS Distribution — Zero-shot\n(N=25, nominal scene)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = out_dir / "zeroshot_pss_dist.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {path}")


def plot_combined(records, out_dir):
    """Single 3-panel figure suitable for a paper column."""
    n       = len(records)
    tsr     = sum(r["success"] for r in records) / n
    lo, hi  = wilson_ci(tsr, n)
    sg_rates = [
        sum(r["reached"] for r in records) / n,
        sum(r["lifted"]  for r in records) / n,
        sum(r["placed"]  for r in records) / n,
    ]
    pss_vals = [r["pss"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A — TSR
    axes[0].bar(["Zero-shot TSR"], [tsr], color=BLUE, alpha=0.85,
                yerr=[[tsr - lo], [hi - tsr]], capsize=10,
                error_kw={"elinewidth": 1.5, "ecolor": "white"})
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("(A) Task Success Rate")
    axes[0].text(0, tsr + (hi - tsr) + 0.04,
                 f"{tsr:.0%}", ha="center", fontsize=13, fontweight="bold", color=BLUE)

    # Panel B — sub-goals
    labels = ["Reached", "Lifted", "Placed"]
    bars = axes[1].bar(labels, sg_rates, color=[GREEN, AMBER, BLUE], alpha=0.85)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("(B) Sub-goal Completion")
    for bar, val in zip(bars, sg_rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                     f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

    # Panel C — PSS histogram
    axes[2].hist(pss_vals, bins=np.arange(0, 1.65, 0.2) - 0.1,
                 color=PURPLE, alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[2].axvline(np.mean(pss_vals), color=AMBER, lw=2,
                    label=f"μ = {np.mean(pss_vals):.3f}")
    axes[2].set_xlabel("PSS")
    axes[2].set_ylabel("Episodes")
    axes[2].set_title("(C) PSS Distribution")
    axes[2].legend(fontsize=9)

    fig.suptitle("SmolVLA Base — Zero-shot Evaluation on Genesis / SO-101  (N=25)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = out_dir / "zeroshot_combined.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {path}")


def write_summary(records, tsr, lo, hi, sg_rates, out_dir):
    n        = len(records)
    mean_pss = np.mean([r["pss"] for r in records])
    mean_steps_success = np.mean([r["steps"] for r in records if r["success"]] or [0])
    mean_steps_fail    = np.mean([r["steps"] for r in records if not r["success"]] or [0])

    lines = [
        "=" * 60,
        "SmolVLA Base — Zero-shot Evaluation Summary",
        "=" * 60,
        f"  N rollouts          : {n}",
        f"  Successes           : {sum(r['success'] for r in records)}",
        f"  TSR                 : {tsr:.1%}",
        f"  95% CI (Wilson)     : [{lo:.3f}, {hi:.3f}]",
        f"  Mean PSS            : {mean_pss:.3f}",
        "",
        "  Sub-goal completion:",
        f"    Reached block     : {sg_rates['Reached\\nblock']:.1%}",
        f"    Lifted block      : {sg_rates['Lifted\\nblock']:.1%}",
        f"    Placed on target  : {sg_rates['Placed\\non target']:.1%}",
        "",
        f"  Mean steps (success): {mean_steps_success:.0f}",
        f"  Mean steps (fail)   : {mean_steps_fail:.0f}",
        "",
        "  LaTeX table row:",
        f"    Zero-shot & {tsr:.0%} & [{lo:.2f},{hi:.2f}] & {mean_pss:.3f} \\\\",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    path = out_dir / "zeroshot_summary.txt"
    path.write_text(text)
    print(f"[ok] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/zeroshot.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        return

    out_dir = csv_path.parent
    records = load_csv(csv_path)
    print(f"Loaded {len(records)} episodes from {csv_path}\n")

    tsr, lo, hi = plot_tsr(records, out_dir)
    sg_rates    = plot_subgoals(records, out_dir)
    plot_pss_distribution(records, out_dir)
    plot_combined(records, out_dir)
    write_summary(records, tsr, lo, hi, sg_rates, out_dir)

    print(f"\nAll outputs written to {out_dir}/")
    print("Videos are in results/videos/  (written during eval run)")


if __name__ == "__main__":
    main()