"""
analyze_all_results.py
-----------------------
Reads ALL evaluation CSV files produced by zeroshot_eval.py and generates
publication-quality figures suitable for a robotics paper.

Usage (run from the project root, where results/ lives):
    python training/analyze_all_results.py --results_root results/

Output (all written to results/figures/):
    fig1_overview_bar.pdf/png       — TSR + PSS for every condition
    fig2_camera_ablation.pdf/png    — Single-camera-only ablation
    fig3_camera_combo.pdf/png       — Two-camera-present ablation
    fig4_missing_camera.pdf/png     — "Absent" (dropped-input) ablation
    fig5_scene_perturb.pdf/png      — Scene-level perturbations
    fig6_combined_4panel.pdf/png    — Paper-ready 4-panel summary
    summary_table.txt               — LaTeX-ready table

Requirements: numpy, matplotlib, scipy (for Wilson CI)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Palette (colourblind-friendly)
C_BLUE   = "#4C8BF5"
C_GREEN  = "#34A853"
C_AMBER  = "#FBBC04"
C_RED    = "#EA4335"
C_PURPLE = "#9B59B6"
C_TEAL   = "#17A589"
C_GRAY   = "#95A5A6"
C_DARK   = "#2C3E50"


# ══════════════════════════════════════════════════════════════════════════════
#  Data registry  — maps condition label → CSV path
#  Edit these paths if you stored files elsewhere.
# ══════════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    # ── Baseline ──────────────────────────────────────────────────────────────
    "Baseline\n(all cams)": {
        "csv":   "results/baseline_nominal/zeroshot.csv",
        "group": "baseline",
        "color": C_GREEN,
    },

    # ── Single-camera-only (blackout: only one camera shown to policy) ────────
    "Context\nonly": {
        "csv":   "results/pert_camnum/zeroshot_context_only.csv",   # context_only
        "group": "single_cam",
        "color": C_BLUE,
        "note":  "context_only",
    },
    "Wrist\nonly": {
        "csv":   "results/pert_camnum/zeroshot.csv",   # wrist_only  — same dir, different file
        "csv_alt": "results/pert_camnum/zeroshot_wrist_only.csv",
        "group": "single_cam",
        "color": C_AMBER,
        "note":  "wrist_only",
    },
    "Top\nonly": {
        "csv":   "results/pert_camnum/zeroshot_top_only.csv",
        "group": "single_cam",
        "color": C_PURPLE,
        "note":  "top_only",
    },

    # ── Two-camera combos (one camera blacked out) ────────────────────────────
    "No wrist\n(ctx+top)": {
        "csv":   "results/pert_camcombo/zeroshot_context_top.csv",
        "group": "two_cam",
        "color": C_BLUE,
    },
    "No top\n(ctx+wrist)": {
        "csv":   "results/pert_camcombo/zeroshot_wrist_context.csv",
        "group": "two_cam",
        "color": C_TEAL,
    },
    "No context\n(wrist+top)": {
        "csv":   "results/pert_camcombo/zeroshot_wrist_top.csv",
        "group": "two_cam",
        "color": C_AMBER,
    },

    # ── Missing / absent cameras (input tensor absent, not zeroed) ─────────────
    "Absent top": {
        "csv":   "results/pert_missing/zeroshot_wrist_context.csv",
        "group": "absent",
        "color": C_TEAL,
    },
    "Absent\nwrist+top": {
        "csv":   "results/pert_missing/zeroshot_context_only.csv",
        "group": "absent",
        "color": C_RED,
    },

    # ── Camera position perturbation ──────────────────────────────────────────
    "Wrist shift\n+3 cm": {
        "csv":   "results/pert_campos/zeroshot.csv",
        "group": "cam_pos",
        "color": C_AMBER,
    },

    # ── Scene perturbation ────────────────────────────────────────────────────
    "Two flanking\nspheres": {
        "csv":   "results/pert_spheres/zeroshot.csv",
        "group": "scene",
        "color": C_RED,
    },
}

# Fallback TSR values from your terminal output (used when CSV is missing or
# when two conditions share the same CSV file):
KNOWN_TSR = {
    "Baseline\n(all cams)":    0.87,
    "Context\nonly":           0.33,
    "Wrist\nonly":             0.00,
    "Top\nonly":               0.00,
    "No wrist\n(ctx+top)":     0.58,
    "No top\n(ctx+wrist)":     0.63,
    "No context\n(wrist+top)": 0.00,
    "Absent top":              0.00,
    "Absent\nwrist+top":       0.00,
    "Wrist shift\n+3 cm":      0.80,
    "Two flanking\nspheres":   0.20,
}

KNOWN_PSS = {
    "Baseline\n(all cams)":    1.388,
    "Context\nonly":           0.592,
    "Wrist\nonly":             0.024,
    "Top\nonly":               0.032,
    "No wrist\n(ctx+top)":     0.944,
    "No top\n(ctx+wrist)":     1.048,
    "No context\n(wrist+top)": 0.044,
    "Absent top":              0.020,
    "Absent\nwrist+top":       0.000,
    "Wrist shift\n+3 cm":      1.326,
    "Two flanking\nspheres":   0.390,
}

N_EPISODES = 100   # all your runs used N=100


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def wilson_ci(p, n, z=1.96):
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            records.append({
                "success": row["success"].strip().lower() == "true",
                "pss":     float(row["pss"]),
                "steps":   int(row["steps"]),
                "reached": row.get("reached", "false").strip().lower() == "true",
                "lifted":  row.get("lifted",  "false").strip().lower() == "true",
                "placed":  row.get("placed",  "false").strip().lower() == "true",
            })
    return records


def load_all_conditions(conditions, n=N_EPISODES):
    """
    Return a dict mapping label → {tsr, pss, lo, hi, records}.
    Falls back to KNOWN_TSR / KNOWN_PSS if CSV is missing or unreadable.
    """
    data = {}
    for label, cfg in conditions.items():
        records = None

        # Try primary CSV first
        primary = cfg.get("csv")
        if primary:
            records = load_csv(primary)

        # Try alternate CSV (some conditions share dirs)
        if records is None:
            alt = cfg.get("csv_alt")
            if alt:
                records = load_csv(alt)

        if records and len(records) >= n // 2:
            tsr = sum(r["success"] for r in records) / len(records)
            pss = np.mean([r["pss"] for r in records])
        else:
            # Use hard-coded fallbacks from terminal output
            tsr = KNOWN_TSR.get(label, 0.0)
            pss = KNOWN_PSS.get(label, 0.0)
            records = None   # signal that we're using fallback

        lo, hi = wilson_ci(tsr, n)
        data[label] = {
            "tsr":     tsr,
            "pss":     pss,
            "lo":      lo,
            "hi":      hi,
            "records": records,
            "group":   cfg["group"],
            "color":   cfg["color"],
        }
        src = "CSV" if records else "fallback"
        print(f"  [{src}] {label.replace(chr(10),' '):35s}  TSR={tsr:.0%}  PSS={pss:.3f}")
    return data


def _save(fig, stem, out_dir):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"[ok] {stem}.pdf/png")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Full overview bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_overview(data, out_dir):
    labels = list(data.keys())
    tsrs   = [data[l]["tsr"] for l in labels]
    psss   = [data[l]["pss"] for l in labels]
    los    = [data[l]["tsr"] - data[l]["lo"] for l in labels]
    his    = [data[l]["hi"] - data[l]["tsr"] for l in labels]
    colors = [data[l]["color"] for l in labels]

    n = len(labels)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # TSR
    bars = ax1.bar(x, tsrs, color=colors, alpha=0.88, width=0.6,
                   yerr=[los, his], capsize=5,
                   error_kw={"elinewidth": 1.4, "ecolor": "#555"})
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Task Success Rate (TSR)", fontweight="bold")
    ax1.axhline(KNOWN_TSR["Baseline\n(all cams)"], color=C_GRAY,
                ls="--", lw=1.2, label="Baseline TSR")
    ax1.legend(fontsize=9, loc="upper right")
    for bar, tsr in zip(bars, tsrs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 min(tsr + 0.06, 1.08),
                 f"{tsr:.0%}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=C_DARK)

    # PSS
    ax2.bar(x, psss, color=colors, alpha=0.75, width=0.6)
    ax2.set_ylim(0, 1.65)
    ax2.set_ylabel("Mean PSS", fontweight="bold")
    ax2.axhline(KNOWN_PSS["Baseline\n(all cams)"], color=C_GRAY,
                ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    for i, pss in enumerate(psss):
        ax2.text(i, pss + 0.04, f"{pss:.2f}", ha="center",
                 fontsize=8, color=C_DARK)

    fig.suptitle(
        "SmolVLA Fine-tuned — Evaluation Across All Conditions  (N=100 per condition)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig1_overview_bar", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Single-camera ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig_single_cam(data, out_dir):
    groups = {
        "Baseline\n(all cams)":  ("All 3 cameras", C_GREEN),
        "Context\nonly":         ("Context only",  C_BLUE),
        "Wrist\nonly":           ("Wrist only",    C_AMBER),
        "Top\nonly":             ("Top only",      C_PURPLE),
    }
    labels_out = ["All 3\ncameras", "Context\nonly", "Wrist\nonly", "Top\nonly"]
    keys = list(groups.keys())
    colors = [v[1] for v in groups.values()]

    tsrs = [data[k]["tsr"] for k in keys]
    psss = [data[k]["pss"] for k in keys]
    los  = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his  = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, ylabel, title in [
        (ax1, (tsrs, [los, his]), "Task Success Rate (TSR)", "(A) TSR"),
        (ax2, (psss, None),       "Mean PSS",               "(B) PSS"),
    ]:
        if vals[1] is not None:
            bars = ax.bar(x, vals[0], color=colors, alpha=0.88, width=0.55,
                          yerr=vals[1], capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
            ax.set_ylim(0, 1.2)
        else:
            bars = ax.bar(x, vals[0], color=colors, alpha=0.88, width=0.55)
            ax.set_ylim(0, 1.7)

        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9.5)
        for bar, v in zip(bars, vals[0]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.03 if vals[1] is None else 0.06),
                    f"{v:.0%}" if vals[1] is not None else f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Single-Camera Ablation — Which camera matters most?\n"
        "(Each condition: only that camera's stream is shown to the policy)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig2_camera_ablation", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Two-camera combo ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig_two_cam(data, out_dir):
    keys  = [
        "Baseline\n(all cams)",
        "No top\n(ctx+wrist)",
        "No wrist\n(ctx+top)",
        "No context\n(wrist+top)",
    ]
    labels_out = [
        "All 3\ncams",
        "Ctx + Wrist\n(no top)",
        "Ctx + Top\n(no wrist)",
        "Wrist + Top\n(no ctx)",
    ]
    colors = [data[k]["color"] for k in keys]
    tsrs   = [data[k]["tsr"] for k in keys]
    psss   = [data[k]["pss"] for k in keys]
    los    = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his    = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, vals, yerr, ylabel, title, ylim in [
        (axes[0], tsrs, [los, his], "TSR", "(A) Task Success Rate", (0, 1.2)),
        (axes[1], psss, None,       "Mean PSS", "(B) Partial Success Score", (0, 1.7)),
    ]:
        if yerr:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55,
                          yerr=yerr, capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
        else:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.06 if yerr else 0.03),
                    f"{v:.0%}" if yerr else f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Two-Camera Combination Ablation\n"
        "(One camera zeroed per condition; policy still receives all 3 tensor slots)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig3_camera_combo", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Absent / missing camera ablation
# ══════════════════════════════════════════════════════════════════════════════

def fig_absent(data, out_dir):
    keys = [
        "Baseline\n(all cams)",
        "Absent top",
        "Absent\nwrist+top",
    ]
    labels_out = [
        "All 3 cams\n(baseline)",
        "Top absent\n(ctx+wrist remain)",
        "Wrist+Top absent\n(ctx only)",
    ]
    colors = [data[k]["color"] for k in keys]
    tsrs   = [data[k]["tsr"] for k in keys]
    psss   = [data[k]["pss"] for k in keys]
    los    = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his    = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, yerr, ylabel, title, ylim in [
        (axes[0], tsrs, [los, his], "TSR", "(A) Task Success Rate", (0, 1.2)),
        (axes[1], psss, None,       "Mean PSS", "(B) Partial Success Score", (0, 1.7)),
    ]:
        if yerr:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55,
                          yerr=yerr, capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
        else:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.06 if yerr else 0.03),
                    f"{v:.0%}" if yerr else f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Missing / Absent Camera Inputs\n"
        "(Camera tensor omitted entirely — tests robustness to missing modalities)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig4_missing_camera", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Scene perturbations
# ══════════════════════════════════════════════════════════════════════════════

def fig_scene(data, out_dir):
    keys = [
        "Baseline\n(all cams)",
        "Wrist shift\n+3 cm",
        "Two flanking\nspheres",
    ]
    labels_out = [
        "Baseline\n(nominal)",
        "Wrist cam\nshifted +3 cm",
        "Two distractor\nspheres",
    ]
    colors = [data[k]["color"] for k in keys]
    tsrs   = [data[k]["tsr"] for k in keys]
    psss   = [data[k]["pss"] for k in keys]
    los    = [data[k]["tsr"] - data[k]["lo"] for k in keys]
    his    = [data[k]["hi"] - data[k]["tsr"] for k in keys]

    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, yerr, ylabel, title, ylim in [
        (axes[0], tsrs, [los, his], "TSR", "(A) Task Success Rate", (0, 1.2)),
        (axes[1], psss, None,       "Mean PSS", "(B) Partial Success Score", (0, 1.7)),
    ]:
        if yerr:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55,
                          yerr=yerr, capsize=6,
                          error_kw={"elinewidth": 1.5, "ecolor": "#555"})
        else:
            bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.55)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_out, fontsize=9.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.06 if yerr else 0.03),
                    f"{v:.0%}" if yerr else f"{v:.3f}",
                    ha="center", fontsize=10.5, fontweight="bold", color=C_DARK)

    fig.suptitle(
        "Scene-Level Perturbations\n"
        "(Camera position shift vs. visual distractors)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "fig5_scene_perturb", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — 4-panel paper summary
# ══════════════════════════════════════════════════════════════════════════════

def fig_4panel(data, out_dir):
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])   # Camera importance (single)
    ax_b = fig.add_subplot(gs[0, 1])   # Two-cam combos
    ax_c = fig.add_subplot(gs[1, 0])   # Scene perturbations
    ax_d = fig.add_subplot(gs[1, 1])   # PSS radar / heatmap strip

    # ── Panel A: single-camera TSR ────────────────────────────────────────────
    cam_keys   = ["Baseline\n(all cams)", "Context\nonly", "Wrist\nonly", "Top\nonly"]
    cam_labels = ["All 3", "Ctx only", "Wrist only", "Top only"]
    cam_colors = [C_GREEN, C_BLUE, C_AMBER, C_PURPLE]
    cam_tsrs   = [data[k]["tsr"] for k in cam_keys]
    cam_lo     = [data[k]["tsr"] - data[k]["lo"] for k in cam_keys]
    cam_hi     = [data[k]["hi"] - data[k]["tsr"] for k in cam_keys]
    x = np.arange(len(cam_keys))

    bars = ax_a.bar(x, cam_tsrs, color=cam_colors, alpha=0.88, width=0.6,
                    yerr=[cam_lo, cam_hi], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": "#555"})
    ax_a.set_ylim(0, 1.2)
    ax_a.set_title("(A) Single-Camera Ablation", fontweight="bold")
    ax_a.set_ylabel("TSR")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cam_labels, fontsize=9)
    for bar, v in zip(bars, cam_tsrs):
        ax_a.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                  f"{v:.0%}", ha="center", fontsize=9, fontweight="bold", color=C_DARK)

    # ── Panel B: two-cam TSR ──────────────────────────────────────────────────
    two_keys   = ["Baseline\n(all cams)", "No top\n(ctx+wrist)", "No wrist\n(ctx+top)", "No context\n(wrist+top)"]
    two_labels = ["All 3", "Ctx+Wrist", "Ctx+Top", "Wrist+Top"]
    two_colors = [C_GREEN, C_TEAL, C_BLUE, C_AMBER]
    two_tsrs   = [data[k]["tsr"] for k in two_keys]
    two_lo     = [data[k]["tsr"] - data[k]["lo"] for k in two_keys]
    two_hi     = [data[k]["hi"] - data[k]["tsr"] for k in two_keys]

    bars = ax_b.bar(np.arange(len(two_keys)), two_tsrs, color=two_colors, alpha=0.88, width=0.6,
                    yerr=[two_lo, two_hi], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": "#555"})
    ax_b.set_ylim(0, 1.2)
    ax_b.set_title("(B) Two-Camera Combos (one zeroed)", fontweight="bold")
    ax_b.set_ylabel("TSR")
    ax_b.set_xticks(np.arange(len(two_keys)))
    ax_b.set_xticklabels(two_labels, fontsize=9)
    for bar, v in zip(bars, two_tsrs):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                  f"{v:.0%}", ha="center", fontsize=9, fontweight="bold", color=C_DARK)

    # ── Panel C: scene perturbations ──────────────────────────────────────────
    sc_keys   = ["Baseline\n(all cams)", "Wrist shift\n+3 cm", "Two flanking\nspheres"]
    sc_labels = ["Baseline", "Wrist +3 cm", "2 Spheres"]
    sc_colors = [C_GREEN, C_AMBER, C_RED]
    sc_tsrs   = [data[k]["tsr"] for k in sc_keys]
    sc_lo     = [data[k]["tsr"] - data[k]["lo"] for k in sc_keys]
    sc_hi     = [data[k]["hi"] - data[k]["tsr"] for k in sc_keys]

    bars = ax_c.bar(np.arange(len(sc_keys)), sc_tsrs, color=sc_colors, alpha=0.88, width=0.55,
                    yerr=[sc_lo, sc_hi], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": "#555"})
    ax_c.set_ylim(0, 1.2)
    ax_c.set_title("(C) Scene Perturbations", fontweight="bold")
    ax_c.set_ylabel("TSR")
    ax_c.set_xticks(np.arange(len(sc_keys)))
    ax_c.set_xticklabels(sc_labels, fontsize=9)
    for bar, v in zip(bars, sc_tsrs):
        ax_c.text(bar.get_x() + bar.get_width() / 2, v + 0.06,
                  f"{v:.0%}", ha="center", fontsize=9, fontweight="bold", color=C_DARK)

    # ── Panel D: summary heatmap (TSR + PSS strip) ────────────────────────────
    summary_labels = [
        "Baseline",
        "Ctx only",
        "Wrist only",
        "Top only",
        "No wrist",
        "No top",
        "No context",
        "Absent top",
        "Absent W+T",
        "Wrist +3cm",
        "2 Spheres",
    ]
    summary_keys = [
        "Baseline\n(all cams)",
        "Context\nonly",
        "Wrist\nonly",
        "Top\nonly",
        "No wrist\n(ctx+top)",
        "No top\n(ctx+wrist)",
        "No context\n(wrist+top)",
        "Absent top",
        "Absent\nwrist+top",
        "Wrist shift\n+3 cm",
        "Two flanking\nspheres",
    ]
    tsr_vals = np.array([data[k]["tsr"] for k in summary_keys])
    pss_vals = np.array([data[k]["pss"] / 1.6 for k in summary_keys])  # normalise PSS to [0,1]

    heatmap = np.vstack([tsr_vals, pss_vals])
    im = ax_d.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                     interpolation="nearest")
    ax_d.set_yticks([0, 1])
    ax_d.set_yticklabels(["TSR", "PSS\n(norm.)"], fontsize=9)
    ax_d.set_xticks(np.arange(len(summary_labels)))
    ax_d.set_xticklabels(summary_labels, fontsize=7.5, rotation=40, ha="right")
    ax_d.set_title("(D) Summary Heatmap (all conditions)", fontweight="bold")

    # Annotate cells
    for j in range(heatmap.shape[1]):
        for i in range(heatmap.shape[0]):
            v = heatmap[i, j]
            ax_d.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=7, color="black" if 0.3 < v < 0.8 else "white",
                      fontweight="bold")

    plt.colorbar(im, ax=ax_d, shrink=0.7, pad=0.02)

    fig.suptitle(
        "SmolVLA SO-101 — Perturbation Robustness Study  (N=100 per condition)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save(fig, "fig6_combined_4panel", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX summary table
# ══════════════════════════════════════════════════════════════════════════════

def write_latex_table(data, out_dir):
    groups = {
        "Baseline":             ["Baseline\n(all cams)"],
        "Single-camera":        ["Context\nonly", "Wrist\nonly", "Top\nonly"],
        "Two-camera combos":    ["No top\n(ctx+wrist)", "No wrist\n(ctx+top)", "No context\n(wrist+top)"],
        "Absent inputs":        ["Absent top", "Absent\nwrist+top"],
        "Scene perturbations":  ["Wrist shift\n+3 cm", "Two flanking\nspheres"],
    }

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{SmolVLA robustness evaluation (N=100 per condition, 95\\% Wilson CI).}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Group & Condition & TSR (\\%) & Mean PSS \\\\",
        "\\midrule",
    ]

    for group, keys in groups.items():
        for i, key in enumerate(keys):
            d    = data[key]
            name = key.replace("\n", " ")
            lo   = d["lo"] * 100
            hi   = d["hi"] * 100
            tsr  = d["tsr"] * 100
            pss  = d["pss"]
            ci   = f"[{lo:.1f}, {hi:.1f}]"
            grp  = group if i == 0 else ""
            lines.append(f"{grp} & {name} & ${tsr:.0f}$ {ci} & ${pss:.3f}$ \\\\")
        lines.append("\\midrule")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    text = "\n".join(lines)
    path = out_dir / "summary_table.txt"
    path.write_text(text)
    print(f"[ok] summary_table.txt")

    # Also write a human-readable version
    hr = ["=" * 72,
          f"SmolVLA Perturbation Robustness — N=100 per condition",
          "=" * 72,
          f"{'Condition':<35}  {'TSR':>6}  {'95% CI':>15}  {'PSS':>6}",
          "-" * 72]
    for label, d in data.items():
        name = label.replace("\n", " ")
        ci   = f"[{d['lo']:.3f},{d['hi']:.3f}]"
        hr.append(f"{name:<35}  {d['tsr']:>6.1%}  {ci:>15}  {d['pss']:>6.3f}")
    hr.append("=" * 72)
    (out_dir / "summary_readable.txt").write_text("\n".join(hr))
    print("[ok] summary_readable.txt")
    print("\n" + "\n".join(hr))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/",
                        help="Root directory containing all result sub-folders")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: results/figures/)")
    args = parser.parse_args()

    root    = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading conditions from {root}...\n")
    data = load_all_conditions(CONDITIONS)

    print(f"\nGenerating figures → {out_dir}/\n")
    fig_overview(data, out_dir)
    fig_single_cam(data, out_dir)
    fig_two_cam(data, out_dir)
    fig_absent(data, out_dir)
    fig_scene(data, out_dir)
    fig_4panel(data, out_dir)
    write_latex_table(data, out_dir)

    print(f"\n✓  All figures written to {out_dir}/")


if __name__ == "__main__":
    main()