"""
visualise_results.py  —  Plot comparison charts for all 5 architectures
------------------------------------------------------------------------
Reads comparison_summary.json and generates:
  1. Grouped bar chart: all metrics across all 5 architectures
  2. Radar chart: architecture profile (hit rate, MRR, NDCG, precision, recall)
  3. Latency vs NDCG scatter: efficiency vs quality trade-off
  4. Per-modality comparison: text vs image contribution per arch

Usage:
    python visualise_results.py
"""

import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "results"
)
SUMMARY_PATH = os.path.join(RESULTS_DIR, "comparison_summary.json")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")


ARCH_COLORS = ["#378ADD", "#1D9E75", "#BA7517", "#7F77DD", "#D85A30", "#0F6E56"]
ARCH_LABELS = {
    "1": "Naive",
    "2": "Hybrid RRF",
    "3": "Meta-Filter",
    "4": "Late Fusion",
    "5": "Query Exp.",
    "6": "Full CDE",
}
METRICS = ["hit_rate", "recall@3", "precision@3", "mrr", "ndcg@3"]
METRIC_LABELS = ["Hit Rate", "Recall@3", "Precision@3", "MRR", "NDCG@3"]


def load_summary():
    with open(SUMMARY_PATH, "r") as f:
        return json.load(f)


def plot_grouped_bars(summary: dict):
    archs  = sorted(summary["architectures"].keys(), key=int)
    n_arch = len(archs)
    n_met  = len(METRICS)
    x      = np.arange(n_met)
    width  = 0.15

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, arch_id in enumerate(archs):
        data   = summary["architectures"][arch_id]["combined"]
        values = [data.get(m, 0) for m in METRICS]
        bars   = ax.bar(x + i * width, values, width,
                        label=ARCH_LABELS.get(arch_id, f"Arch {arch_id}"),
                        color=ARCH_COLORS[i], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (n_arch - 1) / 2)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Retrieval Performance: All 5 Architectures (Combined Text + Image)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, min(1.1, ax.get_ylim()[1] + 0.05))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_grouped_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_radar(summary: dict):
    archs   = sorted(summary["architectures"].keys(), key=int)
    metrics = METRICS
    N       = len(metrics)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])

    for i, arch_id in enumerate(archs):
        data   = summary["architectures"][arch_id]["combined"]
        values = [data.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, color=ARCH_COLORS[i],
                label=ARCH_LABELS.get(arch_id, f"Arch {arch_id}"))
        ax.fill(angles, values, alpha=0.08, color=ARCH_COLORS[i])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Architecture Profile Radar", fontsize=11, pad=20)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_latency_vs_ndcg(summary: dict):
    archs = sorted(summary["architectures"].keys(), key=int)
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, arch_id in enumerate(archs):
        arch = summary["architectures"][arch_id]
        ndcg = arch["combined"].get("ndcg@3", 0)
        lat  = arch["latency_s"] * 1000  # ms
        name = ARCH_LABELS.get(arch_id, f"Arch {arch_id}")
        ax.scatter(lat, ndcg, color=ARCH_COLORS[i], s=120, zorder=5)
        ax.annotate(name, (lat, ndcg), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Average Latency (ms)", fontsize=10)
    ax.set_ylabel("NDCG@3", fontsize=10)
    ax.set_title("Quality vs Speed Trade-off", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_latency_vs_ndcg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_modality_comparison(summary: dict):
    archs      = sorted(summary["architectures"].keys(), key=int)
    arch_names = [ARCH_LABELS.get(a, f"Arch {a}") for a in archs]
    text_ndcg  = [summary["architectures"][a]["text"].get("ndcg@3", 0)  for a in archs]
    image_ndcg = [summary["architectures"][a]["image"].get("ndcg@3", 0) for a in archs]
    combo_ndcg = [summary["architectures"][a]["combined"].get("ndcg@3", 0) for a in archs]

    x     = np.arange(len(archs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, text_ndcg,  width, label="Text only",  color="#378ADD", alpha=0.85)
    ax.bar(x,         image_ndcg, width, label="Image only", color="#D85A30", alpha=0.85)
    ax.bar(x + width, combo_ndcg, width, label="Combined",   color="#1D9E75", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, fontsize=9)
    ax.set_ylabel("NDCG@3", fontsize=10)
    ax.set_title("Text vs Image vs Combined NDCG@3 per Architecture", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_modality_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(SUMMARY_PATH):
        print(f"ERROR: {SUMMARY_PATH} not found.")
        print("Run evaluate.py first to generate results.")
        return

    summary = load_summary()
    n       = summary.get("n_questions", "?")
    print(f"\nGenerating plots from {SUMMARY_PATH} (n={n} questions)...")

    plot_grouped_bars(summary)
    plot_radar(summary)
    plot_latency_vs_ndcg(summary)
    plot_modality_comparison(summary)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("Figures: 01_grouped_bars.png, 02_radar.png, 03_latency_vs_ndcg.png, 04_modality_comparison.png")


if __name__ == "__main__":
    main()