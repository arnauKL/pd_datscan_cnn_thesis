# Reads evaluate/bootstrap_results.csv and produces one boxplot per metric,
# comparing all models side by side.
# Usage:
#     python evaluate/plot_boxplots.py
# Outputs stored in
#     evaluate/figures/boxplot_{metric}.png  for each metric
#     evaluate/figures/boxplot_all.png       combined figure
#
# Again, this works only for the initial models, created with the old scripts

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# use the same font as the report
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)  # force rescan
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")  # explicit add
#[f.name for f in fm.fontManager.ttflist if "Libertine" in f.name or "Libertinus" in f.name]

import matplotlib
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 12

RESULTS_CSV = "evaluate/bootstrap_results.csv"
FIGURES_DIR = "evaluate/figures"
METRICS     = ["auc", "f1", "accuracy", "recall", "precision"]

os.makedirs(FIGURES_DIR, exist_ok=True)

# Shorten model names so they fit on the x-axis
NAME_MAP = {
    "3D_cropping_allimages":           "3D-Crop",
    "2D_suma_rawderiv":                "2D-Sum",
    "2D_nomes_striatum_rawderiv":      "2D-Striatum",
    "2D_suma_rawderiv_128_croppadding":"2D-Sum-128",
    "3D_padding_128_allimages":        "3D-Pad",
}

METRIC_LABELS = {
    "auc":       "AUC-ROC",
    "f1":        "F1 Score",
    "accuracy":  "Accuracy",
    "recall":    "Recall (Sensitivity)",
    "precision": "Precision",
}

COLOR   = "#4C72B0"   # blue, ion like it, it needs to be tweaked

def plot_single_metric(df: pd.DataFrame, metric: str, ax: plt.Axes, model_order: list):
    """Draw one boxplot panel onto ax."""

    data   = []
    labels = []
    colors = []

    for m in model_order:
        subset = df[df["model_name"] == m][metric].dropna()
        if subset.empty:
            continue
        data.append(subset.values)
        labels.append(NAME_MAP.get(m, m))
        colors.append(COLOR)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Add median value as text above each box
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(
            i + 1, med + 0.005,
            f"{med:.3f}",
            ha="center", va="bottom",
            fontsize=7, color="black"
        )

    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)
    ax.set_ylim(max(0, min(np.concatenate(data)) - 0.05), 1.05)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def main():
    df = pd.read_csv(RESULTS_CSV)

    # Consistent model order: sort by median AUC descending
    model_order = (
        df.groupby("model_name")["auc"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Individual plots (one file per metric)
    # for metric in METRICS:
    #     if metric not in df.columns:
    #         print(f"[SKIP] metric '{metric}' not in results CSV")
    #         continue

    #     fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.4), 5))
    #     plot_single_metric(df, metric, ax, model_order)
    #     ax.set_title(f"Model comparison — {METRIC_LABELS.get(metric, metric)}", fontsize=12, pad=12)
    #     fig.tight_layout()

    #     out = os.path.join(FIGURES_DIR, f"boxplot_{metric}.png")
    #     fig.savefig(out, dpi=150)
    #     plt.close(fig)
    #     print(f"Saved: {out}")

    # Combined figure
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(n_metrics * max(7, len(model_order) * 1.4) / 2.5, 5),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        plot_single_metric(df, metric, ax, model_order)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)

    fig.suptitle("Model comparison, all metrics", fontsize=13, y=1.02)
    fig.tight_layout()

    out_combined = os.path.join(FIGURES_DIR, "boxplot_all.png")
    fig.savefig(out_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_combined}")

    print("\n--- Median metrics per model (sorted by AUC) ---")
    summary = (
        df.groupby("model_name")[METRICS]
        .median()
        .loc[model_order]
        .round(3)
    )
    summary.index = [NAME_MAP.get(i, i) for i in summary.index]
    print(summary.to_string())

if __name__ == "__main__":
    main()