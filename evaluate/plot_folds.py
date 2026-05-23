"""
Reads fold_metrics.csv from every outputs/ subfolder and plots one boxplot per metric comparing all models.

Use from project root:
    python evaluate/plot_folds.py

Output:
    evaluate/figures/boxplot_all.svg
    evaluate/figures/boxplot_<metric>.svg  (one x metric)

"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUTS_DIR = "outputs"
FIGURES_DIR = "evaluate/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

METRICS     = ["auc", "f1", "accuracy", "recall", "precision"]

METRIC_LABELS = {
    "auc":       "AUC-ROC",
    "f1":        "F1 Score",
    "accuracy":  "Accuracy",
    "recall":    "Recall (Sensitivity)",
    "precision": "Precision",
}

# Color per data source, still needs to be tweaked
COLOR = {
    "registered": "#444499",   # signature purple
    "raw":        "#55A868",   # 
    "unknown":    "#aaaaaa",   # grey (old-format runs without data tag)
}

# pretty Linux Libertine font for cohesion
import matplotlib.font_manager as fm
import matplotlib
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 14


#  Collect all fold_metrics.csv files
def collect() -> pd.DataFrame:
    rows = []
    for run_name in sorted(os.listdir(OUTPUTS_DIR)):
        run_dir  = os.path.join(OUTPUTS_DIR, run_name)
        fold_csv = os.path.join(run_dir, "fold_metrics.csv")
        cfg_file = os.path.join(run_dir, "config.json")

        if not os.path.isdir(run_dir) or not os.path.exists(fold_csv):
            continue
        if not os.path.exists(cfg_file):
            continue

        with open(cfg_file) as f:
            cfg = json.load(f)

        model_key = cfg.get("model", "unknown")
        data_key  = cfg.get("data",  "unknown")
        folds     = cfg.get("folds", "?")

        # Human-readable label: "3d_crop / raw" etc.
        label = f"{model_key}\n({data_key})"

        fold_df = pd.read_csv(fold_csv)
        fold_df["label"]    = label
        fold_df["model"]    = model_key
        fold_df["data"]     = data_key
        fold_df["folds"]    = folds
        fold_df["run_name"] = run_name
        rows.append(fold_df)

    if not rows:
        raise RuntimeError(
            f"No fold_metrics.csv files found under '{OUTPUTS_DIR}'. "
            "Run train.py first.")

    return pd.concat(rows, ignore_index=True)


# Plotting helpers
def plot_metric(df: pd.DataFrame, metric: str, ax: plt.Axes, model_order: list):
    """Draw one boxplot panel for `metric` onto `ax`."""

    data, labels, colors = [], [], []

    for lbl in model_order:
        subset = df[df["label"] == lbl][metric].dropna()
        if subset.empty:
            continue
        data.append(subset.values)
        labels.append(lbl)
        data_src = df[df["label"] == lbl]["data"].iloc[0]
        colors.append(COLOR.get(data_src, COLOR["unknown"]))

    if not data:
        ax.set_visible(False)
        return

    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
        flierprops=dict(marker="o", markersize=4, alpha=0.45, linestyle="none"),
        widths=0.5,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    # Annotate median value above each box
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + 0.004, f"{med:.3f}", ha="center", va="bottom", fontsize=7.5, color="#222222")

    # Light grid, clean spines
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)

    # y-axis range with a lil headroom
    all_vals = np.concatenate(data)
    lo = max(0.0, all_vals.min() - 0.06)
    ax.set_ylim(lo, 1.06)


#  Main
def main():
    df = collect()

    print(f"Loaded {len(df)} fold rows from {df['run_name'].nunique()} runs\n")

    # Sort models by median AUC descending so best is leftmost
    model_order = (
        df.groupby("label")["auc"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR["registered"], label="Registered images"),
        mpatches.Patch(color=COLOR["raw"],        label="Raw (unregistered) images"),
    ]

    # Individual plot per metric
    for metric in METRICS:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(len(model_order), 6))
        plot_metric(df, metric, ax, model_order)
        ax.set_title(f"Model comparison: {METRIC_LABELS.get(metric, metric)}",
                     fontsize=13, pad=10)
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, f"boxplot_{metric}.svg")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")

    # Combined figure
    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(n * len(model_order), 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        plot_metric(df, metric, ax, model_order)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)

    fig.suptitle("Model comparison — all metrics", fontsize=13, y=1.01)
    axes[-1].legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    out_all = os.path.join(FIGURES_DIR, "boxplot_all.svg")
    fig.savefig(out_all, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_all}")

    # Summary table (claude did this for me)
    print("\nMedian metrics per model (sorted by AUC)")
    summary = (
        df.groupby("label")[METRICS]
        .agg(["median", "std"])
        .round(3)
    )
    # Flatten multi-index columns: (auc, median) -> auc_median
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.loc[model_order]  # apply AUC sort

    # Print neatly: median \pm std for each metric
    print(f"\n{'Model':<30}", end="")
    for m in METRICS:
        print(f"  {m:^13}", end="")
    print()

    for lbl in model_order:
        r = summary.loc[lbl]
        print(f"{lbl.replace(chr(10), ' '):<30}", end="")
        for m in METRICS:
            med = r.get(f"{m}_median", float("nan"))
            std = r.get(f"{m}_std",    float("nan"))
            print(f"  {med:.3f}±{std:.3f}", end="")
        print()

if __name__ == "__main__":
    main()