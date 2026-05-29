import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import  matplotlib.patches as mpatches
import numpy as np
import matplotlib.font_manager as fm
import matplotlib

OUTPUTS_DIR = "outputs"
FIGURES_DIR = "evaluate/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Focus strictly on the core benchmarks for the main text body
METRICS = ["auc", "f1"]
METRIC_LABELS = {"auc": "AUC-ROC Performance", "f1": "F1-Score Performance"}

COLORS = {
    "registered": "#AD9DE6",  # Deep Slate Purple
    "raw": "#eab650",
}

try:
    fm._load_fontmanager(try_read_cache=False)
    fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
    matplotlib.rcParams["font.family"] = "Linux Libertine"
    matplotlib.rcParams["font.size"] = 12
except Exception as e:
    print(f"Font bypass: {e}")

def collect_data():
    rows = []
    for run_name in sorted(os.listdir(OUTPUTS_DIR)):
        run_dir = os.path.join(OUTPUTS_DIR, run_name)
        fold_csv = os.path.join(run_dir, "fold_metrics.csv")
        cfg_file = os.path.join(run_dir, "config.json")

        if not os.path.isdir(run_dir) or not os.path.exists(fold_csv) or not os.path.exists(cfg_file):
            continue

        with open(cfg_file) as f:
            cfg = json.load(f)
        
        # Skip multimodal runs for this specific chart
        model_key = cfg.get("model", "unknown")
        if "late" in model_key or "feature" in model_key:
            continue

        rows.append(pd.read_csv(fold_csv).assign(model=model_key, data=cfg.get("data", "unknown")))
    return pd.concat(rows, ignore_index=True)

def main():
    df = collect_data()
    
    # Clean pair grouping logic
    unique_models = sorted(df["model"].unique())
    paired_order = [(m, c) for m in unique_models for c in ["raw", "registered"] 
                    if not df[(df["model"] == m) & (df["data"] == c)].empty]
    
    x_labels = [f"{m.replace('3d_', '').replace('2d_', '').replace('_encoder', '')}\n({c})" 
                for m, c in paired_order]

    # Clean 1x2 horizontal layout
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, metric in zip(axes, METRICS):
        data = [df[(df["model"] == m) & (df["data"] == c)][metric].dropna().values for m, c in paired_order]
        colors = [COLORS.get(c, "#aaaaaa") for m, c in paired_order]

        bp = ax.boxplot(data, patch_artist=True, labels=x_labels,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor('#333333')

        for i, d in enumerate(data):
            if len(d) == 0: continue
            med = np.median(d)
            y_pos = med - 0.015 if med > 0.98 else med + 0.003
            ax.text(i + 1, y_pos, f"{med:.3f}", ha="center", va="bottom", 
                    fontsize=8, color="white" if med > 0.98 else "#222222",
                    weight='bold' if med > 0.98 else 'normal')

        ax.set_title(METRIC_LABELS[metric], fontsize=13, weight="bold")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5, rotation=10)
        ax.set_ylim(max(0.7, np.concatenate(data).min() - 0.03), 1.02)

    legend_handles = [
        mpatches.Patch(color=COLORS["raw"], label="Raw (Unregistered) Volumetric Regions of Interest"),
        mpatches.Patch(color=COLORS["registered"], label="Anatomically Registered Space (MNI Template)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=11)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "unimodal_baselines_brief_1x2.svg")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Main text graphic generated at: {out_path}")

if __name__ == "__main__":
    main()