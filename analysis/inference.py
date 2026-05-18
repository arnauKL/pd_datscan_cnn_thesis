"""
Zero-shot inference on all patient groups using the best trained CNN.
The model was trained ONLY on HC vs PD, SWEDD and Prodromal are unseen.

SWEDD: Should score HC-like (low PD probability).
In fact I hope so since it confirms it learned the dopaminergic signal
         and not a confound like age or head size.

Prodromal: Expected to score somewhere between HC and PD, they have
           subtle dopaminergic changes before clinical diagnosis.

           If they score PD-like, the model detects deficit before
           clinical threshold. A key result if data is available.
           Sadly, though, the data is not available


Output: analysis/outputs/group_inference/
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset

sys.path.insert(0, os.path.abspath('.'))
from src.architectures import ParkinsonClassifier25D
from src.transforms import get_25d_transforms_padding

# setup fonts for plots
import matplotlib.font_manager as fm
import matplotlib
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 12

CONFIG = {
    "weights_path": "outputs/25d_resnet_raw_2fold_20260508_173440/fold_1/best_model.pth",
    "all_groups_csv": "data/ppmi_raw_allgroups_mapping.csv",
    "roi_size":     (76, 76, 76),
    "dropout":      0.3,
    "batch_size":   8,
    "output_dir":   "analysis/outputs/group_inference",
}

# config to add/remove prodromal in case I'd got them
GROUP_ORDER  = ["HC", "PD", "SWEDD"] #, "Prodromal"]
GROUP_COLORS = {
    "HC":        "#4C72B0",   # blue
    "PD":        "#DD8452",   # orange
    "SWEDD":     "#55A868",   # green
    #"Prodromal": "#C44E52",   # red
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = ParkinsonClassifier25D(dropout_rate=CONFIG["dropout"])
model.load_state_dict(torch.load(CONFIG["weights_path"], map_location=device))
model.to(device)
model.eval()
print(f"Loaded: {CONFIG['weights_path']}\n")

transform = get_25d_transforms_padding(CONFIG["roi_size"])


def run_inference(df, group_name):
    files  = [{"image": p, "label": 0} for p in df["path"]]
    ds     = MonaiDataset(data=files, transform=transform)
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).cpu().numpy().flatten()
            probs.extend(1 / (1 + np.exp(-logits)))
    probs = np.array(probs)
    classified_pd = (probs > 0.5).mean() * 100
    print(f"  {group_name:12s}  n={len(probs):4d}  "
          f"mean={probs.mean():.3f}  median={np.median(probs):.3f}  "
          f"classified as PD={classified_pd:.1f}%")
    return probs


# Load CSV and run per group
all_df = pd.read_csv(CONFIG["all_groups_csv"])
available_groups = all_df["group"].unique()
print(f"Groups found in CSV: {list(available_groups)}")
print(f"\nInference (model outputs PD probability, higher means more PD-like):")

group_probs = {}
for grp in GROUP_ORDER:
    if grp not in available_groups:
        print(f"  {grp:12s}  [not in CSV, skipping]")
        continue
    subset = all_df[all_df["group"] == grp].reset_index(drop=True)
    group_probs[grp] = run_inference(subset, grp)

present_groups = list(group_probs.keys())


# Statistical tests
# this was suggested by Claude
print(f"\nMann-Whitney U tests (non-parametric, no normality assumption):")
stat_rows = []
pairs = [
    ("HC",   "SWEDD",     "Key: SWEDD should look like HC"),
    ("HC",   "Prodromal", "Key: Prodromal may differ from HC"),
    ("PD",   "SWEDD",     "Key: SWEDD should differ from PD"),
    ("PD",   "Prodromal", "Key: Prodromal vs established PD"),
    ("HC",   "PD",        "Sanity check: HC vs PD separation"),
]

for g1, g2, note in pairs:
    if g1 not in group_probs or g2 not in group_probs:
        continue
    u_stat, p_val = stats.mannwhitneyu(
        group_probs[g1], group_probs[g2], alternative="two-sided")
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {g1} vs {g2:12s}  U={u_stat:.0f}  p={p_val:.4f} {sig}   {note}")
    stat_rows.append({"comparison": f"{g1} vs {g2}", "U": u_stat,
                      "p_value": p_val, "significance": sig, "note": note})

pd.DataFrame(stat_rows).to_csv(
    os.path.join(CONFIG["output_dir"], "statistical_tests.csv"), index=False)


# Plot Violin + strip plot
fig, ax = plt.subplots(figsize=(max(6, len(present_groups) * 2), 6))

positions = list(range(1, len(present_groups) + 1))
data      = [group_probs[g] for g in present_groups]
colors    = [GROUP_COLORS[g] for g in present_groups]

parts = ax.violinplot(data, positions=positions,
                      showmedians=True, showextrema=True)
for pc, color in zip(parts["bodies"], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.65)
for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
    parts[part].set_color("black")
    parts[part].set_linewidth(1.2)
parts["cmedians"].set_linewidth(2.5)

# Jittered individual points
rng = np.random.RandomState(42)
for pos, d, color in zip(positions, data, colors):
    jitter = rng.uniform(-0.08, 0.08, size=len(d))
    ax.scatter(pos + jitter, d, alpha=0.4, s=14,
               color=color, zorder=3, linewidths=0)

# Annotate median and n
for pos, d, grp in zip(positions, data, present_groups):
    ax.text(pos, -0.07, f"n={len(d)}", ha="center",
            va="top", fontsize=8, color="#444444")
    ax.text(pos, np.median(d) + 0.03, f"{np.median(d):.2f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.axhline(0.5, color="black", linestyle="--",
           linewidth=1.2, alpha=0.6, label="Decision boundary (0.5)")
ax.set_xticks(positions)
ax.set_xticklabels(present_groups, fontsize=12)
ax.set_ylabel("Model PD probability", fontsize=12)
ax.set_ylim(-0.12, 1.12)
ax.set_title("Zero-shot model output by patient group\n"
             "(trained on HC vs PD only, SWEDD never seen)",
             fontsize=12)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
out = os.path.join(CONFIG["output_dir"], "group_probabilities_violin.svg")
plt.savefig(out, dpi=160)
plt.close()
print(f"\nSaved: {out}")


# Plot: Histogram per group (distributions side by side)
fig, axes = plt.subplots(1, len(present_groups),
                          figsize=(len(present_groups) * 3.5, 4),
                          sharey=True)
if len(present_groups) == 1:
    axes = [axes]

for ax, grp in zip(axes, present_groups):
    d = group_probs[grp]
    ax.hist(d, bins=20, range=(0, 1), color=GROUP_COLORS[grp],
            alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.axvline(np.median(d), color="red", linestyle="-",
               linewidth=1.5, alpha=0.8, label=f"median={np.median(d):.2f}")
    ax.set_title(f"{grp}\n(n={len(d)})", fontsize=11)
    ax.set_xlabel("PD probability", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Count", fontsize=10)
fig.suptitle("Distribution of model PD probabilities by group",
             fontsize=12, y=1.02)
plt.tight_layout()
out2 = os.path.join(CONFIG["output_dir"], "group_probabilities_hist.svg")
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# Save raw probabilities
rows = []
for grp, probs in group_probs.items():
    for p in probs:
        rows.append({
            "group":          grp,
            "pd_probability": round(float(p), 4),
            "predicted":      "PD" if p > 0.5 else "HC",
        })
prob_df = pd.DataFrame(rows)
prob_df.to_csv(os.path.join(CONFIG["output_dir"],
               "group_probabilities.csv"), index=False)

# Summary table
print(f"\nSummary table:")
print(f"{'Group':<12}  {'n':>4}  {'Mean':>6}  {'Median':>7}  "
      f"{'Std':>5}  {'%PD':>6}")
for grp in present_groups:
    d = group_probs[grp]
    print(f"{grp:<12}  {len(d):>4}  {d.mean():>6.3f}  "
          f"{np.median(d):>7.3f}  {d.std():>5.3f}  "
          f"{(d>0.5).mean()*100:>5.1f}%")

print(f"\nAll outputs in: {CONFIG['output_dir']}")