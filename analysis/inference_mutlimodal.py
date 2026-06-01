"""
Multimodal Late-Fusion Zero-Shot Inference on HC, PD, and SWEDD patients.
Combines your best pretrained 2.5D CNN with a baseline clinical tabular model.

Output: analysis/outputs/group_inference_multimodal/
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Setup fonts for publication-quality plots (matching your style)
import matplotlib.font_manager as fm
import matplotlib
try:
    fm._load_fontmanager(try_read_cache=False)
    fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
    matplotlib.rcParams["font.family"] = "Linux Libertine"
except Exception:
    pass
matplotlib.rcParams["font.size"] = 12

CONFIG = {
    "weights_path": "outputs/25d_resnet_raw_2fold_20260508_173440/fold_1/best_model.pth",
    "all_groups_csv": "data/ppmi_raw_allgroups_mapping.csv",
    "tabular_csv": "data/ppmi_tabular.csv",
    "roi_size":     (76, 76, 76),
    "dropout":      0.3,
    "batch_size":   8,
    "output_dir":   "analysis/outputs/group_inference_multimodal",
    "seed":         42,
    "alpha":        0.5  # Equal blending weight between CNN and Tabular branches
}

# Unified clinical features across all groups from multimodal_cnn.py
FEATURE_COLS = ['upsit', 'updrs3_score', 'age', 'SEX', 'moca']

GROUP_ORDER  = ["HC", "PD", "SWEDD"]
GROUP_COLORS = {
    "HC":        "#444499",   # Purple
    "PD":        "#DD8452",   # Orange
    "SWEDD":     "#55A868",   # Green
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CNN Backbone Architecture
sys.path.insert(0, os.path.abspath('.'))
from src.architectures import ParkinsonClassifier25D
from src.transforms import get_25d_transforms_padding

model = ParkinsonClassifier25D(dropout_rate=CONFIG["dropout"])
model.load_state_dict(torch.load(CONFIG["weights_path"], map_location=device))
model.to(device)
model.eval()
print(f"Loaded Pretrained CNN: {CONFIG['weights_path']}\n")

transform = get_25d_transforms_padding(CONFIG["roi_size"])

# Data
print("Loading datasets and matching patient IDs...")
all_df = pd.read_csv(CONFIG["all_groups_csv"])
tab_df = pd.read_csv(CONFIG["tabular_csv"])

def extract_patno(path):
    match = re.search(r'sub-PPMI(\d+)', path)
    return int(match.group(1)) if match else None

all_df['PATNO'] = all_df['path'].apply(extract_patno)
tab_df['PATNO'] = pd.to_numeric(tab_df['PATNO'], errors='coerce')

# Filter for baseline visit only
tab_df = tab_df[tab_df['EVENT_ID'] == 'BL']
tab_subset = tab_df[['PATNO'] + FEATURE_COLS].drop_duplicates(subset='PATNO')

# Merge image tracking sheet with tabular features
merged = all_df.merge(tab_subset, on='PATNO', how='inner')

# Exclude rows with missing clinical variables for a fair complete-case analysis
merged_clean = merged.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"Successfully aligned {len(merged_clean)} complete-record patients:")
for grp in GROUP_ORDER:
    print(f"  {grp:6s}: {len(merged_clean[merged_clean['group'] == grp])} patients")


# CLINICAL TABULAR BRANCH
print("\nTraining tabular model on benchmark groups (HC vs PD)...")
hc_train = merged_clean[merged_clean['group'] == 'HC']
pd_train = merged_clean[merged_clean['group'] == 'PD']

# Balance classes to prevent decision boundary shift
min_samples = min(len(hc_train), len(pd_train))
hc_balanced = hc_train.sample(n=min_samples, random_state=CONFIG["seed"])
pd_balanced = pd_train.sample(n=min_samples, random_state=CONFIG["seed"])
train_tab_df = pd.concat([hc_balanced, pd_balanced]).reset_index(drop=True)
train_tab_df['label'] = train_tab_df['group'].apply(lambda x: 1 if x == 'PD' else 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_tab_df[FEATURE_COLS].values)
y_train = train_tab_df['label'].values

lr_model = LogisticRegression(max_iter=1000, random_state=CONFIG["seed"])
lr_model.fit(X_train, y_train)
print("Tabular Logistic Regression calibrator ready.")


# multimodal lop
def run_cnn_inference(df):
    files  = [{"image": p, "label": 0} for p in df["path"]]
    ds     = MonaiDataset(data=files, transform=transform)
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).cpu().numpy().flatten()
            probs.extend(1 / (1 + np.exp(-logits)))
    return np.array(probs)

results = {}
summary_rows = []

print(f"\nProcessing zero-shot evaluations (Late Fusion Blend Ratio alpha={CONFIG['alpha']}):")
for grp in GROUP_ORDER:
    group_df = merged_clean[merged_clean['group'] == grp].reset_index(drop=True)
    if len(group_df) == 0:
        continue
        
    # Generate image branch probabilities
    cnn_probs = run_cnn_inference(group_df)
    
    # Generate tabular branch probabilities
    X_group = scaler.transform(group_df[FEATURE_COLS].values)
    tab_probs = lr_model.predict_proba(X_group)[:, 1]
    
    # Blend outputs
    fused_probs = CONFIG["alpha"] * cnn_probs + (1 - CONFIG["alpha"]) * tab_probs
    
    results[grp] = {
        "cnn": cnn_probs,
        "tabular": tab_probs,
        "fused": fused_probs
    }
    
    summary_rows.append({
        "Group": grp,
        "n": len(group_df),
        "CNN_Mean": round(cnn_probs.mean(), 3),
        "CNN_Median": round(np.median(cnn_probs), 3),
        "CNN_pctPD": round((cnn_probs > 0.5).mean() * 100, 1),
        "Multimodal_Mean": round(fused_probs.mean(), 3),
        "Multimodal_Median": round(np.median(fused_probs), 3),
        "Multimodal_pctPD": round((fused_probs > 0.5).mean() * 100, 1),
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(CONFIG["output_dir"], "multimodal_vs_cnn_comparison.csv"), index=False)


# --- 4. STATISTICAL HYPOTHESIS TESTING ---
print("\nRunning Mann-Whitney U Tests on Multimodal Late Fusion outputs:")
stat_rows = []
pairs = [("HC", "SWEDD", "SWEDD vs HC"), ("PD", "SWEDD", "SWEDD vs PD"), ("HC", "PD", "Sanity Check")]
for g1, g2, note in pairs:
    if g1 in results and g2 in results:
        u_stat, p_val = stats.mannwhitneyu(results[g1]["fused"], results[g2]["fused"], alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {g1} vs {g2:6s} -> U={u_stat:.1f}, p={p_val:.4f} ({sig}) | {note}")
        stat_rows.append({"comparison": f"{g1} vs {g2}", "U": u_stat, "p_value": p_val, "significance": sig, "note": note})
pd.DataFrame(stat_rows).to_csv(os.path.join(CONFIG["output_dir"], "statistical_tests_multimodal.csv"), index=False)


# --- 5. VISUALIZATION: SIDE-BY-SIDE PANELS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
present_groups = [g for g in GROUP_ORDER if g in results]
positions = list(range(1, len(present_groups) + 1))

# Panel A: Image Only
parts1 = ax1.violinplot([results[g]["cnn"] for g in present_groups], positions=positions, showmedians=True)
ax1.set_title("A: Pretrained CNN Only (Image Only)", fontsize=13, fontweight="bold")

# Panel B: Late Fusion Multimodal
parts2 = ax2.violinplot([results[g]["fused"] for g in present_groups], positions=positions, showmedians=True)
ax2.set_title(f"B: Multimodal Late Fusion (CNN + Tabular, α={CONFIG['alpha']})", fontsize=13, fontweight="bold")

for ax, parts, key in [(ax1, parts1, "cnn"), (ax2, parts2, "fused")]:
    colors = [GROUP_COLORS[g] for g in present_groups]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
        parts[part].set_color("black")
        parts[part].set_linewidth(1.2)
    parts["cmedians"].set_linewidth(2.5)
    
    # Adding jittered strip plot points over violins
    rng = np.random.RandomState(CONFIG["seed"])
    for pos, g in zip(positions, present_groups):
        d = results[g][key]
        jitter = rng.uniform(-0.07, 0.07, size=len(d))
        ax.scatter(pos + jitter, d, alpha=0.35, s=12, color=GROUP_COLORS[g], zorder=3)
        
        # Display sample size and medians
        ax.text(pos, -0.06, f"n={len(d)}", ha="center", va="top", fontsize=9, color="#444444")
        ax.text(pos, np.median(d) + 0.02, f"{np.median(d):.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.5, label="Decision Boundary (0.5)")
    ax.set_xticks(positions)
    ax.set_xticklabels(present_groups, fontsize=12)
    ax.set_ylim(-0.12, 1.12)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.set_ylabel("Predicted PD Probability", fontsize=12)
plt.tight_layout()

out_plot = os.path.join(CONFIG["output_dir"], "multimodal_vs_cnn_violin.svg")
plt.savefig(out_plot, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nSaved comparative visualization to: {out_plot}")
print(f"All tables and tests written to: {CONFIG['output_dir']}/")
print("Execution successfully complete!")