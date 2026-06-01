import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib

# Define the specific directory you want to plot from
RUN_DIR = "outputs/multimodal_20260529_010009"

fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 14
    

# getting data again
print(f"Reading evaluation data from: {RUN_DIR}")

with open(os.path.join(RUN_DIR, "config.json"), "r") as f:
    config = json.load(f)
num_folds = config.get("FOLDS", 2)

results_long = pd.read_csv(os.path.join(RUN_DIR, "results_long.csv"))
all_results = {}
models = results_long['model'].unique()

for m in models:
    fusion_type, feature_group = m.split('_', 1)
    
    if fusion_type == 'feature':
        # Read exact raw values directly from stored subfolder files
        csv_path = os.path.join(RUN_DIR, feature_group, "fold_metrics.csv")
        if os.path.exists(csv_path):
            all_results[m] = pd.read_csv(csv_path)
            
    elif fusion_type == 'late':
        # Reverse-engineer the 2 fold data points from the stored mean and std
        model_summary = results_long[results_long['model'] == m]
        fold_data = {'fold': list(range(1, num_folds + 1))}
        
        for metric in ['auc', 'f1', 'accuracy', 'recall', 'precision']:
            row = model_summary[model_summary['metric'] == metric]
            if not row.empty:
                mean_val = row['mean'].values[0]
                std_val = row['std'].values[0]
                
                if num_folds == 2:
                    # Pure algebraic extraction for N=2 (Bessel-corrected std)
                    offset = std_val / np.sqrt(2)
                    fold_data[metric] = [mean_val - offset, mean_val + offset]
                else:
                    # Fallback generation spread for general testing
                    fold_data[metric] = np.linspace(mean_val - std_val, mean_val + std_val, num_folds)
                    
        all_results[m] = pd.DataFrame(fold_data)


# 2x2plto:
METRICS_TO_PLOT = ['auc', 'f1', 'recall', 'precision'] # Dropped redundant Accuracy
COLORS = {'late': '#444488', 'feature': "#E6BC48"}

# Paired arrangement to ensure side-by-side strategy comparisons
feature_order = ['motor_updrs', 'smell_upsit', 'cognitive', 'demographics', 'ALL']
model_order = []
for f in feature_order:
    model_order.append(f'late_{f}')
    model_order.append(f'feature_{f}')

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, metric in zip(axes, METRICS_TO_PLOT):
    data, labels, colors = [], [], []
    
    for m in model_order:
        if m in all_results:
            fold_df = all_results[m]
            data.append(fold_df[metric].values)
            short_feature = m.split('_')[1]
            suffix = "Late" if "late" in m else "Feat"
            labels.append(f"{short_feature}\n({suffix})")
            colors.append(COLORS[m.split('_')[0]])

    bp = ax.boxplot(data, patch_artist=True, labels=labels,
                    medianprops=dict(color='white', linewidth=2), widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor('#444444')

    # Smart value placement to avoid upper ceiling collisions
    for i, d in enumerate(data):
        med = np.median(d)
        y_offset = -0.006 if med > 0.995 else 0.002
        v_align = 'top' if med > 0.995 else 'bottom'
        text_color = 'white' if med > 0.995 else 'black'
        
        ax.text(i + 1, med + y_offset, f'{med:.3f}',
                ha='center', va=v_align, fontsize=8, 
                color=text_color, weight='bold' if med > 0.99 else 'normal')

    ax.set_title(metric.upper(), fontsize=14, pad=10)
    all_vals = np.concatenate(data)
    ax.set_ylim(max(min(all_vals) - 0.015, 0.6))
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9, rotation=15)

# Universal Legend below charts
legend_handles = [
    mpatches.Patch(color=COLORS['late'],    label='Late Fusion (CNN Probs + Tabular LR)'),
    mpatches.Patch(color=COLORS['feature'], label='Feature Fusion (CNN Embeddings + MLP Head)'),
]
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, fontsize=11, frameon=True, facecolor='white', edgecolor='none')

fig.suptitle('Multimodal Fusion Performance: Information Gain by Feature Group',
             fontsize=16, weight='bold', y=0.97)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])

output_plot_path = os.path.join(RUN_DIR, "multimodal_boxplots_clean_2x2.svg")
fig.savefig(output_plot_path, bbox_inches='tight')
plt.close(fig)

print(f"grid plot exported to: {output_plot_path}")