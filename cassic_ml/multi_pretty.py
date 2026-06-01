import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import shap

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 12


DATA_PATH = "/home/data/PPMI/documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx"
LABEL_COL = "COHORT"
RANDOM_STATE = 42
CV_FOLDS = 5

df = pd.read_excel(DATA_PATH, header=0)
df[LABEL_COL] = df[LABEL_COL].map({1: 1, 2: 0})
df.dropna(subset=[LABEL_COL], inplace=True)
df[LABEL_COL] = df[LABEL_COL].astype(int)

FEATS_DATSCAN_RAW = ['DATSCAN_CAUDATE_L', 'DATSCAN_CAUDATE_R', 'DATSCAN_PUTAMEN_L', 'DATSCAN_PUTAMEN_R']
df.dropna(subset=FEATS_DATSCAN_RAW, inplace=True)

FEATS_DATSCAN_FULL = FEATS_DATSCAN_RAW + [
    "con_caudate", "ips_caudate", "mean_caudate",
    "con_putamen", "ips_putamen", "mean_putamen",
    "con_striatum", "ips_striatum", "mean_striatum"
]

eps = 1e-6
df["AI_Caudate"] = (df["DATSCAN_CAUDATE_L"] - df["DATSCAN_CAUDATE_R"]).abs() / ((df["DATSCAN_CAUDATE_L"] + df["DATSCAN_CAUDATE_R"]) / 2 + eps)
df["AI_Putamen"] = (df["DATSCAN_PUTAMEN_L"] - df["DATSCAN_PUTAMEN_R"]).abs() / ((df["DATSCAN_PUTAMEN_L"] + df["DATSCAN_PUTAMEN_R"]) / 2 + eps)
df["Mean_SBR"]   = df[["DATSCAN_PUTAMEN_L", "DATSCAN_PUTAMEN_R", "DATSCAN_CAUDATE_L", "DATSCAN_CAUDATE_R"]].mean(axis=1)
df["Putamen_Caudate_Ratio"] = ((df["DATSCAN_PUTAMEN_L"] + df["DATSCAN_PUTAMEN_R"]) / 2) / ((df["DATSCAN_CAUDATE_L"] + df["DATSCAN_CAUDATE_R"]) / 2 + eps)

FEATS_DATSCAN_ENG = FEATS_DATSCAN_FULL + ["AI_Caudate", "AI_Putamen", "Mean_SBR", "Putamen_Caudate_Ratio"]
FEATS_DEMO        = FEATS_DATSCAN_ENG + ["age_at_visit", "SEX"]
FEATS_MOTOR       = FEATS_DEMO + ["updrs3_score", "updrs1_score", "updrs2_score", "sym_tremor", "sym_rigid", "sym_brady", "sym_posins", "LEDD"]
FEATS_NONMOTOR    = FEATS_MOTOR + ["upsit", "rem", "ess", "gds", "scopa_gi", "scopa_ur"]
FEATS_BIO         = FEATS_NONMOTOR + ["asyn", "nfl_serum", "urate"]

feature_sets = {
    "DaTscan raw":         FEATS_DATSCAN_RAW,
    "DaTscan full SBR":    FEATS_DATSCAN_FULL,
    "DaTscan engineered":  FEATS_DATSCAN_ENG,
    "+ Demographics":      FEATS_DEMO,
    "+ Motor (UPDRS)":     FEATS_MOTOR,
    "+ Non-motor (UPSIT)": FEATS_NONMOTOR,
    "+ Biomarkers":        FEATS_BIO,
}

def prepare_dataset(df, feature_cols, label_col="COHORT", random_state=42):
    subset = df[feature_cols + [label_col]].copy()
    subset.dropna(how="all", subset=feature_cols, inplace=True)
    
    available_feats = [f for f in feature_cols if f in subset.columns]
    X_raw = subset[available_feats].values
    y_raw = subset[label_col].values

    X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)

    n_hc, n_pd = (y_raw == 0).sum(), (y_raw == 1).sum()
    minority_n = min(n_hc, n_pd)

    rng = np.random.RandomState(random_state)
    hc_sampled = rng.choice(np.where(y_raw == 0)[0], size=minority_n, replace=False)
    pd_sampled = rng.choice(np.where(y_raw == 1)[0], size=minority_n, replace=False)
    idx = np.concatenate([hc_sampled, pd_sampled])

    return X_imp[idx], y_raw[idx], available_feats

prepared = {name: prepare_dataset(df, feats) for name, feats in feature_sets.items()}

classifiers = {
    "SVM (RBF)":           SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "SVM (Linear)":        SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "precision": make_scorer(precision_score, average="macro"),
    "recall": make_scorer(recall_score, average="macro")
}

multimodal_results = {}
table_rows = []

print("\nExecuting Progressive Cross-Validation Cross-Analysis...")
for fs_name, (X_d, y_d, feats) in prepared.items():
    multimodal_results[fs_name] = {}
    for clf_name, clf in classifiers.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_validate(pipe, X_d, y_d, cv=cv, scoring=scoring, n_jobs=-1)
        multimodal_results[fs_name][clf_name] = scores
    
    # Capture statistics for the table
    svm_scores = multimodal_results[fs_name]["SVM (RBF)"]
    table_rows.append({
        "Modality Set": fs_name,
        "Features": len(feats),
        "Samples": len(y_d),
        "AUC": f"{svm_scores['test_roc_auc'].mean():.3f} ± {svm_scores['test_roc_auc'].std():.3f}",
        "B_Acc": f"{svm_scores['test_balanced_accuracy'].mean():.3f} ± {svm_scores['test_balanced_accuracy'].std():.3f}",
        "F1": f"{svm_scores['test_f1'].mean():.3f} ± {svm_scores['test_f1'].std():.3f}"
    })

# Output Markdown Table directly to terminal, incredibly fansi
print(pd.DataFrame(table_rows).to_markdown(index=False))

# Information Gain Plot
plot_metrics = {
    "ROC-AUC": "test_roc_auc",
    "Balanced Accuracy": "test_balanced_accuracy",
    "F1-Score": "test_f1"
}
clf_colors = {"SVM (RBF)": "#4C72B0", "SVM (Linear)": "#DD8452", "Random Forest": "#55A868", "Gradient Boosting": "#C44E52", "Logistic Regression": "#8172B2"}

fs_names = list(multimodal_results.keys())
x = np.arange(len(fs_names))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (metric_label, key) in zip(axes, plot_metrics.items()):
    all_min_vals = []

    for clf_name, color in clf_colors.items():
        means = [multimodal_results[fs][clf_name][key].mean() for fs in fs_names]
        stds  = [multimodal_results[fs][clf_name][key].std()  for fs in fs_names]
        ax.plot(x, means, marker="o", label=clf_name, color=color, lw=2, markersize=5)
        # shadows
        ax.fill_between(x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.1, color=color)
        all_min_vals.extend(np.array(means) - np.array(stds))

    ax.set_xticks(x)
    ax.set_xticklabels(fs_names, rotation=15, ha="right", fontsize=9)
    ax.set_title(metric_label, pad=10)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    

    ax.set_ylim(min(all_min_vals) - 0.002, 1.001)


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, 
    labels, 
    loc="upper center", 
    bbox_to_anchor=(0.5, 0.08), # X=center, Y=just below the x-axis labels
    ncol=5,                     # Puts all 5 models in a clean horizontal row
    fontsize=10, 
    frameon=True
)

fig.suptitle("Progressive Information Gain via Incremental Clinical Modality Integration", fontsize=13, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig("multimodal_information_gain_pt.svg", format="svg", bbox_inches="tight")
plt.close()

#SHAP_TARGET_SET = "+ Non-motor (UPSIT)"
SHAP_TARGET_SET = "+ Biomarkers"
X_shap, y_shap, feats_shap = prepared[SHAP_TARGET_SET]

pipe_shap = Pipeline([("scaler", StandardScaler()), ("clf", classifiers["SVM (RBF)"])])
pipe_shap.fit(X_shap, y_shap)

X_scaled = pipe_shap.named_steps["scaler"].transform(X_shap)

print(f"\nRunning KernelSHAP on complete multimodal track: '{SHAP_TARGET_SET}'...")
# Sample background to keep compute times under control
explainer = shap.KernelExplainer(pipe_shap.named_steps["clf"].predict_proba, shap.sample(X_scaled, 75))
# Calculate shap values for a subset of evaluation cases
shap_values = explainer.shap_values(X_scaled[:150])

# Fix array dimension mapping safely for binary class tracking
if isinstance(shap_values, list):
    # Older SHAP returns a list of arrays per class
    shap_matrix = shap_values[1]
elif shap_values.ndim == 3:
    # Newer SHAP outputs a single 3D array: [samples, features, classes]
    shap_matrix = shap_values[:, :, 1]
else:
    shap_matrix = shap_values

fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_matrix,
    X_scaled[:150],
    feature_names=feats_shap,
    show=False,
    max_display=15 # Focus on top 15 features bds it can be too much
)
plt.title(f"Classical Feature Importance Attribution (SHAP)\nModel: SVM RBF | Set: {SHAP_TARGET_SET}", fontsize=12, pad=15, weight="bold")
plt.tight_layout()
plt.savefig("shap_summary_multimodal_pt.svg", format="svg", bbox_inches="tight")
plt.close()

print("\nProcessing complete. Outputs saved: 'multimodal_information_gain_pt.svg', 'shap_summary_multimodal_pt.svg'")