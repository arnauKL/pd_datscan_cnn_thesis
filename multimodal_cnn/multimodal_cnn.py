# This script implements and compares three fusion strategies for combining your best pretrained CNN with clinical tabular data from PPMI.
# 
# Structure
# 1. Config -> set paths and feature groups here
# 2. Data loading -> merge image paths with tabular features
# 3. Late fusion -> combine existing CNN output probabilities with tabular-only ML
# 4. Feature fusion -> strip CNN head, concatenate embeddings with tabular branch
# 5. Information gain table -> compare all combinations
# 6. Visualisation -> boxplots

import os, sys, json
import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


## pretty fonts in plots
import matplotlib.font_manager as fm
import matplotlib
fm._load_fontmanager(try_read_cache=False)  # force rescan
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")  # explicit add
[f.name for f in fm.fontManager.ttflist if "Libertine" in f.name or "Libertinus" in f.name]
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 14

sys.path.insert(0, os.path.abspath('..'))
from src.architectures import ParkinsonClassifier2D, ParkinsonClassifier25D
from src.transforms import get_2d_sum_transforms_padding, get_25d_transforms_padding

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

fIn = '/home/data/PPMI/documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx'
fOut = '/home/akarel/src_tfg/data/ppmi_tabular.csv'

df = pd.read_excel(fIn, engine='openpyxl')
df.to_csv(fOut, index=False)

IMAGE_CSV    = 'data/ppmi_rawdata_sesBL_mapping.csv'   # raw image mapping
TABULAR_CSV  = 'data/ppmi_tabular.csv'                 # PPMI excel exported as csv

# Best CNN weights, da backbone
CNN_WEIGHTS  = '/home/akarel/src_tfg/outputs/25d_resnet_raw_2fold_20260508_173440/fold_1/best_model.pth'
CNN_CLASS    = ParkinsonClassifier25D   # must match the weights above
CNN_EMBED_DIM = 512                     # 512 for 25d_resnet, 32 for 2D/3D custom CNNs

# Transform must match what was used when training the CNN above
ROI_SIZE    = (76, 76, 76)
TRANSFORM   = get_25d_transforms_padding(ROI_SIZE)

# Training hyperparams
FOLDS       = 5
EPOCHS      = 50       # fusion head trains faster than a full CNN hopefully
LR          = 1e-4
BATCH_SIZE  = 8
DROPOUT     = 0.3
SEED        = 42
FREEZE_CNN  = True     # True = only train the fusion head (faster, less overfitting)
                       # False = fine-tune the whole network end-to-end

# Tabular feature groups 
# Edit column names to match your actual PPMI excel headers.
# Each group is tested independently AND all combined.
FEATURE_GROUPS = {
    'smell_upsit':   ['upsit'],
    'motor_updrs':   ['updrs3_score'], # UPDRS Part III (motor)
    'demographics':  ['age', 'SEX'],
    'cognitive':     ['moca'],
}

# Patient ID column to join image CSV and tabular CSV
PATIENT_ID_COL = 'PATNO'

print('Config loaded.')
print(f'Feature groups: {list(FEATURE_GROUPS.keys())}')

### Load n merge data
# Join image mapping CSV with the tabular PPMI data on patient ID, I'm keeping the rawdata mapping

img_df = pd.read_csv(IMAGE_CSV)
tab_df = pd.read_csv(TABULAR_CSV)

print(f'Image CSV:    {len(img_df)} rows, columns: {list(img_df.columns)}')
print(f'Tabular CSV:  {len(tab_df)} rows')

# print(f'\nTabular CSV columns: {list(tab_df.columns)}')
# print(f'\nImage CSV columns: {list(img_df.columns)}')

# Merge on patient ID
# Keep only baseline visit tabular data if there are multiple timepoints
tab_df = tab_df[tab_df['EVENT_ID'] == 'BL']

# Collect all feature columns we need
all_features = [f for group in FEATURE_GROUPS.values() for f in group]
tab_subset   = tab_df[[PATIENT_ID_COL] + all_features].drop_duplicates(subset=PATIENT_ID_COL)

import re 
# Example path: /home/data/PPMI/rawdata/sub-PPMI4078/ses-BL/...
def extract_patno(path):
    match = re.search(r'sub-PPMI(\d+)', path)
    return match.group(1) if match else None

img_df['PATNO'] = img_df['path'].apply(extract_patno)
img_df['PATNO'] = pd.to_numeric(img_df['PATNO'], errors='coerce')
tab_df['PATNO'] = pd.to_numeric(tab_df['PATNO'], errors='coerce')

merged = img_df.merge(tab_subset, on=PATIENT_ID_COL, how='inner')
print(f'After merge: {len(merged)} patients with both image and tabular data')
print(f'Missing values per feature:')
print(merged[all_features].isnull().sum())

# Drop rows with missing values in any feature group
merged_clean = merged.dropna(subset=all_features).reset_index(drop=True)
print(f'After dropping NaN rows: {len(merged_clean)} patients')

# Balance: downsample PD to match HC
hc_df = merged_clean[merged_clean['label'] == 0]
pd_df = merged_clean[merged_clean['label'] == 1].sample(n=len(hc_df), random_state=SEED)
balanced = pd.concat([hc_df, pd_df]).reset_index(drop=True)
print(f'Balanced: {len(balanced)} total ({len(hc_df)} HC, {len(hc_df)} PD)')
balanced.head()


def compute_metrics(labels, probs):
    preds = (probs > 0.5).astype(int)
    return {
        'auc':       roc_auc_score(labels, probs),
        'f1':        f1_score(labels, preds, zero_division=0),
        'accuracy':  accuracy_score(labels, preds),
        'recall':    recall_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
    }

## Late fusion (baseline)
# 
# No new training needed:
# - THe CNN already outputs a probability for each patient
# - Train a simple LR/SVM on tabular features
# - Average both probability outputs
# 
# This is a quick sanity check.
# If feature fusion later doesn't beat this, something is wrong with the fusion architecture.

def get_cnn_probabilities(df, model_class, weights_path, transform, embed_dim, device, extract_embedding=False):
    """
    Run inference with the full CNN and return probabilities (or embeddings).
    
    extract_embedding=False -> returns sigmoid probs (B,)
    extract_embedding=True  -> strips last layer, returns embeddings (B, embed_dim)
    """
    model = model_class(dropout_rate=DROPOUT)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    if extract_embedding:
        # Strip the final classification layer
        if hasattr(model, 'fc2'):   # custom 2D/3D CNNs
            model.fc2 = nn.Identity()
        elif hasattr(model, 'fc'):  # 25d_resnet
            model.fc = nn.Identity()
    
    model = model.to(device)
    model.eval()
    
    files = [{'image': p, 'label': l} for p, l in zip(df['path'], df['label'])]
    ds    = MonaiDataset(data=files, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_out, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch['image'].to(device)
            labels = batch['label'].numpy().flatten()
            out    = model(imgs).cpu().numpy()
            if not extract_embedding:
                out = 1 / (1 + np.exp(-out.flatten()))  # sigmoid
            all_out.extend(out)
            all_labels.extend(labels)
    
    return np.array(all_out), np.array(all_labels)

def run_late_fusion_cv(df, feature_cols, n_folds=FOLDS, alpha=0.5):
    """
    Late fusion: average CNN probabilities with tabular LR probabilities.
    alpha controls CNN weight: final_prob = alpha*cnn + (1-alpha)*tabular
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]
        
        # CNN probs (inference only, no training)
        cnn_probs_val, labels_val = get_cnn_probabilities(
            val_df, CNN_CLASS, CNN_WEIGHTS, TRANSFORM, CNN_EMBED_DIM, device)
        
        # Tabular LR 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_val   = scaler.transform(val_df[feature_cols].values)
        y_train = train_df['label'].values
        
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_train, y_train)
        tab_probs_val = lr.predict_proba(X_val)[:, 1]
        
        # Combine 
        fused_probs = alpha * cnn_probs_val + (1 - alpha) * tab_probs_val
        
        m = compute_metrics(labels_val, fused_probs)
        m['fold'] = fold
        fold_metrics.append(m)
        print(f'  Fold {fold}: AUC={m["auc"]:.3f}  F1={m["f1"]:.3f}')
    
    return pd.DataFrame(fold_metrics)

# Run late fusion for each feature group + all combined
late_fusion_results = {}

for group_name, feature_cols in FEATURE_GROUPS.items():
    available = [f for f in feature_cols if f in balanced.columns]
    if not available:
        print(f'[SKIP] {group_name}, columns not found in data')
        continue
    print(f'\nLate fusion, {group_name}: {available}')
    late_fusion_results[f'late_{group_name}'] = run_late_fusion_cv(balanced, available)

# All features combined
all_cols = [f for group in FEATURE_GROUPS.values() for f in group if f in balanced.columns]
print(f'\nLate fusion, ALL features: {all_cols}')
late_fusion_results['late_ALL'] = run_late_fusion_cv(balanced, all_cols)

# Feature fusion (bouzas mode)
# 
# Strip the CNN head ->
#     get a learned image embedding ->
#         concatenate with tabular features ->
#             train a small MLP fusion head.
# 
# The CNN backbone can be:
# - Frozen (`FREEZE_CNN=True`): only the fusion head trains. Faster, avoids overfitting on small data. Good first run.
# - Unfrozen (`FREEZE_CNN=False`): full end-to-end fine-tuning.

class MultimodalDataset(Dataset):
    """
    Wraps a MONAI transform pipeline and also returns tabular features.
    Each item: {'image_tensor': ..., 'tabular': ..., 'label': ...}
    """
    def __init__(self, df, feature_cols, transform):
        self.transform    = transform
        self.feature_cols = feature_cols
        self.paths  = df['path'].tolist()
        self.labels = df['label'].tolist()
        # Tabular as float32 tensor — scaler applied externally
        self.tabular = torch.tensor(
            df[feature_cols].values.astype(np.float32))
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Apply MONAI image transform
        item = self.transform({'image': self.paths[idx], 'label': self.labels[idx]})
        return {
            'image':   item['image'],
            'tabular': self.tabular[idx],
            'label':   torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class MultimodalFusionModel(nn.Module):
    """
    Y-shaped architecture:
      - CNN branch: pretrained backbone with head removed -> image_embed_dim
      - Tabular branch: small MLP -> tabular_hidden
      - Fusion head: concat -> MLP -> logit
    """
    def __init__(self, cnn_backbone, image_embed_dim, n_tabular,
                 tabular_hidden=16, dropout_rate=0.3, freeze_cnn=True):
        super().__init__()
        self.cnn = cnn_backbone
        
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
        
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, tabular_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tabular_hidden, tabular_hidden),
            nn.ReLU(),
        )
        fused_dim = image_embed_dim + tabular_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, image, tabular):
        img_emb = self.cnn(image)              # (B, image_embed_dim)
        tab_emb = self.tabular_branch(tabular) # (B, tabular_hidden)
        fused   = torch.cat([img_emb, tab_emb], dim=1)
        return self.head(fused)                # (B, 1)
    
def build_cnn_backbone(model_class, weights_path, embed_dim, device):
    """Load pretrained CNN and remove its classification head."""
    model = model_class(dropout_rate=DROPOUT)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    if hasattr(model, 'fc2'):   # custom 2D/3D CNNs --- embed_dim = 32
        model.fc2 = nn.Identity()
    elif hasattr(model, 'fc'):  # 25d_resnet --- embed_dim = 512
        model.fc = nn.Identity()
    return model.to(device)


def run_feature_fusion_cv(df, feature_cols, n_folds=FOLDS):
    """Train and evaluate the feature-fusion model with k-fold CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        
        # Scale tabular features (fit on train, apply to val)
        scaler = StandardScaler()
        train_df = train_df.copy()
        val_df   = val_df.copy()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
        val_df[feature_cols]   = scaler.transform(val_df[feature_cols].values)
        
        # Datasets
        train_ds = MultimodalDataset(train_df, feature_cols, TRANSFORM)
        val_ds   = MultimodalDataset(val_df,   feature_cols, TRANSFORM)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
        
        # Build fresh model for each fold
        backbone = build_cnn_backbone(CNN_CLASS, CNN_WEIGHTS, CNN_EMBED_DIM, device)
        model = MultimodalFusionModel(
            cnn_backbone    = backbone,
            image_embed_dim = CNN_EMBED_DIM,
            n_tabular       = len(feature_cols),
            tabular_hidden  = 16,
            dropout_rate    = DROPOUT,
            freeze_cnn      = FREEZE_CNN,
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        
        best_auc, best_probs, best_labels = 0, None, None
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            for batch in train_loader:
                imgs    = batch['image'].to(device)
                tabular = batch['tabular'].to(device)
                labels  = batch['label'].to(device).view(-1, 1)
                optimizer.zero_grad()
                loss = criterion(model(imgs, tabular), labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs    = batch['image'].to(device)
                    tabular = batch['tabular'].to(device)
                    labels  = batch['label'].numpy().flatten()
                    logits  = model(imgs, tabular).cpu().numpy().flatten()
                    probs   = 1 / (1 + np.exp(-logits))
                    all_probs.extend(probs)
                    all_labels.extend(labels)
            
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.5
            
            if auc > best_auc:
                best_auc    = auc
                best_probs  = np.array(all_probs)
                best_labels = np.array(all_labels)
        
        m = compute_metrics(best_labels, best_probs)
        m['fold'] = fold
        fold_metrics.append(m)
        print(f'  Fold {fold}: AUC={m["auc"]:.3f}  F1={m["f1"]:.3f}')
    
    return pd.DataFrame(fold_metrics)


# output (same sfuff as train.py)
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name  = f"multimodal_{timestamp}"
run_dir   = os.path.join("outputs", run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"\nOutputs will be saved to: {run_dir}")

# Save the config so the run is reproducible
config_to_save = {
    "CNN_WEIGHTS":   CNN_WEIGHTS,
    "CNN_CLASS":     CNN_CLASS.__name__,
    "CNN_EMBED_DIM": CNN_EMBED_DIM,
    "ROI_SIZE":      list(ROI_SIZE),
    "FOLDS":         FOLDS,
    "EPOCHS":        EPOCHS,
    "LR":            LR,
    "BATCH_SIZE":    BATCH_SIZE,
    "DROPOUT":       DROPOUT,
    "SEED":          SEED,
    "FREEZE_CNN":    FREEZE_CNN,
    "FEATURE_GROUPS": {k: v for k, v in FEATURE_GROUPS.items()},
    "n_patients":    len(balanced),
}
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config_to_save, f, indent=4)

# RE-RUN FEATURE FUSION, saving best model per group per fold
# We re-run the feature fusion loop here so we can intercept the
# trained model at the end of each fold and save its weights.
# The late fusion results don't involve a trainable PyTorch model
# (they use sklearn LR), so there's nothing to save there.

def run_feature_fusion_cv_with_saving(df, feature_cols, group_name,
                                       run_dir, n_folds=FOLDS):
    """
    Same as run_feature_fusion_cv but saves:
      run_dir/
        <group_name>/
          fold_1/
            best_model.pth      : fusion model weights (head + tabular branch)
            scaler.pkl          : fitted StandardScaler (needed for inference)
          fold_2/
            ...
          fold_metrics.csv      : one row per fold
    """
    import pickle
    group_dir = os.path.join(run_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        fold_dir = os.path.join(group_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        # Scale tabular (fit on train only)
        scaler = StandardScaler()
        train_df = train_df.copy()
        val_df   = val_df.copy()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].values)
        val_df[feature_cols]   = scaler.transform(val_df[feature_cols].values)

        # Save the scaler — you need it at inference time to scale new patient data
        with open(os.path.join(fold_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # Datasets and loaders
        train_ds     = MultimodalDataset(train_df, feature_cols, TRANSFORM)
        val_ds       = MultimodalDataset(val_df,   feature_cols, TRANSFORM)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        # Build model
        backbone = build_cnn_backbone(CNN_CLASS, CNN_WEIGHTS, CNN_EMBED_DIM, device)
        model = MultimodalFusionModel(
            cnn_backbone    = backbone,
            image_embed_dim = CNN_EMBED_DIM,
            n_tabular       = len(feature_cols),
            tabular_hidden  = 16,
            dropout_rate    = DROPOUT,
            freeze_cnn      = FREEZE_CNN,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

        best_auc    = 0
        best_probs  = None
        best_labels = None
        best_path   = os.path.join(fold_dir, "best_model.pth")

        log_path = os.path.join(fold_dir, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_auc", "val_f1"])

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                imgs    = batch['image'].to(device)
                tabular = batch['tabular'].to(device)
                labels  = batch['label'].to(device).view(-1, 1)
                optimizer.zero_grad()
                criterion(model(imgs, tabular), labels).backward()
                optimizer.step()

            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs    = batch['image'].to(device)
                    tabular = batch['tabular'].to(device)
                    labels  = batch['label'].numpy().flatten()
                    logits  = model(imgs, tabular).cpu().numpy().flatten()
                    probs   = 1 / (1 + np.exp(-logits))
                    all_probs.extend(probs)
                    all_labels.extend(labels)

            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.5
            preds = (np.array(all_probs) > 0.5).astype(int)
            f1    = f1_score(all_labels, preds, zero_division=0)

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, f"{auc:.4f}", f"{f1:.4f}"])

            print(f"\r    Epoch {epoch+1:3d}/{EPOCHS}  "
                  f"val_auc={auc:.4f}  val_f1={f1:.4f}", end="")

            if auc > best_auc:
                best_auc    = auc
                best_probs  = np.array(all_probs)
                best_labels = np.array(all_labels)
                # Save full model state, includes both CNN backbone and fusion head
                torch.save(model.state_dict(), best_path)
                print("saved ", end="")

        print()  # newline after epoch progress

        m = compute_metrics(best_labels, best_probs)
        m['fold'] = fold
        fold_metrics.append(m)
        print(f"  [Fold {fold}] AUC={m['auc']:.3f}  F1={m['f1']:.3f}  "
              f"saved to {fold_dir}")

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(group_dir, "fold_metrics.csv"), index=False)
    return fold_df


# Run all groups with saving
feature_fusion_results = {}

for group_name, feature_cols in FEATURE_GROUPS.items():
    available = [f for f in feature_cols if f in balanced.columns]
    if not available:
        print(f'[SKIP] {group_name}, columns not found')
        continue
    print(f'\nFeature fusion, {group_name}: {available}')
    feature_fusion_results[f'feature_{group_name}'] = \
        run_feature_fusion_cv_with_saving(balanced, available, group_name, run_dir)

all_cols = [f for group in FEATURE_GROUPS.values()
            for f in group if f in balanced.columns]
print(f'\nFeature fusion — ALL: {all_cols}')
feature_fusion_results['feature_ALL'] = \
    run_feature_fusion_cv_with_saving(balanced, all_cols, 'ALL', run_dir)

#  INFORMATION GAIN TABLE
all_results = {**late_fusion_results, **feature_fusion_results}

rows = []
for name, df_m in all_results.items():
    fusion_type, group = name.split('_', 1)
    for metric in ['auc', 'f1', 'accuracy', 'recall', 'precision']:
        rows.append({
            'model':    name,
            'fusion':   fusion_type,
            'features': group,
            'metric':   metric,
            'mean':     df_m[metric].mean(),
            'std':      df_m[metric].std(),
        })

results_long = pd.DataFrame(rows)

summary = results_long.pivot_table(
    index=['fusion', 'features'], columns='metric', values=['mean', 'std']
).round(3)

print('\nInformation gain table (mean plus.minus std across folds)')
print(summary.to_string())

summary.to_csv(os.path.join(run_dir, "multimodal_results.csv"))
results_long.to_csv(os.path.join(run_dir, "results_long.csv"), index=False)
print(f'\nSaved tables to {run_dir}/')


# BOXPLOTS (saved to run_dir, not just shown)
import csv   # already imported above but safe to repeat bcs just in case

METRICS_TO_PLOT = ['auc', 'f1', 'accuracy', 'recall', 'precision']
COLORS = {
    'late':    '#4C72B0',
    'feature': '#55A868',
}

model_order = (
    results_long[results_long['metric'] == 'auc']
    .groupby('model')['mean']
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

fig, axes = plt.subplots(
    1, len(METRICS_TO_PLOT),
    figsize=(len(model_order) * 1.6 * len(METRICS_TO_PLOT), 5)
)

for ax, metric in zip(axes, METRICS_TO_PLOT):
    data, labels, colors = [], [], []
    for m in model_order:
        fold_df = all_results[m]
        data.append(fold_df[metric].values)
        labels.append(m.replace('_', '\n'))
        colors.append(COLORS[m.split('_')[0]])

    bp = ax.boxplot(
        data, patch_artist=True, labels=labels,
        medianprops=dict(color='white', linewidth=2),
        widths=0.5,
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    for i, d in enumerate(data):
        ax.text(i + 1, np.median(d) + 0.003, f'{np.median(d):.3f}',
                ha='center', va='bottom', fontsize=7)

    ax.set_title(metric.upper(), fontsize=11)
    ax.set_ylim(max(0, min(np.concatenate(data)) - 0.05), 1.02)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

legend_handles = [
    mpatches.Patch(color=COLORS['late'],    label='Late fusion (CNN + tabular LR)'),
    mpatches.Patch(color=COLORS['feature'], label='Feature fusion (CNN embed + MLP)'),
]
axes[-1].legend(handles=legend_handles, fontsize=8, loc='lower right')
fig.suptitle('Multimodal fusion: information gain by feature group',
             fontsize=13, y=1.01)
fig.tight_layout()

plot_path = os.path.join(run_dir, "multimodal_boxplots.svg")
fig.savefig(plot_path, bbox_inches='tight')
plt.close(fig)
print(f'Saved plot to {plot_path}')

print(f'Run complete: {run_name}')
print(f'Output folder: {run_dir}')