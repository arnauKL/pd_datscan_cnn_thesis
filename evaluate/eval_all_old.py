# Loops over every folder in outputs/, loads the saved model,
# runs bootstrap evaluation, and saves a CSV of metrics per run.

# call from root folder
#     python evaluate/eval_all.py

# Outputs are stored in evaluate/bootstrap_results.csv

# I only did this to work with the old runs where I was creating one script per run
# To use with the new train.py-trained models, see evall_all.py

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.utils import resample

# Make sure src/ is on the path (run this script from your project root)
sys.path.insert(0, os.path.abspath("."))
from src.architectures import ParkinsonClassifier2D, ParkinsonClassifier3D
from src.transforms import (
    get_2d_sum_transforms,
    get_2d_sum_transforms_padding,
    get_2d_sum_striatum_transforms,
    get_3d_transforms,
    get_3d_padding_cropping_transforms,
)

# CONFIG: edit DATA_CSV if some runs used a different CSV
OUTPUTS_DIR   = "outputs"
DATA_CSV      = "data/ppmi_baseline_mapping.csv"
RESULTS_OUT   = "evaluate/bootstrap_results.csv"
N_BOOTSTRAP   = 200   # number of bootstrap samples per model
VAL_SIZE      = 0.2   # must match what you used during training
RANDOM_SEED   = 42
BATCH_SIZE    = 4

os.makedirs("evaluate", exist_ok=True)

# Helpers: figure out which architecture + transforms a run used

def get_model_and_transforms(config: dict):
    """
    Returns (model, transform_fn) based on the model_name in config.
    Add more elif branches here if you add new architectures later.
    """
    name       = config["model_name"]
    roi_size   = tuple(config["roi_size"])
    dropout    = config.get("dropout", 0.3)

    # ---- 2D models ----
    if "2D" in name or "2d" in name:
        model = ParkinsonClassifier2D(dropout_rate=dropout)

        if "striatum" in name:
            transform = get_2d_sum_striatum_transforms(roi_size)
        elif "croppadding" in name or "padding" in name:
            transform = get_2d_sum_transforms_padding(roi_size)
        else:
            transform = get_2d_sum_transforms(roi_size)

    # ---- 3D models ----
    else:
        model = ParkinsonClassifier3D(dropout_rate=dropout)

        if "padding" in name or "croppadding" in name:
            transform = get_3d_padding_cropping_transforms(roi_size)
        else:
            transform = get_3d_transforms(roi_size)

    return model, transform


# Bootstrap evaluation
def get_predictions(model, loader, device):
    """Run inference and return (probabilities, true_labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy().flatten()
            logits = model(images).cpu().numpy().flatten()
            probs  = 1 / (1 + np.exp(-logits))   # sigmoid — no torch needed
            all_probs.extend(probs)
            all_labels.extend(labels)

    return np.array(all_probs), np.array(all_labels)


def bootstrap_metrics(probs, labels, n_bootstrap=200, seed=42):
    """
    Resample (probs, labels) n_bootstrap times.
    Returns a DataFrame with one row per bootstrap sample.
    """
    rng  = np.random.RandomState(seed)
    rows = []

    for _ in range(n_bootstrap):
        idx     = resample(range(len(labels)), random_state=rng.randint(0, 99999))
        p_b     = probs[idx]
        l_b     = labels[idx]
        preds_b = (p_b > 0.5).astype(int)

        # Guard: roc_auc needs both classes present in the sample
        if len(np.unique(l_b)) < 2:
            continue

        rows.append({
            "f1":        f1_score(l_b, preds_b, zero_division=0),
            "auc":       roc_auc_score(l_b, p_b),
            "accuracy":  accuracy_score(l_b, preds_b),
            "recall":    recall_score(l_b, preds_b, zero_division=0),    # sensitivity
            "precision": precision_score(l_b, preds_b, zero_division=0),
        })

    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    df_all = pd.read_csv(DATA_CSV)

    all_rows = []

    for run_name in sorted(os.listdir(OUTPUTS_DIR)):
        run_dir    = os.path.join(OUTPUTS_DIR, run_name)
        config_f   = os.path.join(run_dir, "config.json")
        weights_f  = os.path.join(run_dir, "best_model.pth")

        # Skip anything that isn't a proper run folder
        if not os.path.isdir(run_dir):
            continue
        if not os.path.exists(config_f) or not os.path.exists(weights_f):
            print(f"[SKIP] {run_name} — missing config.json or best_model.pth")
            continue

        with open(config_f) as f:
            config = json.load(f)

        print(f"[EVAL] {run_name}")

        # ---- Recreate the same balanced test split used during training ----
        pd_df     = df_all[df_all["label"] == 1]
        hc_df     = df_all[df_all["label"] == 0]
        balanced  = pd.concat([
            pd_df.sample(n=len(hc_df), random_state=RANDOM_SEED),
            hc_df
        ])

        _, test_df = train_test_split(
            balanced,
            test_size=VAL_SIZE,
            stratify=balanced["label"],
            random_state=RANDOM_SEED,
        )

        # ---- Build model and transform ----
        try:
            model, transform = get_model_and_transforms(config)
        except Exception as e:
            print(f"  [WARN] Could not build model/transform: {e} — skipping")
            continue

        # Load saved weights
        state = torch.load(weights_f, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        # ---- DataLoader ----
        test_files = [
            {"image": p, "label": l}
            for p, l in zip(test_df["path"], test_df["label"])
        ]
        test_ds     = MonaiDataset(data=test_files, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # ---- Get predictions ----
        try:
            probs, labels = get_predictions(model, test_loader, device)
        except Exception as e:
            print(f"  [WARN] Inference failed: {e} — skipping")
            continue

        # ---- Point estimate (no bootstrap) on the full test set ----
        preds = (probs > 0.5).astype(int)
        print(f"  Test set size : {len(labels)} ({int(labels.sum())} PD, {int((labels==0).sum())} HC)")
        print(f"  AUC  : {roc_auc_score(labels, probs):.3f}")
        print(f"  F1   : {f1_score(labels, preds):.3f}")
        print(f"  Acc  : {accuracy_score(labels, preds):.3f}\n")

        # ---- Bootstrap ----
        boot_df = bootstrap_metrics(probs, labels, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED)
        boot_df["run"]        = run_name
        boot_df["model_name"] = config.get("model_name", run_name)
        boot_df["balanced"]   = "Unbalanced" not in run_name
        boot_df["roi_size"]   = str(config.get("roi_size"))

        all_rows.append(boot_df)

    if not all_rows:
        print("No runs were evaluated successfully.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(RESULTS_OUT, index=False)
    print(f"\nDone! Results saved to {RESULTS_OUT}")
    print(f"Shape: {results.shape}  ({results['model_name'].nunique()} models {N_BOOTSTRAP} bootstrap samples each)")


if __name__ == "__main__":
    main()