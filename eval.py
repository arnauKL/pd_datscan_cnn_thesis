"""
evaluate.py
-----------
Run after training to compute and persist per-run metrics.
Appends one row to `outputs/all_results.csv` — safe to call
after every training run without keeping the model in memory.

Usage:
    python evaluate.py --run_dir outputs/2D_suma_rawderiv_128_croppadding_20240501_120000
"""

import os
import json
import argparse
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.architectures import ParkinsonClassifier2D
from src.transforms import get_2d_sum_transforms_padding


RESULTS_FILE = os.path.join("outputs", "all_results.csv")
RESULTS_COLUMNS = [
    "run_name", "model_name", "fold",
    "auc_roc", "balanced_acc", "sensitivity", "specificity", "f1",
    "best_val_loss",
]


def load_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def get_best_val_loss(run_dir: str) -> float:
    log_path = os.path.join(run_dir, "training_log.csv")
    df = pd.read_csv(log_path)
    return df["val_loss"].min()


def build_test_loader(config: dict, fold: int = 0) -> DataLoader:
    """
    Reproduces the exact test split used during training.
    `fold` is used as an offset to the random_seed so each CV fold
    gets a different split while remaining reproducible.
    """
    df = pd.read_csv(os.path.join("data", config["data_path"]))

    # Balance
    pd_df = df[df["label"] == 1]
    hc_df = df[df["label"] == 0]
    pd_balanced = pd_df.sample(n=len(hc_df), random_state=config["random_seed"])
    balanced = pd.concat([pd_balanced, hc_df])

    _, test_df = train_test_split(
        balanced,
        test_size=config["val_size"],
        stratify=balanced["label"],
        random_state=config["random_seed"] + fold,
    )

    test_files = [
        {"image": p, "label": l}
        for p, l in zip(test_df["path"], test_df["label"])
    ]
    ds = MonaiDataset(
        data=test_files,
        transform=get_2d_sum_transforms_padding(tuple(config["roi_size"])),
    )
    return DataLoader(ds, batch_size=config["batch_size"])


def evaluate_run(run_dir: str, fold: int = 0) -> dict:
    run_name = os.path.basename(run_dir)
    config = load_config(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParkinsonClassifier2D(dropout_rate=config["dropout"]).to(device)
    model.load_state_dict(
        torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
    )
    model.eval()

    test_loader = build_test_loader(config, fold=fold)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(float)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "run_name":      run_name,
        "model_name":    config["model_name"],
        "fold":          fold,
        "auc_roc":       roc_auc_score(all_labels, all_probs),
        "balanced_acc":  balanced_accuracy_score(all_labels, all_preds),
        "sensitivity":   sensitivity,
        "specificity":   specificity,
        "f1":            f1_score(all_labels, all_preds),
        "best_val_loss": get_best_val_loss(run_dir),
    }

    # Also save a per-run detailed report
    report = classification_report(all_labels, all_preds, target_names=["HC", "PD"])
    with open(os.path.join(run_dir, "eval_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAUC-ROC:        {metrics['auc_roc']:.4f}\n")
        f.write(f"Balanced Acc:   {metrics['balanced_acc']:.4f}\n")
        f.write(f"Sensitivity:    {metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity:    {metrics['specificity']:.4f}\n")

    print(f"[{run_name}] AUC={metrics['auc_roc']:.3f} | "
          f"Bal.Acc={metrics['balanced_acc']:.3f} | "
          f"Sens={metrics['sensitivity']:.3f} | "
          f"Spec={metrics['specificity']:.3f} | "
          f"F1={metrics['f1']:.3f}")

    return metrics


def append_to_results(metrics: dict):
    """Append one row to the shared CSV. Creates the file if missing."""
    os.makedirs("outputs", exist_ok=True)
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics appended to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to the run output directory")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (for CV runs)")
    args = parser.parse_args()

    metrics = evaluate_run(args.run_dir, fold=args.fold)
    append_to_results(metrics)