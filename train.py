"""
train.py
The all new, unified training script for DaTSCAN Parkinson classification

Usage:
python train.py --data registered --model 3d_crop  --folds 2

Optional params (overrides defaults):
python train.py --data registered --model 3d_crop --folds 2 \
    --epochs 80 --lr 0.0001 --batch_size 4 --dropout 0.4

Arguments
--data        'registered' or 'raw'
--model       '3d_crop' / '2d_sum' / '25d_resnet'
--folds       number of CV folds (default 2 for screening, 5 when Adrià allows me)
--roi_size    spatial crop/pad size as 3 ints, default 76 76 76
--epochs      training epochs per fold (default 100)
--lr          learning rate (default 1e-4)
--batch_size  (default 2)
--dropout     dropout rate (default 0.3)
--seed        random seed (default 42)

Outputs (inside outputs/<run_name>/)
config.json           full config for reproducibility in case i forgor
fold_metrics.csv      one row per fold: auc, f1, accuracy, recall, precision
fold_N/
    best_model.pth    best weights for fold N (saved by val AUC)
    training_log.csv  epoch-level train/val loss + acc per fold
"""

import os
import sys
import json
import csv
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    recall_score, precision_score, classification_report,
)

sys.path.insert(0, os.path.abspath("."))
from src.architectures import (
    ParkinsonClassifier3D,
    ParkinsonClassifier3D_deeper,
    ParkinsonClassifier2D,
    ParkinsonClassifier25D,
    ParkinsonClassifierMed3D,
    ParkinsonClassifierMed3DEncoder
)
from src.transforms import (
    get_3d_transforms,
    get_3d_padding_cropping_transforms,
    get_2d_sum_transforms,
    get_2d_sum_transforms_padding,
    get_25d_transforms,
    get_25d_transforms_padding,
)


#  CSV paths for each data source
DATA_CSVS = {
    "registered": "data/ppmi_derivative_sesBL_mapping.csv",
    "raw":        "data/ppmi_rawdata_sesBL_mapping.csv",
}


#  Model + transforms
def build_model_and_transform(model_key: str, data_key: str, roi_size: tuple, dropout: float):
    """
    Returns (model, transform, needs_pretrained_lr_split).

    needs_pretrained_lr_split is True for 25d_resnet so I can use a lower LR for the backbone and a higher one for the head
    I might not even use this.
    

    Oju!
    All the parameters are very case-and-ortography-sensitive
    """
    registered = (data_key == "registered")

    if model_key == "3d_crop":
        model     = ParkinsonClassifier3D(dropout_rate=dropout)
        transform = (get_3d_transforms(roi_size) if registered
                     else get_3d_padding_cropping_transforms(roi_size))
        split_lr  = False

    if model_key == "3d_crop_deeper":
        model     = ParkinsonClassifier3D_deeper(dropout_rate=dropout)
        transform = (get_3d_transforms(roi_size) if registered
                     else get_3d_padding_cropping_transforms(roi_size))
        split_lr  = False

    elif model_key == "2d_sum":
        model     = ParkinsonClassifier2D(dropout_rate=dropout)
        transform = (get_2d_sum_transforms(roi_size) if registered
                     else get_2d_sum_transforms_padding(roi_size))
        split_lr  = False

    elif model_key == "25d_resnet":
        model     = ParkinsonClassifier25D(dropout_rate=dropout, pretrained=True)
        transform = (get_25d_transforms(roi_size) if registered
                     else get_25d_transforms_padding(roi_size))
        split_lr  = True

    elif model_key == "med3d":
        model     = ParkinsonClassifierMed3D(
                        dropout_rate=dropout,
                        weights_path="/home/akarel/src_tfg/mednetWeights/pretrain/resnet_10.pth")
        transform = (get_3d_transforms(roi_size) if registered
                    else get_3d_padding_cropping_transforms(roi_size))
        split_lr  = True   # lower LR on backbone, higher on head: same as 25d

    elif model_key == "med3d_encoder":
        model     = ParkinsonClassifierMed3DEncoder(
                        dropout_rate=dropout,
                        weights_path="mednetWeights/pretrain/resnet_10.pth",
                        roi_size=roi_size)
        transform = (get_3d_transforms(roi_size) if registered
                     else get_3d_padding_cropping_transforms(roi_size))
        split_lr  = True

    else:
        # Thx Francesc Castro
        raise ValueError(f"Unknown model key: '{model_key}'. Choose from: 3d_crop, 3d_crop_deeper, 2d_sum, 25d_resnet, med3d, med3d_encoder")

    return model, transform, split_lr


def build_optimizer(model, base_lr: float, split_lr: bool):
    """
    For pretrained models use a 10x lower LR on the backbone so we don't
    destroy pretrained weights, and a full LR on the new classification head.
    For scratch models use the same LR everywhere.
 
    Works for both 25d_resnet (backbone = model.features) and
    med3d (backbone = model.layer0/1/2/3) by using named parameters
    and checking which ones belong to the head.
    """
    if not split_lr:
        return optim.Adam(model.parameters(), lr=base_lr)
 
    # Head parameters are always fc + dropout, everything else is backbone
    head_names = {"fc", "dropout"}
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        top_level = name.split(".")[0]   # e.g. "layer0", "features", "fc"
        if top_level in head_names:
            head_params.append(param)
        else:
            backbone_params.append(param)
 
    return optim.Adam([
        {"params": backbone_params, "lr": base_lr / 10},
        {"params": head_params,     "lr": base_lr},
    ])
 


#  One fold: train + evaluate
def run_fold(fold_idx, train_df, val_df, transform, model_key, data_key, roi_size, dropout, lr, epochs, batch_size, fold_dir, device):
    # very long function, very many parameters

    os.makedirs(fold_dir, exist_ok=True)

    # DataLoaders
    def make_loader(df, shuffle):
        files = [{"image": p, "label": l}
                 for p, l in zip(df["path"], df["label"])]
        ds = MonaiDataset(data=files, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df,   shuffle=False)

    # Model
    model, _, split_lr = build_model_and_transform(model_key, data_key, roi_size, dropout)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss() # Cuidadu dont touch this, all the architectures use logits
    optimizer = build_optimizer(model, lr, split_lr)

    log_path      = os.path.join(fold_dir, "training_log.csv")
    best_path     = os.path.join(fold_dir, "best_model.pth")
    best_val_auc  = -1.0

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_auc"])

    print(f"\n  [Fold {fold_idx}] train={len(train_df)}  val={len(val_df)}")

    for epoch in range(epochs):
        # Trainin
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            
            imgs   = batch["image"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validatin
        model.eval()
        val_loss  = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].float().to(device).view(-1, 1)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy().flatten())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds  = (all_probs > 0.5).astype(int) # idk if there is a better way to do this using pytorch

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        val_acc   = accuracy_score(all_labels, all_preds)

        # AUC needs both classes present (guard for edge-case folds)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            print(f'value error, setting val_auc to .5')
            val_auc = 0.5 # safe fallback?

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch + 1, f"{avg_train:.4f}", f"{avg_val:.4f}",
                 f"{val_acc:.4f}", f"{val_auc:.4f}"])

        print(f"\r    Epoch {epoch+1:3d}/{epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
              f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}", end="")

        # Save best model (by AUC this time)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(" new best", end="")

    print()  # newline after epoch progress

    # Final evaluation on best weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)
            probs  = torch.sigmoid(model(imgs)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs > 0.5).astype(int)

    # dict to map name ot real fuctions
    metrics = {
        "fold":      fold_idx,
        "auc":       roc_auc_score(all_labels, all_probs),
        "f1":        f1_score(all_labels, all_preds,        zero_division=0),
        "accuracy":  accuracy_score(all_labels, all_preds),
        "recall":    recall_score(all_labels, all_preds,    zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "n_val":     len(all_labels),
    }

    # Save human-readable report for this fold
    report = classification_report(
        all_labels, all_preds, target_names=["Healthy", "PD"])
    with open(os.path.join(fold_dir, "results.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAUC: {metrics['auc']:.4f}\n")

    print(f"  [Fold {fold_idx}] AUC={metrics['auc']:.3f}  "
          f"F1={metrics['f1']:.3f}  Acc={metrics['accuracy']:.3f}")

    return metrics


#  Main stuff
import argparse
def parse_args():
    # I love argparse
    p = argparse.ArgumentParser(description="Train DaTSCAN classifier")
    p.add_argument("--data",       required=True,
                   choices=["registered", "raw"],
                   help="Which image set to use")
    p.add_argument("--model",      required=True,
                   choices=["3d_crop", "3d_crop_deeper", "2d_sum", "25d_resnet", "med3d", "med3d_encoder"],
                   help="Model + transform combination")
    p.add_argument("--folds",      type=int,   default=2,
                   help="Number of CV folds (2 = fast screening, 5 = final)")
    p.add_argument("--roi_size",   type=int,   nargs=3, default=[76, 76, 76],
                   metavar=("H", "W", "D"))
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    roi_size = tuple(args.roi_size)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load n balance data
    csv_path = DATA_CSVS.get(args.data)

    if csv_path is None or not os.path.exists(csv_path):
        sys.exit(f"[ERROR] CSV not found: {csv_path}\n")

    df = pd.read_csv(csv_path)
    pd_df = df[df["label"] == 1]
    hc_df = df[df["label"] == 0]

    balanced = pd.concat([
        pd_df.sample(n=len(hc_df), random_state=args.seed),
        hc_df,
    ]).reset_index(drop=True)

    print(f"Dataset: {args.data} | balanced={len(balanced)} "
          f"(HC={len(hc_df)}, PD={len(hc_df)})")
    print(f"Model: {args.model} | folds={args.folds} | "
          f"roi_size={roi_size} | epochs={args.epochs}")

    # Build transform (same for all folds)
    _, transform, _ = build_model_and_transform(args.model, args.data, roi_size, args.dropout)

    # Output dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{args.model}_{args.data}_{args.folds}fold_{timestamp}"
    run_dir   = os.path.join("outputs", run_name)
    os.makedirs(run_dir, exist_ok=True)


    config = vars(args)
    config["roi_size"] = list(roi_size)   # make JSON-serialisable
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"Output: {run_dir}\n")

    # Stratified K-Fold
    skf       = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.seed)
    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(balanced, balanced["label"]), start=1):

        train_df = balanced.iloc[train_idx].reset_index(drop=True)
        val_df   = balanced.iloc[val_idx].reset_index(drop=True)
        fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")

        metrics = run_fold(
            fold_idx   = fold_idx,
            train_df   = train_df,
            val_df     = val_df,
            transform  = transform,
            model_key  = args.model,
            data_key   = args.data,
            roi_size   = roi_size,
            dropout    = args.dropout,
            lr         = args.lr,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            fold_dir   = fold_dir,
            device     = device,
        )
        fold_rows.append(metrics)

    # Summary
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False)

    print()
    print()

    print(f"Run complete: {run_name}")
    print(f"{'Metric':<12}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    for col in ["auc", "f1", "accuracy", "recall", "precision"]:
        s = fold_df[col]
        print(f"{col:<12}  {s.mean():.3f}   {s.std():.3f}   "
              f"{s.min():.3f}   {s.max():.3f}")
    print()
    print(f"Results saved to: {run_dir}/fold_metrics.csv")


if __name__ == "__main__":
    main()

    print()
    print()
    print()