"""
Get typst tables from fold metrics
"""

import os
import json
import pandas as pd
import numpy as np

OUTPUTS_DIR = "outputs"
TARGET_METRICS = ["auc", "accuracy", "f1"]

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

        fold_df = pd.read_csv(fold_csv)
        fold_df["model"]    = model_key
        fold_df["data"]     = data_key
        fold_df["run_name"] = run_name
        rows.append(fold_df)

    return pd.concat(rows, ignore_index=True)

def main():
    df = collect()

    # Ensure all target metrics exist in the dataframe to prevent KeyErrors
    for metric in TARGET_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan

    # Calculate median and standard deviation for target metrics
    summary = df.groupby("model").agg({
        "auc": ["median", "std"],
        "accuracy": ["median", "std"],
        "f1": ["median", "std"],
    })

    # Flatten the multi-index columns for cleaner iterations
    summary.columns = [
        "auc_med", "auc_std", 
        "acc_med", "acc_std", 
        "f1_med", "f1_std"
    ]
    summary = summary.reset_index()

    # Sort models by median AUC descending (Winners at the top!)
    summary = summary.sort_values(by="auc_med", ascending=False)

    # Calculate dynamic column padding based on the longest model name
    max_model_len = max(summary["model"].str.len().max(), 14)

    # Helper function to format metric values safely
    def format_metric(med, std):
        return f"{med:.3f} $plus.minus$ {std:.3f}"

    # Print Table Header
    print(f"columns: ({'auto, '*int(len(summary.columns)/2)}auto),")
    print(f"[ {'Model':<{max_model_len}} ],[ AUC             ],[ Acc           ],[ F1              ],")

    # Print Table Rows
    for _, row in summary.iterrows():
        model_name = row["model"]
        
        auc_str  = format_metric(row["auc_med"], row["auc_std"])
        acc_str = format_metric(row["acc_med"], row["acc_std"])
        f1_str   = format_metric(row["f1_med"], row["f1_std"])

        print(f"[ {model_name:<{max_model_len}} ],[ {auc_str:<15} ],[ {acc_str:<15} ],[ {f1_str:<15} ],")

if __name__ == "__main__":
    main()