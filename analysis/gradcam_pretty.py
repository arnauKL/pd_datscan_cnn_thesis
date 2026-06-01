"""
GradCAM Explainability Pipeline for 2.5D CNN Models.

Generates structured multi-patient panel grids comparing raw orthogonal 
MIP views with their respective GradCAM activation heatmaps side-by-side.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('.'))
from src.architectures import ParkinsonClassifier25D
from src.transforms import get_25d_transforms_padding
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


import matplotlib.font_manager as fm
import matplotlib

fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 11
    

CONFIG = {
    "weights_path": "outputs/25d_resnet_raw_2fold_20260508_173440/fold_1/best_model.pth",
    "data_csv":     "data/ppmi_rawdata_sesBL_mapping.csv",
    "roi_size":     (76, 76, 76),
    "dropout":      0.3,
    "batch_size":   1,
    "seed":         42,
    "max_display_patients": 3,  # Fits landscape page
    "output_dir":   "analysis/outputs/gradcam/pretty",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading Pipeline
model = ParkinsonClassifier25D(dropout_rate=CONFIG["dropout"])
model.load_state_dict(torch.load(CONFIG["weights_path"], map_location=device))
model.to(device).eval()

# Select final convolutional block layer within ResNet18
target_layers = [model.features[7][1].conv2]

transform = get_25d_transforms_padding(CONFIG["roi_size"])

# Dataset Stratification Block
df = pd.read_csv(CONFIG["data_csv"])
balanced = pd.concat([df[df["label"] == 1].sample(n=len(df[df["label"] == 0]), random_state=CONFIG["seed"]), df[df["label"] == 0]]).reset_index(drop=True)
_, test_df = train_test_split(balanced, test_size=0.2, stratify=balanced["label"], random_state=CONFIG["seed"])
pd_test = test_df[test_df["label"] == 1].reset_index(drop=True)
hc_test = test_df[test_df["label"] == 0].reset_index(drop=True)

def load_image_tensor(row):
    item = transform({"image": row["path"], "label": row["label"]})
    return item["image"].unsqueeze(0)

def run_gradcam_inference(group_df, max_patients):
    cam = GradCAM(model=model, target_layers=target_layers)
    cams, imgs_raw, predictions = [], [], []

    for i in range(min(max_patients, len(group_df))):
        row = group_df.iloc[i]
        tensor = load_image_tensor(row).to(device)

        grayscale_cam = cam(input_tensor=tensor, targets=None)[0]
        if grayscale_cam.ndim == 3:
            grayscale_cam = grayscale_cam.mean(axis=0)

        with torch.no_grad():
            prob = 1 / (1 + np.exp(-model(tensor).item()))

        cams.append(grayscale_cam)
        imgs_raw.append(tensor.squeeze(0).cpu().numpy())
        predictions.append(prob)
        
    return cams, imgs_raw, predictions

def plot_publication_gradcam(cams, imgs_tensor, preds, true_label, filename_base):
    """
    Generates a perfectly aligned grid layout where each row corresponds to an 
    orthogonal slice, and columns are grouped side-by-side: [Raw View | GradCAM Overlay]
    """
    n_patients = len(cams)
    CHANNEL_NAMES = ["Axial View", "Coronal View", "Sagittal View"]
    label_str = "Parkinson's Disease (PD)" if true_label == 1 else "Healthy Control (HC)"
    
    # 3 Rows (Planes), 2 Columns per Patient (Raw + Overlay)
    fig, axes = plt.subplots(3, n_patients * 2, figsize=(n_patients * 5.5, 8.5))
    
    for i in range(n_patients):
        img_np = imgs_tensor[i]
        col_raw = i * 2
        col_cam = i * 2 + 1
        
        for ch in range(3):
            ax_raw = axes[ch, col_raw]
            ax_cam = axes[ch, col_cam]
            
            # Extract and normalize individual 2D channel slices
            channel = img_np[ch, :, :]
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            channel_rgb = np.stack([channel_norm] * 3, axis=-1).astype(np.float32)
            
            # Generate the true color heatmask overlay
            overlay = show_cam_on_image(channel_rgb, cams[i], use_rgb=True)
            
            # Render Raw Slice
            ax_raw.imshow(channel_norm, cmap="gray")
            ax_raw.axis("off")
            
            # Render GradCAM Overlay Slice
            ax_cam.imshow(overlay)
            ax_cam.axis("off")
            
            # Row Labels (only applied to the absolute leftmost edge)
            if i == 0:
                ax_raw.text(-12, 38, CHANNEL_NAMES[ch], rotation=90, 
                            ha="center", va="center", fontsize=11)
            
            # Subheaders differentiating the paired views
            if ch == 0:
                ax_raw.set_title("Raw Input", fontsize=9, style="italic", pad=4)
                ax_cam.set_title("CAM Overlay", fontsize=9, style="italic", pad=4)

        # Draw a clear global boundary boundary around each patient column block
        # We place a text banner across the top of the patient's paired columns
        axes[0, col_raw].text(78, -12, f"Patient Sample #{i+1}\nP (PD) = {preds[i]:.3f}", 
                              ha="center", va="bottom", fontsize=11)

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    out_svg = os.path.join(CONFIG["output_dir"], f"{filename_base}.svg")
    fig.savefig(out_svg, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Publication graphic successfully saved to: {out_svg}")

def plot_mean_gradcam(cams, filename, title):
    mean_cam = np.stack(cams).mean(axis=0)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(mean_cam, cmap="turbo", interpolation="bilinear")
    ax.set_title(title, fontsize=12, pad=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = os.path.join(CONFIG["output_dir"], filename)
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close()

def main():
    print(f"\nEvaluating visual attention vectors via GradCAM on {device}...")
    
    pd_cams, pd_imgs, pd_preds = run_gradcam_inference(pd_test, CONFIG["max_display_patients"])
    hc_cams, hc_imgs, hc_preds = run_gradcam_inference(hc_test, CONFIG["max_display_patients"])

    # Generate multi-patient side-by-side panel structures
    plot_publication_gradcam(pd_cams, pd_imgs, pd_preds, 1, "gradcam_25d_raw_PD_panel")
    plot_publication_gradcam(hc_cams, hc_imgs, hc_preds, 0, "gradcam_25d_raw_HC_panel")

    # Generate consolidated global mean maps
    plot_mean_gradcam(pd_cams, "gradcam_mean_25d_raw_PD.svg", "Mean Attentional Heatmap: PD Group")
    plot_mean_gradcam(hc_cams, "gradcam_mean_25d_raw_HC.svg", "Mean Attentional Heatmap: HC Group")
    print("\nVisual explainability routines completed.")

if __name__ == "__main__":
    main()