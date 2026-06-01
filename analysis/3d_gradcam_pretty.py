"""
GradCAM Explainability Pipeline for True 3D CNN Models.

Generates structured multi-patient panel grids extracting center orthogonal 
slices from 3D volumes alongside an interactive 3D HTML visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from itertools import product, combinations

sys.path.insert(0, os.path.abspath('.'))
from src.architectures import ParkinsonClassifier3D
from src.transforms import get_3d_padding_cropping_transforms # Updated for 3D
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


import matplotlib.font_manager as fm
import matplotlib
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 11

CONFIG = {
    "weights_path": "outputs/3d_crop_raw_2fold_20260508_030602/fold_1/best_model.pth",
    "data_csv":     "data/ppmi_rawdata_sesBL_mapping.csv",
    "roi_size":     (76, 76, 76),
    "dropout":      0.3,
    "batch_size":   1,
    "seed":         42,
    "max_display_patients": 3,
    "output_dir":   "analysis/outputs/gradcam/pretty_3d",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Model Initialization
model = ParkinsonClassifier3D(dropout_rate=CONFIG["dropout"])
model.load_state_dict(torch.load(CONFIG["weights_path"], map_location=device))
model.to(device).eval()

# Point directly to the final Conv3D layer inside features sequential
target_layers = [model.conv3]

# Placeholder for your 3D data pipeline transformation
transform = get_3d_padding_cropping_transforms(CONFIG["roi_size"])


# Dataset Stratification Block
df = pd.read_csv(CONFIG["data_csv"])
balanced = pd.concat([df[df["label"] == 1].sample(n=len(df[df["label"] == 0]), random_state=CONFIG["seed"]), df[df["label"] == 0]]).reset_index(drop=True)
_, test_df = train_test_split(balanced, test_size=0.2, stratify=balanced["label"], random_state=CONFIG["seed"])
pd_test = test_df[test_df["label"] == 1].reset_index(drop=True)
hc_test = test_df[test_df["label"] == 0].reset_index(drop=True)


def load_image_tensor(row):
    item = transform({"image": row["path"], "label": row["label"]})
    return item["image"].unsqueeze(0) # Returns shape (1, 1, H, W, D)

def run_gradcam_inference(group_df, max_patients):
    cam = GradCAM(model=model, target_layers=target_layers)
    cams, imgs_raw, predictions = [], [], []

    for i in range(min(max_patients, len(group_df))):
        row = group_df.iloc[i]
        tensor = load_image_tensor(row).to(device)

        # pytorch-grad-cam natively outputs (H, W, D) for 3D inputs
        grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

        with torch.no_grad():
            logit = model(tensor).item()
            prob = 1 / (1 + np.exp(-logit))

        cams.append(grayscale_cam)
        # Squeezing outputs to cleanly isolate spatial configurations: (H, W, D)
        imgs_raw.append(tensor.squeeze(0).squeeze(0).cpu().numpy())
        predictions.append(prob)
        
    return cams, imgs_raw, predictions

def plot_gradcam(cams, imgs_tensor, preds, true_label, filename_base):
    n_patients = len(cams)
    PLANE_NAMES = ["Axial View", "Coronal View", "Sagittal View"]
    
    fig, axes = plt.subplots(3, n_patients * 2, figsize=(n_patients * 5.5, 8.5))
    
    for i in range(n_patients):
        img_3d = imgs_tensor[i]
        cam_3d = cams[i]
        
        col_raw = i * 2
        col_cam = i * 2 + 1
        
        # Compute geometric centers for slicing a 3D matrix
        h, w, d = img_3d.shape
        cx, cy, cz = h // 2, w // 2, d // 2
        
        # Extract corresponding 2D slices along standard radiology orientations
        slices_raw = [img_3d[:, :, cz], img_3d[:, cy, :], img_3d[cx, :, :]]
        slices_cam = [cam_3d[:, :, cz], cam_3d[:, cy, :], cam_3d[cx, :, :]]
        
        for plane_idx in range(3):
            ax_raw = axes[plane_idx, col_raw]
            ax_cam = axes[plane_idx, col_cam]
            
            raw_slice = slices_raw[plane_idx]
            cam_slice = slices_cam[plane_idx]
            
            # Normalize slice scale values
            raw_norm = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min() + 1e-8)
            raw_rgb = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)
            
            overlay = show_cam_on_image(raw_rgb, cam_slice, use_rgb=True)
            
            ax_raw.imshow(raw_norm, cmap="gray")
            ax_raw.axis("off")
            
            ax_cam.imshow(overlay)
            ax_cam.axis("off")
            
            if i == 0:
                ax_raw.text(-12, raw_norm.shape[0]//2, PLANE_NAMES[plane_idx], rotation=90, 
                            ha="center", va="center", fontsize=11)
            
            if plane_idx == 0:
                ax_raw.set_title("Raw Input", fontsize=9, style="italic", pad=4)
                ax_cam.set_title("CAM Overlay", fontsize=9, style="italic", pad=4)

        axes[0, col_raw].text(h, -12, f"Patient Sample #{i+1}\nP (PD) = {preds[i]:.3f}", 
                              ha="center", va="bottom", fontsize=11)

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    out_svg = os.path.join(CONFIG["output_dir"], f"{filename_base}.svg")
    fig.savefig(out_svg, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Graphic successfully saved to: {out_svg}")

def plot_interactive_3d_cam(cam_3d, filename):
    """Generates an interactive, rotatable 3D HTML volumetric heatmap."""
    # Create coordinate grid indices
    X, Y, Z = np.mgrid[0:cam_3d.shape[0], 0:cam_3d.shape[1], 0:cam_3d.shape[2]]
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=cam_3d.flatten(),
        isomin=0.15,          # Drops background noise out of view
        isomax=1.0,
        opacity=0.15,         # Soft opacity to peek inside clusters
        surface_count=20,     # Level of voxel gradient rendering detail
        colorscale='Turbo',
    ))
    
    fig.update_layout(
        title="Interactive 3D Grad-CAM Activation Volume",
        scene=dict(xaxis_title='H', yaxis_title='W', zaxis_title='D'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_html = os.path.join(CONFIG["output_dir"], filename)
    fig.write_html(out_html)
    print(f"Interactive 3D view rendered to: {out_html}")

def plot_static_3d_scatter_cam(cam_3d, filename, threshold_percentile=95.0):
    """
    Generates a static 3D scatter point cloud of high-intensity Grad-CAM activations
    bounded by a wireframe spatial cube framework.
    """
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    dx, dy, dz = cam_3d.shape
    
    # --- Draw a clean gray wireframe bounding box around the volume dimensions ---
    r = [[0, dx], [0, dy], [0, dz]]
    for s, e in combinations(np.array(list(product(*r))), 2):
        if np.sum(np.abs(s - e)) in [dx, dy, dz]:
            ax.plot3D(*zip(s, e), color="darkgray", linewidth=1.2, linestyle="--")
            
    # --- Isolate high-intensity voxels ---
    # NOTE: Since your model overfits to boundaries, setting this to 95.0 or 98.5
    # will cleanly map the exact edge artifacts in 3D space.
    thresh = np.percentile(cam_3d, threshold_percentile)
    x_idx, y_idx, z_idx = np.where(cam_3d > thresh)
    intensities = cam_3d[cam_3d > thresh]
    
    # --- Plot the 3D point cloud ---
    # Increased marker size (s=5) and opacity (alpha=0.4) slightly so small focus regions stay visible
    sc = ax.scatter(x_idx, y_idx, z_idx, c=intensities, cmap='turbo', s=5, alpha=0.4)
    
    ax.set_title("Grad-CAM Voxel Activation Cloud\n(Spatial Grid Structure)", pad=15)
    ax.set_xlim(0, dx)
    ax.set_ylim(0, dy)
    ax.set_zlim(0, dz)
    ax.axis('off')  
    
    # Add a clean colorbar tracking activation scale weights
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=12, pad=0.05)
    cbar.set_label('CAM Intensity', rotation=270, labelpad=15, fontsize=9)
    
    # Save output image file
    out_img = os.path.join(CONFIG["output_dir"], filename)
    fig.savefig(out_img, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Static 3D scatter plot successfully saved to: {out_img}")


def main():
    print(f"\nEvaluating 3D visual attention vectors via GradCAM on {device}...")
    
    pd_cams, pd_imgs, pd_preds = run_gradcam_inference(pd_test, CONFIG["max_display_patients"])
    hc_cams, hc_imgs, hc_preds = run_gradcam_inference(hc_test, CONFIG["max_display_patients"])

    if len(pd_cams) > 0:
        # Standard Multi-patient panel configuration (Static 2D center slices)
        plot_gradcam(pd_cams, pd_imgs, pd_preds, 1, "gradcam_3d_PD_panel")
        plot_gradcam(hc_cams, hc_imgs, hc_preds, 1, "gradcam_3d_HC_panel")
        
        plot_static_3d_scatter_cam(
            cam_3d=pd_cams[0], 
            filename="gradcam_3d_scatter_sample.png", 
            threshold_percentile=97.0
        )
        
if __name__ == "__main__":
    main()