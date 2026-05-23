"""
GradCAM explainability for 2.5D CNN models.

GradCAM produces a heatmap showing which spatial regions of the input
image most influenced the model's prediction. For DaTSCAN this will hopefully
highlight the striatum (caudate + putamen) if the model is working correctly.

For 2.5D (3-channel orthogonal MIP input):
  - GradCAM runs on the last conv layer of the ResNet
  - Produces one heatmap per input channel (axial/coronal/sagittal), TODO

Output: analysis/outputs/gradcam/
  gradcam_PD_examples.svg
  gradcam_HC_examples.svg
  gradcam_mean_PD.svg        average GradCAM across all PD patients
  gradcam_mean_HC.svg        average GradCAM across all HC patients
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('.'))
from src.architectures import ParkinsonClassifier25D
from src.transforms import get_25d_transforms_padding
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# setup fonts for plots
import matplotlib.font_manager as fm
import matplotlib
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont("/home/akarel/.local/share/fonts/LinLibertine_R.ttf")
matplotlib.rcParams["font.family"] = "Linux Libertine"
matplotlib.rcParams["font.size"] = 12



CONFIG = {
    "weights_path": "outputs/25d_resnet_raw_2fold_20260508_173440/fold_1/best_model.pth",
    "data_csv":     "data/ppmi_rawdata_sesBL_mapping.csv",
    "roi_size":     (76, 76, 76),
    "dropout":      0.3,
    "batch_size":   1, # GradCAM works one image at a time
    "seed":         42,
    "n_examples":   6, # perexemple
    "output_dir":   "analysis/outputs/gradcam",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = ParkinsonClassifier25D(dropout_rate=CONFIG["dropout"])
model.load_state_dict(torch.load(CONFIG["weights_path"], map_location=device))
model.to(device)
model.eval()

# Target layer for GradCAM: last conv layer in the ResNet backbone.
# For ResNet18, this is the last layer of layer4.
# model.features is the backbone (everything before the FC head).
#target_layers = [model.features[-1][-1].conv2]
target_layers = [model.features[7][1].conv2]  # layer4, second BasicBlock

transform = get_25d_transforms_padding(CONFIG["roi_size"])

# Load test images
df       = pd.read_csv(CONFIG["data_csv"])
pd_df    = df[df["label"] == 1]
hc_df    = df[df["label"] == 0]
balanced = pd.concat([
    pd_df.sample(n=len(hc_df), random_state=CONFIG["seed"]), hc_df
]).reset_index(drop=True)

_, test_df = train_test_split(
    balanced, test_size=0.2,
    stratify=balanced["label"], random_state=CONFIG["seed"])

pd_test = test_df[test_df["label"] == 1].reset_index(drop=True)
hc_test = test_df[test_df["label"] == 0].reset_index(drop=True)
print(f"Test: {len(pd_test)} PD, {len(hc_test)} HC")


def load_image_tensor(row):
    """Load one image through the transform pipeline."""
    item = transform({"image": row["path"], "label": row["label"]})
    return item["image"].unsqueeze(0)   # add batch dim: (1, 3, H, W)


def normalise_for_display(img_tensor):
    """
    Normalise a (3, H, W) tensor to (H, W, 3) float in [0,1] for display.
    We average the 3 channels (axial/coronal/sagittal) into one greyscale,
    then stack to RGB.
    """
    img = img_tensor.mean(dim=0).numpy()   # (H, W)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.stack([img, img, img], axis=-1).astype(np.float32)  # (H, W, 3)


def run_gradcam_batch(group_df, group_name, n_show):
    """Run GradCAM on the first n_show patients and return cams + images."""
    cam = GradCAM(model=model, target_layers=target_layers)
    #cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    cams, imgs_display, imgs_raw, preds = [], [], [], []

    for i in range(min(n_show, len(group_df))):
        row    = group_df.iloc[i]
        tensor = load_image_tensor(row).to(device)

        # targets=None means it explains the top predicted class
        grayscale_cam = cam(input_tensor=tensor, targets=None)
        grayscale_cam = grayscale_cam[0]   # (H, W)

        # Get model prediction
        with torch.no_grad():
            logit = model(tensor).item()
            prob  = 1 / (1 + np.exp(-logit))

        img_rgb = normalise_for_display(tensor.squeeze(0).cpu())
        imgs_raw.append(tensor.squeeze(0).cpu().numpy())  # (3, H, W), raw channels
        imgs_display.append(img_rgb)                        # keep for other uses
        # grayscale_cam shape is (3, H, W) for multi-channel input, average to (H, W)
        if grayscale_cam.ndim == 3:
            grayscale_cam = grayscale_cam.mean(axis=0)
        cams.append(grayscale_cam)  # now always (H, W)
        preds.append(prob)
        print(f"  {group_name} patient {i+1}: PD prob={prob:.3f}")

    return cams, imgs_display, imgs_raw, preds


def plot_gradcam_examples_averaged(cams, imgs, preds, true_label, filename):
    """ I was trying stuff out but this just blends all the slices which is not very useful """
    n   = len(cams)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))

    for i in range(n):
        overlay = show_cam_on_image(imgs[i], cams[i], use_rgb=True)

        axes[0, i].imshow(imgs[i])
        axes[0, i].set_title(f"PD prob={preds[i]:.2f}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Input (avg channels)", fontsize=9)
    axes[1, 0].set_ylabel("GradCAM overlay", fontsize=9)
    label_str = "PD" if true_label == 1 else "HC"
    fig.suptitle(f"GradCAMPlusPlus: {label_str} patients "
                 f"(red = most important regions)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(CONFIG["output_dir"], filename)
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")

# Show each of the 3 orthogonal MIP channels separately
CHANNEL_NAMES = ["Axial MIP", "Coronal MIP", "Sagittal MIP"]

def plot_gradcam_examples(cams, imgs_tensor, preds, true_label, filename):
    n = len(cams)
    # 4 rows: 3 input channels + 1 GradCAM overlay on the avg
    fig, axes = plt.subplots(4, n, figsize=(n * 2.5, 8))
    for i in range(n):
        img_np = imgs_tensor[i]  # shape: (3, H, W) numpy array
        for ch in range(3):
            channel = img_np[ch, :, :]    # (H, W)
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            axes[ch, i].imshow(channel_norm, cmap="gray")
            axes[ch, i].set_title(f"prob={preds[i]:.2f}", fontsize=7)
            axes[ch, i].axis("off")
            if i == 0:
                axes[ch, i].set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
        
        # i do not need the average, this should be in the 'for ch in range(3)' loop
        avg = img_np.mean(axis=0)     # (H, W), average over the 3 channels
        avg_norm = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
        avg_rgb = np.stack([avg_norm, avg_norm, avg_norm], axis=-1).astype(np.float32)
        overlay = show_cam_on_image(avg_rgb, cams[i], use_rgb=True)
        axes[3, i].imshow(overlay)
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_ylabel("GradCAM", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

CHANNEL_NAMES = ["Axial MIP", "Coronal MIP", "Sagittal MIP"]

def plot_gradcam_examples_overkill(cams, imgs_tensor, preds, true_label, filename):
    n = len(cams)
    label_str = "PD" if true_label == 1 else "HC"
    # 6 rows: for each channel, raw image + GradCAM overlay
    fig, axes = plt.subplots(6, n, figsize=(n * 2.5, 12))

    for i in range(n):
        img_np = imgs_tensor[i]  # (3, H, W)

        for ch in range(3):
            row_img     = ch * 2        # rows 0, 2, 4 — raw image
            row_gradcam = ch * 2 + 1   # rows 1, 3, 5 — GradCAM overlay

            channel = img_np[ch, :, :]
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            channel_rgb  = np.stack([channel_norm] * 3, axis=-1).astype(np.float32)

            # Raw image
            axes[row_img, i].imshow(channel_norm, cmap="gray")
            axes[row_img, i].set_title(f"prob={preds[i]:.2f}", fontsize=7)
            axes[row_img, i].axis("off")
            if i == 0:
                axes[row_img, i].set_ylabel(f"{CHANNEL_NAMES[ch]}", fontsize=8)

            # GradCAM overlay on this channel's image
            overlay = show_cam_on_image(channel_rgb, cams[i], use_rgb=True)
            axes[row_gradcam, i].imshow(overlay)
            axes[row_gradcam, i].axis("off")
            if i == 0:
                axes[row_gradcam, i].set_ylabel(f"{CHANNEL_NAMES[ch]}\nGradCAM", fontsize=8)

    fig.suptitle(f"GradCAM, {label_str} patients", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close()

def plot_mean_gradcam(cams, filename, title):
    """Average GradCAM across all patients in a group."""
    mean_cam = np.stack(cams).mean(axis=0)   # (H, W)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mean_cam, cmap="hot", interpolation="bilinear")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(CONFIG["output_dir"], filename)
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


# Run GradCAM
print("\nRunning GradCAM on PD patients...")
pd_cams, pd_imgs, pd_imgs_raw, pd_preds = run_gradcam_batch(pd_test, "PD", len(pd_test))

print("\nRunning GradCAM on HC patients...")
hc_cams, hc_imgs, hc_imgs_raw, hc_preds = run_gradcam_batch(hc_test, "HC", len(hc_test))

# Individual examples
plot_gradcam_examples_overkill(pd_cams[:CONFIG["n_examples"]], pd_imgs_raw[:CONFIG["n_examples"]], 
                      pd_preds[:CONFIG["n_examples"]], 1, "gradcam_25d_raw_PD_examples.svg")
plot_gradcam_examples_overkill(hc_cams[:CONFIG["n_examples"]], hc_imgs_raw[:CONFIG["n_examples"]], 
                      hc_preds[:CONFIG["n_examples"]], 0, "gradcam_25d_raw_HC_examples.svg")

# Mean GradCAM
plot_mean_gradcam(pd_cams, "gradcam_mean_25d_raw_PD.svg",
                  "Mean GradCAM, PD patients\n(average attention across all PD test images)")
plot_mean_gradcam(hc_cams, "gradcam_mean_25d_raw_HC.svg",
                  "Mean GradCAM, HC patients\n(average attention across all HC test images)")

print("\nDone. Check analysis/outputs/gradcam/")