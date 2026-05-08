import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose, LoadImaged, 
    EnsureChannelFirstd, 
    CenterSpatialCropd, 
    NormalizeIntensityd, 
    Lambdad,
    ResizeWithPadOrCropd,
    Orientationd
)

############# 3D

def get_3d_transforms(roi_size=(76, 76, 76)):
    """
    Center-crop to roi_size.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        NormalizeIntensityd(keys=["image"]),
    ])


def get_3d_padding_cropping_transforms(spatial_size):
    """
    Standardise orientation then pad-or-crop to spatial_size.
    Used for raw (unregistered) images with variable sizes.
    """

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Standardize orientation first
        Orientationd(keys=["image"], axcodes="RAS"), 
        # This handles BOTH cases: pads if < 128, crops if > 128
        ResizeWithPadOrCropd(
            keys=["image"], 
            spatial_size=spatial_size, 
            mode="constant"
        ),
        NormalizeIntensityd(keys=["image"]),
    ])


############# 2D


def _sum_slices(data):
    # Collapse the depth axis (last dim) by summing
    return torch.sum(data, dim=-1)

def get_2d_sum_transforms(roi_size=(76, 76, 76)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        Lambdad(keys=["image"], func=_sum_slices),
        NormalizeIntensityd(keys=["image"]),
    ])

def get_2d_sum_transforms_padding(spatial_size):
    # Full-volume sum projection with pad-or-crop. For raw images.

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Standardize orientation first
        Orientationd(keys=["image"], axcodes="RAS"), 
        # This handles BOTH cases: pads if < 128, crops if > 128 (128 being spatial_size)
        ResizeWithPadOrCropd(
            keys=["image"], 
            spatial_size=spatial_size, 
            mode="constant"
        ),
        Lambdad(keys=["image"], func=_sum_slices),
        NormalizeIntensityd(keys=["image"]),
    ])


# (test): Only sum slices 30 to 45 where the striatum usually lives
# This gave really bad results, probably because I'm hardcoding the values instead of using percentiles or proportions,
# It feels too fragile and artificial anyway
# def sum_striatum_only(data):
#     # this is just an idea, Im not even calling this function for now
#     return torch.sum(data[:, :, :, 30:45], dim=-1) # q això realment podrien ser percentils/ proporcions de la imatge
def _sum_striatum_only(x):
    # Sum only the central third of slices along the depth axis instead of hard-coding. Might be better, not tried yet
    d     = x.shape[-1]
    start = d // 3
    end   = 2 * d // 3
    return torch.sum(x[..., start:end], dim=-1)


def get_2d_sum_striatum_transforms(roi_size=(76, 76, 76)):
    # Central-third sum projection, targeting the striatum. For registered images only.
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        Lambdad(keys=["image"], func=_sum_striatum_only),
        NormalizeIntensityd(keys=["image"]),
    ])


############# 2.5D
# Why 2.5D?
# Pretrained 2D backbones (ResNet18, EfficientNet, ...) expect a
# 3-channel input. By treating each orthogonal view as one channel we
# get richer spatial context than a single sum-projection --- the model
# sees the striatum from three directions --- while still benefiting from
# strong ImageNet initialisation without the complexity of a full 3D CNN.


def _make_orthogonal_extractor(out_hw):
    """
    Returns a function that extracts the 3 central orthogonal slices
    (axial, coronal, sagittal) and stacks them as a (3, H, W) tensor.
 
    out_hw : (height, width) that every slice is resized to.
    """
    def extract(x):
        # x: (1, H, W, D)  after EnsureChannelFirst
        _, H, W, D = x.shape
        axial    = x[0, :, :, D // 2]   # (H, W) — top-down / axial
        coronal  = x[0, :, W // 2, :]   # (H, D) — front / coronal
        sagittal = x[0, H // 2, :, :]   # (W, D) — side / sagittal
 
        size = (out_hw[0], out_hw[1])
 
        def resize(s):
            return F.interpolate(
                s.unsqueeze(0).unsqueeze(0).float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
 
        # Stack → (3, H, W)  — looks like an RGB image to ResNet
        return torch.stack([resize(axial), resize(coronal), resize(sagittal)], dim=0)
 
    return extract
 
def get_25d_transforms(roi_size=(76, 76, 76)):
    """
    3-channel orthogonal-slice transform for registered images.
    Output shape: (3, roi_size[0], roi_size[1])

    Designed for ParkinsonClassifier25D (pretrained ResNet18).
    Center-crop then extract 3 orthogonal central slices.
    Use for registered images + a 2D pretrained backbone.
    """
    out_hw = (roi_size[0], roi_size[1])
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        NormalizeIntensityd(keys=["image"]),
        Lambdad(keys=["image"], func=_make_orthogonal_extractor(out_hw)),
    ])
 
 
def get_25d_transforms_padding(spatial_size=(76, 76, 76)):
    """
    Pad-or-crop then extract 3 orthogonal central slices.
    Use for raw (unregistered) images + a 2D pretrained backbone.
    """
    out_hw = (spatial_size[0], spatial_size[1])
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size, mode="constant"),
        NormalizeIntensityd(keys=["image"]),
        Lambdad(keys=["image"], func=_make_orthogonal_extractor(out_hw)),
    ])