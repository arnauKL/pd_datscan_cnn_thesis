import torch
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


############# 2.5D


def _extract_orthogonal_slices(x, out_size=76):
    """
    Takes the 3 central slices of a 3D volume (one per axis) and stacks
    them into a 3-channel 2D image.
 
    Input:  x  shape (1, H, W, D)  - single-channel 3D volume
    Output:    shape (3, out_size, out_size)
 
    The three channels are:
      0 -> axial    (z = D//2)   - the "top-down" view, most informative for
                                   the comma/full-stop DaT pattern
      1 -> coronal  (y = W//2)   - front-back view
      2 -> sagittal (x = H//2)   - left-right view
 
    All three are bilinearly resized to `out_size x out_size` so the tensor
    has a fixed shape regardless of input volume dimensions.
    """
    h, w, d = x.shape[1], x.shape[2], x.shape[3]
 
    axial    = x[0, :,    :,    d // 2]   # (H, W)
    coronal  = x[0, :,    w//2, :]        # (H, D)
    sagittal = x[0, h//2, :,    :]        # (W, D)
 
    def resize(s):
        # s is a 2D tensor; add batch+channel dims, interpolate, remove them
        return F.interpolate(
            s.unsqueeze(0).unsqueeze(0).float(),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
 
    return torch.stack([resize(axial), resize(coronal), resize(sagittal)], dim=0)
 
 
def get_25d_transforms(roi_size=(76, 76, 76)):
    """
    3-channel orthogonal-slice transform for registered images.
    Output shape: (3, roi_size[0], roi_size[1])
    Designed for ParkinsonClassifier25D (pretrained ResNet18).
    """
    out_size = roi_size[0]
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        NormalizeIntensityd(keys=["image"]),
        Lambdad(keys=["image"], func=lambda x: _extract_orthogonal_slices(x, out_size)),
    ])
 
 
def get_25d_transforms_padding(spatial_size=(76, 76, 76)):
    """
    3-channel orthogonal-slice transform for raw (unregistered) images.
    Pads/crops first to make variable-size volumes uniform, then slices.
    Output shape: (3, spatial_size[0], spatial_size[1])
    """
    out_size = spatial_size[0]
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size, mode="constant"),
        NormalizeIntensityd(keys=["image"]),
        Lambdad(keys=["image"], func=lambda x: _extract_orthogonal_slices(x, out_size)),
    ])