import torch
import torch.nn as nn
import torch.nn.functional as F


# architectures for the CNNs


class ParkinsonClassifier3D(nn.Module):
    """
    Small 3d (custom, trained from scratch)
    3D 3-layer CNN
    Input: (B, 1, H, W, D)
    Output: (B, 1) raw logit (needs BCEWithLogitsLoss during trainin)
    """
    def __init__(self, dropout_rate=0.3):

        #super(ParkinsonClassifier3D, self).__init__()
        super().__init__()
        
        # Layer 1: Conv -> Batch Norm -> ReLU -> Pool
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(16)
        
        # Layer 2: Conv -> Batch Norm -> ReLU -> Pool
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(32)
        
        # Layer 3: Conv -> Batch Norm -> ReLU -> Pool
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm3d(64)
        
        self.pool = nn.MaxPool3d(2)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        
        # Fully Connected Layers with Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # take into account: using RAW LOGITS. 
        # nn.BCEWithLogitsLoss()
        return self.fc2(x)



# Deeper 3d
# (custom, trained from scratch)
class ParkinsonClassifier3D_deeper(nn.Module):
    """
    4-layer 3D-CNN (16->32->64->128 filters).
    Input : (B, 1, H, W, D)
    Output: (B, 1)  raw logit (same as before, cuidadu with BCEWithLogitsLoss when training)
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16),  nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1     = nn.Linear(128, 64)
        self.fc2     = nn.Linear(64, 1)
 
    def forward(self, x):
        x = self.gap(self.features(x)).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
 

# architecture for the 2D models

#  same as the small 3D
class ParkinsonClassifier2D(nn.Module):
    """
    Small 2d (custom, trained from scratch)
    2D 3-layer CNN
    Input: (B, 1, H, W) -- the projection / sum of slices
    Output: (B, 1) raw logit
    """
    def __init__(self, dropout_rate=0.3):
        #super(ParkinsonClassifier2D, self).__init__()
        super().__init__()
        
        # Changed to 2D from the 3D architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    

#  2.5D ResNet18 (pretrained on ImageNet)
# Training a 3D-CNN from scratch needs a lot of data to converge well.
# Reusing `ImageNet` weights gives the network a strong visual prior from
# the start: edges, textures, shapes: all useful even for medical images.
# The 2.5D trick lets us exploit those 2D weights on volumetric data.

import torchvision.models as tv_models
 
class ParkinsonClassifier25D(nn.Module):
    """
    ResNet18 pretrained on ImageNet, fine-tuned for binary PD classification.
 
    Input : (B, 3, H, W)
            The 3 channels are the axial, coronal and sagittal central slices
            produced by get_25d_transforms(). They look like an RGB image to
            the backbone, but each channel carries a different spatial view
            of the DaTSCAN volume.
 
    Output: (B, 1)  raw logit, use BCEWithLogitsLoss.
    """
    def __init__(self, dropout_rate=0.3, pretrained=True):
        super().__init__()
        weights  = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
 
        # Strip the original ImageNet classification head
        # Everything up to (and including) the global avg pool is kept.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # backbone.children() ends with: avgpool -> (B,512,1,1)
        # then fc -> (B,1000)  we replace this
 
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(512, 1)
 
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.features(x)       # -> (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B, 512)
        x = self.dropout(x)
        return self.fc(x)          # -> (B, 1)
 
import os

# ResNet-10 for MedicalNet. Why this over ImageNet transfer?
# The Med3D weights come from tasks on real medical volumes, so
# the low-level filters already respond to the kinds of intensity
# patterns found in nuclear medicine imaging. This is a much
# closer domain match than natural images.

"""
It uses MedicalNet's ResNet class directly so weight keys match exactly.
The segmentation head (conv_seg) is stripped and replaced with a
global average pool + classification head.
"""

from src.resnet import resnet10

class ParkinsonClassifierMed3D(nn.Module):
    """
    MedicalNet ResNet-10 backbone for DaTSCAN classification.

    The MedicalNet weights were trained on a segmentation task, so
    the architecture has a conv_seg head ion want.
      1. Build the full ResNet from resnet.py (so weight keys match)
      2. Load pretrained weights (now ALL backbone keys will match)
      3. Replace conv_seg with GlobalAvgPool + classification head bcs classifier

    Input : (B, 1, H, W, D)
    Output: (B, 1) raw logit yea
    """

    def __init__(self,
                 dropout_rate=0.3,
                 weights_path="mednetWeights/pretrain/resnet_10.pth",
                 roi_size=(76, 76, 76)):
        super().__init__()

        # Build full MedicalNet ResNet (with conv_seg)
        # num_seg_classes=2 is arbitrary since Im removin conv_seg anyway
        backbone = resnet10(
            sample_input_D=roi_size[0],
            sample_input_H=roi_size[1],
            sample_input_W=roi_size[2],
            num_seg_classes=2,
            shortcut_type='B',
            no_cuda=False, # please
        )

        # Load weights
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            # MedicalNet saves under 'state_dict' key
            state = checkpoint.get('state_dict', checkpoint)
            # Strip 'module.' prefix if saved with DataParallel
            state = {k.replace('module.', ''): v for k, v in state.items()}

            missing, unexpected = backbone.load_state_dict(state, strict=False)
            # Now 'missing' should only be the conv_seg keys (which we remove)
            # and 'unexpected' should be empty hopefully
            backbone_missing  = [k for k in missing    if 'conv_seg' not in k]
            backbone_unexpected = [k for k in unexpected if 'conv_seg' not in k]
            print(f"  Med3D: backbone missing (excl. conv_seg): {len(backbone_missing)}")
            print(f"  Med3D: unexpected keys (excl. conv_seg):  {len(backbone_unexpected)}")
            if backbone_missing:
                print(f"    Missing: {backbone_missing[:5]}")
            print(f"  Med3D weights loaded from {weights_path}")
        else:
            print(f"  [WARN] Weights not found at {weights_path} — training from scratch")

        # Keep only the backbone, discard conv_seg
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        # conv_seg is NOT kept, dummy warning

        # Classification head
        # layer4 outputs 512 channels (BasicBlock.expansion=1)
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).view(x.size(0), -1)   # (B, 512)
        x = self.dropout(x)
        return self.fc(x)                     # (B, 1)

class ParkinsonClassifierMed3DEncoder(nn.Module):
    """
    MedicalNet ResNet-10 used as a pure ENCODER for classification.

    Difference from ParkinsonClassifierMed3D:
    - Explicitly loads ONLY encoder layer weights (conv1, bn1, layer1-4)
    this is what I should've done from the start but i forgor

    - conv_seg (decoder/segmentation head) is never instantiated
    - Cleaner weight loading with explicit key filtering
    - More aggressive classification head to compensate for
      the smaller feature set vs the full segmentation network

    Input : (B, 1, H, W, D)
    Output: (B, 1)  raw logit
    """
    def __init__(self,
                 dropout_rate=0.3,
                 weights_path="mednetWeights/pretrain/resnet_10.pth",
                 roi_size=(76, 76, 76)):
        super().__init__()

        from src.resnet import resnet10

        # Build full network (needed to load weights correctly)
        full_net = resnet10(
            sample_input_D=roi_size[0],
            sample_input_H=roi_size[1],
            sample_input_W=roi_size[2],
            num_seg_classes=2,
            shortcut_type="B",
            no_cuda=False,
        )

        # Load pretrained weights
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu")
            state = checkpoint.get("state_dict", checkpoint)
            state = {k.replace("module.", ""): v for k, v in state.items()}

            # ENCODER ONLY: explicitly filter to just the encoder keys
            encoder_keys = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}
            encoder_state = {
                k: v for k, v in state.items()
                if k.split(".")[0] in encoder_keys
            }
            missing, unexpected = full_net.load_state_dict(
                encoder_state, strict=False)
            loaded = len(encoder_state)
            print(f"  Med3D encoder: loaded {loaded} weight tensors")
            print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")
        else:
            print(f"  [WARN] No weights at {weights_path}, training from scratch")

        # Copy ONLY encoder layers
        self.conv1   = full_net.conv1
        self.bn1     = full_net.bn1
        self.relu    = full_net.relu
        self.maxpool = full_net.maxpool
        self.layer1  = full_net.layer1
        self.layer2  = full_net.layer2
        self.layer3  = full_net.layer3
        self.layer4  = full_net.layer4

        # Classification head, slightly deeper than before
        # layer4 outputs 512 channels with BasicBlock
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.head    = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.head(x)