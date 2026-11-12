"""
torchvision 활용
- num_classes: 2 => Binary Floor Segmentation
- weights_backbone = MobileNet_V3_Large_Weights.DEFAULT => ImageNet 사전학습 Backbone
"""

import torch
import torch.nn as nn

from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def build_lraspp(
    num_classes: int = 2,
    pretrained_backbone: bool = True,
) -> nn.Module:
    
    """
    LR-ASPP + MobileNetV3-Large.
    
    Classes:
        0: Non-floor
        1: floor
        
    Output:
        dict["out"] -> [B, num_classes, H, W]
    """
    
    backbone_weights = (MobileNet_V3_Large_Weights.IMAGENET1K_V1
                        if pretrained_backbone else None)
    
    model = lraspp_mobilenet_v3_large(weights=None,
                                      weights_backbone=backbone_weights,
                                      num_classes=num_classes)
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_lraspp(num_classes=2,
                         pretrained_backbone=False,).to(device)
    
    model.eval()
    
    x = torch.randn(1, 3, 240, 320, device=device)
    with torch.inference_mode():
        logits = model(x)["out"]
        
    print("Device: ", device)
    print("Input: ", x.shape)
    print("Output: ", logits.shape)
    print("Parameters:",
          f"{sum(p.numel() for p in model.parameters()):,}")
    