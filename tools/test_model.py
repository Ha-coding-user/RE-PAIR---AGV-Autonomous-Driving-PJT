from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from data.ade20k_dataset import ADE20KFloorDataset
from models.lraspp import build_lraspp

def main():
    # 1. Device
    assert torch.cuda.is_available(), (
        "CUDA is not available. Check PyTorch installation."
    )
    
    device = torch.device("cuda")
    
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    
    # 2. Dataset
    dataset = ADE20KFloorDataset(ROOT / "datasets",
                                 split="training",
                                 image_size=(240, 320))
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=0)
    
    images, masks = next(iter(loader))
    
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    
    print("Image:", images.shape)
    print("Mask:", masks.shape)
    print("Mask classes:", torch.unique(masks).tolist())
    
    # 3. Model
    model = build_lraspp(num_classes=2,
                         pretrained_backbone=True).to(device)
    
    model.train()
    
    # 4. Forward
    outputs = model(images)
    logits = outputs["out"]
    
    print("Logits:", logits.shape)
    
    assert logits.shape == (images.shape[0], 2, 240, 320)
    
    # 5. Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss = criterion(logits, masks)
    
    print("Loss:", loss.item())
    assert torch.isfinite(loss), "Non-finite loss"
    
    # 6. Backward
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4,
                                  weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Backward Pass: SUCCESS!")
    
    # 7. Prediction
    model.eval()
    
    with torch.no_grad():
        pred_logits = model(images)["out"]
        prediction = pred_logits.argmax(dim=1)
        
    print("Prediction:", prediction.shape)
    print("Prediction classes:",
          torch.unique(prediction).tolist())
    
    print("\nModel integration test PASSED!")
        
if __name__ == "__main__":
    main()