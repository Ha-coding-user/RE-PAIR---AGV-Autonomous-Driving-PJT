from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from data.ade20k_dataset import ADE20KFloorDataset

def main():
    dataset = ADE20KFloorDataset(root=ROOT / "datasets",
                                 split="training",
                                 image_size=(240, 320))
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True)
    
    images, masks = next(iter(loader))
    
    print("Dataset size:", len(dataset))
    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
    print("Mask values:", torch.unique(masks))
    
    assert images.shape == (4, 3, 240, 320)
    assert masks.shape == (4, 240, 320)
    
    assert set(torch.unique(masks).tolist()).issubset({0, 1, 255})
    
    print("DataLoader test passed!")


if __name__ == "__main__":
    main()