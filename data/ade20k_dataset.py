from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class ADE20KFloorDataset(Dataset):
    def __init__(self, root, split="training", image_size=(240, 320)):
        """
        image_size: (height, width)
        
        Mask:
            0   -> Non-floor
            1   -> Floor
            255 -> Ignore
        """
        
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        
        self.image_dir = (self.root / "ADEChallengeData2016/images" / split)
        self.mask_dir = (self.root / "floor_binary/masks" / split)
        
        self.images = sorted(self.image_dir.glob("*.jpg"))
        if not self.images:
            raise RuntimeError(f"No image found: {self.image_dir}")
        
        self.samples = []
        
        for image_path in self.images:
            mask_path = ( self.mask_dir / f"{image_path.stem}.png" )
            
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            
            self.samples.append((image_path, mask_path))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            
        with Image.open(mask_path) as mask_img:
            mask = mask_img.copy()
            
        height, width = self.image_size
        
        image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [height, width], interpolation=InterpolationMode.NEAREST)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask
        