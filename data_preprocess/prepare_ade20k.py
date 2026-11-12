from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# Repository Root
ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets" / "ADEChallengeData2016"
TARGET = ROOT / "datasets" / "floor_binary"

FLOOR_ID = 4
IGNORE_ID = 255

def convert_mask(mask_path, save_path):
    """
    ADE20K original annotation:
        0 -> Ignore (255)
        4 -> Floor (1)
        Others -> Non-floor (0)
    """
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L"))
        
    binary = np.zeros(mask.shape, dtype=np.uint8)
    
    binary[mask == FLOOR_ID] = 1
    binary[mask == 0] = IGNORE_ID
    
    if not os.path.exists(save_path.parent):
        os.makedirs(save_path.parent, exist_ok=True)
        
    Image.fromarray(binary, mode="L").save(save_path)
    

def prepare_split(split):
    """
    summary:
        pass
    params:
        split(string):  pass
    """
    
    image_dir = SOURCE / "images" / split
    mask_dir = SOURCE / "annotations" / split
    
    output_dir = TARGET / "masks" / split
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    image_paths = sorted(image_dir.glob("*.jpg"))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found: {image_dir}")
    
    floor_images = 0
    
    for image_path in tqdm(image_paths, desc=split):
        mask_path = mask_dir / (image_path.stem + ".png")
        
        if not mask_path:
            raise FileNotFoundError(mask_path)
        
        save_path = output_dir / mask_path.name
        
        convert_mask(mask_path, save_path)
        
        with Image.open(save_path) as im:
            mask = np.array(im)
            
        if np.any(mask == 1):
            floor_images += 1
            
    print(f"\n[{split}]")
    print(f"Total images: {len(image_paths)}")
    print(f"Images containing floor: {floor_images}")
    print(f"Binary masks saved: {output_dir}")
    
    

def main():
    for split in ["training", "validation"]:
        prepare_split(split)
        
    print("\nDataset preparation completed!")


if __name__ == "__main__":
    main()