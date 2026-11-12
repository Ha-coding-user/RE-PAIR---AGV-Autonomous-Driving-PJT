from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = ROOT / "datasets/ADEChallengeData2016/images"
MASK_ROOT = ROOT / "datasets/floor_binary/masks"

VALID_VALUES = {0, 1, 255}


def verify_split(split):
    image_dir = IMAGE_ROOT / split
    mask_dir = MASK_ROOT / split
    
    images = sorted(image_dir.glob("*.jpg"))
    
    assert images, f"No images: {image_dir}"
    
    floor_count = 0
    errors = []
    
    for image_path in tqdm(images, desc=split):
        mask_path = mask_dir / f"{image_path.stem}.png"
        
        if not mask_path.exists():
            errors.append(f"Missing mask: {mask_path}")
            continue
        
        with Image.open(image_path) as img, Image.open(mask_path) as mask_img:
            if img.size != mask_img.size:
                errors.append(f"Size mismatch: {image_path.name}")
                
            mask = np.asarray(mask_img)
            
        values = set(np.unique(mask).tolist())
        
        if not values.issubset(VALID_VALUES):
            errors.append(f"Invalid values: {mask_path.name}: {values}")
            
        if 1 in values:
            floor_count += 1
            
    print(f"\n[{split}]")
    print(f"Total images: {len(images)}")
    print(f"Floor images: {floor_count}")
    print(f"Errors: {len(errors)}")
    
    for error in errors[:10]:
        print(error)
        
    if errors:
        raise RuntimeError(f"{len(errors)} dataset erorrs detected.")

if __name__ == "__main__":
    verify_split("training")
    verify_split("validation")