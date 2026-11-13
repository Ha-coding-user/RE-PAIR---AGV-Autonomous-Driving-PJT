import argparse
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ade20k_dataset import ADE20KFloorDataset
from models.lraspp import build_lraspp


def compute_metrics(cm):
    tn, fp = cm[0].tolist()
    fn, tp = cm[1].tolist()
    
    def divide(a, b):
        return a / b if b else 0.0
    
    floor_iou = divide(tp, tp + fp + fn)
    nonfloor_iou = divide(tn, tn + fp + fn)
    
    return {
        "floor_iou": floor_iou,
        "nonfloor_iou": nonfloor_iou,
        "miou": (floor_iou + nonfloor_iou) / 2,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "false_drivable_rate": divide(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/lraspp_mobilenetv3_baseline/best.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    dataset = ADE20KFloorDataset(root=ROOT / "datasets",
                                 split="validation",
                                 image_size=(240, 320))
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0)
    
    checkpoint = torch.load(ROOT / args.checkpoint,
                            map_location="cpu",
                            weights_only=True)
    
    model = build_lraspp(num_classes=2,
                         pretrained_backbone=False)
    
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    confusion = confusion.to(device)
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)["out"]
            preds = logits.argmax(dim=1)
            
            valid = masks != 255
            
            gt = masks[valid]
            pred = preds[valid]
            
            indices = gt * 2 + pred
            counts = torch.bincount(indices, minlength=4).reshape(2, 2)
            confusion += counts
            
    metrics = compute_metrics(confusion)
    
    print("\nEvaluation Results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
        
    output_dir = ROOT / "outputs" / "evaluation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nSaved:", output_dir / "metrics.json")
    
    

if __name__ == "__main__":
    main()