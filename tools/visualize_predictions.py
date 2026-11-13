import argparse
import csv
import random
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from PIL import Image, ImageDraw
from tqdm import tqdm

from data.ade20k_dataset import ADE20KFloorDataset
from models.lraspp import build_lraspp

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Error Map Colors (RGB)
COLOR_GT = (59, 130, 246)
COLOR_PRED = (32, 184, 121)

COLOR_TN = (0, 0, 0)
COLOR_TP = (32, 184, 121)
COLOR_FP = (239, 68, 68)
COLOR_FN = (246, 196, 83)
COLOR_IGNORE = (139, 146, 156)


def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Visualize ADE20K Floor Segmentation "
        "predictions with transparent overlays."
    ))
    parser.add_argument("--checkpoint", type=str, default="outputs/lraspp_mobilenetv3_baseline/best.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation/visualizations")
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--scan_count", type=int, default=100, help="Number of samples to evaluate. 0 = all")
    parser.add_argument("--save_count", type=int, default=20, help="Number of visualizations to save")
    parser.add_argument("--selection", choices=["first", "random"], default="random", help="How to select samples for evaluation")
    parser.add_argument("--rank", choices=["none", "worst_iou", "most_fp", "highest_fpr", "best_iou"], default="worst_iou", hlep="How to rank evaluated samples")
    parser.add_argument("--alpha", type=float, default=0.40, help="Overlay opacity: 0.0 to 1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    
    return parser.parse_args()

def resolve_path(path):
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    return ROOT / path

def denormalize(image_tensor):
    image = (image_tensor.detach().cpu().permute(1, 2, 0).numpy())
    image = image * STD + MEAN
    image = np.clip(image, 0.0, 1.0)
    
    return (image * 255.0).astype(np.uint8)

def calculate_metrics(gt, pred):
    valid = gt != 255
    
    tp = int(np.count_nonzero(valid & (gt == 1) & (pred == 1)))
    fp = int(np.count_nonzero(valid & (gt == 0) & (pred == 1)))
    fn = int(np.count_nonzero(valid & (gt == 1) & (pred == 0)))
    tn = int(np.count_nonzero(valid & (gt == 0) & (pred == 0)))
    
    def safe_div(a, b, empty=0.0):
        return a / b if b > 0 else empty
    
    union = tp + fp + fn
    
    floor_iou = safe_div(tp, union, empty=1.0)
    
    return {
        "floor_iou": floor_iou,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "false_positive_rate": safe_div(fp, fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_floor_pixels": tp + fn,
        "pred_floor_pixels": tp + fp,
    }
    
    
    
def main():
    args = parse_args()
    
    if args.scan_count < 0:
        raise ValueError("--scan_count must be >= 0")
    
    if args.save_count < 0:
        raise ValueError("--save_count must be >= 0")
    
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        
    device = torch.device(args.device)
    
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_thread()))
        
    output_dir = resolve_path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Dataset
    dataset = ADE20KFloorDataset(root=ROOT / "datasets",
                                 split=args.split,
                                 image_size=(args.height, args.width))
    indices = list(range(len(dataset)))
    
    if args.selection == "random":
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        
    if args.scan_count > 0:
        indices = indices[:args.scan_count]
        
    if not indices:
        raise RuntimeError("No dataset samples selected.")
    
    # 2. Load Checkpoint
    checkpoint_path = resolve_path(args.checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path,
                            map_location="cpu",
                            weights_only=True)
    
    # 3. Build Model
    model = build_lraspp(num_classes=2,
                         pretrained_backbone=False)
    model.load_state_dict(checkpoint["model"])
    
    model = model.to(device)
    model.eval()
    
    print("\n==============================")
    print(" ADE20K Floor Visualization")
    print("\n==============================")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Epoch       : {checkpoint.get('epoch', 'unknown')}")
    print(f"Selected img: {len(indices)}")
    print(f"Save count  : {args.save_count}")
    print(f"Rank mode   : {args.rank}")
    print(f"Opacity     : {args.alpha}")
    print("\n==============================")
    
    # 4. Run inference
    results = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating predictions"):
            image, gt_tensor = dataset[idx]
            
            input_tensor = (image.unsqueeze(0)).to(device)
            logits = model(input_tensor)["out"]
            pred = (
                logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            )
            gt = (
                gt_tensor.cpu().numpy().astype(np.uint8)
            )
            
            metrics = calculate_metrics(gt, pred)
            
            results.append({
                "index": idx,
                "image_name": (
                    dataset.samples[idx][0].name
                ),
                **metrics
            })
            
    # 5. Rank Results
    if args.rank == "worst_iou":
        results.sort(key=lambda x: (x["floor_iou"], -x["fp"]))
    elif args.rank == "best_iou":
        results.sort(key=lambda x: x["floor_iou"], reverse=True)
    elif args.rank == "most_fp":
        results.sort(key=lambda x: x["fp"], reverse=True)
    elif args.rank == "highest_fpr":
        results.sort(key=lambda x: x["false_positive_rate"], reverse=True)
        
    # 6. Save CSV
    csv_path = output_dir / "sample_metrics.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        
        writer.writeheader()
        writer.writerows(results)
    
    
if __name__ == "__main__":
    main()