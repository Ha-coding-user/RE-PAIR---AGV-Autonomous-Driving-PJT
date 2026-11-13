import argparse
import csv
import random
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.ade20k_dataset import ADE20KFloorDataset
from models.lraspp import build_lraspp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def update_confusion(confusion, logits, targets, ignore_index):
    preds = logits.argmax(dim=1)
    
    valid = targets != ignore_index
    targets = targets[valid]
    preds = preds[valid]
    
    if targets.numel() == 0:
        return
    
    indices = targets * 2 + preds
    
    counts = torch.bincount(indices,
                            minlength=4).reshape(2, 2)
    
    confusion += counts.cpu().to(torch.int64)
    
    
def compute_metrics(confusion):
    # confusion[GT, Prediction]
    tn = confusion[0, 0].item()
    fp = confusion[0, 1].item()
    fn = confusion[1, 0].item()
    tp = confusion[1, 1].item()
    
    def safe_div(num, den):
        return num / den if den > 0 else 0.0
    
    floor_iou = safe_div(tp, tp + fp + fn)
    nonfloor_iou = safe_div(tn, tn + fp + fn)
    
    return {
        "floor_iou": floor_iou,
        "miou": (floor_iou + nonfloor_iou) / 2,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "false_drivable_rate": safe_div(fp, fp + tn),
    }
    
    
def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    ignore_index=255,
):
    is_train = optimizer is not None
    model.train(is_train)
    
    total_loss = 0.0
    total_pixels = 0
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    
    pbar = tqdm(loader,
                desc="Train" if is_train else "Val",
                leave=False)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        valid_pixels = (masks != ignore_index).sum().item()
        if valid_pixels == 0:
            continue
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)["out"]
                loss = criterion(logits, masks)
                
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    
        total_loss += loss.item() * valid_pixels
        total_pixels += valid_pixels
        
        update_confusion(confusion,
                         logits.detach(),
                         masks,
                         ignore_index)
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )
        
    metrics = compute_metrics(confusion)
    metrics["loss"] = total_loss / max(total_pixels, 1)
    
    return metrics

def save_checkpoint(path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_iou,
                    config):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "best_iou": best_iou,
            "config": config,
        },
        path,
    )
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    set_seed(cfg["experiment"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    
    dataset_root = ROOT / data_cfg["root"]
    
    # Load Dataset
    train_dataset = ADE20KFloorDataset(root=dataset_root,
                                       split="training",
                                       image_size=data_cfg["image_size"])
    val_dataset = ADE20KFloorDataset(root=dataset_root,
                                     split="validation",
                                     image_size=data_cfg["image_size"])
    
    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            **loader_kwargs)
    
    model = build_lraspp(
        num_classes=data_cfg["num_classes"],
        pretrained_backbone=(cfg["model"]["pretrained_backbone"]
                             and args.resume is None)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=data_cfg["ignore_index"]
    )
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    
    epochs = train_cfg["epochs"]
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs,
                                                           eta_min=cfg["scheduler"]["min_lr"])
    
    amp_enabled = (train_cfg["amp"] and device.type == "cuda")
    
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=amp_enabled) if device.type == "cuda" else None
    
    output_dir = ROOT / cfg["output"]["root"] / cfg["experiment"]["name"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    
    start_epoch = 0
    best_iou = -1.0
    
    if args.resume:
        checkpoint = torch.load(ROOT / args.resume,
                                map_location=device,
                                weights_only=False)
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
            
        start_epoch = checkpoint["epoch"]
        best_iou = checkpoint["best_iou"]
        
        print(f"Resumed from epoch {start_epoch}")
        
    csv_path = output_dir / "metrics.csv"
    
    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "epoch", "lr",
                "train_loss", "val_loss",
                "floor_iou", "miou",
                "precision", "recall",
                "false_drivable_rate",
            ])
            
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        lr = optimizer.param_groups[0]["lr"]
        
        train_metrics = run_epoch(model,
                                  train_loader,
                                  criterion,
                                  device,
                                  optimizer=optimizer,
                                  scaler=scaler,
                                  amp_enabled=amp_enabled,
                                  ignore_index=data_cfg["ignore_index"])
        val_metrics = run_epoch(model,
                                val_loader,
                                criterion,
                                device,
                                amp_enabled=amp_enabled,
                                ignore_index=data_cfg["ignore_index"])
        
        scheduler.step()
        
        print(f"Train Loss: {train_metrics['loss']:.4f} |"
              f"Val Loss: {val_metrics['loss']:.4f} |"
              f"Floor Iou: {val_metrics['floor_iou']:.4f} |"
              f"mIoU: {val_metrics['miou']:.4f}")
        
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch+1)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch+1)
        
        for key, value in val_metrics.items():
            if key != "loss":
                writer.add_scalar(f"Validation/{key}", value, epoch+1)
                
        writer.add_scalar("LearningRate", lr, epoch+1)
        
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1,
                lr,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["floor_iou"],
                val_metrics["miou"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["false_drivable_rate"]
            ])
            
        if val_metrics["floor_iou"] > best_iou:
            best_iou = val_metrics["floor_iou"]
            
            save_checkpoint(output_dir / "best.pth",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch + 1,
                            best_iou,
                            cfg)
            
            print("Best model saved!")
            
        save_checkpoint(output_dir / "last.pth",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch + 1,
                        best_iou,
                        cfg)
        
    writer.close()
    print("\nTraining completed!")
    print(f"Best Floor Iou: {best_iou:.4f}")
            
        
    
if __name__ == "__main__":
    main()