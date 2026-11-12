import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.ade20k_dataset import ADE20KFloorDataset
from models.lraspp import build_lraspp

ROOT = Path(__file__).resolve().parents[1]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
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
    
if __name__ == "__main__":
    main(0)