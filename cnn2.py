import json, os, math, time, re
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import random, numpy as np
import torch.nn.functional as F    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import Subset

import dataset            # <- your existing dataset.py with get_dataloaders

# ---------- Utility ----------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Model ----------
from torchvision.models import resnet18, ResNet18_Weights

class WordCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.conv1.stride = (1, 1)
        net.layer4[0].conv1.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.dropout  = nn.Dropout(dropout_p)
        self.dropblock = DropBlock2d(p=0.1, block_size=5)
        self.head     = nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropblock(x)
        x = x.flatten(1)
        return self.head(x)

class DropBlock2d(nn.Module):
    def __init__(self, p: float = 0.1, block_size: int = 5):
        super().__init__()
        self.p = p; self.block_size = block_size
    def forward(self, x):
        if not self.training or self.p == 0.: return x
        gamma = self.p / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :1]) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size,
                             stride=1, padding=self.block_size//2)
        return x * (1 - mask)

# ---------- Training helpers ----------
class CosineWithWarmup(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, total_steps):
        self.warmup = warmup; self.total = total_steps
        super().__init__(optimizer)
    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup:
            return [base * step / self.warmup for base in self.base_lrs]
        cos = 0.5 * (1 + math.cos(math.pi * (step - self.warmup) /
                                   max(1, self.total - self.warmup)))
        return [base * cos for base in self.base_lrs]

def mixup(x, y, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, (y, y[idx], lam)

# compute one epoch
def epoch_loop(model, loader, criterion, optimizer, device, scaler,
               phase: str, log_every: int = 100) -> Tuple[float, float]:
    is_train = (phase == "train")
    model.train(is_train)
    running_loss = correct = total = 0
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        if is_train and random.random() < 0.8:
            x, y = mixup(x, y, alpha=0.2)
        with autocast(device.type, enabled=(device.type=="cuda")):
            out = model(x)
            if isinstance(y, tuple):
                y1, y2, lam = y
                loss = lam * criterion(out, y1) + (1-lam) * criterion(out, y2)
            else:
                loss = criterion(out, y)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        running_loss += loss.item() * x.size(0)
        y_true = y[0] if isinstance(y, tuple) else y
        preds = out.argmax(1)
        correct += preds.eq(y_true).sum().item()
        total += y_true.size(0)
        if log_every and i % log_every == 0:
            print(f"{phase:5} [{i}/{len(loader)}] loss {loss.item():.3f}")
    return running_loss/total, correct/total

# ---------- Main training ----------
def train(num_epochs: int = 25, patience: int = 5):
    set_seed(42)
    train_dl, val_dl = dataset.get_dataloaders()

    # --------- filter punctuation labels for Subset ---------
    pattern = re.compile(r'^[A-Za-z]+$')
    ds = train_dl.dataset
    if isinstance(ds, Subset):
        full = ds.dataset
        # keep only indices with alphabetic labels
        new_idx = [i for i in ds.indices if pattern.match(full.samples[i][1])]
        ds.indices = new_idx
        # rebuild mapping on full.dataset
        labels = sorted({full.samples[i][1] for i in new_idx})
        full.label2idx = {l: j for j, l in enumerate(labels)}
        full.idx2label = {j: l for l, j in full.label2idx.items()}
    else:
        # direct dataset.samples
        full = ds
        filtered = [(p, l) for p, l in full.samples if pattern.match(l)]
        full.samples = filtered
        labels = sorted({l for _, l in filtered})
        full.label2idx = {l: j for j, l in enumerate(labels)}
        full.idx2label = {j: l for l, j in full.label2idx.items()}

    # transforms
    def keep_ratio_resize(h=128):
        return transforms.Lambda(lambda img: transforms.functional.resize(img, size=(h, int(img.width * h / img.height))))
    train_tf = transforms.Compose([
        keep_ratio_resize(128),
        transforms.Pad((0,0,16,0), fill=255),
        transforms.RandomAffine(degrees=4, translate=(0.03,0.03), scale=(0.95,1.05)),
        transforms.ToTensor(),
    ])
    clean_tf = transforms.Compose([
        keep_ratio_resize(128), transforms.Pad((0,0,16,0), fill=255), transforms.ToTensor()
    ])
    train_dl.dataset.transform = train_tf
    val_dl.dataset.transform = clean_tf

    num_classes = len(full.label2idx)
    device = get_device(); print("Using", device)
    model = WordCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    swa_start = 25; swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=2e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)
    scaler = GradScaler(device.type, enabled=(device.type=="cuda"))

    best_val = 0.0; wait = 0; best_epoch = 0
    history = {"tr_loss":[], "tr_acc":[], "val_loss":[], "val_acc":[]}

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        tl, _ = epoch_loop(model, train_dl, criterion, optimizer, device, scaler, "train")
        # clean train accuracy
        saved_tf = train_dl.dataset.transform
        train_dl.dataset.transform = clean_tf
        _, ta = epoch_loop(model, train_dl, criterion, optimizer, device, None, "val")
        train_dl.dataset.transform = saved_tf
        # val accuracy
        vl, va = epoch_loop(model, val_dl, criterion, optimizer, device, None, "val")

        history["tr_loss"].append(tl)
        history["tr_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"summary: train {ta*100:5.2f}% | val {va*100:5.2f}%")

        # save & early-stop
        if va > best_val:
            best_val = va; best_epoch = epoch; wait = 0
            torch.save(model.state_dict(), "model.dump")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping."); break

        if epoch >= swa_start:
            swa_model.update_parameters(model); swa_scheduler.step()
        else:
            scheduler.step()

    # finalize SWA
    val_dl.dataset.transform = clean_tf
    torch.optim.swa_utils.update_bn(val_dl, swa_model, device=device)
    torch.save(swa_model.state_dict(), "model.dump")

    with open("history.json", "w") as f: json.dump(history, f, indent=2)
    print(f"\nBest validation accuracy {best_val*100:.2f}% (epoch {best_epoch})")
    print("Model weights  -> model.dump")
    print("Training curve -> history.json")

if __name__ == "__main__":
    start = time.time(); train(); print(f"Total wall-time {time.time()-start:.1f}s")