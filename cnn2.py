import json, os, math, time
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

import dataset            # <- your existing dataset.py with get_dataloaders

# ---------- Utility ----------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- Model ----------
class WordCNN(nn.Module):
    """Slightly deeper but still fast CNN for IAM word images"""
    def __init__(self, num_classes: int, p_drop: float = 0.2):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p_drop),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)

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

def epoch_loop(model, loader, criterion, optimizer, device, scaler,
               phase: str, log_every: int = 100) -> Tuple[float, float]:
    is_train = phase == "train"
    model.train(is_train)
    running_loss = correct = total = 0

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        with autocast(enabled=scaler is not None):
            out = model(x)
            loss = criterion(out, y)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)

        if log_every and i % log_every == 0:
            print(f"{phase:5} [{i:>3}/{len(loader)}] "
                  f"loss {loss.item():.3f}")

    return running_loss/total, correct/total

# ---------- Main training ----------
def train(num_epochs: int = 25, patience: int = 5):
    set_seed(42)
    train_dl, val_dl = dataset.get_dataloaders(
        transform=transforms.Compose([
            transforms.RandomResizedCrop((128, 512), scale=(0.9, 1.1)),
            transforms.RandomAffine(degrees=5, translate=(.05,.05)),
            transforms.RandomPerspective(distortion_scale=.2, p=.3),
            transforms.ToTensor()
        ]),
        val_transform=transforms.ToTensor(),
        batch_size=64, num_workers=4
    )

    num_classes = len(train_dl.dataset.label2idx)
    device = get_device()
    print("Using", device)

    model = WordCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = num_epochs * len(train_dl)
    scheduler = CosineWithWarmup(optimizer, warmup=500, total_steps=total_steps)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val = 0.0
    history: Dict[str, List[float]] = {"tr_loss": [], "tr_acc": [],
                                       "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        tl, ta = epoch_loop(model, train_dl, criterion,
                            optimizer, device, scaler, "train")
        vl, va = epoch_loop(model, val_dl, criterion,
                            optimizer, device, None, "val")
        scheduler.step()

        history["tr_loss"].append(tl); history["tr_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)

        print(f"summary: train {ta*100:5.2f}% | val {va*100:5.2f}%")

        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), "model.dump")
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # write history
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy {best_val*100:.2f}% "
          f"(epoch {best_epoch})")
    print("Model weights  -> model.dump")
    print("Training curve -> history.json")


if __name__ == "__main__":
    start = time.time()
    train()
    print(f"Total wall-time {time.time()-start:.1f}s")