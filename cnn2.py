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
from torchvision.models import resnet18, ResNet18_Weights

class WordCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=w)
        # 1) accept 1-channel or 3-channel: IAM is grayscale PNG but loader returns 3-ch RGB
        net.conv1.stride = (1, 1)          # keep more spatial info
        net.layer4[0].conv1.stride = (1, 1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # no FC
        self.dropout  = nn.Dropout(dropout_p)
        self.head     = nn.Linear(net.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)

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

    train_dl, val_dl = dataset.get_dataloaders()

    # --- safer augmentation for IAM ------------------------------------
    def keep_ratio_resize(h=128):
        return transforms.Lambda(
            lambda img: transforms.functional.resize(
                img, size=(h, int(img.width * h / img.height))
            )
        )

    train_dl.dataset.transform = transforms.Compose([
        keep_ratio_resize(128),                      # keep aspect ratio
        transforms.Pad((0, 0, 16, 0), fill=255),     # pad RHS 16 px
        transforms.RandomAffine(degrees=4,
                                translate=(0.03, 0.03),
                                scale=(0.95, 1.05)),
        transforms.ToTensor()
    ])
    val_dl.dataset.transform = transforms.ToTensor()
    # --------------------------------------------------------------------

    num_classes = len(train_dl.dataset.label2idx)
    device = get_device()
    print("Using", device)

    model = WordCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()                # ← no smoothing

    # ↑ learning-rate from start, decay each epoch
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)
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
        scheduler.step()                             # now really steps

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