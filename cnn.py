# True CNN for the project test_cnn.py is just a toy cnn for testing
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
import json
from typing import Tuple, Optional, Dict, List

def get_device() -> torch.device:
    # returns best available compute device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WordCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int)->nn.Sequential:
            return nn.Sequential(
                # 3x3 kernel padding first
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                # normalizes outputs
                nn.BatchNorm2d(out_ch),
                # introduces non-linearity
                nn.ReLU(inplace=True),
                # downsamples feature map
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            # stacks 3 conv blocks
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute logits
        x = self.features(x)
        x = self.dropout(x.flatten(1))
        return self.classifier(x)

"""
    Training
"""

# train and validation loop
def _run_epoch(
        model: nn.Module,
        loader,
        criterion,
        optimizer: Optional[optim.Optimizer],
        scaler: Optional[torch.cuda.amp.GradScaler],
        device: torch.device) -> Tuple[float, float]:

    train_mode = optimizer is not None
    model.train(train_mode)
    torch.set_grad_enabled(train_mode)

    total_loss = correct = seen = 0
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        if train_mode:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        seen += imgs.size(0)

    return total_loss / seen, correct / seen


def train(model: nn.Module,
          train_loader,
          val_loader=None,
          epochs: int = 20,
          lr: float = 0.03,
          wd: float = 1e-4,
          device: Optional[torch.device] = None) -> Dict[str, List[float]]:

    if device is None:
        device = get_device()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    hist: Dict[str, List[float]] = {k: [] for k in ("tr_loss", "tr_acc", "va_loss", "va_acc")}

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        hist["tr_loss"].append(tr_loss)
        hist["tr_acc"].append(tr_acc)

        va_loss = va_acc = 0.0
        if val_loader is not None:
            with torch.no_grad():
                va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, None, device)
            hist["va_loss"].append(va_loss)
            hist["va_acc"].append(va_acc)
            scheduler.step(va_loss)
        else:
            scheduler.step(tr_loss)

        print(f"Epoch {ep:02}/{epochs} | loss {tr_loss:.3f} | train {tr_acc:.3f} | val {va_acc:.3f}")

    return hist


if __name__ == "__main__":
    set_seed()
    from dataset import get_dataloaders

    pin = torch.cuda.is_available()
    tr_dl, va_dl = get_dataloaders()

    device = get_device()
    net = WordCNN(len(tr_dl.dataset.label2idx))

    history = train(net, tr_dl, va_dl, epochs=20, device=device)

    print("\nFinal val accuracy:", history["va_acc"][-1] if history["va_acc"] else "N/A")
    #torch.save(net.state_dict(), "cnn_weights.pth")