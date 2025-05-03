# True CNN for the project test_cnn.py is just a toy cnn for testing
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

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
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute logits
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

"""
    Training
"""

# train and validation loop
def _run_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        log_interval: int = 100,
        phase: str = "train") -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    # sum of per batch losses
    running_loss = 0.0
    # num of correct predictions
    correct = 0
    # total num samples
    total = 0

    total_batches = len(loader)
    torch.set_grad_enabled(is_train)
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # Console progress (only if requested)
        if log_interval and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
            percent = 100.0 * (batch_idx + 1) / total_batches
            print(f"[{phase}] {batch_idx+1:>4}/{total_batches} ({percent:5.1f}%) | loss {loss.item():.4f}")

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: Optional[torch.utils.data.DataLoader] = None,
          num_epochs : int = 10,
          lr: float = 0.001,
          weight_decay: float = 0.0001,
          device: torch.device | None = None,
          log_interval: int = 100) -> None:
    if device is None:
        device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=log_interval, phase="train",
        )

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc = _run_epoch(
                    model, val_loader, criterion, optimizer=None, device=device,
                    log_interval=log_interval, phase="val",
                )
            print(
                f"[epoch summary] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
        else:
            print(f"[epoch summary] train loss {tr_loss:.4f} acc {tr_acc:.4f}")



if __name__ == '__main__':
    set_seed(42)

    from dataset import get_dataloaders
    train_dl, val_dl = get_dataloaders()

    num_classes = len(train_dl.dataset.label2idx)
    device = get_device()

    print("Device       :", device)
    print("Train samples:", len(train_dl.dataset))
    print("Val samples  :", len(val_dl.dataset))
    print("Num classes  :", num_classes)

    net = WordCNN(num_classes)
    train(
        net,
        train_dl,
        val_dl,
        num_epochs=10,
        log_interval=100,
        device=device,
    )