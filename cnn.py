# True CNN for the project test_cnn.py is just a toy cnn for testing
import dataset
import torch
import torch.nn as nn
import torch.optim as optim

def get_device() -> torch.device:
    # returns best available compute device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(model: nn.Module,
          dataloader: torch.utils.data.dataloader.DataLoader,
          num_epochs : int = 10,
          lr: float = 0.001,
          weight_decay: float = 0.0001,
          device: torch.device | None = None
          ) -> None:
    if device is None:
        device = get_device()
    model.to(device)

    # optimizer / loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    for epoch in range(num_epochs):
        model.train()
        # sum of per batch losses
        running_loss = 0.0
        # num of correct predictions
        correct = 0
        # total num samples
        total = 0

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # clear old gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # back propagate
            loss.backward()
            # param updates
            optimizer.step()

            # metrics
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        # epoch
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"loss = {epoch_loss:.4f} | acc = {epoch_acc:.4f}")

if __name__ == '__main__':
    # import data
    from dataset import get_dataloader
    dl = get_dataloader()
    # num classes
    num_classes = len(dl.dataset.label2idx)
    # model + train
    net = WordCNN(num_classes)
    train(net, dl, num_epochs=10)