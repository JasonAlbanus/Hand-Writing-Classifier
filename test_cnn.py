import torch.nn as nn

class testCNN(nn.Module):
    # this would only work for 28 x 28 digits
    def __init(self, num_classes=10):
        super().__init__()
        # feature extraction
        # each conv learns patterns about the image.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # only keep positive activations (removes negative noise)
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        # classifier head - transofrms flattened vector into class logits
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x