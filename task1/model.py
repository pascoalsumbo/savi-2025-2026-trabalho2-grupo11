import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelBetterCNN(nn.Module):
    """
    CNN mais robusta para MNIST:
    - 3 blocos conv + BN + ReLU
    - Dropout
    - Head fully-connected
    """
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.25):
        super().__init__()

        self.features = nn.Sequential(
            # 1x28x28 -> 32x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32x28x28 -> 64x28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # -> 64x14x14
            nn.Dropout(dropout_p),

            # 64x14x14 -> 128x14x14
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # -> 128x7x7
            nn.Dropout(dropout_p),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
