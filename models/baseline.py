from collections import OrderedDict

import torch.nn as nn

from .base import Classifier


class BaselineModel(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 100, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 200, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
        )

        self.head = Classifier(400, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
