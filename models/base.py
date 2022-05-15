import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, *args, size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Classifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.Linear(in_channels, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_channels),
        )

    def forward(self, x):
        return self.model(x)

