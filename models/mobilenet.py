from collections import OrderedDict

import torch
import torch.nn as nn

from .base import Classifier


class MobileNetV2(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        mobilenet2 = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=pretrained
        )
        self.encoder = nn.Sequential(
            OrderedDict(list(mobilenet2.named_children())[:-1])
        )

        # freeze pretrained part except the last two layers
        # FIXME: will fail if the naming format in torch hub changes
        for name, param in self.encoder.named_parameters():
            layer = int(name.split('.')[1])
            if layer < 17:
                param.requires_grad = False

        self.head = Classifier(2560, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
