from collections import OrderedDict

import torch
import torch.nn as nn

from .base import BaseCNN, Classifier


class ResNet50(BaseCNN):
    def __init__(self, *args, pretrained: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=pretrained)
        self.encoder = nn.Sequential(
            OrderedDict(list(resnet50.named_children())[:-2])
        )
        
        # freeze pretrained part except last two layers
        for param in list(self.encoder.parameters())[:-6]:
            param.requires_grad = False

        self.head = Classifier(4096, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
