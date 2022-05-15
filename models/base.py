import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from torch.optim import Adam


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, *args, size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class BaseCNN(pl.LightningModule):
    def __init__(
        self,
        *args,
        lr: float = 1e-3,
        weight_decay: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.crit = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_f1 = F1Score(num_classes=3, average="weighted")
        self.valid_f1 = F1Score(num_classes=3, average="weighted")
        self.test_f1 = F1Score(num_classes=3, average="weighted")

    def training_step(self, batch, batch_idx):
        x, targs = batch

        logits = self(x)
        loss = self.crit(logits, targs)
        preds = logits.argmax(dim=-1)

        self.log(f"train/loss", loss.item())
        self.log(f"train/acc", self.train_acc(preds, targs), prog_bar=True)
        self.log(f'train/f1', self.train_f1(preds, targs), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, targs = batch
        logits = self(x)
        loss = self.crit(logits, targs)
        preds = logits.argmax(dim=-1)

        self.log(f"valid/loss", loss.item())
        self.log(f"valid/acc", self.valid_acc(preds, targs))
        self.log(f'valid/f1', self.valid_f1(preds, targs))

    def test_step(self, batch, batch_idx):
        x, targs = batch
        logits = self(x)
        preds = logits.argmax(dim=-1)
        self.log(f"test/acc", self.test_acc(preds, targs))
        self.log(f'test/f1', self.test_f1(preds, targs))
        # mainly used for optuna
        loss = self.crit(logits, targs)
        self.log(f"test/loss", loss.item())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)



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

