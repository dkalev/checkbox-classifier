import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torchmetrics import Accuracy

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from dataset import CheckBoxDataset, download_dataset, get_split, split_dataset, data_augs


class TrainingModule(pl.LightningModule):
    def __init__(
        self, model: nn.Module, crit: nn.Module, *args, lr: float = 1e-3, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.crit = crit
        self.lr = lr

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        x, targs = batch

        logits = self.model(x)
        loss = self.crit(logits, targs)
        preds = logits.argmax(dim=-1)

        self.log(f"train/loss", loss.item())
        self.log(f"train/acc", self.train_acc(preds, targs), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, targs = batch
        logits = self.model(x)
        loss = self.crit(logits, targs)
        preds = logits.argmax(dim=-1)

        self.log(f"valid/loss", loss.item())
        self.log(f"valid/acc", self.valid_acc(preds, targs))

    def test_step(self, batch, batch_idx):
        x, targs = batch
        logits = self.model(x)
        preds = logits.argmax(dim=-1)
        self.log(f"test/acc", self.test_acc(preds, targs))

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_url",
        type=str,
        default="https://storage.googleapis.com/autify-dev-dl/data/checkbox_state_v2.zip",
    )
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--split_ratio", nargs="+", default=["0.6,0.2,0.2"])

    config = parser.parse_args()
    config.split_ratio = [float(frac) for frac in config.split_ratio[0].split(",")]
    config.dataset_dir = Path(config.dataset_dir)

    if not config.dataset_dir.exists():
        download_dataset(config.dataset_url, config.dataset_dir)
        data_df, idx2label = split_dataset(config.dataset_dir, config.split_ratio)

        data_df.to_csv(config.dataset_dir / "splits.csv")
        with open(config.dataset_dir / "label_mapping.json", "w") as f:
            json.dump(idx2label, f)
    else:
        data_df = pd.read_csv(config.dataset_dir / "splits.csv")

    images_train = get_split(data_df, "train")
    images_valid = get_split(data_df, "valid")
    images_test = get_split(data_df, "test")

    ds_train = CheckBoxDataset(config.dataset_dir, images_train, data_augs["common"])
    ds_valid = CheckBoxDataset(config.dataset_dir, images_valid, data_augs["common"])
    ds_test = CheckBoxDataset(config.dataset_dir, images_test, data_augs["common"])
