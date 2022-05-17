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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from dataset import (
    CheckBoxDataset,
    download_dataset,
    get_data_augs,
    get_split,
    split_dataset,
    get_dataset_stats,
)
from models import BaselineModel, MobileNetV2, ResNet50


def get_model(model_name: str) -> nn.Module:
    return {
        "baseline": BaselineModel,
        "mobilenet": MobileNetV2,
        "resnet50": ResNet50,
    }[model_name]


def train(config: argparse.Namespace) -> None:
    if not config.dataset_dir.exists():
        download_dataset(config.dataset_url, config.dataset_dir)
        data_df, idx2label = split_dataset(config.dataset_dir, config.split_ratio)

        data_df.to_csv(config.dataset_dir / "splits.csv")
        with open(config.dataset_dir / "label_mapping.json", "w") as f:
            json.dump(idx2label, f)
        ds_mean, ds_std = get_dataset_stats(config.dataset_dir, data_df)
        with open(config.dataset_dir / "ds_stats.json", "w") as f:
            json.dump({ "mean": ds_mean, "std": ds_std }, f)
        
    else:
        data_df = pd.read_csv(config.dataset_dir / "splits.csv")

    images_train = get_split(data_df, "train")
    images_valid = get_split(data_df, "valid")

    data_augs = get_data_augs(config.dataset_dir)

    model_data_augs = {
        BaselineModel: data_augs["common"] + data_augs["baseline"],
        MobileNetV2: data_augs["common"] + data_augs["image_net"],
        ResNet50: data_augs["common"] + data_augs["image_net"],
    }

    model = get_model(config.model)(
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    ds_train = CheckBoxDataset(
        config.dataset_dir,
        images_train,
        data_augs["train"] + model_data_augs[model.__class__],
    )
    ds_valid = CheckBoxDataset(
        config.dataset_dir, images_valid, model_data_augs[model.__class__]
    )

    dl_train = DataLoader(
        ds_train,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    dl_valid = DataLoader(
        ds_valid, batch_size=config.batch_size, num_workers=config.num_workers
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=TensorBoardLogger(config.log_dir),
        log_every_n_steps=3,
        callbacks=[EarlyStopping(monitor="valid/loss", mode="min")],
    )
    trainer.fit(model, dl_train, dl_valid)

    return trainer.test(model, dl_valid, ckpt_path='best')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_url",
        type=str,
        default="https://storage.googleapis.com/autify-dev-dl/data/checkbox_state_v2.zip",
    )
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--split_ratio", nargs="+", default=["0.6,0.2,0.2"])
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "mobilenet", "resnet50"],
    )
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    config = parser.parse_args()
    config.split_ratio = [float(frac) for frac in config.split_ratio[0].split(",")]
    config.dataset_dir = Path(config.dataset_dir)

    score = train(config)
    print(score)
