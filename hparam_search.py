import argparse
import pickle as pkl
import time
import numpy as np
import traceback
from pathlib import Path

import optuna

from train import train as train_fn


class Objective:
    def __init__(self, config: argparse.Namespace, n_epochs: int = 10) -> None:
        self.config = config
        self.config.n_epochs = n_epochs

    def __call__(self, trial: optuna.trial.Trial) -> float:
        config = self.config
        config.model = trial.suggest_categorical(
            "model", ["baseline", "mobilenet", "resnet50"]
        )
        config.lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        config.batch_size = trial.suggest_int("batch_size", 4, 64, log=True)

        try:
            res = train_fn(config)
            return res[0]["test/loss"]
        except Exception:
            print(traceback.format_exc())
            return np.inf



def search(config: argparse.Namespace) -> None:
    sampler = optuna.samplers.TPESampler(seed=17)
    study = optuna.create_study(
        study_name="check_box_hparam_search", direction="minimize", sampler=sampler
    )
    study.optimize(Objective(config), n_trials=config.n_trials)

    with open(config.log_dir / f"study_{int(time.time())}.pkl", "wb") as f:
        pkl.dump(study, f)

    res = []
    for trial in study.trials:
        res.append([trial.value] + list(trial.params.items()))
    for trial_summary in sorted(res, key=lambda x: x[0])[:10]:
        print(trial_summary)


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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_trials", type=int, default=50)

    config = parser.parse_args()
    config.split_ratio = [float(frac) for frac in config.split_ratio[0].split(",")]
    config.dataset_dir = Path(config.dataset_dir)
    config.log_dir = Path(config.log_dir)

    search(config)
