
from train import train as train_fn
from pathlib import Path
import optuna
import argparse

def objective(trial: optuna.trial.Trial) -> float:
    config = argparse.Namespace

    config.dataset_url = "https://storage.googleapis.com/autify-dev-dl/data/checkbox_state_v2.zip"
    config.dataset_dir = Path("data")
    config.log_dir = "logs"
    config.split_ratio = [0.6, 0.2, 0.2] 
    config.num_workers = 4

    config.n_epochs = 2
    config.model = trial.suggest_categorical("model", ["baseline", "mobilenet", "resnet50"])
    config.lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    config.batch_size = trial.suggest_int("batch_size", 4, 64, log=True)

    res = train_fn(config)
    return res[0]["test/loss"]

def search() -> None:
    sampler = optuna.samplers.TPESampler(seed=17)
    study = optuna.create_study(study_name="check_box_hparam_search", direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=100)

if __name__ == "__main__":
    search()