import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import requests


def download_dataset(
    url: Union[str, Path], output_dir: Union[str, Path], wrapper_folder: str = "data"
) -> None:
    filename = url.split("/")[-1]
    output_dir = Path(output_dir)

    res = requests.get(url, allow_redirects=True)
    with open(filename, "wb") as f:
        f.write(res.content)

    with zipfile.ZipFile(filename) as f:
        f.extractall(output_dir)

    os.remove(filename)
    
    for folder in os.listdir(output_dir / wrapper_folder):
        shutil.move(output_dir / wrapper_folder / folder, output_dir / folder)

    os.rmdir(output_dir / wrapper_folder)


def get_splits(
    samples_per_class: List[List[str]], ratio: Tuple[int, int, int]
) -> Dict[str, str]:
    splits = {}
    for class_samples in samples_per_class:
        n_samples = len(class_samples)
        n_train = int(ratio[0] * n_samples)
        n_valid = int(ratio[1] * n_samples)
        n_test = int(ratio[2] * n_samples)
        n_train += n_samples - (n_train + n_valid + n_test)

        class_samples = np.random.permutation(class_samples)
        for fpath in class_samples[:n_train]:
            splits[fpath] = "train"
        for fpath in class_samples[n_train : n_train + n_valid]:
            splits[fpath] = "valid"
        for fpath in class_samples[n_train + n_valid :]:
            splits[fpath] = "test"

    return splits


def get_samples_per_class(
    data_dir: Path,
    categories: list[str],
    idx2label: Dict[int, str],
    label2idx: Dict[str, int],
) -> List[List[Path]]:
    samples_per_class = [[] for _ in range(len(idx2label))]
    for cat in categories:
        for img_path in os.listdir(data_dir / cat):
            label_idx = label2idx[cat]
            samples_per_class[label_idx].append(Path(cat, img_path))
    return samples_per_class


def split_dataset(data_dir: Union[str, Path], ratio: Tuple[int, int, int]) -> None:
    categories = [cat for cat in os.listdir(data_dir) if (data_dir / cat).is_dir()]
    idx2label = {idx: label for idx, label in enumerate(categories)}
    label2idx = {label: idx for idx, label in enumerate(categories)}

    samples_per_class = get_samples_per_class(data_dir, categories, idx2label, label2idx)

    splits = get_splits(samples_per_class, ratio)

    res = []
    for label_idx, class_samples in enumerate(samples_per_class):
        for fpath in class_samples:
            res.append([fpath, label_idx, splits[fpath]])
    data_df = pd.DataFrame(res, columns=["filepath", "label", "split"])
    data_df.set_index("filepath", inplace=True)

    return data_df, idx2label
