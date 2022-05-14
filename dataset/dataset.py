from pathlib import Path
from typing import Any

import pandas as pd
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CheckBoxDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        images_meta: list[tuple[str, int]],
        data_augs: list[Any],
        grayscale: bool = False,
    ):
        self.color_mode = "L" if grayscale else "RGB"
        self.images_meta = [[data_dir / fpath, label] for fpath, label in images_meta]
        self.transforms = transforms.Compose(data_augs)

    def __len__(self):
        return len(self.images_meta)

    def __getitem__(self, idx):
        x, y = self.images_meta[idx]
        x = Image.open(x).convert(self.color_mode)
        return self.transforms(x), y


def get_split(
    data_df: pd.DataFrame, split: str, subset: tuple[str] = ("filepath", "label")
) -> list[list[Any]]:
    return data_df.loc[data_df["split"] == split, subset].values.tolist()




class SquarePad:
    """A transformation that pads a rectangular image to a square
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
    """

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


data_augs = {
    "common": [SquarePad(), transforms.ToTensor(), transforms.Resize(size=224)],
    "image_net": [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ],
    "train": [
        transforms.ColorJitter(),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(degrees=0),  # no rotations
    ],
}
