import argparse
import json
from PIL import Image

from models import MobileNetV2
from dataset import get_data_augs
from pathlib import Path
from torchvision.transforms import Compose


def predict(config: argparse.Namespace) -> None:
    model = MobileNetV2.load_from_checkpoint(config.model_weights_url)
    model.eval()

    with open("data/label_mapping.json") as f:
        label_mapping = json.load(f)

    image = Image.open(config.image_path).convert("RGB")
    data_augs = get_data_augs(Path("data"))
    transforms = Compose(data_augs["common"] + data_augs["image_net"])
    image = transforms(image).unsqueeze(0)

    pred = model(image).argmax(dim=-1)
    print("The model predictions is: ", label_mapping[str(pred.item())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weights_url",
        type=str,
        default=f"https://drive.google.com/uc?export=download&id=1FKkPtQGu3i7EOMuH5qJWt5goV0Rwl21L",
    )
    parser.add_argument("--image_path", type=str)

    config = parser.parse_args()

    predict(config)
