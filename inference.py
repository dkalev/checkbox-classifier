import json
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.transforms import Compose

from dataset import get_data_augs
from models import MobileNetV2

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MobileNetV2.load_from_checkpoint("https://drive.google.com/uc?export=download&id=11dbv1XSgiR1QI50fzXwbZnRnfOPy_Ts3")
model.to(device)
model.eval()

with open("data/label_mapping.json") as f:
    label_mapping = json.load(f)

@app.post("/predict/")
def predict(file: UploadFile = File(...)) -> None:
    image = Image.open(file.file).convert("RGB")
    data_augs = get_data_augs(Path("data"))
    transforms = Compose(data_augs["common"] + data_augs["image_net"])
    image = transforms(image).unsqueeze(0).to(device)

    pred = model(image).argmax(dim=-1)
    return JSONResponse({
        "pred": label_mapping[str(pred.item())]
    })
