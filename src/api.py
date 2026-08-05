# src/api.py
import io
from typing import List, Tuple

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import build_model

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

checkpoint = torch.load("models/best_model.pt", map_location=DEVICE)
classes = checkpoint["classes"]
model = build_model(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = infer_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, 1)

    predicted_class = classes[top_idx.item()]
    confidence = float(top_prob.item())

    return {
        "class": predicted_class,
        "confidence": confidence
    }