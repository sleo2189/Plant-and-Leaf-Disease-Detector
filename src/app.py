# src/app.py
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from model import build_model

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

st.title("Plant Leaf Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", width=600)

    tensor = infer_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.max(probs, 0)
    predicted_class = classes[top_idx.item()]
    confidence = float(top_prob.item())

    st.write(f"Prediction: **{predicted_class}** (confidence {confidence:.2%})")