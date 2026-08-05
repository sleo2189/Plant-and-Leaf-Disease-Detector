# Plant-and-Leaf-Disease-Detector
Transfer-learning CNN for multi-class plant/leaf disease detection, served via FastAPI and Streamlit.

## About this project

This repository implements an end-to-end computer vision system for detecting plant and leaf diseases from images.

- Uses a CNN with transfer learning (PyTorch) trained on a public plant disease dataset.
- Provides a FastAPI backend for model inference via REST API.
- Includes a Streamlit web app where users can upload leaf images and get predicted disease labels with confidence scores.

The goal is to demonstrate a real-world ML pipeline — from data and model training to deployment — that can help farmers, agronomists, and researchers quickly identify potential plant diseases from photos.




## 1. Recommended project structure

Create this structure in your local repo folder (e.g. `plant-and-leaf-disease-detector/`):

```text
plant-and-leaf-disease-detector/
├─ .streamlit/
│  └─ config.toml
├─ app/
│  ├─ __init__.py
│  ├─ model.py
│  ├─ inference.py
│  └─ app.py
├─ notebooks/
│  └─ train_model.ipynb
├─ models/
│  └─ best_model.pt         # will be added after training
├─ requirements.txt
├─ README.md
└─ runtime.txt
```

You can put all Python files under `app/` so the Streamlit main file is `app/app.py`.

***

## 2. Python 3.13–compatible `requirements.txt`

This is what Streamlit Cloud will use to install dependencies.

```txt
# Core
torch
torchvision
torchaudio
numpy
pillow
streamlit

# Optional but useful
opencv-python-headless
tqdm
```

Notes:

- On Streamlit Cloud, you may need to force a specific Python version (see `runtime.txt` below) if you encounter issues with PyTorch + 3.13. [discuss.streamlit](https://discuss.streamlit.io/t/modulenotfounderror-no-module-named-torch-when-deploying-streamlit-app/120604)
- If you hit compatibility problems, you can temporarily pin versions, e.g.:

  ```txt
  torch==2.5.0
  torchvision==0.20.0
  torchaudio==2.5.0
  streamlit==1.38.0
  ```

  but start with the unpinned versions above.

***

## 3. Force Python version (optional but recommended)

Create `runtime.txt` in the repo root:

```text
python-3.13
```

This tells platforms like Streamlit Cloud to use Python 3.13. [discuss.streamlit](https://discuss.streamlit.io/t/streamlit-cloud-using-python-3-13-despite-runtime-txt-specifying-3-11/113759)

***

## 4. Streamlit config (optional)

Create `.streamlit/config.toml`:

```toml
[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
```

This is optional but clean for deployment.

***

## 5. Model definition (`app/model.py`)

This must match the architecture you used during training.

```python
# app/model.py
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 23  # adjust to your dataset (e.g. 23 for Plant Disease Recognition)

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build a ResNet-18 based classifier for plant disease detection.
    Must match the architecture used during training.
    """
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load model weights from a checkpoint saved with torch.save({...}).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes
```

Notes:

- `weights_only=True` is good practice for loading untrusted checkpoints in newer PyTorch versions.
- Ensure `NUM_CLASSES` or the logic matches what you saved in `best_model.pt`.

***

## 6. Inference utilities (`app/inference.py`)

This handles image preprocessing and prediction.

```python
# app/inference.py
from PIL import Image
import torch
from torchvision import transforms

from .model import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Same transforms as used in training (validation-time)
infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_model_and_classes(checkpoint_path: str = "models/best_model.pt"):
    """
    Load model and class list from checkpoint.
    Adjust path if your model is stored elsewhere.
    """
    return load_model(checkpoint_path, device=DEVICE)

def predict_image(model, classes, image: Image.Image, top_k: int = 3):
    """
    Predict top-k classes for a PIL image.
    Returns list of (label, confidence) tuples.
    """
    tensor = infer_tfms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(top_k, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    results = []
    for p, idx in zip(top_probs, top_idxs):
        label = classes[idx]
        results.append((label, float(p)))

    return results
```

***

## 7. Streamlit app (`app/app.py`)

This is your main Streamlit file. It avoids deprecated `use_column_width` and uses `width` instead.

```python
# app/app.py
import streamlit as st
from PIL import Image

from .inference import get_model_and_classes, predict_image

st.set_page_config(
    page_title="Plant & Leaf Disease Detector",
    page_icon="🌿"
)

st.title("🌿 Plant & Leaf Disease Detector")
st.markdown(
    "Upload a leaf image to get a predicted disease label and confidence score."
)

# Load model once (cached)
@st.cache_resource
def load_model_once():
    return get_model_and_classes("models/best_model.pt")

try:
    model, classes = load_model_once()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Use width instead of deprecated use_column_width [62]
    st.image(image, caption="Uploaded image", width=600)

    with st.spinner("Running inference..."):
        results = predict_image(model, classes, image, top_k=3)

    st.subheader("Top predictions")
    for label, conf in results:
        st.write(f"- **{label}**: {conf:.2%}")
```

Important:

- The path `"models/best_model.pt"` assumes the `models/` folder is at the repo root. Streamlit Cloud runs from the repo root, so this relative path works.
- If you change the structure, adjust the path accordingly.

***

## 8. Example training notebook (outline)

You don’t need full code here, but your `notebooks/train_model.ipynb` should:

1. Load the dataset (e.g. from `data/raw` with `ImageFolder`).
2. Define the same model architecture as in `app/model.py`.
3. Train and save a checkpoint like:

   ```python
   torch.save({
       "model_state": model.state_dict(),
       "classes": train_dataset.classes
   }, "models/best_model.pt")
   ```

Make sure the **classes list order** matches what you use at inference time (it will, if you save/load it as above).

***

## 9. README.md template

Create `README.md` in the repo root:

```markdown
# Plant & Leaf Disease Detector

Deep learning app to detect plant and leaf diseases from images using PyTorch, with a Streamlit web UI.

## Features

- CNN-based multi-class disease classifier (ResNet-18 backbone).
- Simple web interface to upload leaf images and view predictions.
- Easily extendable to more plants/diseases.

## Tech stack

- Python 3.13
- PyTorch + torchvision
- Streamlit
- OpenCV (optional)

## Project structure

- `app/app.py` – Streamlit UI.
- `app/model.py` – Model definition.
- `app/inference.py` – Image preprocessing and prediction logic.
- `notebooks/train_model.ipynb` – Training code (example).
- `models/best_model.pt` – Trained model checkpoint (add after training).

## Local setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sleo2189/plant-and-leaf-disease-detector.git
   cd plant-and-leaf-disease-detector
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained model at `models/best_model.pt`.

5. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

## Deployment

To deploy on Streamlit Community Cloud:

1. Push this repo to GitHub (already done).
2. In Streamlit Cloud, create a new app:
   - Repository: `sleo2189/plant-and-leaf-disease-detector`
   - Branch: `main`
   - Main file path: `app/app.py`
3. Ensure `requirements.txt` and `runtime.txt` are present.

## Dataset

This project uses a public plant disease dataset (e.g., Plant Disease Recognition on Kaggle) with multiple plant types and disease classes.

## License

MIT
```

***

## 10. Git commands to update your repo (from your machine)

In your project folder:

```bash
git add .
git commit -m "Add Streamlit app, model code, and project setup for Python 3.13"
git push origin main
```

Then in Streamlit Cloud, redeploy or let auto‑deploy kick in.
