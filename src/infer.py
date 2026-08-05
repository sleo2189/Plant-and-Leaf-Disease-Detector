# src/infer.py
import torch
from torchvision import transforms
from PIL import Image

from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    classes = checkpoint["classes"]
    num_classes = len(classes)
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, classes

def predict_image(model, classes, img_path: str, topk: int = 3):
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("No class labels were found in the checkpoint.")

    valid_topk = min(topk, len(classes))
    if valid_topk <= 0:
        raise ValueError("topk must be greater than 0.")

    img = Image.open(img_path).convert("RGB")
    tensor = infer_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(valid_topk, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    results = []
    for p, idx in zip(top_probs, top_idxs):
        label = classes[idx]
        results.append((label, float(p)))

    return results

if __name__ == "__main__":
    model, classes = load_model("models/best_model.pt")
    preds = predict_image(model, classes, "sample_leaf.jpg")
    print(preds)