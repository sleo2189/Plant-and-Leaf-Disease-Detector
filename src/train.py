# src/train.py
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("DATA_DIR", str(REPO_ROOT / "data" / "raw"))
MODEL_PATH = REPO_ROOT / "models" / "best_model.pt"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, batch_size=32)
    num_classes = len(classes)

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch+1}: "
              f"Train loss {train_loss:.4f}, acc {train_acc:.4f}, "
              f"Val loss {val_loss:.4f}, acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes
            }, MODEL_PATH)
            print("Saved new best model")

    # Optional: evaluate on test set
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")

if __name__ == "__main__":
    main()