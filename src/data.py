# src/data.py
import os
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224


def _resolve_dataset_root(data_dir):
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    if data_dir:
        candidates.append(Path(data_dir))

    candidates.extend([
        Path("data/raw"),
        Path("data/procesed"),
        Path("data/processed"),
        Path("data"),
        repo_root / "data" / "raw",
        repo_root / "data" / "procesed",
        repo_root / "data" / "processed",
        repo_root / "data",
    ])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "No dataset directory found. Expected a folder such as 'data/raw' or 'data/procesed' "
        "containing 'train', 'validation', and 'test' subfolders."
    )


def _resolve_split_dir(dataset_root, split_name):
    split_dir = dataset_root / split_name
    if split_dir.is_dir():
        return split_dir.resolve()

    raise FileNotFoundError(
        f"Missing dataset split '{split_name}' under '{dataset_root}'. "
        f"Create '{split_dir}' with class subfolders or point DATA_DIR to the correct dataset root."
    )


def get_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_tfms, val_tfms

def get_dataloaders(data_dir, batch_size=32):
    train_tfms, val_tfms = get_transforms()

    dataset_root = _resolve_dataset_root(data_dir)
    train_dir = _resolve_split_dir(dataset_root, "train")
    val_dir = _resolve_split_dir(dataset_root, "validation")
    test_dir = _resolve_split_dir(dataset_root, "test")

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tfms)
    test_ds = datasets.ImageFolder(str(test_dir), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.classes