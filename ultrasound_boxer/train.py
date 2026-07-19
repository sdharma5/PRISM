"""
Train a fine-tuned ResNet18 patch classifier on MMOTU.

Replaces the previous HOG + logistic regression approach with a pretrained CNN
that is fine-tuned on the ultrasound patches — substantially more accurate.

Dataset: MMOTU OTU_2d (already downloaded to data/MMOTU/OTU_2d)

Usage:
  python train.py data/MMOTU/OTU_2d
  python train.py data/MMOTU/OTU_2d --out us_classifier.pt --epochs 30
"""

from __future__ import annotations

import argparse

import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

PATCH_SIZE = 64
POS_PER_IMAGE = 20
NEG_PER_IMAGE = 30
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 6
SEED = 42

# ImageNet normalization (grayscale repeated to 3 channels)
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── data loading ─────────────────────────────────────────────────────────────

def load_mmotu(data_dir: Path):
    img_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    for img_path in sorted(img_dir.glob("*.JPG")) + sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        mask_path = ann_dir / (img_path.stem + "_binary.PNG")
        if not mask_path.exists():
            mask_path = ann_dir / (img_path.stem + "_binary.png")
        if not mask_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path)) > 0
        yield img, mask


def _border_mask(shape: tuple[int, int], half: int) -> np.ndarray:
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    m[half:h - half, half:w - half] = True
    return m


def build_dataset(data_dir: Path, rng: np.random.Generator):
    half = PATCH_SIZE // 2
    patches: list[np.ndarray] = []
    labels: list[int] = []
    n_img = 0

    for img, mask in load_mmotu(data_dir):
        n_img += 1
        border = _border_mask(img.shape, half)

        pos_coords = np.argwhere(mask & border)
        if len(pos_coords) > 0:
            chosen = pos_coords[rng.choice(len(pos_coords), min(POS_PER_IMAGE, len(pos_coords)), replace=False)]
            for r, c in chosen:
                patches.append(img[r - half:r + half, c - half:c + half])
                labels.append(1)

        dilated = ndi.binary_dilation(mask, iterations=half)
        neg_region = ~dilated & border
        neg_coords = np.argwhere(neg_region)
        if len(neg_coords) > 0:
            chosen = neg_coords[rng.choice(len(neg_coords), min(NEG_PER_IMAGE, len(neg_coords)), replace=False)]
            for r, c in chosen:
                patches.append(img[r - half:r + half, c - half:c + half])
                labels.append(0)

    y = np.array(labels, dtype=np.int64)
    X = np.stack(patches, axis=0)  # (N, H, W) uint8
    print(f"  {n_img} images → {(y == 1).sum()} pos / {(y == 0).sum()} neg patches")
    return X, y


# ── dataset & transforms ─────────────────────────────────────────────────────

class PatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.patches = patches
        self.labels = labels
        base = [T.ToPILImage(), T.Resize((64, 64))]
        if augment:
            base += [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(brightness=0.3, contrast=0.3),
            ]
        base += [T.ToTensor()]
        self.transform = T.Compose(base)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]  # (H, W) uint8
        tensor = self.transform(patch)           # (1, 64, 64) float
        tensor = tensor.repeat(3, 1, 1)          # (3, 64, 64)
        tensor = NORMALIZE(tensor)
        return tensor, int(self.labels[idx])


# ── model ────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze early feature extraction; fine-tune the deeper layers
    for name, param in model.named_parameters():
        if not any(k in name for k in ("layer3", "layer4", "fc")):
            param.requires_grad = False
    model.fc = nn.Linear(512, 2)
    return model


# ── training loop ────────────────────────────────────────────────────────────

def train(data_dir: str, out: str, epochs: int) -> None:
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Building dataset from {data_dir} ...")
    X, y = build_dataset(Path(data_dir), rng)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    train_loader = DataLoader(
        PatchDataset(X_tr, y_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        PatchDataset(X_val, y_val, augment=False),
        batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
    )

    model = build_model().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_train:,}")

    # upweight positives since negatives are 1.5x more common
    pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += len(yb)
        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)

        tr_acc = tr_correct / tr_total
        val_acc = val_correct / val_total
        print(f"  Epoch {epoch:02d}/{epochs}  train={tr_acc:.3f}  val={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(
                {"model_state": model.state_dict(), "arch": "resnet18", "patch_size": PATCH_SIZE},
                out,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val={best_val_acc:.3f})")
                break

    print(f"\nBest val accuracy: {best_val_acc:.3f}  →  {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on MMOTU ultrasound patches.")
    parser.add_argument("data_dir", help="Path to OTU_2d/")
    parser.add_argument("--out", default="us_classifier.pt")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.data_dir, args.out, args.epochs)


if __name__ == "__main__":
    main()
