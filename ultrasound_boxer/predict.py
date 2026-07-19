"""
Draw bounding boxes around suspicious regions in a uterine/pelvic ultrasound PNG.

Loads a fine-tuned ResNet18 (.pt from train.py) for accurate detection.
Falls back to Otsu intensity thresholding if no model is provided.

Usage:
  python predict.py scan.png --model us_classifier.pt
  python predict.py scan.png --model us_classifier.pt --out result.png
  python predict.py scan.png --conf 0.65 --min-px 200
"""

from __future__ import annotations

import argparse

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

STRIDE = 16
CONF_THRESHOLD = 0.65
MIN_REGION_PX = 200

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str) -> tuple[nn.Module, int, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    patch_size = ckpt.get("patch_size", 64)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, patch_size, device


def patch_to_tensor(patch: np.ndarray, patch_size: int) -> torch.Tensor:
    """uint8 (H,W) → normalised (3, patch_size, patch_size) float tensor."""
    pil = Image.fromarray(patch).resize((patch_size, patch_size))
    t = T.ToTensor()(pil)          # (1, H, W)
    t = t.repeat(3, 1, 1)          # (3, H, W)
    return NORMALIZE(t)


# ── CNN sliding-window heatmap ────────────────────────────────────────────────

def classifier_heatmap(
    img: np.ndarray, model: nn.Module, patch_size: int, device: torch.device, conf: float
) -> np.ndarray:
    h, w = img.shape
    half = patch_size // 2
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    # collect all patch positions
    rows = range(half, h - half, STRIDE)
    cols = range(half, w - half, STRIDE)
    positions = [(r, c) for r in rows for c in cols]

    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)

    # batch through GPU
    INFER_BATCH = 256
    probs_all: list[float] = []
    for start in range(0, len(positions), INFER_BATCH):
        batch_pos = positions[start:start + INFER_BATCH]
        tensors = torch.stack([
            patch_to_tensor(img_u8[r - half:r + half, c - half:c + half], patch_size)
            for r, c in batch_pos
        ]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(tensors), dim=1)[:, 1].cpu().numpy()
        probs_all.extend(p.tolist())

    for (r, c), p in zip(positions, probs_all):
        heatmap[r - half:r + half, c - half:c + half] += p
        counts[r - half:r + half, c - half:c + half] += 1

    averaged = heatmap / np.clip(counts, 1, None)
    mask = averaged > conf
    return ndi.binary_opening(mask, structure=np.ones((5, 5)))


# ── threshold fallback (no model) ────────────────────────────────────────────

def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=128)
    c = (edges[:-1] + edges[1:]) / 2
    wb = np.cumsum(hist).astype(float)
    wf = wb[-1] - wb
    mb = np.cumsum(hist * c) / np.clip(wb, 1e-9, None)
    mf = (hist * c).sum() - np.cumsum(hist * c)
    mf /= np.clip(wf, 1e-9, None)
    return float(c[np.argmax(wb * wf * (mb - mf) ** 2)])


def threshold_fallback(img: np.ndarray) -> np.ndarray:
    smoothed = ndi.gaussian_filter(img, sigma=1.5)
    bright = smoothed > _otsu(smoothed.ravel())
    closed = ndi.binary_closing(bright, structure=np.ones((5, 5)))
    closed = ndi.binary_fill_holes(closed)
    labelled, n = ndi.label(closed)
    if n == 0:
        return np.zeros_like(img, dtype=bool)
    sizes = [(labelled == i).sum() for i in range(1, n + 1)]
    tissue = labelled == (np.argmax(sizes) + 1)
    interior = ndi.binary_erosion(tissue, structure=np.ones((3, 3)))
    if not interior.any():
        interior = tissue
    suspicious = (smoothed <= _otsu(smoothed[interior])) & interior
    return ndi.binary_opening(suspicious, structure=np.ones((3, 3)))


# ── box extraction ───────────────────────────────────────────────────────────

def boxes_from_mask(mask: np.ndarray, min_px: int = MIN_REGION_PX):
    labelled, n = ndi.label(mask)
    boxes = []
    for i in range(1, n + 1):
        region = labelled == i
        if region.sum() < min_px:
            continue
        coords = np.argwhere(region)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


# ── main ─────────────────────────────────────────────────────────────────────

def predict(image_path: str, model_path: str | None, output_path: str | None,
            conf: float, min_px: int) -> None:
    img_pil = Image.open(image_path).convert("L")
    img = np.array(img_pil, dtype=np.float32) / 255.0

    if model_path and Path(model_path).exists():
        model, patch_size, device = load_model(model_path)
        print(f"Using model: {model_path}  (device={device})")
        smoothed = ndi.gaussian_filter(img, sigma=1.0)
        mask = classifier_heatmap(smoothed, model, patch_size, device, conf)
    else:
        if model_path:
            print(f"Model not found at {model_path} — using threshold fallback.")
        else:
            print("No model specified — using threshold fallback.")
        mask = threshold_fallback(img)

    boxes = boxes_from_mask(mask, min_px=min_px)
    print(f"Found {len(boxes)} suspicious region(s).")

    rgb = img_pil.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for r0, c0, r1, c1 in boxes:
        draw.rectangle([c0, r0, c1, r1], outline="red", width=3)

    out = output_path or str(Path(image_path).stem) + "_annotated.png"
    rgb.save(out)
    print(f"Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Box suspicious regions in ultrasound PNG.")
    parser.add_argument("image")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help=f"Confidence threshold (default {CONF_THRESHOLD})")
    parser.add_argument("--min-px", type=int, default=MIN_REGION_PX,
                        help=f"Min region size in pixels (default {MIN_REGION_PX})")
    args = parser.parse_args()
    predict(args.image, args.model, args.out, args.conf, args.min_px)


if __name__ == "__main__":
    main()
