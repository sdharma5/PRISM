"""ResNet18 patch classifier for ovarian follicle detection.

Builds a dense score heatmap by averaging patch confidences, restricts to
dark (anechoic) pixels, then finds local maxima to localise individual
follicles. Results are drawn as circles rather than boxes so each marker
sits inside one follicle rather than spanning several.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PatchClassifier"]

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts/encoders/ovarian_ultrasound/us_classifier.pt"
)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pixels darker than this (0-255) are treated as anechoic follicle interior.
_DARK_THRESHOLD = 70
# Gaussian smoothing sigma applied to the score heatmap before peak finding.
_SMOOTH_SIGMA = 10
# Minimum separation between two peak centres (pixels).
_PEAK_SEPARATION = 50
# Only keep peaks whose smoothed score exceeds this fraction of the map max.
_PEAK_FRACTION = 0.45
# Dense heatmap stride (smaller = more accurate localisation, slower).
_STRIDE = 8


def _to_base64_png(rgb: np.ndarray) -> str:
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _load_image(data: bytes) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


class PatchClassifier:
    """Sliding-window follicle detector backed by a ResNet18."""

    def __init__(self, model: Any, patch_size: int = 64) -> None:
        self.model = model
        self.patch_size = patch_size

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PatchClassifier":
        import torch  # noqa: PLC0415
        from torchvision.models import resnet18  # noqa: PLC0415

        checkpoint_path = Path(path) if path else _DEFAULT_CHECKPOINT
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Patch classifier checkpoint not found at {checkpoint_path}."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = resnet18(num_classes=2)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return cls(model=model, patch_size=int(checkpoint.get("patch_size", 64)))

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run the classifier on raw image bytes.

        Returns
        -------
        dict with:
            follicle_count        – number of detected follicles
            follicle_confidence   – mean peak score of retained detections
            annotated_image_b64   – base64-encoded PNG with green circles
            image_shape           – [H, W]
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from scipy import ndimage  # noqa: PLC0415

        rgb = _load_image(image_bytes)
        gray = np.mean(rgb, axis=2)
        h, w = rgb.shape[:2]
        ps = self.patch_size

        # --- dense score heatmap -------------------------------------------
        heatmap = np.zeros((h, w), dtype=np.float32)
        hits = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for y in range(0, h - ps + 1, _STRIDE):
                for x in range(0, w - ps + 1, _STRIDE):
                    patch = rgb[y : y + ps, x : x + ps].astype(np.float32) / 255.0
                    patch = (patch - _IMAGENET_MEAN) / _IMAGENET_STD
                    t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)
                    prob = float(F.softmax(self.model(t), dim=1)[0, 1])
                    heatmap[y : y + ps, x : x + ps] += prob
                    hits[y : y + ps, x : x + ps] += 1

        with np.errstate(invalid="ignore"):
            heatmap = np.where(hits > 0, heatmap / hits, 0.0)

        # --- restrict to anechoic pixels INSIDE the scan area --------------
        # Pixels in the black background corners have no tissue in any
        # direction; eroding the "neighbourhood has some tissue" mask removes
        # them while keeping dark follicles surrounded by ovary tissue.
        local_bright = ndimage.maximum_filter(gray, size=60)
        scan_mask = ndimage.binary_erosion(local_bright > 50, iterations=15).astype(np.float32)

        dark_mask = (gray < _DARK_THRESHOLD).astype(np.float32)
        weighted = heatmap * dark_mask * scan_mask
        smoothed = ndimage.gaussian_filter(weighted, sigma=_SMOOTH_SIGMA)

        # --- local maxima --------------------------------------------------
        local_max = ndimage.maximum_filter(smoothed, size=_PEAK_SEPARATION)
        threshold = smoothed.max() * _PEAK_FRACTION if smoothed.max() > 0 else 1.0
        peaks_mask = (smoothed == local_max) & (smoothed > threshold)
        peak_coords = np.argwhere(peaks_mask)  # rows of (y, x)

        # --- annotate image ------------------------------------------------
        from PIL import Image, ImageDraw  # noqa: PLC0415

        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        confs: list[float] = []

        detections: list[dict[str, Any]] = []
        for py, px in peak_coords:
            conf = float(smoothed[py, px])
            confs.append(conf)
            # Estimate circle radius from dark area density nearby
            half = 40
            region = gray[max(0, py - half) : py + half, max(0, px - half) : px + half]
            dark_frac = float((region < _DARK_THRESHOLD).mean())
            r = int(15 + dark_frac * 20)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 230, 100), width=3)
            detections.append({"cx": int(px), "cy": int(py), "radius": r, "confidence": conf})

        return {
            "follicle_count": len(peak_coords),
            "follicle_confidence": float(np.mean(confs)) if confs else 0.0,
            "follicle_detections": detections,
            "annotated_image_b64": _to_base64_png(np.array(pil)),
            "image_shape": [h, w],
        }
