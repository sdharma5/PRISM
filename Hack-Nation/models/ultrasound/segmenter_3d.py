"""3D ovary/follicle segmentation — the OPTIONAL enhanced-mode pathway.

A dedicated 3D volume acquisition is not the routine clinical input for PMOS
ovarian assessment; 2D transvaginal imaging is (see
:mod:`models.ultrasound.segmenter_2d`, the primary path). A volume is worth
having when it exists, because it is the only acquisition that supports a *true*
per-ovary follicle count and a genuine ovarian volume — but the pipeline must
never assume one is available.

USOVA3D is a 3D dataset and is used here, but it is a **label resource**: it is
one of very few public datasets carrying expert ovary *and* individual-follicle
annotations. Its volumes are sliced into labelled 2D frames for pretraining the
2D model (see :mod:`ingestion.ultrasound.slice_extraction`). Treating USOVA3D as
a model of the deployment input is the architectural mistake this module's
demotion corrects.

Two implementations share one interface, ``segment(volume) -> SegmentationOutput``:

* :class:`UNet3D` — the trainable model, imported lazily so torch stays optional.
* :class:`ThresholdSegmenter` — the deterministic intensity + morphology
  heuristic, inheriting the rank-agnostic implementation from the 2D module.

Class indices: 0 = background, 1 = ovary stroma, 2 = follicle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.ultrasound.losses import CLASS_FOLLICLE, CLASS_OVARY, N_CLASSES
from models.ultrasound.segmenter_2d import (
    SegmentationOutput,
    ThresholdSegmenterBase,
    ball,
    labels_from_probs,
    mean_confidence,
    otsu_threshold,
)

__all__ = [
    "SegmentationOutput",
    "ThresholdSegmenter",
    "UNet3D",
    "ball",
    "build_segmenter",
    "labels_from_probs",
    "otsu_threshold",
]


class ThresholdSegmenter(ThresholdSegmenterBase):
    """Torch-free 3D ovary/follicle segmentation.

    Behaviourally identical to the rank-agnostic base implementation; only the
    defaults are those tuned for volumes, where the ovary occupies a far smaller
    fraction of the field of view than it does of a single cross-sectional frame.
    """

    model_version = "threshold-0.1.0"


class UNet3D:
    """Small 3D U-Net over ``(N, 1, D, H, W)`` producing 3-class logits.

    Deliberately small (two down/up levels, 16 base channels): ovarian ultrasound
    datasets that can legitimately be used here are on the order of hundreds of
    studies, and a large network would memorise rather than generalise. The
    architecture is built lazily on first use so importing this module never
    requires torch.
    """

    model_version = "unet3d-0.1.0"

    def __init__(
        self,
        *,
        in_channels: int = 1,
        n_classes: int = N_CLASSES,
        base_channels: int = 16,
        depth: int = 2,
        with_quality_head: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.depth = depth
        self.with_quality_head = with_quality_head
        self._module: Any | None = None

    @staticmethod
    def is_available() -> bool:
        """True when torch is importable."""
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def build(self) -> Any:
        """Instantiate and cache the underlying ``torch.nn.Module``."""
        if self._module is not None:
            return self._module
        try:
            import torch  # noqa: PLC0415
            from torch import nn  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "UNet3D requires the optional 'torch' extra: pip install '.[torch]'. "
                "Use ThresholdSegmenter for a torch-free path."
            ) from exc

        def block(cin: int, cout: int) -> Any:
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.InstanceNorm3d(cout),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1),
                nn.InstanceNorm3d(cout),
                nn.LeakyReLU(0.01, inplace=True),
            )

        n_classes = self.n_classes
        base = self.base_channels
        depth = self.depth
        with_quality = self.with_quality_head

        class _UNet3D(nn.Module):
            """Encoder/decoder with skip connections and an optional quality head."""

            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.encoders = nn.ModuleList()
                self.pools = nn.ModuleList()
                channels = in_channels
                widths: list[int] = []
                for level in range(depth):
                    width = base * (2**level)
                    self.encoders.append(block(channels, width))
                    self.pools.append(nn.MaxPool3d(2))
                    channels, widths = width, [*widths, width]
                self.bottleneck = block(channels, base * (2**depth))
                self.ups = nn.ModuleList()
                self.decoders = nn.ModuleList()
                channels = base * (2**depth)
                for width in reversed(widths):
                    self.ups.append(nn.ConvTranspose3d(channels, width, 2, stride=2))
                    self.decoders.append(block(width * 2, width))
                    channels = width
                self.head = nn.Conv3d(channels, n_classes, 1)
                self.quality_head = (
                    nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(channels, 7))
                    if with_quality
                    else None
                )

            def forward(self, x: Any) -> Any:
                skips: list[Any] = []
                for encoder, pool in zip(self.encoders, self.pools, strict=True):
                    x = encoder(x)
                    skips.append(x)
                    x = pool(x)
                x = self.bottleneck(x)
                for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
                    x = up(x)
                    if x.shape[2:] != skip.shape[2:]:
                        x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="nearest")
                    x = decoder(torch.cat([skip, x], dim=1))
                logits = self.head(x)
                if self.quality_head is None:
                    return logits
                return logits, self.quality_head(x)

        self._module = _UNet3D(self.in_channels)
        return self._module

    def predict_proba(self, volume: np.ndarray) -> np.ndarray:
        """Run a forward pass and return ``(3, ...)`` probabilities."""
        import torch  # noqa: PLC0415

        module = self.build()
        module.eval()
        array = np.asarray(volume, dtype=np.float32)
        tensor = torch.from_numpy(array)[None, None]
        with torch.no_grad():
            out = module(tensor)
        logits = out[0] if isinstance(out, tuple) else out
        return torch.softmax(logits, dim=1)[0].numpy()

    def segment(self, volume: np.ndarray) -> SegmentationOutput:
        """Segment a volume, matching the :class:`ThresholdSegmenter` contract."""
        probs = self.predict_proba(volume)
        labels = labels_from_probs(probs)
        ovary = labels == CLASS_OVARY
        follicle = labels == CLASS_FOLLICLE
        return SegmentationOutput(
            probs=probs,
            labels=labels,
            ovary_mask=ovary,
            follicle_mask=follicle,
            ovary_confidence=mean_confidence(probs, ovary | follicle, CLASS_OVARY),
            follicle_confidence=mean_confidence(probs, follicle, CLASS_FOLLICLE),
        )


def build_segmenter(kind: str = "auto", **kwargs: Any) -> ThresholdSegmenter | UNet3D:
    """Return a 3D segmenter, falling back to the torch-free one when needed.

    Args:
        kind: ``"auto"``, ``"unet3d"`` or ``"threshold"``.
        **kwargs: Forwarded to the chosen constructor.

    Returns:
        A segmenter exposing ``segment`` and ``predict_proba``.
    """
    if kind == "threshold":
        return ThresholdSegmenter(**kwargs)
    if kind == "unet3d":
        return UNet3D(**kwargs)
    if kind == "auto":
        return UNet3D(**kwargs) if UNet3D.is_available() else ThresholdSegmenter()
    raise ValueError(f"Unknown segmenter kind '{kind}'.")
