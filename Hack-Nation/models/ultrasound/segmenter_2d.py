"""2D ovary/follicle segmentation — the PRIMARY segmentation pathway.

Routine PMOS ultrasound assessment is 2D transvaginal imaging. The sonographer
sweeps the probe through the ovary and reads follicles off individual B-mode
frames; a dedicated 3D volume acquisition is an optional extra that most clinics
do not perform. This module is therefore the default segmenter, and
:mod:`models.ultrasound.segmenter_3d` is the optional enhanced-mode counterpart.

Two implementations share one interface, ``segment(frame) -> SegmentationOutput``:

* :class:`UNet2D` — the trainable model, imported lazily so torch stays optional.
* :class:`ThresholdSegmenter2D` — a deterministic intensity + morphology
  heuristic built on ``scipy.ndimage``.

The fallback is not a toy. B-mode ultrasound shows follicles as anechoic (dark)
fluid inside echogenic (bright) ovarian stroma, and that contrast is the same
physical signal a learned model exploits. Having a torch-free path means the
counting, tracking, measurement and quality-gating logic — which is where the
clinically consequential bugs live — is exercised by CI on every commit.

Class indices: 0 = background, 1 = ovary stroma, 2 = follicle.

This module also owns the dimension-agnostic primitives (:class:`SegmentationOutput`,
:func:`otsu_threshold`, :func:`ball`, :class:`ThresholdSegmenterBase`) that the 3D
module imports. The import direction runs 3D -> 2D deliberately: 2D is the primary
pathway, so it is the one that must not depend on the optional one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from models.ultrasound.losses import CLASS_BACKGROUND, CLASS_FOLLICLE, CLASS_OVARY, N_CLASSES

__all__ = [
    "SegmentationOutput",
    "ThresholdSegmenter2D",
    "ThresholdSegmenterBase",
    "UNet2D",
    "ball",
    "build_segmenter_2d",
    "labels_from_probs",
    "otsu_threshold",
    "segment_frames",
]


@dataclass
class SegmentationOutput:
    """Probabilities plus the derived hard masks.

    Shape-agnostic: ``probs`` is ``(3, H, W)`` for a 2D frame and ``(3, D, H, W)``
    for a volume.
    """

    probs: np.ndarray
    labels: np.ndarray
    ovary_mask: np.ndarray
    follicle_mask: np.ndarray
    ovary_confidence: float
    follicle_confidence: float

    @property
    def ovary_region_mask(self) -> np.ndarray:
        """Ovary stroma plus follicles: the full ovarian region."""
        return self.ovary_mask | self.follicle_mask


def labels_from_probs(probs: np.ndarray) -> np.ndarray:
    """Argmax over the class axis."""
    return np.asarray(probs).argmax(axis=0).astype(np.int16)


def mean_confidence(probs: np.ndarray, mask: np.ndarray, class_index: int) -> float:
    """Mean predicted probability of ``class_index`` inside ``mask``."""
    if not mask.any():
        return 0.0
    return float(np.clip(probs[class_index][mask].mean(), 0.0, 1.0))


def ball(radius: int, ndim: int) -> np.ndarray:
    """Binary structuring element approximating a disc/sphere of ``radius``."""
    radius = max(int(radius), 1)
    axes = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(ndim))]
    squared = np.zeros((1,) * ndim, dtype=float)
    for axis in axes:
        squared = squared + np.asarray(axis, dtype=float) ** 2
    return np.asarray(squared <= radius**2 + 1e-9)


def otsu_threshold(array: np.ndarray, bins: int = 128) -> float:
    """Otsu's threshold, implemented locally so scikit-image stays optional."""
    flat = np.asarray(array, dtype=float).ravel()
    if flat.size == 0 or float(flat.max() - flat.min()) < 1e-8:
        return float(flat.max() if flat.size else 0.0)
    hist, edges = np.histogram(flat, bins=bins)
    hist = hist.astype(float)
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = weight_bg[-1] - weight_bg
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bg = np.cumsum(hist * centres) / np.clip(weight_bg, 1e-9, None)
        total = float((hist * centres).sum())
        mean_fg = (total - np.cumsum(hist * centres)) / np.clip(weight_fg, 1e-9, None)
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance = np.nan_to_num(variance)
    return float(centres[int(np.argmax(variance))])


class ThresholdSegmenterBase:
    """Torch-free ovary/follicle segmentation from intensity and morphology.

    Pipeline, and why each step exists:

    1. Smooth with a small Gaussian — ultrasound speckle is high-frequency
       multiplicative noise; without smoothing every speckle grain becomes a
       spurious connected component.
    2. Find the ovary as the largest bright, compact region (Otsu split of the
       smoothed intensity, then binary closing and largest-component selection).
       Closing fills the anechoic follicle lumina, which is required because a
       follicle is *inside* the ovary even though it is dark.
    3. Find follicles as dark pixels *within* the closed ovary region. Restricting
       to the interior enforces the same anatomical constraint the ``L_outside``
       penalty imposes on the learned model.

    The implementation is deliberately rank-agnostic — every ``scipy.ndimage``
    call and the structuring element adapt to ``array.ndim`` — so the identical
    logic serves a 2D frame and a 3D volume. Only the tuned defaults differ, in
    the two concrete subclasses.
    """

    model_version = "threshold-base-0.1.0"

    def __init__(
        self,
        *,
        smoothing_sigma: float = 1.0,
        closing_radius: int = 3,
        follicle_percentile: float = 30.0,
        min_ovary_fraction: float = 0.002,
        soft_temperature: float = 0.08,
    ) -> None:
        """
        Args:
            smoothing_sigma: Gaussian sigma in pixels for speckle suppression.
            closing_radius: Morphological closing radius used to fill follicles
                back into the ovary region.
            follicle_percentile: Intensity percentile within the ovary below
                which a pixel is considered anechoic.
            min_ovary_fraction: Minimum ovary size as a fraction of the image;
                below this the segmenter reports no ovary rather than guessing.
            soft_temperature: Temperature converting distances to pseudo-probabilities.
        """
        self.smoothing_sigma = smoothing_sigma
        self.closing_radius = closing_radius
        self.follicle_percentile = follicle_percentile
        self.min_ovary_fraction = min_ovary_fraction
        self.soft_temperature = soft_temperature

    # -- masks -------------------------------------------------------------

    def _ovary_region(self, smoothed: np.ndarray) -> np.ndarray:
        """Largest bright compact component, with follicle lumina filled in."""
        threshold = otsu_threshold(smoothed)
        bright = smoothed > threshold
        if bright.sum() < self.min_ovary_fraction * bright.size:
            return np.zeros_like(bright)

        structure = ball(self.closing_radius, bright.ndim)
        # Closing is done as explicit dilation-then-erosion with border_value=1 on
        # the erosion. scipy's default erodes inward from the array boundary,
        # which would pull a truncated ovary away from the edge of the field of
        # view and hide the very truncation the quality gate must detect.
        closed = ndi.binary_erosion(
            ndi.binary_dilation(bright, structure=structure, border_value=0),
            structure=structure,
            border_value=1,
        )
        closed = ndi.binary_fill_holes(closed)

        labelled, n = ndi.label(closed)
        if n == 0:
            return np.zeros_like(bright)
        sizes = ndi.sum(np.ones_like(labelled), labelled, index=range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        region = labelled == largest
        if region.sum() < self.min_ovary_fraction * region.size:
            return np.zeros_like(bright)
        return region

    def _follicles(self, smoothed: np.ndarray, region: np.ndarray) -> np.ndarray:
        """Anechoic pixels strictly inside the ovary region."""
        if not region.any():
            return np.zeros_like(region)
        interior = ndi.binary_erosion(region, structure=ball(1, region.ndim))
        if not interior.any():
            interior = region
        # Otsu *within the ovary* splits anechoic lumen from echogenic stroma at
        # the actual bimodal minimum. A fixed percentile would instead force a
        # preset follicle fraction and systematically over- or under-size every
        # follicle depending on how many the ovary happens to contain.
        values = smoothed[interior]
        cutoff = otsu_threshold(values)
        # Guard the degenerate unimodal case (no follicles): Otsu still returns a
        # split, so require the dark mode to be genuinely darker than the stroma.
        if cutoff >= float(np.median(values)):
            cutoff = float(np.percentile(values, self.follicle_percentile))
        dark = (smoothed <= cutoff) & interior
        # Opening removes single-pixel speckle holes that are not follicles.
        return ndi.binary_opening(dark, structure=ball(1, region.ndim))

    # -- public API --------------------------------------------------------

    def segment(self, image: np.ndarray) -> SegmentationOutput:
        """Segment one frame or volume into background / ovary / follicle."""
        array = np.asarray(image, dtype=float)
        if array.size and float(array.max() - array.min()) > 1e-8:
            array = (array - array.min()) / (array.max() - array.min())
        smoothed = ndi.gaussian_filter(array, self.smoothing_sigma)

        region = self._ovary_region(smoothed)
        follicle = self._follicles(smoothed, region)
        ovary = region & ~follicle

        probs = self._soft_probs(smoothed, ovary, follicle)
        return SegmentationOutput(
            probs=probs,
            labels=labels_from_probs(probs),
            ovary_mask=ovary,
            follicle_mask=follicle,
            ovary_confidence=mean_confidence(probs, region, CLASS_OVARY) if region.any() else 0.0,
            follicle_confidence=mean_confidence(probs, follicle, CLASS_FOLLICLE),
        )

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """Return ``(3, ...)`` class probabilities."""
        return self.segment(image).probs

    def _soft_probs(
        self, smoothed: np.ndarray, ovary: np.ndarray, follicle: np.ndarray
    ) -> np.ndarray:
        """Turn hard masks into calibrated-ish soft probabilities.

        Pixels near a boundary get intermediate probabilities via a distance
        transform, so downstream confidence figures reflect boundary uncertainty
        instead of claiming certainty everywhere.
        """
        probs = np.zeros((N_CLASSES, *smoothed.shape), dtype=float)
        background = ~(ovary | follicle)
        for class_index, mask in (
            (CLASS_BACKGROUND, background),
            (CLASS_OVARY, ovary),
            (CLASS_FOLLICLE, follicle),
        ):
            if not mask.any():
                probs[class_index] = 1e-3
                continue
            inside = ndi.distance_transform_edt(mask)
            outside = ndi.distance_transform_edt(~mask)
            signed = inside - outside
            temperature = max(self.soft_temperature * 10, 1e-3)
            probs[class_index] = 1.0 / (1.0 + np.exp(-signed / temperature))
        return probs / np.clip(probs.sum(axis=0, keepdims=True), 1e-9, None)


class ThresholdSegmenter2D(ThresholdSegmenterBase):
    """Torch-free 2D frame segmentation.

    Defaults differ from the 3D case for one geometric reason: a 2D transvaginal
    frame is a *cross-section*, so the ovary occupies a much larger fraction of
    the image than an ovary occupies of an acquisition volume. ``min_ovary_fraction``
    is correspondingly larger (0.5% of the frame rather than 0.2% of the volume),
    and the closing radius is smaller because in-plane pixels are typically finer
    than the through-plane slice step that the 3D radius was tuned against.
    """

    model_version = "threshold2d-0.1.0"

    def __init__(
        self,
        *,
        smoothing_sigma: float = 1.0,
        closing_radius: int = 2,
        follicle_percentile: float = 30.0,
        min_ovary_fraction: float = 0.005,
        soft_temperature: float = 0.08,
    ) -> None:
        super().__init__(
            smoothing_sigma=smoothing_sigma,
            closing_radius=closing_radius,
            follicle_percentile=follicle_percentile,
            min_ovary_fraction=min_ovary_fraction,
            soft_temperature=soft_temperature,
        )

    def segment(self, image: np.ndarray) -> SegmentationOutput:
        """Segment one 2D frame, rejecting anything that is not 2D."""
        array = np.asarray(image, dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"ThresholdSegmenter2D expects a 2D frame, got shape {array.shape}. "
                "Use segment_frames() for a stack, or segmenter_3d for a volume."
            )
        return super().segment(array)


class UNet2D:
    """Small 2D U-Net over ``(N, 1, H, W)`` producing 3-class logits.

    Deliberately small (two down/up levels, 16 base channels). The datasets that
    may legitimately be used here are on the order of hundreds of studies, and a
    large network would memorise rather than generalise. The architecture is built
    lazily on first use so importing this module never requires torch.

    A 2D network is the right shape for this problem even though USOVA3D ships
    volumes: the clinical input is a frame, so the model must be able to read a
    frame. USOVA3D is used by extracting labelled 2D slices from its volumes (see
    :mod:`ingestion.ultrasound.slice_extraction`), which is a pretraining resource
    rather than a model of the deployment input.
    """

    model_version = "unet2d-0.1.0"

    def __init__(
        self,
        *,
        in_channels: int = 1,
        n_classes: int = N_CLASSES,
        base_channels: int = 16,
        depth: int = 3,
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
                "UNet2D requires the optional 'torch' extra: pip install '.[torch]'. "
                "Use ThresholdSegmenter2D for a torch-free path."
            ) from exc

        def block(cin: int, cout: int) -> Any:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.InstanceNorm2d(cout),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.InstanceNorm2d(cout),
                nn.LeakyReLU(0.01, inplace=True),
            )

        n_classes = self.n_classes
        base = self.base_channels
        depth = self.depth
        with_quality = self.with_quality_head

        class _UNet2D(nn.Module):
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
                    self.pools.append(nn.MaxPool2d(2))
                    channels, widths = width, [*widths, width]
                self.bottleneck = block(channels, base * (2**depth))
                self.ups = nn.ModuleList()
                self.decoders = nn.ModuleList()
                channels = base * (2**depth)
                for width in reversed(widths):
                    self.ups.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
                    self.decoders.append(block(width * 2, width))
                    channels = width
                self.head = nn.Conv2d(channels, n_classes, 1)
                self.quality_head = (
                    nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 7))
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

        self._module = _UNet2D(self.in_channels)
        return self._module

    def predict_proba(self, frame: np.ndarray) -> np.ndarray:
        """Run a forward pass on one 2D frame and return ``(3, H, W)``."""
        import torch  # noqa: PLC0415

        module = self.build()
        module.eval()
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"UNet2D expects a 2D frame, got shape {array.shape}.")
        tensor = torch.from_numpy(array)[None, None]
        with torch.no_grad():
            out = module(tensor)
        logits = out[0] if isinstance(out, tuple) else out
        return torch.softmax(logits, dim=1)[0].numpy()

    def segment(self, frame: np.ndarray) -> SegmentationOutput:
        """Segment a frame, matching the :class:`ThresholdSegmenter2D` contract."""
        probs = self.predict_proba(frame)
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


def build_segmenter_2d(kind: str = "auto", **kwargs: Any) -> ThresholdSegmenter2D | UNet2D:
    """Return a 2D segmenter, falling back to the torch-free one when needed.

    Args:
        kind: ``"auto"``, ``"unet2d"`` or ``"threshold"``.
        **kwargs: Forwarded to the chosen constructor.

    Returns:
        A segmenter exposing ``segment`` and ``predict_proba``.
    """
    if kind == "threshold":
        return ThresholdSegmenter2D(**kwargs)
    if kind in ("unet2d", "unet"):
        return UNet2D(**kwargs)
    if kind == "auto":
        return UNet2D(**kwargs) if UNet2D.is_available() else ThresholdSegmenter2D()
    raise ValueError(f"Unknown 2D segmenter kind '{kind}'.")


def segment_frames(
    frames: np.ndarray | list[np.ndarray],
    segmenter: Any | None = None,
    *,
    kind: str = "threshold",
) -> list[SegmentationOutput]:
    """Segment every frame of a stack independently.

    Frames are segmented independently *by design*. Temporal smoothing of the
    masks would couple neighbouring frames and make the tracking step in
    :mod:`models.ultrasound.cine_tracking` partly circular: a follicle would then
    appear on frame ``k+1`` partly because it appeared on frame ``k``, which is
    exactly the evidence the tracker is supposed to weigh independently.

    Args:
        frames: ``(T, H, W)`` array or a list of 2D frames.
        segmenter: Prebuilt segmenter; built from ``kind`` when omitted.
        kind: Segmenter kind used when ``segmenter`` is None.

    Returns:
        One :class:`SegmentationOutput` per frame, in frame order.
    """
    model = segmenter if segmenter is not None else build_segmenter_2d(kind)
    stack = [np.asarray(f, dtype=float) for f in frames]
    return [model.segment(frame) for frame in stack]
