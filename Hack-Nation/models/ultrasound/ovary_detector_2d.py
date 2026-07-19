"""Localize the ovary within a 2D transvaginal frame.

Detection is a separate step from segmentation for two clinical reasons.

First, **most frames of a real cine loop do not contain the ovary at all.** The
sonographer approaches, sweeps through, and leaves. A segmenter run on an
ovary-free frame will still return its largest bright compact blob — bowel,
myometrium, a vessel wall — and that blob then enters the tracker as a phantom
follicle carrier. A cheap up-front detector that can answer "there is no ovary
here" is what keeps those frames out of the count.

Second, detection gives the tracker a **region-of-interest prior**. Follicle
matching across frames is only meaningful within the same anatomical structure;
knowing the ovary bounding box on each frame lets the tracker reject matches that
would link a follicle in one ovary to a structure outside it.

The detector is deliberately torch-free and deterministic. It is a localizer, not
a classifier: it never claims that what it found *is* an ovary, only that it is
the best ovary-shaped candidate and how confident that shape evidence is. The
quality gate in :mod:`models.ultrasound.qc_2d` makes the accept/reject decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from models.ultrasound.segmenter_2d import ball, otsu_threshold

__all__ = ["OvaryDetection", "detect_ovary_2d", "detect_ovary_in_frames"]

#: A candidate below this fraction of the frame is too small to be a resolvable
#: ovary at transvaginal field settings.
MIN_CANDIDATE_FRAME_FRACTION = 0.005

#: A candidate above this fraction of the frame is not an ovary. See
#: :data:`models.ultrasound.qc_2d.MAX_OVARY_PIXEL_FRACTION` for the reasoning;
#: the detector uses the same ceiling so it fails before the segmenter runs.
MAX_CANDIDATE_FRAME_FRACTION = 0.60

#: Minimum solidity (area / convex-hull area proxy) for an ovary-like blob. An
#: ovary is a convex ellipsoid in cross-section; a ragged, branching region is
#: bowel or a shadow artefact, not an ovary.
MIN_SOLIDITY = 0.55


@dataclass
class OvaryDetection:
    """Where the ovary is on one frame, and how much to believe it."""

    found: bool
    #: ``(row_min, row_max, col_min, col_max)`` inclusive-exclusive, or None.
    bbox: tuple[int, int, int, int] | None = None
    centroid_px: tuple[float, float] | None = None
    mask: np.ndarray | None = None
    area_px: int = 0
    frame_fraction: float = 0.0
    #: Shape-and-contrast evidence in [0, 1]. NOT a probability that this is an
    #: ovary — no anatomical classifier is trained here.
    confidence: float = 0.0
    solidity: float = 0.0
    #: Ovary-vs-surroundings intensity difference; the detection signal itself.
    contrast: float = 0.0
    reasons: list[str] | None = None

    def region_of_interest(self, frame: np.ndarray, *, margin_px: int = 8) -> np.ndarray:
        """Crop ``frame`` to the detected ovary plus a margin.

        The margin exists because the detector's boundary is a threshold result
        and systematically under-covers the low-contrast ovarian rim; cropping
        tight to the detection would clip the very boundary the segmenter needs.
        """
        array = np.asarray(frame)
        if self.bbox is None:
            return array
        r0, r1, c0, c1 = self.bbox
        return array[
            max(r0 - margin_px, 0) : min(r1 + margin_px, array.shape[0]),
            max(c0 - margin_px, 0) : min(c1 + margin_px, array.shape[1]),
        ]


def _solidity(mask: np.ndarray) -> float:
    """Area divided by the area of its filled bounding ellipse-ish hull.

    A true convex-hull area needs scipy.spatial; the filled bounding box is used
    as a cheap, dependency-free proxy. It is scale-free and adequate for
    rejecting the branching, high-perimeter regions that bowel produces.
    """
    if not mask.any():
        return 0.0
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    box_area = float((r1 - r0) * (c1 - c0))
    if box_area <= 0:
        return 0.0
    # An ellipse inscribed in the bounding box covers pi/4 of it, so a perfect
    # ellipse scores 1.0 under this normalisation.
    return float(np.clip(mask.sum() / (box_area * np.pi / 4.0), 0.0, 1.0))


def detect_ovary_2d(
    frame: np.ndarray,
    *,
    smoothing_sigma: float = 1.5,
    closing_radius: int = 3,
    min_frame_fraction: float = MIN_CANDIDATE_FRAME_FRACTION,
    max_frame_fraction: float = MAX_CANDIDATE_FRAME_FRACTION,
    min_solidity: float = MIN_SOLIDITY,
) -> OvaryDetection:
    """Find the single best ovary candidate in one 2D frame.

    The ovary is sought as the largest bright, compact, convex region after
    speckle suppression and closing. Closing runs before component selection so
    the anechoic follicle lumina are filled back into the ovary — otherwise a
    follicle-rich ovary fragments into a ring of stroma slivers and the "largest
    component" is a fragment rather than the organ.

    Args:
        frame: 2D B-mode frame, any intensity scale.
        smoothing_sigma: Gaussian sigma in pixels for speckle suppression.
        closing_radius: Morphological closing radius, in pixels.
        min_frame_fraction: Below this the candidate is too small to be an ovary.
        max_frame_fraction: Above this the candidate is a segmentation failure.
        min_solidity: Below this the candidate is too ragged to be an ovary.

    Returns:
        An :class:`OvaryDetection`. ``found`` is False with populated ``reasons``
        whenever no candidate survives; the frame is then excluded from counting.
    """
    array = np.asarray(frame, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"detect_ovary_2d expects a 2D frame, got shape {array.shape}.")

    reasons: list[str] = []
    if array.size == 0 or float(array.max() - array.min()) < 1e-8:
        return OvaryDetection(found=False, reasons=["Frame is blank: no structure to detect."])

    normalized = (array - array.min()) / (array.max() - array.min())
    smoothed = ndi.gaussian_filter(normalized, smoothing_sigma)

    bright = smoothed > otsu_threshold(smoothed)
    structure = ball(closing_radius, 2)
    closed = ndi.binary_erosion(
        ndi.binary_dilation(bright, structure=structure, border_value=0),
        structure=structure,
        border_value=1,
    )
    closed = ndi.binary_fill_holes(closed)

    labelled, n = ndi.label(closed)
    if n == 0:
        return OvaryDetection(found=False, reasons=["No bright region found in frame."])

    sizes = np.bincount(labelled.ravel())[1:]
    best = int(np.argmax(sizes)) + 1
    mask = labelled == best

    area = int(mask.sum())
    fraction = float(area / mask.size)
    solidity = _solidity(mask)
    outside = ~mask
    contrast = (
        float(abs(normalized[mask].mean() - normalized[outside].mean()))
        if outside.any() and mask.any()
        else 0.0
    )

    if fraction < min_frame_fraction:
        reasons.append(
            f"Best candidate occupies {fraction:.3%} of the frame, below the "
            f"{min_frame_fraction:.1%} floor; no ovary in this frame."
        )
    if fraction > max_frame_fraction:
        reasons.append(
            f"Best candidate occupies {fraction:.1%} of the frame, above the "
            f"{max_frame_fraction:.0%} ceiling; this is a detection failure, not an ovary."
        )
    if solidity < min_solidity:
        reasons.append(
            f"Best candidate solidity {solidity:.2f} is below the {min_solidity} floor; "
            "the region is too ragged to be an ovarian cross-section."
        )

    coords = np.argwhere(mask)
    r0, c0 = (int(v) for v in coords.min(axis=0))
    r1, c1 = (int(v) + 1 for v in coords.max(axis=0))
    centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    found = not reasons
    # Confidence blends the two independent pieces of evidence — how ovary-shaped
    # the region is, and how distinguishable it is from its surroundings. A
    # perfectly elliptical region with no contrast is not a detection.
    confidence = (
        float(np.clip(0.5 * solidity + 0.5 * np.clip(contrast / 0.25, 0.0, 1.0), 0.0, 1.0))
        if found
        else 0.0
    )

    return OvaryDetection(
        found=found,
        bbox=(r0, r1, c0, c1),
        centroid_px=centroid,
        mask=mask,
        area_px=area,
        frame_fraction=fraction,
        confidence=confidence,
        solidity=solidity,
        contrast=contrast,
        reasons=reasons,
    )


def detect_ovary_in_frames(
    frames: np.ndarray | list[np.ndarray], **kwargs: object
) -> list[OvaryDetection]:
    """Run :func:`detect_ovary_2d` over a frame stack, in frame order."""
    return [detect_ovary_2d(np.asarray(f, dtype=float), **kwargs) for f in frames]  # type: ignore[arg-type]
