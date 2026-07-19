"""Frame-level image quality: the gate that lets the 2D pipeline refuse to measure.

A measurement from a 2D transvaginal frame is only meaningful when the ovary is
actually in that frame, the cross-section is not cut off by the edge of the
sector, the in-plane pixel spacing is known, and the follicle/stroma contrast has
survived. When those conditions do not hold the correct behaviour is to abstain.
A per-section follicle count read off a degraded frame is worse than no count,
because it looks like evidence.

**The safety property carried over from the 3D gate.** The 3D gate achieves a 0.0
unsafe-acceptance rate on the phantom suite through two defensive checks that are
reproduced here rather than reinvented:

1. an **ovary-fraction ceiling** — a candidate covering most of the frame is a
   segmentation failure, not an ovary. Without it, a structureless noise frame is
   split roughly in half by any thresholding step and the larger half is
   confidently measured as an ovary; and
2. an **ovary-vs-background contrast floor** — if the candidate and its
   surroundings have effectively the same echogenicity, nothing has been
   detected, whatever the mask says.

Both are calibrated differently in 2D than in 3D, because a *cross-section* of an
ovary occupies a far larger share of a frame than an ovary occupies of an
acquisition volume. Using the 3D ceiling of 40% on 2D frames would reject
correctly segmented ovaries; the 2D ceiling is 60%.

**Volume is structurally infeasible in 2D.** ``ovarian_volume_feasible`` is
always False from this module, regardless of how good the frame is. A cross-section
has two dimensions and ovarian volume needs three; no amount of image quality
changes that. The schema's ``model_validator`` enforces the same rule downstream,
so the two guards are independent.

For cine loops, :func:`assess_cine_quality` aggregates the per-frame assessments
and reports how many frames were actually usable — the denominator behind
``tracking_coverage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from schemas.imaging import ImageQualityAssessment, UltrasoundStudyMetadata

__all__ = [
    "MAX_BORDER_CONTACT_FRACTION_2D",
    "MAX_OVARY_PIXEL_FRACTION",
    "MIN_CONTRAST_FOR_COUNTING_2D",
    "MIN_OVARY_BACKGROUND_CONTRAST_2D",
    "MIN_OVARY_PIXEL_FRACTION",
    "CineQualityAssessment",
    "FrameQualityEvidence",
    "assess_cine_quality",
    "assess_frame_quality",
    "compute_frame_evidence",
]

#: An ovary cross-section below this fraction of the frame is not resolvable.
#: Lower than a 3D volume's floor would suggest is wrong: in-plane the ovary is
#: large, so a genuinely tiny candidate is speckle or a vessel.
MIN_OVARY_PIXEL_FRACTION = 0.005

#: An "ovary" occupying more than this fraction of the frame is not an ovary.
#: The 3D counterpart is 40%; 2D is looser at 60% because a transvaginal frame is
#: zoomed onto the adnexa and a correctly segmented ovary legitimately fills much
#: of it. The ceiling still catches the failure it exists for: a structureless
#: frame thresholded into two halves.
MAX_OVARY_PIXEL_FRACTION = 0.60

#: The ovary must be distinguishable in intensity from what surrounds it. Same
#: role and same value as the 3D floor: below it, nothing has been detected.
MIN_OVARY_BACKGROUND_CONTRAST_2D = 0.06

#: Fraction of the ovary mask allowed to touch the frame border before the
#: cross-section is judged truncated by the sector edge.
MAX_BORDER_CONTACT_FRACTION_2D = 0.02

#: Contrast-to-noise floor below which follicle lumina cannot be separated from
#: speckle, so per-section counting is infeasible even if the ovary is visible.
MIN_CONTRAST_FOR_COUNTING_2D = 0.08


@dataclass
class FrameQualityEvidence:
    """The measured quantities behind one frame's quality decision, for auditing."""

    ovary_pixel_fraction: float
    border_contact_fraction: float
    #: Follicle-lumen vs ovarian-stroma intensity difference. Gates counting.
    contrast: float
    #: Ovary vs surrounding-background intensity difference. Gates detection.
    ovary_background_contrast: float
    sharpness: float
    dynamic_range: float


def _border_contact_fraction(mask: np.ndarray) -> float:
    """Fraction of mask pixels sitting on the frame boundary.

    A high value means the ovary runs off the edge of the sector, so its
    cross-sectional area and per-section follicle count are both truncated.
    """
    if not mask.any():
        return 0.0
    border = np.zeros_like(mask, dtype=bool)
    for axis in range(mask.ndim):
        index: list[Any] = [slice(None)] * mask.ndim
        index[axis] = 0
        border[tuple(index)] = True
        index[axis] = -1
        border[tuple(index)] = True
    return float((mask & border).sum() / mask.sum())


def _sharpness(frame: np.ndarray) -> float:
    """Normalised gradient magnitude: a blur/defocus proxy."""
    array = np.asarray(frame, dtype=float)
    spread = float(array.max() - array.min())
    if spread < 1e-8:
        return 0.0
    normalized = (array - array.min()) / spread
    grads = np.gradient(normalized)
    magnitude = np.sqrt(sum(g**2 for g in grads))
    return float(np.clip(magnitude.mean() * 10.0, 0.0, 1.0))


def compute_frame_evidence(
    frame: np.ndarray,
    ovary_mask: np.ndarray | None,
    follicle_mask: np.ndarray | None,
) -> FrameQualityEvidence:
    """Measure the raw quantities one frame's quality decision is based on."""
    array = np.asarray(frame, dtype=float)
    if ovary_mask is None:
        ovary_mask = np.zeros(array.shape, dtype=bool)
    if follicle_mask is None:
        follicle_mask = np.zeros(array.shape, dtype=bool)

    region = np.asarray(ovary_mask, dtype=bool) | np.asarray(follicle_mask, dtype=bool)
    fraction = float(region.sum() / region.size) if region.size else 0.0

    contrast = 0.0
    if follicle_mask.any() and ovary_mask.any():
        contrast = float(abs(array[ovary_mask].mean() - array[follicle_mask].mean()))

    background_contrast = 0.0
    outside = ~region
    if region.any() and outside.any():
        background_contrast = float(abs(array[region].mean() - array[outside].mean()))

    spread = float(np.percentile(array, 99) - np.percentile(array, 1)) if array.size else 0.0
    return FrameQualityEvidence(
        ovary_pixel_fraction=fraction,
        border_contact_fraction=_border_contact_fraction(region),
        contrast=contrast,
        ovary_background_contrast=background_contrast,
        sharpness=_sharpness(array),
        dynamic_range=spread,
    )


def assess_frame_quality(
    frame: np.ndarray,
    metadata: UltrasoundStudyMetadata,
    *,
    ovary_mask: np.ndarray | None = None,
    follicle_mask: np.ndarray | None = None,
    ovary_confidence: float = 0.0,
) -> ImageQualityAssessment:
    """Produce the quality assessment gating all measurement on one 2D frame.

    Args:
        frame: The (preprocessed) 2D pixel array.
        metadata: Study metadata; supplies spacing and laterality availability.
        ovary_mask: Predicted ovary stroma mask for this frame.
        follicle_mask: Predicted follicle mask for this frame.
        ovary_confidence: Mean predicted ovary probability inside the mask.

    Returns:
        An :class:`ImageQualityAssessment` whose ``reasons`` list explains every
        capability that is switched off. ``ovarian_volume_feasible`` is always
        False: a 2D cross-section cannot support a volume.
    """
    evidence = compute_frame_evidence(frame, ovary_mask, follicle_mask)
    reasons: list[str] = []

    too_small = evidence.ovary_pixel_fraction < MIN_OVARY_PIXEL_FRACTION
    too_large = evidence.ovary_pixel_fraction > MAX_OVARY_PIXEL_FRACTION
    no_contrast = evidence.ovary_background_contrast < MIN_OVARY_BACKGROUND_CONTRAST_2D

    ovary_visible = not (too_small or too_large or no_contrast)
    if too_small:
        reasons.append(
            f"No ovary in this frame (candidate occupies {evidence.ovary_pixel_fraction:.4%} of "
            f"the frame, below the {MIN_OVARY_PIXEL_FRACTION:.2%} floor)."
        )
    if too_large:
        reasons.append(
            f"Candidate ovary occupies {evidence.ovary_pixel_fraction:.1%} of the frame, above "
            f"the {MAX_OVARY_PIXEL_FRACTION:.0%} ceiling; this is a segmentation failure, "
            "not an ovary."
        )
    if no_contrast:
        reasons.append(
            f"Candidate ovary is indistinguishable from its surroundings (intensity difference "
            f"{evidence.ovary_background_contrast:.3f}, below the "
            f"{MIN_OVARY_BACKGROUND_CONTRAST_2D} floor); no ovary has been detected."
        )

    whole_ovary_visible = ovary_visible and (
        evidence.border_contact_fraction <= MAX_BORDER_CONTACT_FRACTION_2D
    )
    if ovary_visible and not whole_ovary_visible:
        reasons.append(
            f"Ovary cross-section touches the frame border "
            f"({evidence.border_contact_fraction:.1%} of its pixels); the per-section count and "
            "cross-sectional area would be truncated."
        )

    spacing_available = metadata.spacing_mm is not None
    if not spacing_available:
        reasons.append(
            "In-plane pixel spacing unknown; no measurement can be expressed in millimetres."
        )

    laterality_available = metadata.laterality in ("left", "right")
    if not laterality_available:
        reasons.append("Laterality unknown; per-ovary attribution and asymmetry unavailable.")

    contrast_ok = evidence.contrast >= MIN_CONTRAST_FOR_COUNTING_2D
    if ovary_visible and not contrast_ok:
        reasons.append(
            f"Follicle/stroma contrast {evidence.contrast:.3f} is below the "
            f"{MIN_CONTRAST_FOR_COUNTING_2D} floor; follicles cannot be distinguished from speckle."
        )

    counting_feasible = ovary_visible and spacing_available and contrast_ok

    reasons.append(
        "Ovarian volume is not feasible from 2D: a cross-section has two dimensions and volume "
        "requires three. Report ovary_area_mm2 instead."
    )

    return ImageQualityAssessment(
        ovary_visible=ovary_visible,
        whole_ovary_visible=whole_ovary_visible,
        laterality_available=laterality_available,
        pixel_spacing_available=spacing_available,
        follicle_counting_feasible=counting_feasible,
        # Structurally impossible in 2D, independent of image quality.
        ovarian_volume_feasible=False,
        overall_quality_score=_frame_score(
            evidence,
            ovary_visible=ovary_visible,
            whole_ovary_visible=whole_ovary_visible,
            spacing_available=spacing_available,
            ovary_confidence=ovary_confidence,
        ),
        reasons=reasons,
    )


def _frame_score(
    evidence: FrameQualityEvidence,
    *,
    ovary_visible: bool,
    whole_ovary_visible: bool,
    spacing_available: bool,
    ovary_confidence: float,
) -> float:
    """Combine one frame's evidence into a single 0-1 score.

    The score is a *screening* number: ``measurement_feasible`` on the schema
    requires it to exceed 0.5 in addition to the hard boolean gates, so a high
    score can never on its own unlock a measurement whose preconditions failed.
    """
    if not ovary_visible:
        # Cap hard: without a visible ovary nothing else can rescue the frame.
        return float(np.clip(0.15 * evidence.sharpness, 0.0, 0.2))
    terms = [
        0.30,
        0.20 if whole_ovary_visible else 0.0,
        0.15 if spacing_available else 0.0,
        0.15 * float(np.clip(evidence.contrast / 0.3, 0.0, 1.0)),
        0.10 * float(np.clip(evidence.sharpness, 0.0, 1.0)),
        0.10 * float(np.clip(ovary_confidence, 0.0, 1.0)),
    ]
    return float(np.clip(sum(terms), 0.0, 1.0))


@dataclass
class CineQualityAssessment:
    """Aggregate quality over a cine loop, plus the usable-frame bookkeeping."""

    per_frame: list[ImageQualityAssessment] = field(default_factory=list)
    #: Indices of frames that passed the gate and may contribute to measurement.
    usable_frame_indices: list[int] = field(default_factory=list)
    aggregate: ImageQualityAssessment = field(default_factory=ImageQualityAssessment)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.per_frame)

    @property
    def n_usable_frames(self) -> int:
        return len(self.usable_frame_indices)

    @property
    def usable_fraction(self) -> float:
        """Fraction of frames that contributed a usable ovary mask.

        This is the quantity reported as ``tracking_coverage`` on the morphology
        output. It is *frame* coverage, not anatomical coverage: it says nothing
        about whether the sweep actually traversed the whole ovary.
        """
        return float(self.n_usable_frames / self.n_frames) if self.n_frames else 0.0


#: Below this usable fraction a cine loop's unique-count estimate is too weak to
#: report as a per-ovary estimate at all.
MIN_USABLE_FRACTION_FOR_ESTIMATE = 0.30


def assess_cine_quality(
    frames: np.ndarray | list[np.ndarray],
    metadata: UltrasoundStudyMetadata,
    *,
    ovary_masks: list[np.ndarray] | None = None,
    follicle_masks: list[np.ndarray] | None = None,
    ovary_confidences: list[float] | None = None,
) -> CineQualityAssessment:
    """Assess every frame of a cine loop and aggregate into one decision.

    Aggregation is deliberately **not** an average of frame scores. A loop where
    the sonographer spends 40 excellent frames on the ovary and 60 approaching it
    is a good loop, and averaging would score it as mediocre. What matters is
    whether *enough* frames were usable, so the aggregate takes the median score
    over the usable frames and records the usable fraction separately.

    Args:
        frames: ``(T, H, W)`` array or a list of 2D frames.
        metadata: Study metadata; supplies spacing and laterality.
        ovary_masks: Per-frame predicted ovary masks.
        follicle_masks: Per-frame predicted follicle masks.
        ovary_confidences: Per-frame mean ovary probability.

    Returns:
        A :class:`CineQualityAssessment`.
    """
    stack = [np.asarray(f, dtype=float) for f in frames]
    n = len(stack)
    ovary_masks = ovary_masks or [None] * n  # type: ignore[list-item]
    follicle_masks = follicle_masks or [None] * n  # type: ignore[list-item]
    ovary_confidences = ovary_confidences or [0.0] * n

    per_frame = [
        assess_frame_quality(
            frame,
            metadata,
            ovary_mask=ovary_masks[i],
            follicle_mask=follicle_masks[i],
            ovary_confidence=ovary_confidences[i],
        )
        for i, frame in enumerate(stack)
    ]
    usable = [i for i, q in enumerate(per_frame) if q.measurement_feasible]

    warnings: list[str] = []
    if n == 0:
        warnings.append("Cine loop contains no frames.")
    usable_fraction = float(len(usable) / n) if n else 0.0
    if n and not usable:
        warnings.append(
            f"No usable frames in this {n}-frame loop; all quantitative measurement is withheld."
        )
    elif usable_fraction < MIN_USABLE_FRACTION_FOR_ESTIMATE:
        warnings.append(
            f"Only {len(usable)}/{n} frames ({usable_fraction:.0%}) were usable, below the "
            f"{MIN_USABLE_FRACTION_FOR_ESTIMATE:.0%} floor; the unique-follicle estimate from "
            "this loop is unreliable and its confidence is reduced accordingly."
        )
    elif usable_fraction < 1.0:
        warnings.append(
            f"{n - len(usable)}/{n} frames were unusable and excluded from tracking; the "
            "unique-follicle count is an estimate over the usable frames only."
        )

    aggregate = _aggregate_quality(per_frame, usable, metadata)
    return CineQualityAssessment(
        per_frame=per_frame,
        usable_frame_indices=usable,
        aggregate=aggregate,
        warnings=warnings,
    )


def _aggregate_quality(
    per_frame: list[ImageQualityAssessment],
    usable: list[int],
    metadata: UltrasoundStudyMetadata,
) -> ImageQualityAssessment:
    """Collapse per-frame assessments into the loop-level gate."""
    if not per_frame:
        return ImageQualityAssessment(
            reasons=["Cine loop contains no frames; nothing can be measured."]
        )
    if not usable:
        merged = sorted({r for q in per_frame for r in q.reasons})
        return ImageQualityAssessment(
            pixel_spacing_available=metadata.spacing_mm is not None,
            laterality_available=metadata.laterality in ("left", "right"),
            overall_quality_score=float(max(q.overall_quality_score for q in per_frame)),
            reasons=[
                "No frame in this loop passed the quality gate.",
                *merged,
            ],
        )

    good = [per_frame[i] for i in usable]
    return ImageQualityAssessment(
        ovary_visible=True,
        # The loop shows the whole ovary if ANY usable frame captured an
        # untruncated cross-section; a sweep legitimately clips the ovary at its
        # extremes, and requiring every frame to be untruncated would reject
        # every real loop.
        whole_ovary_visible=any(q.whole_ovary_visible for q in good),
        laterality_available=metadata.laterality in ("left", "right"),
        pixel_spacing_available=metadata.spacing_mm is not None,
        follicle_counting_feasible=any(q.follicle_counting_feasible for q in good),
        # Still structurally infeasible: many cross-sections without calibrated
        # out-of-plane geometry do not reconstruct a volume.
        ovarian_volume_feasible=False,
        overall_quality_score=float(np.median([q.overall_quality_score for q in good])),
        reasons=[
            f"{len(usable)}/{len(per_frame)} frames usable.",
            "Ovarian volume is not feasible from a freehand 2D sweep: the frames have no "
            "calibrated out-of-plane spacing, so they do not reconstruct a volume.",
        ],
    )
