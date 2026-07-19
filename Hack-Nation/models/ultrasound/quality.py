"""Image-quality head: the gate that lets the pipeline refuse to measure.

An ovarian ultrasound measurement is only meaningful when the ovary is actually
in the field of view, the *whole* ovary is captured (a partially imaged ovary
always under-reports volume and follicle count), the physical spacing is known,
and the image is not so degraded that the follicle/stroma contrast is lost.

When those conditions do not hold the correct behaviour is to abstain. An
under-counted antral follicle count on a poor scan is worse than no count,
because it looks like evidence. :func:`assess_quality` therefore returns an
:class:`ImageQualityAssessment` whose ``measurement_feasible`` property is False
with explicit reasons, and the measurement code refuses to emit numbers.

This module holds the **volumetric** gate and the shared routing entry point.
Frame-level and cine-level gating live in :mod:`models.ultrasound.qc_2d`, which
reimplements the same two defensive checks — an ovary-fraction ceiling and an
ovary-vs-background contrast floor — with thresholds calibrated for
cross-sections. Use :func:`assess_quality_for_mode` to dispatch on acquisition
mode rather than picking a gate by hand; picking by hand is how a 2D frame ends up
assessed against volumetric thresholds it can never satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from schemas.imaging import ImageQualityAssessment, UltrasoundStudyMetadata

#: Fields of the quality head, in the fixed order used by the torch head.
QUALITY_FIELDS: tuple[str, ...] = (
    "ovary_visible",
    "whole_ovary_visible",
    "laterality_available",
    "pixel_spacing_available",
    "follicle_counting_feasible",
    "ovarian_volume_feasible",
    "overall_quality_score",
)

#: An ovary occupying less than this fraction of the volume is not resolvable.
MIN_OVARY_VOXEL_FRACTION = 0.002

#: An "ovary" occupying more than this fraction of the field of view is not an
#: ovary. A normal ovary is a few millilitres inside a much larger acquisition
#: volume, so a mask this large means the segmentation latched onto noise or
#: onto surrounding tissue. Without this ceiling a structureless noise volume
#: gets split roughly in half by any thresholding step and the larger half is
#: mistaken for an ovary — a confidently wrong measurement on an empty image.
MAX_OVARY_VOXEL_FRACTION = 0.40

#: The ovary must be distinguishable in intensity from what surrounds it. If the
#: putative ovary and the background have effectively the same echogenicity,
#: nothing has been detected, whatever the mask says.
MIN_OVARY_BACKGROUND_CONTRAST = 0.06

#: Fraction of the ovary mask allowed to touch the volume border before the
#: ovary is judged to be cut off by the field of view.
MAX_BORDER_CONTACT_FRACTION = 0.02

#: Contrast-to-noise floor below which follicle lumina cannot be separated from
#: speckle, so counting is infeasible even if the ovary is visible.
MIN_CONTRAST_FOR_COUNTING = 0.08


@dataclass
class QualityEvidence:
    """The measured quantities behind a quality decision, kept for auditing."""

    ovary_voxel_fraction: float
    border_contact_fraction: float
    #: Follicle-lumen vs ovarian-stroma intensity difference. Gates counting.
    contrast: float
    #: Ovary vs surrounding-background intensity difference. Gates detection.
    ovary_background_contrast: float
    sharpness: float
    dynamic_range: float


def _border_contact_fraction(mask: np.ndarray) -> float:
    """Fraction of mask voxels sitting on the volume boundary.

    A high value means the ovary runs off the edge of the acquisition, so its
    volume and follicle count are both truncated.
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


def _sharpness(volume: np.ndarray) -> float:
    """Normalised gradient magnitude: a blur/defocus proxy."""
    array = np.asarray(volume, dtype=float)
    spread = float(array.max() - array.min())
    if spread < 1e-8:
        return 0.0
    normalized = (array - array.min()) / spread
    grads = np.gradient(normalized)
    magnitude = np.sqrt(sum(g**2 for g in grads))
    return float(np.clip(magnitude.mean() * 10.0, 0.0, 1.0))


def compute_quality_evidence(
    volume: np.ndarray,
    ovary_mask: np.ndarray | None,
    follicle_mask: np.ndarray | None,
) -> QualityEvidence:
    """Measure the raw quantities the quality decision is based on."""
    array = np.asarray(volume, dtype=float)
    if ovary_mask is None:
        ovary_mask = np.zeros(array.shape, dtype=bool)
    if follicle_mask is None:
        follicle_mask = np.zeros(array.shape, dtype=bool)

    region = ovary_mask | follicle_mask
    fraction = float(region.sum() / region.size) if region.size else 0.0

    contrast = 0.0
    if follicle_mask.any() and ovary_mask.any():
        contrast = float(abs(array[ovary_mask].mean() - array[follicle_mask].mean()))

    background_contrast = 0.0
    outside = ~region
    if region.any() and outside.any():
        background_contrast = float(abs(array[region].mean() - array[outside].mean()))

    spread = float(np.percentile(array, 99) - np.percentile(array, 1)) if array.size else 0.0
    return QualityEvidence(
        ovary_voxel_fraction=fraction,
        border_contact_fraction=_border_contact_fraction(region),
        contrast=contrast,
        ovary_background_contrast=background_contrast,
        sharpness=_sharpness(array),
        dynamic_range=spread,
    )


def assess_quality(
    volume: np.ndarray,
    metadata: UltrasoundStudyMetadata,
    *,
    ovary_mask: np.ndarray | None = None,
    follicle_mask: np.ndarray | None = None,
    ovary_confidence: float = 0.0,
) -> ImageQualityAssessment:
    """Produce the quality assessment that gates all quantitative measurement.

    Args:
        volume: The (preprocessed) pixel array.
        metadata: Study metadata; supplies spacing and laterality availability.
        ovary_mask: Predicted ovary stroma mask, if segmentation has run.
        follicle_mask: Predicted follicle mask, if segmentation has run.
        ovary_confidence: Mean predicted ovary probability inside the mask.

    Returns:
        An :class:`ImageQualityAssessment` whose ``reasons`` list explains every
        capability that is switched off.
    """
    evidence = compute_quality_evidence(volume, ovary_mask, follicle_mask)
    reasons: list[str] = []

    too_small = evidence.ovary_voxel_fraction < MIN_OVARY_VOXEL_FRACTION
    too_large = evidence.ovary_voxel_fraction > MAX_OVARY_VOXEL_FRACTION
    no_contrast = evidence.ovary_background_contrast < MIN_OVARY_BACKGROUND_CONTRAST

    ovary_visible = not (too_small or too_large or no_contrast)
    if too_small:
        reasons.append(
            f"No ovary detected (candidate occupies {evidence.ovary_voxel_fraction:.4%} of the "
            f"volume, below the {MIN_OVARY_VOXEL_FRACTION:.2%} floor)."
        )
    if too_large:
        reasons.append(
            f"Candidate ovary occupies {evidence.ovary_voxel_fraction:.1%} of the field of view, "
            f"above the {MAX_OVARY_VOXEL_FRACTION:.0%} ceiling; this is a segmentation failure, "
            "not an ovary."
        )
    if no_contrast:
        reasons.append(
            f"Candidate ovary is indistinguishable from its surroundings "
            f"(intensity difference {evidence.ovary_background_contrast:.3f}, below the "
            f"{MIN_OVARY_BACKGROUND_CONTRAST} floor); no ovary has been detected."
        )

    whole_ovary_visible = ovary_visible and (
        evidence.border_contact_fraction <= MAX_BORDER_CONTACT_FRACTION
    )
    if ovary_visible and not whole_ovary_visible:
        reasons.append(
            f"Ovary touches the field-of-view border ({evidence.border_contact_fraction:.1%} of "
            "its voxels); volume and follicle count would be truncated."
        )

    spacing_available = metadata.spacing_mm is not None
    if not spacing_available:
        reasons.append("Pixel spacing unknown; no measurement can be expressed in millimetres.")

    laterality_available = metadata.laterality in ("left", "right")
    if not laterality_available:
        reasons.append("Laterality unknown; per-ovary attribution and asymmetry unavailable.")

    contrast_ok = evidence.contrast >= MIN_CONTRAST_FOR_COUNTING
    if ovary_visible and not contrast_ok:
        reasons.append(
            f"Follicle/stroma contrast {evidence.contrast:.3f} is below the "
            f"{MIN_CONTRAST_FOR_COUNTING} floor; follicles cannot be distinguished from speckle."
        )

    counting_feasible = ovary_visible and spacing_available and contrast_ok
    volume_feasible = ovary_visible and whole_ovary_visible and spacing_available

    score = _overall_score(
        evidence,
        ovary_visible=ovary_visible,
        whole_ovary_visible=whole_ovary_visible,
        spacing_available=spacing_available,
        ovary_confidence=ovary_confidence,
    )

    return ImageQualityAssessment(
        ovary_visible=ovary_visible,
        whole_ovary_visible=whole_ovary_visible,
        laterality_available=laterality_available,
        pixel_spacing_available=spacing_available,
        follicle_counting_feasible=counting_feasible,
        ovarian_volume_feasible=volume_feasible,
        overall_quality_score=score,
        reasons=reasons,
    )


def _overall_score(
    evidence: QualityEvidence,
    *,
    ovary_visible: bool,
    whole_ovary_visible: bool,
    spacing_available: bool,
    ovary_confidence: float,
) -> float:
    """Combine the evidence into a single 0-1 score.

    The score is a *screening* number: ``measurement_feasible`` on the schema
    requires it to exceed 0.5 in addition to the hard boolean gates, so a high
    score can never on its own unlock a measurement whose preconditions failed.
    """
    if not ovary_visible:
        # Cap hard: without a visible ovary nothing else can rescue the study.
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


def assess_quality_for_mode(
    image: np.ndarray,
    metadata: UltrasoundStudyMetadata,
    *,
    acquisition_mode: str,
    ovary_mask: np.ndarray | None = None,
    follicle_mask: np.ndarray | None = None,
    ovary_confidence: float = 0.0,
) -> ImageQualityAssessment:
    """Route to the gate that matches the acquisition.

    A 2D frame assessed against volumetric thresholds fails for the wrong reason
    (its cross-section legitimately fills far more of the image than an ovary
    fills a volume), and a volume assessed against 2D thresholds passes for the
    wrong reason. Dispatching on the declared acquisition mode keeps the two
    calibrations from being mixed.

    Args:
        image: A 2D frame, a ``(T, H, W)`` frame stack, or a 3D volume.
        metadata: Study metadata.
        acquisition_mode: One of the schema's ``AcquisitionMode`` values.
        ovary_mask: Predicted ovary mask matching ``image``'s shape.
        follicle_mask: Predicted follicle mask matching ``image``'s shape.
        ovary_confidence: Mean predicted ovary probability inside the mask.

    Returns:
        An :class:`ImageQualityAssessment`. For multi-frame modes this is the
        aggregate; use :func:`models.ultrasound.qc_2d.assess_cine_quality`
        directly when the per-frame detail is needed.
    """
    from models.ultrasound.qc_2d import (  # noqa: PLC0415 - avoids an import cycle
        assess_cine_quality,
        assess_frame_quality,
    )

    if acquisition_mode == "single_frame":
        return assess_frame_quality(
            image,
            metadata,
            ovary_mask=ovary_mask,
            follicle_mask=follicle_mask,
            ovary_confidence=ovary_confidence,
        )
    if acquisition_mode in ("cine_loop", "multi_frame"):
        return assess_cine_quality(
            image,
            metadata,
            ovary_masks=list(ovary_mask) if ovary_mask is not None else None,
            follicle_masks=list(follicle_mask) if follicle_mask is not None else None,
        ).aggregate
    if acquisition_mode == "volume_3d":
        return assess_quality(
            image,
            metadata,
            ovary_mask=ovary_mask,
            follicle_mask=follicle_mask,
            ovary_confidence=ovary_confidence,
        )
    raise ValueError(
        f"Unknown acquisition_mode '{acquisition_mode}'. Refusing to guess which quality gate "
        "applies: the wrong gate is worse than no gate."
    )


def abstention_reasons(assessment: ImageQualityAssessment) -> list[str]:
    """Reasons the pipeline must abstain, or an empty list if it may measure."""
    if assessment.measurement_feasible:
        return []
    reasons = list(assessment.reasons)
    if not reasons:
        reasons.append(
            f"Overall quality score {assessment.overall_quality_score:.2f} is at or below the "
            "0.50 threshold required for quantitative measurement."
        )
    return reasons


def quality_to_vector(assessment: ImageQualityAssessment) -> np.ndarray:
    """Serialise an assessment into the fixed-order vector the head predicts."""
    return np.array(
        [float(getattr(assessment, field)) for field in QUALITY_FIELDS],
        dtype=float,
    )


def quality_from_vector(
    vector: np.ndarray, *, reasons: list[str] | None = None
) -> ImageQualityAssessment:
    """Inverse of :func:`quality_to_vector`, used to decode a torch head."""
    values = np.asarray(vector, dtype=float).ravel()
    if values.size != len(QUALITY_FIELDS):
        raise ValueError(f"Expected {len(QUALITY_FIELDS)} quality outputs, got {values.size}.")
    payload: dict[str, Any] = {
        field: bool(values[i] >= 0.5) for i, field in enumerate(QUALITY_FIELDS[:-1])
    }
    payload["overall_quality_score"] = float(np.clip(values[-1], 0.0, 1.0))
    payload["reasons"] = list(reasons or [])
    return ImageQualityAssessment(**payload)
