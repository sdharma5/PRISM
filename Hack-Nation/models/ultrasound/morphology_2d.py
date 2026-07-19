"""Ovarian morphology from 2D frames — the primary measurement pathway.

What a 2D acquisition can and cannot support is the whole subject of this module.

* **A single frame** supports a follicle count *in that cross-section*
  (``follicle_number_per_section``) and the cross-sectional ovary area
  (``ovary_area_mm2``). It supports nothing else. It cannot support a per-ovary
  count, estimated or otherwise, and it cannot support an ovarian volume: a
  cross-section has two dimensions and volume needs three.
* **A cine loop** additionally supports an *estimated* unique follicle count via
  :mod:`models.ultrasound.cine_tracking`, plus in-plane ovary dimensions taken
  from the frame showing the largest cross-section. It still does not support a
  true per-ovary count or a volume, because a freehand sweep has no calibrated
  out-of-plane spacing.

The 2023 international guideline treats per-section and per-ovary follicle counts
as distinct quantities, and explicitly permits per-section counting *because*
complete counting is often unreliable. Reporting a per-section count as though it
were an antral follicle count would inflate nothing and deflate everything — it is
simply a different measurement, and the method must travel with the number.

**Physical units require known spacing.** Every function here returns ``None``
and records a warning when the in-plane pixel spacing is unknown. A follicle
diameter in pixels is clinically meaningless, and a count of "follicles above 2
pixels" is not comparable to any published threshold.

Spacing convention
------------------
``UltrasoundStudyMetadata.spacing_mm`` is a 3-tuple ``(z, row, col)`` because the
schema was written for volumes. For a 2D acquisition the *in-plane* spacing is the
last two elements, and the first element is a placeholder that this module never
reads. :func:`in_plane_spacing_mm` is the only sanctioned accessor, so no 2D code
path can accidentally multiply by a through-plane spacing that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.ultrasound.cine_tracking import (
    TrackingResult,
    follicle_label_image,
    observations_from_labels,
    track_follicles,
)
from models.ultrasound.follicle_instances import (
    LARGE_STRUCTURE_DIAMETER_MM,
    MIN_FOLLICLE_DIAMETER_MM,
)
from schemas.imaging import ImageQualityAssessment, UltrasoundStudyMetadata

#: Below this tracking coverage the unique-follicle estimate is withheld: too few
#: frames contributed usable masks for cross-frame following to mean anything.
MIN_COVERAGE_FOR_ESTIMATE = 0.30

#: A count of ZERO is only reportable when tracking was confident enough that an
#: empty result is informative. Below this, zero tracks means "we could not
#: follow anything", not "there are no follicles", and the estimate is withheld.
MIN_CONFIDENCE_FOR_ZERO_ESTIMATE = 0.40

__all__ = [
    "MIN_CONFIDENCE_FOR_ZERO_ESTIMATE",
    "MIN_COVERAGE_FOR_ESTIMATE",
    "CineMorphology2D",
    "FrameMorphology2D",
    "compute_cine_morphology",
    "compute_frame_morphology",
    "follicle_number_per_section",
    "in_plane_spacing_mm",
    "ovary_area_mm2",
    "ovary_in_plane_dimensions_mm",
]


def in_plane_spacing_mm(
    metadata: UltrasoundStudyMetadata,
) -> tuple[float, float] | None:
    """In-plane ``(row_mm, col_mm)`` spacing, or ``None`` when unknown.

    The only sanctioned way for 2D code to read spacing. See the module docstring
    for why the through-plane element of ``metadata.spacing_mm`` is never used.
    """
    if metadata.spacing_mm is None:
        return None
    spacing = tuple(float(s) for s in metadata.spacing_mm)
    row_mm, col_mm = spacing[-2], spacing[-1]
    if row_mm <= 0 or col_mm <= 0 or not np.isfinite(row_mm) or not np.isfinite(col_mm):
        return None
    return (row_mm, col_mm)


def ovary_area_mm2(
    ovary_region_mask: np.ndarray, pixel_spacing_mm: tuple[float, float] | None
) -> float | None:
    """Cross-sectional ovary area in mm^2, or ``None`` without spacing.

    Area is the 2D analogue of ovarian volume and is the quantity a single frame
    can honestly report. It is NOT convertible to a volume: doing so would require
    assuming a shape and an out-of-plane extent that were never imaged.
    """
    if pixel_spacing_mm is None:
        return None
    mask = np.asarray(ovary_region_mask, dtype=bool)
    row_mm, col_mm = (float(s) for s in pixel_spacing_mm)
    return float(mask.sum() * row_mm * col_mm)


def ovary_in_plane_dimensions_mm(
    ovary_region_mask: np.ndarray, pixel_spacing_mm: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Major and minor in-plane axes in mm, largest first.

    Principal axes (via the mask's inertia tensor) rather than the bounding box,
    because a bounding box in image coordinates over-reports the dimensions of any
    ovary that is not axis-aligned — and none are.

    Returns ``None`` without spacing. Note this is a **2-tuple**: the third
    dimension of the ovary does not exist in a single plane and is not invented.
    """
    if pixel_spacing_mm is None:
        return None
    mask = np.asarray(ovary_region_mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0.0, 0.0)
    physical = coords.astype(float) * np.asarray(pixel_spacing_mm, dtype=float)[None, :]
    centred = physical - physical.mean(axis=0)
    if centred.shape[0] < 3:
        extent = physical.max(axis=0) - physical.min(axis=0)
        dims = sorted((float(v) for v in extent), reverse=True)
        return (dims[0], dims[1])
    cov = np.cov(centred, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    # For a uniform ellipse, variance along a semi-axis a is a^2/4.
    semi_axes = np.sqrt(4.0 * eigenvalues)
    dims = sorted((float(2.0 * a) for a in semi_axes), reverse=True)
    return (dims[0], dims[1])


def follicle_number_per_section(
    follicle_label_image_2d: np.ndarray,
    *,
    pixel_spacing_mm: tuple[float, float] | None,
    min_diameter_mm: float = MIN_FOLLICLE_DIAMETER_MM,
    large_structure_diameter_mm: float = LARGE_STRUCTURE_DIAMETER_MM,
) -> tuple[int | None, list[float], list[float]]:
    """Count follicles in one cross-section and measure their diameters.

    Args:
        follicle_label_image_2d: Integer instance labels for one frame.
        pixel_spacing_mm: In-plane ``(row_mm, col_mm)``; ``None`` forces abstention.
        min_diameter_mm: Components below this are speckle, not follicles.
        large_structure_diameter_mm: At or above this a structure is flagged as
            large/uncertain and EXCLUDED from the count. It is never named.

    Returns:
        ``(count, small_diameters_mm, large_diameters_mm)``. ``count`` is ``None``
        when spacing is unknown, because an unfiltered component count is not a
        follicle count.
    """
    if pixel_spacing_mm is None:
        return None, [], []

    labels = np.asarray(follicle_label_image_2d)
    row_mm, col_mm = (float(s) for s in pixel_spacing_mm)
    pixel_area = row_mm * col_mm

    small: list[float] = []
    large: list[float] = []
    for label_id in (int(v) for v in np.unique(labels) if v != 0):
        area_mm2 = float((labels == label_id).sum() * pixel_area)
        # Equivalent-circle diameter: what a caliper measurement of this
        # cross-section would read.
        diameter = float(2.0 * np.sqrt(area_mm2 / np.pi))
        if diameter < min_diameter_mm:
            continue
        (large if diameter >= large_structure_diameter_mm else small).append(diameter)

    return len(small), sorted(small), sorted(large)


@dataclass
class FrameMorphology2D:
    """Everything one 2D frame can honestly support."""

    frame_index: int = 0
    measurement_feasible: bool = False
    follicle_number_per_section: int | None = None
    ovary_area_mm2: float | None = None
    ovary_in_plane_dimensions_mm: tuple[float, float] | None = None
    follicle_diameters_mm: list[float] = field(default_factory=list)
    large_structure_diameters_mm: list[float] = field(default_factory=list)
    ovary_region_mask: np.ndarray | None = None
    follicle_labels: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def large_structure_flag(self) -> bool:
        return bool(self.large_structure_diameters_mm)


def _large_structure_warning(diameters: list[float], large_structure_diameter_mm: float) -> str:
    joined = ", ".join(f"{d:.1f} mm" for d in diameters)
    return (
        f"{len(diameters)} structure(s) at or above the {large_structure_diameter_mm} mm "
        f"threshold ({joined}) were flagged as large/uncertain and EXCLUDED from the follicle "
        "count. This is a descriptive size observation only; it is not a diagnosis and no "
        "pathological interpretation is offered. Clinician review required."
    )


def compute_frame_morphology(
    *,
    metadata: UltrasoundStudyMetadata,
    quality: ImageQualityAssessment,
    ovary_region_mask: np.ndarray,
    follicle_mask: np.ndarray,
    frame_index: int = 0,
    min_diameter_mm: float = MIN_FOLLICLE_DIAMETER_MM,
    large_structure_diameter_mm: float = LARGE_STRUCTURE_DIAMETER_MM,
) -> FrameMorphology2D:
    """Measure one 2D frame, abstaining when the frame cannot support it.

    Args:
        metadata: Study metadata; supplies the in-plane spacing.
        quality: The gating per-frame quality assessment.
        ovary_region_mask: Ovary stroma plus follicles for this frame.
        follicle_mask: Predicted follicle mask for this frame.
        frame_index: Index within the loop, for provenance.
        min_diameter_mm: Documented small-object rejection threshold.
        large_structure_diameter_mm: Documented large-structure threshold.

    Returns:
        A :class:`FrameMorphology2D`. No per-ovary quantity and no volume is
        produced here under any circumstances.
    """
    warnings: list[str] = []
    spacing = in_plane_spacing_mm(metadata)

    if spacing is None:
        warnings.append(
            "In-plane pixel spacing unknown: refusing to emit any measurement in physical units. "
            "A follicle diameter in pixels is not comparable to any published threshold."
        )
    if not quality.measurement_feasible or spacing is None:
        warnings.extend(quality.reasons)
        warnings.insert(
            0, "Quantitative measurement withheld: frame quality or spacing is insufficient."
        )
        return FrameMorphology2D(
            frame_index=frame_index,
            measurement_feasible=False,
            warnings=sorted(set([*warnings, *metadata.warnings])),
        )

    region = np.asarray(ovary_region_mask, dtype=bool)
    labels = follicle_label_image(
        follicle_mask, pixel_spacing_mm=spacing, min_diameter_mm=min_diameter_mm
    )

    count, small, large = follicle_number_per_section(
        labels,
        pixel_spacing_mm=spacing,
        min_diameter_mm=min_diameter_mm,
        large_structure_diameter_mm=large_structure_diameter_mm,
    )
    if not quality.follicle_counting_feasible:
        count, small = None, []
        warnings.append(
            "Per-section follicle counting not feasible for this frame; count withheld. "
            + "; ".join(quality.reasons)
        )
    if large:
        warnings.append(_large_structure_warning(large, large_structure_diameter_mm))

    return FrameMorphology2D(
        frame_index=frame_index,
        measurement_feasible=True,
        follicle_number_per_section=count,
        ovary_area_mm2=ovary_area_mm2(region, spacing),
        ovary_in_plane_dimensions_mm=ovary_in_plane_dimensions_mm(region, spacing),
        follicle_diameters_mm=small,
        large_structure_diameters_mm=large,
        ovary_region_mask=region,
        follicle_labels=labels,
        warnings=sorted(set([*warnings, *metadata.warnings])),
    )


@dataclass
class CineMorphology2D:
    """Everything a cine loop can honestly support.

    Note what is absent: ``ovary_volume_ml`` and any true per-ovary count. A
    freehand sweep records no calibrated out-of-plane spacing, so its frames do not
    reconstruct a volume however many of them there are.
    """

    per_frame: list[FrameMorphology2D] = field(default_factory=list)
    tracking: TrackingResult | None = None
    estimated_follicle_number_per_ovary: int | None = None
    #: Per-section count from the frame with the largest ovary cross-section —
    #: the representative section a sonographer would freeze and report.
    representative_follicle_number_per_section: int | None = None
    representative_frame_index: int | None = None
    max_ovary_area_mm2: float | None = None
    ovary_in_plane_dimensions_mm: tuple[float, float] | None = None
    follicle_diameters_mm: list[float] = field(default_factory=list)
    large_structure_diameters_mm: list[float] = field(default_factory=list)
    frames_analyzed: int = 0
    frames_total: int = 0
    tracking_coverage: float = 0.0
    tracking_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def large_structure_flag(self) -> bool:
        return bool(self.large_structure_diameters_mm)


def compute_cine_morphology(
    *,
    metadata: UltrasoundStudyMetadata,
    per_frame_quality: list[ImageQualityAssessment],
    ovary_region_masks: list[np.ndarray],
    follicle_masks: list[np.ndarray],
    min_diameter_mm: float = MIN_FOLLICLE_DIAMETER_MM,
    large_structure_diameter_mm: float = LARGE_STRUCTURE_DIAMETER_MM,
    **tracking_kwargs: object,
) -> CineMorphology2D:
    """Measure a cine loop: per-frame morphology plus cross-frame tracking.

    Only frames that pass the quality gate contribute observations to the tracker.
    Letting a failed frame contribute is how a shadow artefact becomes a follicle:
    the tracker has no way to tell a spurious blob from a real one, so the filter
    has to happen before it.

    The representative per-section count comes from the frame with the **largest**
    ovary cross-section, not the mean or the maximum count. That frame is the one
    closest to the ovary's mid-plane, which is the section a sonographer freezes
    and reports; taking the maximum count instead would systematically pick the
    frame with the most segmentation noise.

    Args:
        metadata: Study metadata; supplies the in-plane spacing.
        per_frame_quality: One assessment per frame, in frame order.
        ovary_region_masks: One ovary region mask per frame.
        follicle_masks: One follicle mask per frame.
        min_diameter_mm: Documented small-object rejection threshold.
        large_structure_diameter_mm: Documented large-structure threshold.
        **tracking_kwargs: Forwarded to :func:`track_follicles`.

    Returns:
        A :class:`CineMorphology2D`.
    """
    spacing = in_plane_spacing_mm(metadata)
    n_frames = len(per_frame_quality)
    warnings: list[str] = []

    per_frame = [
        compute_frame_morphology(
            metadata=metadata,
            quality=per_frame_quality[i],
            ovary_region_mask=ovary_region_masks[i],
            follicle_mask=follicle_masks[i],
            frame_index=i,
            min_diameter_mm=min_diameter_mm,
            large_structure_diameter_mm=large_structure_diameter_mm,
        )
        for i in range(n_frames)
    ]
    usable = [f for f in per_frame if f.measurement_feasible and f.follicle_labels is not None]

    if spacing is None:
        warnings.append(
            "In-plane pixel spacing unknown: no cine measurement can be expressed in millimetres."
        )
    if not usable:
        warnings.append(
            f"No usable frames in this {n_frames}-frame loop; all quantitative measurement is "
            "withheld."
        )
        return CineMorphology2D(
            per_frame=per_frame,
            frames_analyzed=0,
            frames_total=n_frames,
            warnings=sorted(set([*warnings, *metadata.warnings])),
        )

    label_images = {f.frame_index: f.follicle_labels for f in usable}
    observations = observations_from_labels(
        label_images,  # type: ignore[arg-type]
        pixel_spacing_mm=spacing,
    )
    tracking = track_follicles(
        observations,
        frames_total=n_frames,
        pixel_spacing_mm=spacing,
        **tracking_kwargs,  # type: ignore[arg-type]
    )
    warnings.extend(tracking.warnings)

    representative = max(
        (f for f in usable if f.ovary_area_mm2 is not None),
        key=lambda f: f.ovary_area_mm2 or 0.0,
        default=None,
    )

    estimated = tracking.estimated_unique_count if spacing is not None else None
    if estimated is not None and not any(q.follicle_counting_feasible for q in per_frame_quality):
        estimated = None
        warnings.append(
            "Follicle counting was not feasible on any frame; the unique-count estimate is "
            "withheld rather than reported from unusable masks."
        )

    # A reported count of zero is a strong claim — "this ovary has no follicles".
    # When tracking has broken down, zero surviving tracks means "we could not
    # follow anything", which is a completely different statement and must not be
    # emitted as a measurement. The two are only distinguishable when coverage and
    # tracking confidence are high enough for an empty result to be informative.
    if estimated is not None and tracking.tracking_coverage < MIN_COVERAGE_FOR_ESTIMATE:
        warnings.append(
            f"Tracking coverage {tracking.tracking_coverage:.0%} is below the "
            f"{MIN_COVERAGE_FOR_ESTIMATE:.0%} floor; the unique-follicle estimate is withheld "
            "because too few frames contributed usable masks to follow anything across them."
        )
        estimated = None
    elif estimated == 0 and tracking.confidence < MIN_CONFIDENCE_FOR_ZERO_ESTIMATE:
        warnings.append(
            f"No follicle track survived, but tracking confidence is only "
            f"{tracking.confidence:.2f}. A count of zero cannot be distinguished from a tracking "
            "failure at this confidence, so the unique-follicle estimate is withheld rather than "
            "reported as zero."
        )
        estimated = None

    large = sorted({d for f in usable for d in f.large_structure_diameters_mm})
    if large:
        warnings.append(_large_structure_warning(large, large_structure_diameter_mm))

    return CineMorphology2D(
        per_frame=per_frame,
        tracking=tracking,
        estimated_follicle_number_per_ovary=estimated,
        representative_follicle_number_per_section=(
            representative.follicle_number_per_section if representative else None
        ),
        representative_frame_index=representative.frame_index if representative else None,
        max_ovary_area_mm2=representative.ovary_area_mm2 if representative else None,
        ovary_in_plane_dimensions_mm=(
            representative.ovary_in_plane_dimensions_mm if representative else None
        ),
        follicle_diameters_mm=tracking.diameters_mm,
        large_structure_diameters_mm=large,
        frames_analyzed=tracking.frames_analyzed,
        frames_total=n_frames,
        tracking_coverage=tracking.tracking_coverage,
        tracking_confidence=tracking.confidence,
        warnings=sorted(set([*warnings, *metadata.warnings])),
    )
