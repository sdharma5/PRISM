"""Volumetric ovarian morphology — the OPTIONAL enhanced-mode measurement path.

A 3D volume is not the routine clinical acquisition (see
:mod:`models.ultrasound.morphology_2d` for the primary 2D path), but when one
exists it is the only acquisition that supports two quantities nothing else can:

* a **true per-ovary follicle count** (``follicle_number_per_ovary``), because
  every follicle is imaged once with known geometry in all three axes; and
* a genuine **ovarian volume in millilitres**.

Everything here is gated on image quality and physical spacing, and two
safeguards are structural rather than advisory:

* Every output carries ``clinician_review_status='model_generated'``. Nothing in
  this repository may promote that status; only a human review step may.
* Nothing here names a pathology. A large cystic structure is reported as
  ``large_structure_flag`` with a diameter and a warning. It is never called a
  cyst, a corpus luteum, an endometrioma, or a sign of any condition. Naming a
  finding is a diagnosis, and this is a research artifact.

Volume is computed two ways and cross-checked: voxel-count integration (exact
for the segmented mask) and the prolate-ellipsoid formula D1 x D2 x D3 x 0.523
(what clinical reports use). A large disagreement means the mask is not
ellipsoid-like, which usually means the segmentation failed, so the disagreement
is surfaced as a warning rather than averaged away.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from models.ultrasound.follicle_instances import (
    LARGE_STRUCTURE_DIAMETER_MM,
    InstanceExtractionResult,
    size_distribution,
)
from models.ultrasound.output_schema import build_abstained_output, build_volume_output
from schemas.imaging import (
    ImageQualityAssessment,
    OvarianMorphologyOutput,
    UltrasoundStudyMetadata,
)

__all__ = [
    "ELLIPSOID_COEFFICIENT",
    "MM3_PER_ML",
    "VOLUME_CROSSCHECK_TOLERANCE",
    "compute_morphology",
    "count_follicle_voxels_outside_ovary",
    "largest_component_fraction",
    "lateral_asymmetry",
    "morphology_summary",
    "ovary_dimensions_mm",
    "ovary_surface_area_mm2",
    "ovary_volume_ml_ellipsoid",
    "ovary_volume_ml_voxelwise",
]

#: Conventional prolate-ellipsoid coefficient used in clinical ovarian volumetry.
ELLIPSOID_COEFFICIENT = 0.523

#: Relative disagreement between the two volume estimates that triggers a warning.
VOLUME_CROSSCHECK_TOLERANCE = 0.35

MM3_PER_ML = 1000.0


def ovary_dimensions_mm(
    ovary_region_mask: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Principal-axis dimensions of the ovary in mm, largest first.

    Principal axes (via the mask's inertia tensor) are used rather than the
    bounding box, because a bounding box in scanner coordinates over-reports the
    dimensions of any ovary that is not axis-aligned — and none are.
    """
    coords = np.argwhere(ovary_region_mask)
    if coords.size == 0:
        return (0.0, 0.0, 0.0)
    physical = coords.astype(float) * np.asarray(spacing_mm, dtype=float)[None, :]
    centred = physical - physical.mean(axis=0)
    if centred.shape[0] < 4:
        extent = physical.max(axis=0) - physical.min(axis=0)
        return tuple(sorted((float(v) for v in extent), reverse=True))  # type: ignore[return-value]
    cov = np.cov(centred, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    # For a uniform ellipsoid, variance along a semi-axis a is a^2/5.
    semi_axes = np.sqrt(5.0 * eigenvalues)
    dims = sorted((float(2.0 * a) for a in semi_axes), reverse=True)
    return (dims[0], dims[1], dims[2])


def ovary_volume_ml_voxelwise(
    ovary_region_mask: np.ndarray, spacing_mm: tuple[float, float, float]
) -> float:
    """Volume by direct voxel integration, in millilitres."""
    voxel_mm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    return float(np.asarray(ovary_region_mask, dtype=bool).sum() * voxel_mm3 / MM3_PER_ML)


def ovary_volume_ml_ellipsoid(dimensions_mm: tuple[float, float, float]) -> float:
    """Clinical prolate-ellipsoid volume estimate, in millilitres."""
    d1, d2, d3 = dimensions_mm
    return float(d1 * d2 * d3 * ELLIPSOID_COEFFICIENT / MM3_PER_ML)


def ovary_surface_area_mm2(
    ovary_region_mask: np.ndarray, spacing_mm: tuple[float, float, float]
) -> float:
    """Approximate mask surface area from exposed voxel faces."""
    mask = np.asarray(ovary_region_mask, dtype=bool)
    if not mask.any():
        return 0.0
    sx, sy, sz = (float(s) for s in spacing_mm)
    face_areas = (sy * sz, sx * sz, sx * sy)
    total = 0.0
    for axis, area in enumerate(face_areas):
        shifted_up = np.roll(mask, 1, axis=axis)
        shifted_down = np.roll(mask, -1, axis=axis)
        total += float(((mask & ~shifted_up).sum() + (mask & ~shifted_down).sum()) * area)
    return total


def compute_morphology(
    *,
    metadata: UltrasoundStudyMetadata,
    quality: ImageQualityAssessment,
    ovary_region_mask: np.ndarray,
    instance_result: InstanceExtractionResult,
    ovary_confidence: float = 0.0,
    follicle_confidence: float = 0.0,
    false_follicle_voxels_outside_ovary: int = 0,
    large_structure_diameter_mm: float = LARGE_STRUCTURE_DIAMETER_MM,
) -> OvarianMorphologyOutput:
    """Assemble the volumetric morphology output, abstaining when unsupported.

    Args:
        metadata: Study metadata; supplies spacing, laterality and provenance.
        quality: The gating quality assessment.
        ovary_region_mask: Ovary stroma plus follicles (the full ovarian region).
        instance_result: Output of the follicle instance extractor.
        ovary_confidence: Mean ovary probability inside the mask.
        follicle_confidence: Mean follicle probability inside follicle instances.
        false_follicle_voxels_outside_ovary: Predicted follicle voxels lying
            outside the ovary; a direct read-out of the ``L_outside`` violation.
        large_structure_diameter_mm: Documented flagging threshold.

    Returns:
        An :class:`OvarianMorphologyOutput` with ``acquisition_mode='volume_3d'``,
        always ``model_generated``.
    """
    warnings: list[str] = list(instance_result.warnings)

    if not quality.measurement_feasible or metadata.spacing_mm is None:
        if metadata.spacing_mm is None:
            warnings.append("Spacing unknown: refusing to emit any measurement in physical units.")
        return build_abstained_output(
            metadata=metadata,
            quality=quality,
            acquisition_mode="volume_3d",
            extra_warnings=warnings,
        )

    spacing = metadata.spacing_mm
    region = np.asarray(ovary_region_mask, dtype=bool)

    dimensions = ovary_dimensions_mm(region, spacing)
    volume_voxel = ovary_volume_ml_voxelwise(region, spacing)
    volume_ellipsoid = ovary_volume_ml_ellipsoid(dimensions)

    if volume_voxel > 0:
        disagreement = abs(volume_ellipsoid - volume_voxel) / volume_voxel
        if disagreement > VOLUME_CROSSCHECK_TOLERANCE:
            warnings.append(
                f"Voxel-count volume ({volume_voxel:.2f} ml) and ellipsoid-formula volume "
                f"({volume_ellipsoid:.2f} ml) disagree by {disagreement:.0%}; the ovary mask is "
                "not ellipsoid-like, so the reported volume is uncertain."
            )

    all_instances = list(instance_result.instances)
    large = [inst for inst in all_instances if inst.is_large_or_uncertain]
    small = [inst for inst in all_instances if not inst.is_large_or_uncertain]

    if large:
        diameters = ", ".join(
            f"{inst.max_diameter_mm:.1f} mm" for inst in large if inst.max_diameter_mm is not None
        )
        warnings.append(
            f"{len(large)} structure(s) at or above the {large_structure_diameter_mm} mm "
            f"threshold ({diameters}) were flagged as large/uncertain and EXCLUDED from the "
            "follicle count. This is a descriptive size observation only; it is not a diagnosis "
            "and no pathological interpretation is offered. Clinician review required."
        )

    count = len(small) if quality.follicle_counting_feasible else None
    if count is None:
        warnings.append(
            "Follicle counting not feasible for this study; count withheld. "
            + "; ".join(quality.reasons)
        )

    volume_ml: float | None = volume_voxel if quality.ovarian_volume_feasible else None
    if volume_ml is None:
        # Carry the gate's reasons through, so the withheld measurement explains
        # itself wherever the morphology output is read.
        warnings.append(
            "Ovarian volume not feasible for this study; volume withheld. "
            + "; ".join(quality.reasons)
        )

    density: float | None = None
    if count is not None and volume_ml is not None and volume_ml > 0:
        density = float(count / volume_ml)

    diameters_mm = sorted(
        float(inst.mean_diameter_mm) for inst in small if inst.mean_diameter_mm is not None
    )

    if false_follicle_voxels_outside_ovary > 0:
        warnings.append(
            f"{false_follicle_voxels_outside_ovary} predicted follicle voxel(s) fall outside the "
            "predicted ovary; these are anatomically impossible and indicate segmentation error."
        )

    return build_volume_output(
        metadata=metadata,
        quality=quality,
        ovary_volume_ml=volume_ml,
        ovary_dimensions_mm=dimensions,
        follicle_number_per_ovary=count,
        follicle_diameters_mm=diameters_mm,
        instances=all_instances,
        follicle_density_per_ml=density,
        ovary_confidence=ovary_confidence,
        follicle_confidence=follicle_confidence,
        false_follicle_voxels_outside_ovary=false_follicle_voxels_outside_ovary,
        large_structure_flag=bool(large),
        extra_warnings=warnings,
    )


def morphology_summary(output: OvarianMorphologyOutput) -> dict[str, float]:
    """Flatten a morphology output into scalar features for token export.

    The follicle count is exported under a key naming the method that produced it,
    never under a generic ``follicle_count``. A consumer that averages a
    per-section count with a per-ovary count is averaging two different
    quantities, and the key names are what make that mistake visible.
    """
    summary: dict[str, float] = {
        "quality_score": float(output.quality_score),
        "measurement_feasible": float(output.measurement_feasible),
        "ovary_mask_confidence": float(output.ovary_mask_confidence),
        "follicle_detection_confidence": float(output.follicle_detection_confidence),
    }
    if output.ovary_volume_ml is not None:
        summary["ovary_volume_ml"] = float(output.ovary_volume_ml)
    if output.ovary_area_mm2 is not None:
        summary["ovary_area_mm2"] = float(output.ovary_area_mm2)
    if output.follicle_number_per_section is not None:
        summary["follicle_number_per_section"] = float(output.follicle_number_per_section)
    if output.estimated_follicle_number_per_ovary is not None:
        summary["estimated_follicle_number_per_ovary"] = float(
            output.estimated_follicle_number_per_ovary
        )
    if output.follicle_number_per_ovary is not None:
        summary["follicle_number_per_ovary"] = float(output.follicle_number_per_ovary)
    if output.follicle_density_per_ml is not None:
        summary["follicle_density_per_ml"] = float(output.follicle_density_per_ml)
    if output.tracking_coverage is not None:
        summary["tracking_coverage"] = float(output.tracking_coverage)
    if output.instances:
        summary.update(size_distribution(output.instances))
    elif output.follicle_diameters_mm:
        summary.update(_diameter_summary(output.follicle_diameters_mm))
    return summary


def _diameter_summary(diameters_mm: list[float]) -> dict[str, float]:
    """Size-distribution statistics from bare diameters (the 2D paths).

    The 2D pathways report diameters without full :class:`FollicleInstance`
    records, because a 2D cross-section has no voxel volume to record. The same
    distribution statistics are still meaningful and are computed here.
    """
    array = np.asarray(diameters_mm, dtype=float)
    if array.size == 0:
        return {}
    return {
        "n": float(array.size),
        "mean_diameter_mm": float(array.mean()),
        "median_diameter_mm": float(np.median(array)),
        "sd_diameter_mm": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min_diameter_mm": float(array.min()),
        "max_diameter_mm": float(array.max()),
        "n_2_to_9_mm": float(((array >= 2.0) & (array < 9.0)).sum()),
        "n_ge_9_mm": float((array >= 9.0).sum()),
    }


def lateral_asymmetry(
    left: OvarianMorphologyOutput | None, right: OvarianMorphologyOutput | None
) -> dict[str, float | None]:
    """Left/right differences, computed only where both sides measured.

    Asymmetry is reported as a raw difference and a normalised index. Both are
    ``None`` when either side abstained, because an asymmetry computed against a
    withheld measurement would be a fabricated finding.

    The count difference is additionally ``None`` when the two sides were counted
    by **different methods**. A per-section count on the left minus a per-ovary
    count on the right is not an asymmetry; it is a category error, and it would
    read as a large genuine difference.
    """
    result: dict[str, float | None] = {
        "volume_difference_ml": None,
        "volume_asymmetry_index": None,
        "area_difference_mm2": None,
        "follicle_count_difference": None,
    }
    if left is None or right is None:
        return result
    if left.ovary_volume_ml is not None and right.ovary_volume_ml is not None:
        diff = float(left.ovary_volume_ml - right.ovary_volume_ml)
        total = float(left.ovary_volume_ml + right.ovary_volume_ml)
        result["volume_difference_ml"] = diff
        result["volume_asymmetry_index"] = float(abs(diff) / total) if total > 0 else None
    if left.ovary_area_mm2 is not None and right.ovary_area_mm2 is not None:
        result["area_difference_mm2"] = float(left.ovary_area_mm2 - right.ovary_area_mm2)

    left_count, left_method = left.reportable_follicle_count
    right_count, right_method = right.reportable_follicle_count
    if (
        left_count is not None
        and right_count is not None
        and left_method == right_method
        and left_method != "not_assessed"
    ):
        result["follicle_count_difference"] = float(left_count - right_count)
    return result


def count_follicle_voxels_outside_ovary(
    follicle_mask: np.ndarray, ovary_region_mask: np.ndarray
) -> int:
    """Count predicted follicle voxels lying outside the predicted ovary.

    This is the discrete counterpart of the ``L_outside`` training penalty and is
    reported on every study as a segmentation-sanity read-out. Works unchanged on
    2D frames, where it counts pixels.
    """
    follicle = np.asarray(follicle_mask, dtype=bool)
    region = np.asarray(ovary_region_mask, dtype=bool)
    return int((follicle & ~region).sum())


def largest_component_fraction(mask: np.ndarray) -> float:
    """Fraction of mask voxels in its largest connected component.

    A well-segmented ovary is one object; a low value means the mask is
    fragmented and its volume should not be trusted.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0
    labels, n = ndi.label(mask)
    if n <= 1:
        return 1.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / sizes.sum())
