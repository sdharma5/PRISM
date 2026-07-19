"""Assembly of measurements into :class:`OvarianMorphologyOutput`.

This is the single place where a number acquires a *claim*. Everything upstream
computes quantities; this module decides which schema field each quantity is
allowed to occupy, and therefore what the pipeline is asserting about the ovary.

The governing rule: **an acquisition may never claim a measurement it cannot
support.** Three builders exist, one per acquisition class, and each is
structurally incapable of over-claiming:

===================  ===============================  ====================
Acquisition          Count it may report              Volume?
===================  ===============================  ====================
``single_frame``     ``follicle_number_per_section``  no — area only
``multi_frame`` /    ``estimated_follicle_number_``   no
``cine_loop``        ``per_ovary`` (tracked estimate)
``volume_3d``        ``follicle_number_per_ovary``    yes
===================  ===============================  ====================

The schema's own ``model_validator`` enforces the same rules independently. That
redundancy is deliberate: if a future caller assembles an output by hand rather
than through these builders, pydantic still refuses the over-claim. Neither guard
is a substitute for the other.

Every output leaves here as ``clinician_review_status='model_generated'`` and
carries the "Clinician confirmation pending" warning. Nothing in this repository
may promote that status; only a human review step may.
"""

from __future__ import annotations

import numpy as np

from models.ultrasound.morphology_2d import CineMorphology2D, FrameMorphology2D
from schemas.imaging import (
    FollicleInstance,
    ImageQualityAssessment,
    OvarianMorphologyOutput,
    UltrasoundStudyMetadata,
)

__all__ = [
    "CLINICIAN_PENDING_WARNING",
    "build_abstained_output",
    "build_cine_output",
    "build_single_frame_output",
    "build_volume_output",
]

#: Carried on every ultrasound output and token without exception.
CLINICIAN_PENDING_WARNING = "Clinician confirmation pending"

#: Stated on every single-frame output, so a consumer reading only the warnings
#: still learns that the missing per-ovary count is a property of the acquisition
#: rather than a failure of the model.
SINGLE_FRAME_WARNING = (
    "Single frame: per-ovary count unavailable. Only follicle_number_per_section "
    "(follicles visible in this one cross-section) is reported, and it is not "
    "comparable to a per-ovary antral follicle count."
)

#: Stated on every cine output.
CINE_ESTIMATE_WARNING = (
    "Cine loop: the per-ovary follicle count is an ESTIMATE from cross-frame tracking, "
    "not a census. A true per-ovary count requires a complete volumetric acquisition."
)

#: Stated whenever a 2D acquisition is asked about volume.
NO_VOLUME_FROM_2D_WARNING = (
    "Ovarian volume not reported: a 2D acquisition has no calibrated out-of-plane "
    "geometry, so volume cannot be computed. Cross-sectional area is reported instead."
)


def _common_warnings(
    metadata: UltrasoundStudyMetadata, extra: list[str] | None = None
) -> list[str]:
    return [CLINICIAN_PENDING_WARNING, *metadata.warnings, *(extra or [])]


def build_abstained_output(
    *,
    metadata: UltrasoundStudyMetadata,
    quality: ImageQualityAssessment,
    acquisition_mode: str,
    extra_warnings: list[str] | None = None,
) -> OvarianMorphologyOutput:
    """An output with no measurements and explicit abstention reasons.

    Abstention is a result, not an error. It is returned as a fully valid output
    with ``measurement_feasible=False`` and every measurement literally ``None``,
    so a downstream consumer cannot mistake a withheld number for a zero.
    """
    reasons = list(quality.reasons) or [
        f"Overall quality score {quality.overall_quality_score:.2f} is at or below the "
        "0.50 threshold required for quantitative measurement."
    ]
    return OvarianMorphologyOutput(
        study_id=metadata.study_id,
        patient_id=metadata.patient_id,
        laterality=metadata.laterality,
        acquisition_mode=acquisition_mode,  # type: ignore[arg-type]
        quality_score=quality.overall_quality_score,
        full_ovary_visible=quality.whole_ovary_visible,
        measurement_feasible=False,
        follicle_count_method="not_assessed",
        warnings=sorted(
            set(
                _common_warnings(
                    metadata,
                    [
                        "Quantitative measurement withheld: image quality is insufficient.",
                        *reasons,
                        *(extra_warnings or []),
                    ],
                )
            )
        ),
        clinician_review_status="model_generated",
    )


def build_single_frame_output(
    *,
    metadata: UltrasoundStudyMetadata,
    quality: ImageQualityAssessment,
    frame: FrameMorphology2D,
    ovary_confidence: float = 0.0,
    follicle_confidence: float = 0.0,
    false_follicle_pixels_outside_ovary: int = 0,
    extra_warnings: list[str] | None = None,
) -> OvarianMorphologyOutput:
    """Assemble a single-frame result.

    Structurally incapable of claiming a per-ovary count or a volume: neither is
    passed to the constructor under any code path, and the schema validator would
    reject them if they were.
    """
    if not frame.measurement_feasible:
        return build_abstained_output(
            metadata=metadata,
            quality=quality,
            acquisition_mode="single_frame",
            extra_warnings=[SINGLE_FRAME_WARNING, *frame.warnings, *(extra_warnings or [])],
        )

    count = frame.follicle_number_per_section
    warnings = [
        SINGLE_FRAME_WARNING,
        NO_VOLUME_FROM_2D_WARNING,
        *frame.warnings,
        *(extra_warnings or []),
    ]
    if false_follicle_pixels_outside_ovary > 0:
        warnings.append(
            f"{false_follicle_pixels_outside_ovary} predicted follicle pixel(s) fall outside the "
            "predicted ovary; these are anatomically impossible and indicate segmentation error."
        )

    return OvarianMorphologyOutput(
        study_id=metadata.study_id,
        patient_id=metadata.patient_id,
        laterality=metadata.laterality,
        acquisition_mode="single_frame",
        quality_score=quality.overall_quality_score,
        full_ovary_visible=quality.whole_ovary_visible,
        measurement_feasible=True,
        ovary_volume_ml=None,
        ovary_area_mm2=frame.ovary_area_mm2,
        ovary_dimensions_mm=None,
        follicle_number_per_section=count,
        follicle_count_method="per_section" if count is not None else "not_assessed",
        follicle_diameters_mm=list(frame.follicle_diameters_mm),
        frames_analyzed=1,
        tracking_coverage=None,
        ovary_mask_confidence=float(np.clip(ovary_confidence, 0.0, 1.0)),
        follicle_detection_confidence=float(np.clip(follicle_confidence, 0.0, 1.0)),
        false_follicle_voxels_outside_ovary=int(false_follicle_pixels_outside_ovary),
        large_structure_flag=frame.large_structure_flag,
        warnings=sorted(set(_common_warnings(metadata, warnings))),
        clinician_review_status="model_generated",
    )


def build_cine_output(
    *,
    metadata: UltrasoundStudyMetadata,
    quality: ImageQualityAssessment,
    cine: CineMorphology2D,
    acquisition_mode: str = "cine_loop",
    ovary_confidence: float = 0.0,
    follicle_confidence: float = 0.0,
    false_follicle_pixels_outside_ovary: int = 0,
    extra_warnings: list[str] | None = None,
) -> OvarianMorphologyOutput:
    """Assemble a cine-loop or multi-frame result.

    Reports both the representative per-section count and the tracked per-ovary
    *estimate*, because they answer different questions and a reader who only
    trusts one should be able to find it. ``follicle_count_method`` names the
    estimate as the headline quantity when tracking succeeded, and falls back to
    the per-section count when it did not.

    ``follicle_detection_confidence`` is multiplied by the tracking confidence:
    a unique-count estimate from a loop with poor coverage is exactly as weak as
    its tracking, however crisp the individual masks were.
    """
    if cine.frames_analyzed == 0:
        return build_abstained_output(
            metadata=metadata,
            quality=quality,
            acquisition_mode=acquisition_mode,
            extra_warnings=[CINE_ESTIMATE_WARNING, *cine.warnings, *(extra_warnings or [])],
        )

    estimated = cine.estimated_follicle_number_per_ovary
    per_section = cine.representative_follicle_number_per_section

    method = "not_assessed"
    if estimated is not None:
        method = "estimated_per_ovary"
    elif per_section is not None:
        method = "per_section"

    warnings = [
        CINE_ESTIMATE_WARNING,
        NO_VOLUME_FROM_2D_WARNING,
        *cine.warnings,
        *(extra_warnings or []),
    ]
    if cine.ovary_in_plane_dimensions_mm is not None:
        major, minor = cine.ovary_in_plane_dimensions_mm
        warnings.append(
            f"Ovary in-plane dimensions {major:.1f} x {minor:.1f} mm, measured on frame "
            f"{cine.representative_frame_index} (largest cross-section). The third ovarian "
            "dimension was not imaged and is not reported."
        )
    if false_follicle_pixels_outside_ovary > 0:
        warnings.append(
            f"{false_follicle_pixels_outside_ovary} predicted follicle pixel(s) fall outside the "
            "predicted ovary; these are anatomically impossible and indicate segmentation error."
        )

    return OvarianMorphologyOutput(
        study_id=metadata.study_id,
        patient_id=metadata.patient_id,
        laterality=metadata.laterality,
        acquisition_mode=acquisition_mode,  # type: ignore[arg-type]
        quality_score=quality.overall_quality_score,
        full_ovary_visible=quality.whole_ovary_visible,
        measurement_feasible=True,
        # Never a volume from 2D, whatever the frame count.
        ovary_volume_ml=None,
        ovary_area_mm2=cine.max_ovary_area_mm2,
        # The schema's ovary_dimensions_mm is a 3-tuple describing a volume. A
        # sweep measures two dimensions, so the field stays None and the in-plane
        # pair is reported in the warnings rather than padded to three.
        ovary_dimensions_mm=None,
        follicle_number_per_section=per_section,
        estimated_follicle_number_per_ovary=estimated,
        follicle_number_per_ovary=None,
        follicle_count_method=method,  # type: ignore[arg-type]
        follicle_diameters_mm=list(cine.follicle_diameters_mm),
        frames_analyzed=cine.frames_analyzed,
        tracking_coverage=float(np.clip(cine.tracking_coverage, 0.0, 1.0)),
        ovary_mask_confidence=float(np.clip(ovary_confidence, 0.0, 1.0)),
        follicle_detection_confidence=float(
            np.clip(follicle_confidence * cine.tracking_confidence, 0.0, 1.0)
        ),
        false_follicle_voxels_outside_ovary=int(false_follicle_pixels_outside_ovary),
        large_structure_flag=cine.large_structure_flag,
        warnings=sorted(set(_common_warnings(metadata, warnings))),
        clinician_review_status="model_generated",
    )


def build_volume_output(
    *,
    metadata: UltrasoundStudyMetadata,
    quality: ImageQualityAssessment,
    ovary_volume_ml: float | None,
    ovary_dimensions_mm: tuple[float, float, float] | None,
    follicle_number_per_ovary: int | None,
    follicle_diameters_mm: list[float],
    instances: list[FollicleInstance],
    follicle_density_per_ml: float | None = None,
    ovary_confidence: float = 0.0,
    follicle_confidence: float = 0.0,
    false_follicle_voxels_outside_ovary: int = 0,
    large_structure_flag: bool = False,
    extra_warnings: list[str] | None = None,
) -> OvarianMorphologyOutput:
    """Assemble a 3D volume result — the only acquisition permitted a true count.

    A volume is the only acquisition where every follicle is imaged with known
    geometry in all three axes, so ``follicle_number_per_ovary`` is a count rather
    than an estimate, and ``ovary_volume_ml`` is measurable.
    """
    return OvarianMorphologyOutput(
        study_id=metadata.study_id,
        patient_id=metadata.patient_id,
        laterality=metadata.laterality,
        acquisition_mode="volume_3d",
        quality_score=quality.overall_quality_score,
        full_ovary_visible=quality.whole_ovary_visible,
        measurement_feasible=True,
        ovary_volume_ml=ovary_volume_ml,
        ovary_dimensions_mm=ovary_dimensions_mm,
        follicle_number_per_ovary=follicle_number_per_ovary,
        follicle_count_method=(
            "per_ovary" if follicle_number_per_ovary is not None else "not_assessed"
        ),
        follicle_diameters_mm=list(follicle_diameters_mm),
        follicle_density_per_ml=follicle_density_per_ml,
        instances=list(instances),
        frames_analyzed=None,
        tracking_coverage=None,
        ovary_mask_confidence=float(np.clip(ovary_confidence, 0.0, 1.0)),
        follicle_detection_confidence=float(np.clip(follicle_confidence, 0.0, 1.0)),
        false_follicle_voxels_outside_ovary=int(false_follicle_voxels_outside_ovary),
        large_structure_flag=large_structure_flag,
        warnings=sorted(set(_common_warnings(metadata, extra_warnings))),
        clinician_review_status="model_generated",
    )
