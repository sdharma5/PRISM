"""Ultrasound inputs, quality gating, and ovarian morphology outputs.

Every measurement here is model-generated until a clinician confirms it. The
``clinician_review_status`` field is the contract that keeps that distinction
visible downstream.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Laterality = Literal["left", "right", "unknown"]
Route = Literal["transvaginal", "transabdominal", "unknown"]
ReviewStatus = Literal[
    "model_generated", "awaiting_clinician_review", "clinician_confirmed", "rejected"
]

#: How the study was acquired. 2D transvaginal imaging is the routine clinical
#: input; a dedicated 3D volume is an optional enhanced mode, not the norm.
AcquisitionMode = Literal[
    "single_frame",
    "multi_frame",
    "cine_loop",
    "volume_3d",
    "unknown",
]

#: Which follicle-counting quantity a number represents. These are not
#: interchangeable and the method must travel with the count.
FollicleCountMethod = Literal[
    "per_section",
    "estimated_per_ovary",
    "per_ovary",
    "not_assessed",
]


class UltrasoundStudyMetadata(BaseModel):
    study_id: str
    patient_id: str
    laterality: Laterality = "unknown"
    route: Route = "unknown"
    #: Physical spacing in mm per voxel (x, y, z). Required for real measurements.
    spacing_mm: tuple[float, float, float] | None = None
    shape: tuple[int, ...] | None = None
    is_3d: bool = False
    deidentified: bool = False
    source_dataset: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ImageQualityAssessment(BaseModel):
    """Gate that decides whether quantitative measurement is even attempted."""

    ovary_visible: bool = False
    whole_ovary_visible: bool = False
    laterality_available: bool = False
    pixel_spacing_available: bool = False
    follicle_counting_feasible: bool = False
    ovarian_volume_feasible: bool = False
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

    @property
    def measurement_feasible(self) -> bool:
        return (
            self.ovary_visible and self.pixel_spacing_available and self.overall_quality_score > 0.5
        )


class FollicleInstance(BaseModel):
    instance_id: int
    voxel_count: int
    volume_mm3: float | None = None
    mean_diameter_mm: float | None = None
    max_diameter_mm: float | None = None
    centroid_voxel: tuple[float, ...] | None = None
    inside_ovary_fraction: float = 1.0
    is_large_or_uncertain: bool = False


class OvarianMorphologyOutput(BaseModel):
    """Semantic ovarian measurements produced by the imaging module.

    Follicle counting is split into three **non-interchangeable** quantities,
    because the acquisition determines which one is even obtainable:

    ``follicle_number_per_section``
        Follicles visible in one cross-section. A single 2D frame can support
        this and nothing more.
    ``estimated_follicle_number_per_ovary``
        Unique follicles estimated by tracking across cine-loop frames. An
        estimate with tracking error, not a census.
    ``follicle_number_per_ovary``
        A true per-ovary count. Only a full 3D volume (or a genuinely complete
        sweep) can support this.

    The 2023 international guideline treats per-section and per-ovary counts as
    distinct, and permits per-section counting precisely *because* complete
    counting is often unreliable. Collapsing them into one integer would let a
    single still frame silently claim a whole-ovary count.
    """

    study_id: str
    patient_id: str
    laterality: Laterality = "unknown"
    acquisition_mode: AcquisitionMode = "unknown"

    quality_score: float = Field(ge=0.0, le=1.0)
    full_ovary_visible: bool = False
    measurement_feasible: bool = False

    ovary_volume_ml: float | None = None
    ovary_area_mm2: float | None = None
    ovary_dimensions_mm: tuple[float, float, float] | None = None

    follicle_number_per_section: int | None = None
    estimated_follicle_number_per_ovary: int | None = None
    follicle_number_per_ovary: int | None = None
    follicle_count_method: FollicleCountMethod = "not_assessed"

    follicle_diameters_mm: list[float] = Field(default_factory=list)
    follicle_density_per_ml: float | None = None
    instances: list[FollicleInstance] = Field(default_factory=list)

    #: Cine-loop tracking quality. Low coverage means frames were skipped or
    #: tracking broke, so the unique-count estimate is correspondingly weaker.
    frames_analyzed: int | None = None
    tracking_coverage: float | None = Field(default=None, ge=0.0, le=1.0)

    ovary_mask_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    follicle_detection_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    false_follicle_voxels_outside_ovary: int = 0

    large_structure_flag: bool = False
    warnings: list[str] = Field(default_factory=list)

    model_version: str = "0.1.0"
    clinician_review_status: ReviewStatus = "model_generated"

    @model_validator(mode="after")
    def _check_count_matches_acquisition(self) -> OvarianMorphologyOutput:
        """A count may only be claimed if the acquisition can support it."""
        if self.acquisition_mode == "single_frame":
            if self.estimated_follicle_number_per_ovary is not None:
                raise ValueError(
                    "A single frame cannot support an estimated per-ovary follicle count. "
                    "Report follicle_number_per_section instead."
                )
            if self.follicle_number_per_ovary is not None:
                raise ValueError("A single frame cannot support a true per-ovary follicle count.")

        if self.acquisition_mode in {"cine_loop", "multi_frame"} and (
            self.follicle_number_per_ovary is not None
        ):
            raise ValueError(
                "2D frames cannot support a true per-ovary count — that requires a complete "
                "volumetric acquisition. Use estimated_follicle_number_per_ovary."
            )

        # Ovarian volume needs three dimensions; a 2D cross-section has two.
        if self.acquisition_mode == "single_frame" and self.ovary_volume_ml is not None:
            raise ValueError(
                "Ovarian volume cannot be computed from a single cross-section. "
                "Report ovary_area_mm2 instead."
            )

        declared = {
            "per_section": self.follicle_number_per_section,
            "estimated_per_ovary": self.estimated_follicle_number_per_ovary,
            "per_ovary": self.follicle_number_per_ovary,
        }
        if (
            self.follicle_count_method != "not_assessed"
            and declared[self.follicle_count_method] is None
        ):
            raise ValueError(
                f"follicle_count_method='{self.follicle_count_method}' but that count is None."
            )
        return self

    @property
    def is_clinically_confirmed(self) -> bool:
        return self.clinician_review_status == "clinician_confirmed"

    @property
    def reportable_follicle_count(self) -> tuple[int | None, FollicleCountMethod]:
        """The best available count *with* the method that produced it.

        Callers must never unpack only the integer — the method is what makes
        the number interpretable.
        """
        if self.follicle_number_per_ovary is not None:
            return self.follicle_number_per_ovary, "per_ovary"
        if self.estimated_follicle_number_per_ovary is not None:
            return self.estimated_follicle_number_per_ovary, "estimated_per_ovary"
        if self.follicle_number_per_section is not None:
            return self.follicle_number_per_section, "per_section"
        return None, "not_assessed"
