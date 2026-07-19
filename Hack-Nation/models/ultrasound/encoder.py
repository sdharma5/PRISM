"""The ovarian-ultrasound encoder: frames in, :class:`ModalityToken` out.

This is the only public entry point for the imaging modality. It dispatches on
the **acquisition mode** and runs the pathway that acquisition can actually
support:

===================  ==========================================================
``single_frame``     preprocess -> 2D segment -> frame QC -> per-section count
``multi_frame`` /    preprocess -> 2D segment per frame -> cine QC -> tracking
``cine_loop``        -> estimated unique count
``volume_3d``        preprocess -> 3D segment -> volumetric QC -> instances ->
                     true per-ovary count and ovarian volume
===================  ==========================================================

2D is the default. :meth:`UltrasoundEncoder.encode` infers the mode from the
array rank when the caller does not declare one, and a rank-2 array is a frame,
not a degenerate volume.

Every token this module emits carries the warning **"Clinician confirmation
pending"**. Ultrasound is the modality where an automated number most easily
reads as a finding, so the disclaimer travels *with the data* rather than living
in documentation that a downstream consumer may never read.

The exported embedding names each follicle count by the **method** that produced
it. There is deliberately no generic ``follicle_count`` slot: a per-section count
and a per-ovary count occupying the same vector position is exactly the confusion
this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ingestion.ultrasound.preprocessing import preprocess_volume
from models.ultrasound.cine_tracking import TrackingResult
from models.ultrasound.follicle_instances import (
    InstanceExtractionResult,
    extract_follicle_instances,
    track_instances_across_slices,
)
from models.ultrasound.morphology_2d import (
    CineMorphology2D,
    FrameMorphology2D,
    compute_cine_morphology,
    compute_frame_morphology,
    in_plane_spacing_mm,
)
from models.ultrasound.morphology_3d import (
    compute_morphology,
    count_follicle_voxels_outside_ovary,
    largest_component_fraction,
    morphology_summary,
)
from models.ultrasound.output_schema import (
    CLINICIAN_PENDING_WARNING,
    build_cine_output,
    build_single_frame_output,
)
from models.ultrasound.qc_2d import assess_cine_quality, assess_frame_quality
from models.ultrasound.quality import assess_quality
from models.ultrasound.segmenter_2d import (
    SegmentationOutput,
    ThresholdSegmenter2D,
    UNet2D,
    build_segmenter_2d,
)
from models.ultrasound.segmenter_3d import ThresholdSegmenter, UNet3D, build_segmenter
from schemas.imaging import ImageQualityAssessment, OvarianMorphologyOutput, UltrasoundStudyMetadata
from schemas.modality_token import ModalityToken
from schemas.model_output import ModelCardMetadata

__all__ = [
    "CLINICIAN_PENDING_WARNING",
    "EMBEDDING_FIELDS",
    "ThresholdSegmenter2D",
    "UltrasoundEncoder",
    "UltrasoundEncoding",
    "encode_study",
    "infer_acquisition_mode",
]

#: Fixed order of the exported embedding, so the vector is interpretable.
#: The three counts occupy three distinct slots, never one shared slot.
EMBEDDING_FIELDS: tuple[str, ...] = (
    "follicle_number_per_section",
    "estimated_follicle_number_per_ovary",
    "follicle_number_per_ovary",
    "ovary_area_mm2",
    "ovary_volume_ml",
    "follicle_density_per_ml",
    "mean_diameter_mm",
    "median_diameter_mm",
    "sd_diameter_mm",
    "max_diameter_mm",
    "n_2_to_9_mm",
    "n_ge_9_mm",
    "tracking_coverage",
    "ovary_mask_confidence",
    "follicle_detection_confidence",
    "quality_score",
)

#: Schema fields that must be listed in ``missing_fields`` when unmeasured, rather
#: than zero-filled. A zero ovarian volume and an unmeasured one are different
#: claims, and only one of them is ever true here.
_NULLABLE_MEASUREMENTS: tuple[str, ...] = (
    "ovary_volume_ml",
    "ovary_area_mm2",
    "follicle_number_per_section",
    "estimated_follicle_number_per_ovary",
    "follicle_number_per_ovary",
    "follicle_density_per_ml",
)


def infer_acquisition_mode(array: np.ndarray, metadata: UltrasoundStudyMetadata) -> str:
    """Infer the acquisition mode from the array rank and metadata.

    A rank-2 array is a single frame. A rank-3 array is ambiguous — it is either a
    stack of 2D frames or a volume — and is resolved by ``metadata.is_3d``, which
    the loader sets from the source. The default for an ambiguous rank-3 array is
    ``cine_loop`` rather than ``volume_3d``, because 2D is the routine acquisition
    and mis-classifying a sweep as a volume would let it claim a true per-ovary
    count and an ovarian volume it cannot support.

    Args:
        array: The pixel data.
        metadata: Study metadata carrying the ``is_3d`` flag.

    Returns:
        One of the schema's ``AcquisitionMode`` values.
    """
    data = np.asarray(array)
    if data.ndim == 2:
        return "single_frame"
    if data.ndim >= 3:
        return "volume_3d" if metadata.is_3d else "cine_loop"
    raise ValueError(f"Cannot infer an acquisition mode from an array of shape {data.shape}.")


@dataclass
class UltrasoundEncoding:
    """Everything one study produced, for callers that want the intermediates.

    Which fields are populated depends on the acquisition mode; the ones that do
    not apply stay ``None`` rather than being filled with an empty stand-in.
    """

    token: ModalityToken
    morphology: OvarianMorphologyOutput
    quality: ImageQualityAssessment
    acquisition_mode: str = "unknown"
    #: 3D path only.
    segmentation: SegmentationOutput | None = None
    instances: InstanceExtractionResult | None = None
    slice_tracks: dict[int, list[int]] = field(default_factory=dict)
    #: 2D paths.
    frame_segmentations: list[SegmentationOutput] = field(default_factory=list)
    frame_morphology: FrameMorphology2D | None = None
    cine_morphology: CineMorphology2D | None = None
    per_frame_quality: list[ImageQualityAssessment] = field(default_factory=list)

    @property
    def tracking(self) -> TrackingResult | None:
        """The cine tracking result, when this was a cine/multi-frame study."""
        return self.cine_morphology.tracking if self.cine_morphology is not None else None


class UltrasoundEncoder:
    """Encode an ovarian ultrasound study into a ``ModalityToken``.

    The encoder is 2D-first: constructing one with defaults gives a 2D segmenter
    and the 2D measurement path. The 3D segmenter is built lazily, only if a
    volume is actually encoded, so the optional pathway costs nothing when unused.
    """

    model_name = "ovarian_ultrasound_encoder"
    model_version = "0.2.0"
    modality = "ovarian_ultrasound"

    def __init__(
        self,
        *,
        segmenter_2d: Any | None = None,
        segmenter_3d: Any | None = None,
        segmenter_kind: str = "auto",
        target_spacing_mm: tuple[float, float, float] | None = None,
        normalization: str = "percentile",
        min_diameter_mm: float | None = None,
        large_structure_diameter_mm: float | None = None,
        tracking_kwargs: dict[str, Any] | None = None,
        segmenter: Any | None = None,
    ) -> None:
        """
        Args:
            segmenter_2d: Prebuilt 2D segmenter; overrides ``segmenter_kind``.
            segmenter_3d: Prebuilt 3D segmenter; overrides ``segmenter_kind``.
            segmenter_kind: ``"auto"``, ``"threshold"``, ``"unet2d"``/``"unet3d"``.
            target_spacing_mm: Optional resample target for the 3D path. Ignored
                when the study has no known spacing, since resampling would
                invent a scale.
            normalization: Intensity normalization mode.
            min_diameter_mm: Override the documented small-object threshold.
            large_structure_diameter_mm: Override the large-structure threshold.
            tracking_kwargs: Overrides forwarded to the cine tracker.
            segmenter: Backwards-compatible alias applied to the 3D path.
        """
        self.segmenter_kind = segmenter_kind
        self._segmenter_2d = segmenter_2d
        self._segmenter_3d = segmenter_3d if segmenter_3d is not None else segmenter
        self.target_spacing_mm = target_spacing_mm
        self.normalization = normalization
        self.min_diameter_mm = min_diameter_mm
        self.large_structure_diameter_mm = large_structure_diameter_mm
        self.tracking_kwargs = dict(tracking_kwargs or {})

    # -- segmenters (built lazily) -----------------------------------------

    @property
    def segmenter_2d(self) -> Any:
        """The 2D segmenter — the primary pathway."""
        if self._segmenter_2d is None:
            kind = "threshold" if self.segmenter_kind == "threshold" else self.segmenter_kind
            self._segmenter_2d = build_segmenter_2d(
                "unet2d" if kind == "unet3d" else kind  # a 3D request still needs a 2D frame model
            )
        return self._segmenter_2d

    @property
    def segmenter_3d(self) -> Any:
        """The 3D segmenter — built only if a volume is actually encoded."""
        if self._segmenter_3d is None:
            self._segmenter_3d = build_segmenter(self.segmenter_kind)
        return self._segmenter_3d

    def _instance_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.min_diameter_mm is not None:
            kwargs["min_diameter_mm"] = self.min_diameter_mm
        if self.large_structure_diameter_mm is not None:
            kwargs["large_structure_diameter_mm"] = self.large_structure_diameter_mm
        return kwargs

    # -- main entry point --------------------------------------------------

    def encode(
        self,
        image: np.ndarray,
        metadata: UltrasoundStudyMetadata,
        *,
        acquisition_mode: str | None = None,
        observed_at: str | None = None,
    ) -> UltrasoundEncoding:
        """Run the pathway matching this study's acquisition.

        Args:
            image: A 2D frame, a ``(T, H, W)`` frame stack, or a 3D volume.
            metadata: Study metadata carrying spacing, laterality and route.
            acquisition_mode: Declared mode. Inferred from rank and
                ``metadata.is_3d`` when omitted.
            observed_at: ISO timestamp recorded on the token.

        Returns:
            An :class:`UltrasoundEncoding`.
        """
        mode = acquisition_mode or infer_acquisition_mode(image, metadata)
        if mode == "single_frame":
            return self.encode_frame(image, metadata, observed_at=observed_at)
        if mode in ("cine_loop", "multi_frame"):
            return self.encode_cine(image, metadata, acquisition_mode=mode, observed_at=observed_at)
        if mode == "volume_3d":
            return self.encode_volume(image, metadata, observed_at=observed_at)
        raise ValueError(
            f"Unsupported acquisition_mode '{mode}'. Refusing to guess which measurements the "
            "acquisition can support."
        )

    # -- 2D single frame (primary) -----------------------------------------

    def encode_frame(
        self,
        frame: np.ndarray,
        metadata: UltrasoundStudyMetadata,
        *,
        observed_at: str | None = None,
    ) -> UltrasoundEncoding:
        """Encode one 2D frame. Produces a per-section count and nothing more."""
        array = np.asarray(frame, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"encode_frame expects a 2D frame, got shape {array.shape}.")

        prepared = preprocess_volume(
            array, spacing_mm=None, target_spacing_mm=None, normalization=self.normalization
        )
        segmentation = self.segmenter_2d.segment(prepared.volume)
        quality = assess_frame_quality(
            prepared.volume,
            metadata,
            ovary_mask=segmentation.ovary_mask,
            follicle_mask=segmentation.follicle_mask,
            ovary_confidence=segmentation.ovary_confidence,
        )
        morphology_2d = compute_frame_morphology(
            metadata=metadata,
            quality=quality,
            ovary_region_mask=segmentation.ovary_region_mask,
            follicle_mask=segmentation.follicle_mask,
            **self._instance_kwargs(),
        )
        outside = count_follicle_voxels_outside_ovary(
            segmentation.follicle_mask, segmentation.ovary_region_mask
        )
        morphology = build_single_frame_output(
            metadata=metadata,
            quality=quality,
            frame=morphology_2d,
            ovary_confidence=segmentation.ovary_confidence,
            follicle_confidence=segmentation.follicle_confidence,
            false_follicle_pixels_outside_ovary=outside,
            extra_warnings=prepared.warnings,
        )
        token = self.build_token(
            morphology,
            quality,
            metadata,
            fragmentation=largest_component_fraction(segmentation.ovary_region_mask),
            preprocessing_warnings=prepared.warnings,
            observed_at=observed_at,
        )
        return UltrasoundEncoding(
            token=token,
            morphology=morphology,
            quality=quality,
            acquisition_mode="single_frame",
            segmentation=segmentation,
            frame_segmentations=[segmentation],
            frame_morphology=morphology_2d,
            per_frame_quality=[quality],
        )

    # -- 2D cine loop / multi-frame (primary) ------------------------------

    def encode_cine(
        self,
        frames: np.ndarray | list[np.ndarray],
        metadata: UltrasoundStudyMetadata,
        *,
        acquisition_mode: str = "cine_loop",
        observed_at: str | None = None,
    ) -> UltrasoundEncoding:
        """Encode a frame stack: per-frame measurement plus cross-frame tracking.

        The tracked unique count is an estimate, never a census. See
        :mod:`models.ultrasound.cine_tracking` for the failure modes that make it
        one.
        """
        stack = [np.asarray(f, dtype=float) for f in frames]
        if any(f.ndim != 2 for f in stack):
            raise ValueError("encode_cine expects a stack of 2D frames.")

        prepared = [
            preprocess_volume(
                f, spacing_mm=None, target_spacing_mm=None, normalization=self.normalization
            )
            for f in stack
        ]
        segmentations = [self.segmenter_2d.segment(p.volume) for p in prepared]

        cine_quality = assess_cine_quality(
            [p.volume for p in prepared],
            metadata,
            ovary_masks=[s.ovary_mask for s in segmentations],
            follicle_masks=[s.follicle_mask for s in segmentations],
            ovary_confidences=[s.ovary_confidence for s in segmentations],
        )
        cine = compute_cine_morphology(
            metadata=metadata,
            per_frame_quality=cine_quality.per_frame,
            ovary_region_masks=[s.ovary_region_mask for s in segmentations],
            follicle_masks=[s.follicle_mask for s in segmentations],
            **self._instance_kwargs(),
            **self.tracking_kwargs,
        )
        outside = int(
            sum(
                count_follicle_voxels_outside_ovary(s.follicle_mask, s.ovary_region_mask)
                for s in segmentations
            )
        )
        usable = cine_quality.usable_frame_indices
        morphology = build_cine_output(
            metadata=metadata,
            quality=cine_quality.aggregate,
            cine=cine,
            acquisition_mode=acquisition_mode,
            ovary_confidence=(
                float(np.mean([segmentations[i].ovary_confidence for i in usable]))
                if usable
                else 0.0
            ),
            follicle_confidence=(
                float(np.mean([segmentations[i].follicle_confidence for i in usable]))
                if usable
                else 0.0
            ),
            false_follicle_pixels_outside_ovary=outside,
            extra_warnings=[*cine_quality.warnings, *prepared[0].warnings] if prepared else [],
        )
        token = self.build_token(
            morphology,
            cine_quality.aggregate,
            metadata,
            fragmentation=(
                float(
                    np.mean(
                        [
                            largest_component_fraction(segmentations[i].ovary_region_mask)
                            for i in usable
                        ]
                    )
                )
                if usable
                else 1.0
            ),
            preprocessing_warnings=cine_quality.warnings,
            observed_at=observed_at,
        )
        return UltrasoundEncoding(
            token=token,
            morphology=morphology,
            quality=cine_quality.aggregate,
            acquisition_mode=acquisition_mode,
            frame_segmentations=segmentations,
            cine_morphology=cine,
            per_frame_quality=cine_quality.per_frame,
        )

    # -- 3D volume (optional enhanced mode) --------------------------------

    def encode_volume(
        self,
        volume: np.ndarray,
        metadata: UltrasoundStudyMetadata,
        *,
        observed_at: str | None = None,
    ) -> UltrasoundEncoding:
        """Encode a 3D volume: the only path that may report a true per-ovary count."""
        prepared = preprocess_volume(
            volume,
            spacing_mm=metadata.spacing_mm,
            target_spacing_mm=self.target_spacing_mm,
            normalization=self.normalization,
        )

        segmentation = self.segmenter_3d.segment(prepared.volume)
        region = segmentation.ovary_region_mask

        quality = assess_quality(
            prepared.volume,
            metadata,
            ovary_mask=segmentation.ovary_mask,
            follicle_mask=segmentation.follicle_mask,
            ovary_confidence=segmentation.ovary_confidence,
        )

        # Measurement always uses the ORIGINAL acquisition spacing.
        instances = extract_follicle_instances(
            segmentation.follicle_mask,
            spacing_mm=prepared.measurement_spacing_mm,
            ovary_mask=region,
            follicle_prob=segmentation.probs[2],
            **self._instance_kwargs(),
        )

        outside = count_follicle_voxels_outside_ovary(segmentation.follicle_mask, region)
        morphology = compute_morphology(
            metadata=metadata,
            quality=quality,
            ovary_region_mask=region,
            instance_result=instances,
            ovary_confidence=segmentation.ovary_confidence,
            follicle_confidence=segmentation.follicle_confidence,
            false_follicle_voxels_outside_ovary=outside,
            **(
                {"large_structure_diameter_mm": self.large_structure_diameter_mm}
                if self.large_structure_diameter_mm is not None
                else {}
            ),
        )

        tracks = (
            track_instances_across_slices(instances.label_volume)
            if instances.label_volume is not None and instances.label_volume.ndim >= 3
            else {}
        )

        token = self.build_token(
            morphology,
            quality,
            metadata,
            fragmentation=largest_component_fraction(region),
            preprocessing_warnings=prepared.warnings,
            observed_at=observed_at,
        )
        return UltrasoundEncoding(
            token=token,
            morphology=morphology,
            quality=quality,
            acquisition_mode="volume_3d",
            segmentation=segmentation,
            instances=instances,
            slice_tracks=tracks,
        )

    # -- token assembly ----------------------------------------------------

    def build_token(
        self,
        morphology: OvarianMorphologyOutput,
        quality: ImageQualityAssessment,
        metadata: UltrasoundStudyMetadata,
        *,
        fragmentation: float = 1.0,
        preprocessing_warnings: list[str] | None = None,
        observed_at: str | None = None,
    ) -> ModalityToken:
        """Package morphology into a :class:`ModalityToken`.

        Missing measurements become entries in ``missing_fields`` rather than
        zeros. A zero ovarian volume and an unmeasured ovarian volume are
        completely different claims, and only one of them is true here.

        The count is exported alongside ``follicle_count_method``, and the
        structured features carry all three count fields under their own names.
        A consumer that reads a number without reading the method has read an
        uninterpretable number.
        """
        summary = morphology_summary(morphology)
        embedding = [float(summary.get(field_name, 0.0)) for field_name in EMBEDDING_FIELDS]

        missing = [name for name in _NULLABLE_MEASUREMENTS if getattr(morphology, name) is None]
        if metadata.spacing_mm is None:
            missing.append("spacing_mm")
        if metadata.laterality == "unknown":
            missing.append("laterality")
        if metadata.route == "unknown":
            missing.append("acquisition_route")

        count, method = morphology.reportable_follicle_count
        structured: dict[str, Any] = {
            "laterality": metadata.laterality,
            "acquisition_route": metadata.route,
            "acquisition_mode": morphology.acquisition_mode,
            "measurement_feasible": morphology.measurement_feasible,
            "full_ovary_visible": morphology.full_ovary_visible,
            "large_structure_flag": morphology.large_structure_flag,
            "clinician_review_status": morphology.clinician_review_status,
            "ovary_volume_ml": morphology.ovary_volume_ml,
            "ovary_area_mm2": morphology.ovary_area_mm2,
            "follicle_number_per_section": morphology.follicle_number_per_section,
            "estimated_follicle_number_per_ovary": (morphology.estimated_follicle_number_per_ovary),
            "follicle_number_per_ovary": morphology.follicle_number_per_ovary,
            "reportable_follicle_count": count,
            "follicle_count_method": method,
            "follicle_density_per_ml": morphology.follicle_density_per_ml,
            "frames_analyzed": morphology.frames_analyzed,
            "tracking_coverage": morphology.tracking_coverage,
            "false_follicle_voxels_outside_ovary": morphology.false_follicle_voxels_outside_ovary,
            "ovary_mask_largest_component_fraction": float(fragmentation),
        }
        for key in ("mean_diameter_mm", "median_diameter_mm", "max_diameter_mm", "n_2_to_9_mm"):
            if key in summary:
                structured[f"follicle_{key}"] = float(summary[key])

        warnings = [
            CLINICIAN_PENDING_WARNING,
            "Model-generated imaging measurement; not a diagnosis and not clinically validated.",
            *morphology.warnings,
            *(preprocessing_warnings or []),
        ]
        if fragmentation < 0.8 and morphology.measurement_feasible:
            warnings.append(
                f"Ovary mask is fragmented (largest component holds {fragmentation:.0%} of "
                "pixels); the reported measurement may be unreliable."
            )

        confidence = (
            float(np.clip(morphology.ovary_mask_confidence, 0.0, 1.0))
            if morphology.measurement_feasible
            else 0.0
        )

        return ModalityToken(
            patient_id=morphology.patient_id,
            modality="ovarian_ultrasound",
            embedding=embedding,
            structured_features=structured,
            quality_score=float(np.clip(quality.overall_quality_score, 0.0, 1.0)),
            confidence_score=confidence,
            observed_at=observed_at,
            model_version=self.model_version,
            source_dataset=metadata.source_dataset,
            provenance_ids=[metadata.study_id],
            missing_fields=sorted(set(missing)),
            warnings=sorted(set(warnings)),
        )

    # -- model card --------------------------------------------------------

    def export_model_card_metadata(self) -> ModelCardMetadata:
        """Model card describing what this encoder may and may not be used for."""
        backend_2d = "unet2d" if isinstance(self._segmenter_2d, UNet2D) else "threshold_heuristic"
        backend_3d = "unet3d" if isinstance(self._segmenter_3d, UNet3D) else "threshold_heuristic"
        return ModelCardMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            intended_use=(
                "Research-only extraction of descriptive ovarian morphology from ovarian "
                "ultrasound. The primary input is 2D transvaginal imaging: a single frame yields "
                "a per-section follicle count and cross-sectional area; a cine loop additionally "
                "yields an ESTIMATED per-ovary count via cross-frame tracking. A 3D volume is an "
                "optional enhanced mode and is the only acquisition that yields a true per-ovary "
                f"count and an ovarian volume. Backends: 2D {backend_2d}, 3D {backend_3d}. All "
                "outputs are model-generated and require clinician confirmation."
            ),
            out_of_scope_uses=[
                "Any diagnostic use, including polycystic ovarian morphology determination.",
                "Clinical decision-making of any kind.",
                "Measurement on studies without known physical pixel spacing.",
                "Naming or characterising cystic structures as any specific pathology.",
                "Comparison of follicle counts across different acquisition routes.",
                "Comparing or averaging a per-section count with a per-ovary count; they are "
                "different quantities and the method must travel with every number.",
                "Reporting a cine-tracked unique count as a complete antral follicle count.",
            ],
            limitations=[
                "Validated only on synthetic phantoms in this repository.",
                "Follicle counts are route-dependent and not comparable across scanners.",
                "Partially imaged ovaries systematically under-report area and count.",
                "The torch-free fallback segmenter is a heuristic, not a trained model.",
                "Cine tracking cannot detect a probe sweep that re-images the same plane, so a "
                "back-and-forth sweep over-counts; and it cannot detect an incomplete sweep, so "
                "a partial sweep under-counts. Neither is visible in tracking_coverage.",
                "A single frame supports no per-ovary quantity of any kind.",
                "2D pretraining slices derived from 3D volumes are not an independent test set.",
            ],
            ethical_considerations=[
                "Imaging measurements can be over-read as findings; every output is labelled "
                "model_generated and carries 'Clinician confirmation pending'.",
                "Studies retaining identifying DICOM tags or burned-in annotation are refused.",
                "A per-section count reported without its method could be mistaken for an antral "
                "follicle count, which would materially misrepresent the ovary.",
            ],
        )


def encode_study(
    image: np.ndarray,
    metadata: UltrasoundStudyMetadata,
    **kwargs: Any,
) -> UltrasoundEncoding:
    """Convenience one-shot encode using default settings."""
    call_keys = {"observed_at", "acquisition_mode"}
    encoder_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k not in call_keys}
    return UltrasoundEncoder(**encoder_kwargs).encode(image, metadata, **kwargs)


# Re-exported so callers keep a single import site for the 2D spacing accessor.
__all__ = [*__all__, "ThresholdSegmenter", "in_plane_spacing_mm"]
