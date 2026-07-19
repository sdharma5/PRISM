"""Construction and merging of :class:`UltrasoundStudyMetadata`.

Metadata for ultrasound is not bookkeeping — it is what decides whether a
measurement may be emitted at all. Spacing determines whether a millimetre
figure exists; laterality determines whether left/right asymmetry is meaningful;
route determines whether a follicle count is comparable to any published
threshold. Each of those gaps is recorded as an explicit warning rather than
being defaulted away.
"""

from __future__ import annotations

from typing import Any

from schemas.imaging import UltrasoundStudyMetadata

#: Sidecar keys accepted when spacing is supplied outside DICOM (e.g. NIfTI/JSON).
SPACING_KEYS: tuple[str, ...] = ("spacing_mm", "spacing", "voxel_size_mm", "pixel_spacing_mm")

#: Modes for which a 2-element in-plane spacing is sufficient and correct.
_TWO_D_MODES = frozenset({"single_frame", "multi_frame", "cine_loop"})


def _coerce_spacing(value: Any, *, allow_2d: bool = False) -> tuple[float, float, float] | None:
    """Coerce a sidecar spacing entry to a 3-tuple, or ``None`` if unusable.

    A 2-element ``(row_mm, col_mm)`` in-plane spacing is complete information for a
    2D acquisition and is accepted when ``allow_2d`` is set, by repeating the row
    spacing into the unused through-plane slot. The schema's ``spacing_mm`` is a
    3-tuple because it was written for volumes; the 2D measurement code reads only
    the last two elements (see
    :func:`models.ultrasound.morphology_2d.in_plane_spacing_mm`) and no 2D path can
    reach a volumetric computation, because the schema validator forbids a volume
    for every 2D acquisition mode.

    For a 3D acquisition a 2-element spacing is still refused: there the
    through-plane spacing is genuinely unknown and genuinely needed.
    """
    if value is None:
        return None
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if len(values) == 2:
        if not allow_2d or min(values) <= 0:
            return None
        return (values[0], values[0], values[1])
    if len(values) != 3 or min(values) <= 0:
        return None
    return (values[0], values[1], values[2])


def build_metadata(
    *,
    study_id: str,
    patient_id: str,
    shape: tuple[int, ...],
    spacing_mm: tuple[float, float, float] | None = None,
    laterality: str = "unknown",
    route: str = "unknown",
    is_3d: bool | None = None,
    deidentified: bool = False,
    source_dataset: str | None = None,
    acquisition_mode: str | None = None,
    extra_warnings: list[str] | None = None,
) -> UltrasoundStudyMetadata:
    """Assemble study metadata, attaching a warning for every missing gate field.

    Args:
        study_id: Study identifier.
        patient_id: Dataset-scoped pseudonymous patient identifier.
        shape: Array shape in voxels or pixels.
        spacing_mm: Physical spacing in mm, or ``None`` when unknown.
        laterality: ``"left"``, ``"right"`` or ``"unknown"``.
        route: ``"transvaginal"``, ``"transabdominal"`` or ``"unknown"``.
        is_3d: Override for 3D detection; derived from ``acquisition_mode`` or
            from rank when omitted.
        deidentified: Whether the de-identification gate has passed.
        source_dataset: Registry dataset id.
        acquisition_mode: The detected acquisition mode. Recorded as a warning so
            it survives on the metadata, which has no dedicated field for it.
        extra_warnings: Additional warnings to carry.

    Returns:
        A validated :class:`UltrasoundStudyMetadata`.
    """
    warnings = list(extra_warnings or [])
    if acquisition_mode is not None:
        warnings.append(f"Acquisition mode: {acquisition_mode}.")
        if acquisition_mode in _TWO_D_MODES:
            warnings.append(
                "2D acquisition: no ovarian volume and no true per-ovary follicle count can be "
                "derived from it, however many frames it contains."
            )
    if spacing_mm is None:
        warnings.append("PixelSpacing unknown: physical measurements will be withheld.")
    if laterality == "unknown":
        warnings.append("Laterality unknown: left/right asymmetry cannot be computed.")
    if route == "unknown":
        warnings.append(
            "Acquisition route unknown: follicle counts are not comparable across "
            "transvaginal and transabdominal scanning."
        )
    if not deidentified:
        warnings.append("De-identification not confirmed for this study.")

    if is_3d is None and acquisition_mode is not None:
        is_3d = acquisition_mode == "volume_3d"

    return UltrasoundStudyMetadata(
        study_id=study_id,
        patient_id=patient_id,
        laterality=laterality,  # type: ignore[arg-type]
        route=route,  # type: ignore[arg-type]
        spacing_mm=spacing_mm,
        shape=tuple(int(n) for n in shape),
        is_3d=bool(len(shape) >= 3) if is_3d is None else bool(is_3d),
        deidentified=deidentified,
        source_dataset=source_dataset,
        warnings=sorted(set(warnings)),
    )


def metadata_from_sidecar(
    sidecar: dict[str, Any],
    *,
    study_id: str,
    patient_id: str,
    shape: tuple[int, ...],
    source_dataset: str | None = None,
    acquisition_mode: str | None = None,
    extra_warnings: list[str] | None = None,
) -> UltrasoundStudyMetadata:
    """Build metadata from a JSON/YAML sidecar accompanying a non-DICOM study."""
    allow_2d = acquisition_mode in _TWO_D_MODES
    spacing: tuple[float, float, float] | None = None
    for key in SPACING_KEYS:
        spacing = _coerce_spacing(sidecar.get(key), allow_2d=allow_2d)
        if spacing is not None:
            break
    return build_metadata(
        study_id=str(sidecar.get("study_id", study_id)),
        patient_id=str(sidecar.get("patient_id", patient_id)),
        shape=shape,
        spacing_mm=spacing,
        laterality=str(sidecar.get("laterality", "unknown")),
        route=str(sidecar.get("route", "unknown")),
        deidentified=bool(sidecar.get("deidentified", False)),
        source_dataset=source_dataset or sidecar.get("source_dataset"),
        acquisition_mode=acquisition_mode,
        extra_warnings=extra_warnings,
    )


def merge_lateral_metadata(
    left: UltrasoundStudyMetadata, right: UltrasoundStudyMetadata
) -> UltrasoundStudyMetadata:
    """Combine a left and a right study for one participant into one record.

    Spacing is kept only when both sides agree, because a per-patient summary
    computed from two different physical scales would be meaningless.
    """
    if left.patient_id != right.patient_id:
        raise ValueError("Refusing to merge studies from different patients.")
    spacing = left.spacing_mm if left.spacing_mm == right.spacing_mm else None
    warnings = sorted(set(left.warnings) | set(right.warnings))
    if spacing is None and left.spacing_mm is not None:
        warnings.append(
            "Left and right studies have different spacing; not merged for measurement."
        )
    return UltrasoundStudyMetadata(
        study_id=f"{left.study_id}+{right.study_id}",
        patient_id=left.patient_id,
        laterality="unknown",
        route=left.route if left.route == right.route else "unknown",
        spacing_mm=spacing,
        shape=left.shape,
        is_3d=left.is_3d and right.is_3d,
        deidentified=left.deidentified and right.deidentified,
        source_dataset=left.source_dataset,
        warnings=sorted(set(warnings)),
    )
