"""DICOM reading and de-identification for ovarian ultrasound.

``pydicom`` is an optional dependency: the whole ultrasound pipeline must be
testable on synthetic phantoms without it, so it is imported lazily and only
when an actual DICOM path is read.

De-identification is *fail-loud* by design. Ultrasound is the modality where PHI
most often survives anonymisation, because the identifying text is frequently
**burned into the pixel data** by the scanner rather than stored in a tag. A
pipeline that quietly proceeds on an identified study creates a disclosure
incident, so :func:`assert_deidentified` raises instead of warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from schemas.imaging import UltrasoundStudyMetadata

#: Tags whose presence means the study is still identifiable.
IDENTIFYING_TAGS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "InstitutionName",
)

#: Extra tags stripped by :func:`deidentify` even though they are not gating.
ADDITIONAL_STRIPPED_TAGS: tuple[str, ...] = (
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionAddress",
    "StationName",
    "AccessionNumber",
    "StudyID",
    "DeviceSerialNumber",
)

#: Set when the scanner has drawn text (often the patient name) into the pixels.
BURNED_IN_TAG = "BurnedInAnnotation"


class DeidentificationError(RuntimeError):
    """Raised when a study still carries identifying information."""


class DicomUnavailableError(RuntimeError):
    """Raised when a DICOM path is requested but ``pydicom`` is not installed."""


@dataclass(frozen=True)
class DicomLoadResult:
    """A loaded DICOM study: pixels plus the metadata we can trust."""

    array: np.ndarray
    metadata: UltrasoundStudyMetadata
    #: Schema ``AcquisitionMode``: what this study may claim downstream.
    acquisition_mode: str = "unknown"

    @property
    def kind(self) -> str:
        """Legacy alias for :attr:`acquisition_mode`."""
        return self.acquisition_mode


def _require_pydicom() -> Any:
    """Import ``pydicom`` lazily, with an actionable error if it is absent."""
    try:
        import pydicom  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DicomUnavailableError(
            "Reading DICOM requires the optional 'imaging' extra: pip install '.[imaging]'. "
            "Synthetic phantoms and array inputs work without it."
        ) from exc
    return pydicom


def present_identifying_tags(dataset: Any) -> list[str]:
    """Return the identifying tags that are present and non-empty.

    ``BurnedInAnnotation == 'YES'`` counts as identifying because pixel-level
    text cannot be removed by stripping tags; such a study must be rejected or
    routed to pixel redaction, never measured.
    """
    found: list[str] = []
    for tag in IDENTIFYING_TAGS:
        value = getattr(dataset, tag, None)
        if value not in (None, "", b""):
            found.append(tag)
    burned = getattr(dataset, BURNED_IN_TAG, None)
    if isinstance(burned, str) and burned.strip().upper() == "YES":
        found.append(BURNED_IN_TAG)
    return found


def is_deidentified(dataset: Any) -> bool:
    """True when no identifying tag and no burned-in annotation is present."""
    return not present_identifying_tags(dataset)


def assert_deidentified(dataset: Any, *, study_id: str = "<unknown>") -> None:
    """Raise :class:`DeidentificationError` if any identifier survives.

    Args:
        dataset: A ``pydicom`` dataset or any object exposing the tags as attrs.
        study_id: Identifier used in the error message.

    Raises:
        DeidentificationError: If identifying tags or burned-in text are present.
    """
    found = present_identifying_tags(dataset)
    if found:
        raise DeidentificationError(
            f"Study {study_id} is NOT de-identified: {', '.join(found)}. "
            "Refusing to process. Run deidentify() first; if BurnedInAnnotation is "
            "present the pixel data itself must be redacted before use."
        )


def deidentify(dataset: Any, *, drop_private_tags: bool = True) -> Any:
    """Strip identifying tags in place and return the dataset.

    Note that this cannot remove burned-in pixel text; ``BurnedInAnnotation`` is
    therefore left truthful rather than being reset to ``'NO'``, so downstream
    :func:`assert_deidentified` still refuses the study. Lying about redaction
    would be worse than failing.
    """
    for tag in (*IDENTIFYING_TAGS, *ADDITIONAL_STRIPPED_TAGS):
        if hasattr(dataset, tag):
            try:
                setattr(dataset, tag, "")
            except Exception:  # noqa: BLE001 - some tags are read-only in odd files
                continue
    if drop_private_tags and hasattr(dataset, "remove_private_tags"):
        dataset.remove_private_tags()
    return dataset


def _spacing_from(dataset: Any) -> tuple[float, float, float] | None:
    """Extract (z, y, x) spacing in mm, or ``None`` when it is not recorded.

    Returning ``None`` is load-bearing: every physical measurement downstream is
    gated on spacing being known, because a follicle diameter in pixels is
    clinically meaningless.
    """
    pixel_spacing = getattr(dataset, "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) < 2:
        return None
    try:
        row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    except (TypeError, ValueError):
        return None
    thickness = getattr(dataset, "SliceThickness", None) or getattr(
        dataset, "SpacingBetweenSlices", None
    )
    try:
        z_mm = float(thickness) if thickness is not None else float(row_mm)
    except (TypeError, ValueError):
        z_mm = float(row_mm)
    if min(row_mm, col_mm, z_mm) <= 0:
        return None
    return (z_mm, row_mm, col_mm)


def _laterality_from(dataset: Any) -> str:
    """Map DICOM laterality codes onto the schema's ``Laterality`` literal."""
    for tag in ("ImageLaterality", "Laterality", "FrameLaterality"):
        raw = getattr(dataset, tag, None)
        if not raw:
            continue
        code = str(raw).strip().upper()
        if code.startswith("L"):
            return "left"
        if code.startswith("R"):
            return "right"
    body = str(getattr(dataset, "BodyPartExamined", "") or "").upper()
    if "LEFT" in body:
        return "left"
    if "RIGHT" in body:
        return "right"
    return "unknown"


def _route_from(dataset: Any) -> str:
    """Infer transvaginal vs transabdominal from the free-text-ish tags.

    Route matters scientifically: transvaginal scanning resolves follicles that
    transabdominal scanning cannot, so a follicle count is only comparable to a
    published threshold when the route is known and matched.
    """
    haystack = " ".join(
        str(getattr(dataset, tag, "") or "")
        for tag in (
            "BodyPartExamined",
            "StudyDescription",
            "SeriesDescription",
            "ProtocolName",
            "TransducerType",
            "TransducerData",
        )
    ).upper()
    if any(k in haystack for k in ("TRANSVAGINAL", "ENDOVAGINAL", "TVUS", "TV US", "INTRACAVITY")):
        return "transvaginal"
    if any(k in haystack for k in ("TRANSABDOMINAL", "ABDOMINAL", "TAUS", "SUPRAPUBIC")):
        return "transabdominal"
    return "unknown"


#: Below this frame count a multi-frame object is treated as separate stills
#: rather than a continuous sweep. Mirrors ``loader.MIN_FRAMES_FOR_CINE``.
MIN_FRAMES_FOR_CINE = 5

#: SOP Class UIDs that identify an ultrasound *image* (single or multi-frame)
#: as opposed to a volumetric object. A multi-frame US image is a cine loop over
#: time; it carries no through-plane geometry whatsoever.
ULTRASOUND_IMAGE_SOP_CLASSES = frozenset(
    {
        "1.2.840.10008.5.1.4.1.1.6.1",  # Ultrasound Image Storage
        "1.2.840.10008.5.1.4.1.1.3.1",  # Ultrasound Multi-frame Image Storage
    }
)


def _classify(array: np.ndarray, dataset: Any) -> str:
    """Determine the acquisition mode from the pixel array and DICOM header.

    The decisive question is whether the frames have **through-plane geometry**.
    A multi-frame ultrasound object with no ``SpacingBetweenSlices`` and no
    ``SliceThickness`` is a cine loop over *time*: its frames are cross-sections
    at unknown, uncontrolled positions, so they do not reconstruct a volume and
    cannot support an ovarian volume or a true per-ovary follicle count.

    The default for the ambiguous case is the 2D interpretation. Calling a real
    volume a cine loop forfeits measurements; calling a sweep a volume fabricates
    them.
    """
    if array.ndim == 2:
        return "single_frame"
    if array.ndim < 2:
        return "unknown"

    n_frames = getattr(dataset, "NumberOfFrames", None)
    frames = int(n_frames) if n_frames is not None else int(array.shape[0])

    sop_class = str(getattr(dataset, "SOPClassUID", "") or "")
    has_slice_geometry = (
        getattr(dataset, "SpacingBetweenSlices", None) is not None
        or getattr(dataset, "SliceThickness", None) is not None
    )
    # An ultrasound image SOP class is never a volume, whatever tags it carries.
    if sop_class in ULTRASOUND_IMAGE_SOP_CLASSES or not has_slice_geometry:
        if frames <= 1:
            return "single_frame"
        return "cine_loop" if frames >= MIN_FRAMES_FOR_CINE else "multi_frame"
    return "volume_3d"


def _pixel_array(dataset: Any) -> np.ndarray:
    """Return the pixel array as float, collapsing any colour channel."""
    array = np.asarray(dataset.pixel_array)
    if array.ndim == 4 and array.shape[-1] in (3, 4):
        # RGB screen-capture ultrasound: use luminance, not an arbitrary channel.
        array = array[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    elif array.ndim == 3 and array.shape[-1] in (3, 4) and array.shape[0] not in (3, 4):
        array = array[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    return array.astype(float)


def read_dicom(
    path: Path | str,
    *,
    patient_id: str | None = None,
    study_id: str | None = None,
    require_deidentified: bool = True,
    source_dataset: str | None = None,
) -> DicomLoadResult:
    """Read one DICOM file into pixels plus :class:`UltrasoundStudyMetadata`.

    Args:
        path: Path to the DICOM file.
        patient_id: Dataset-scoped pseudonymous ID. Never taken from PatientID.
        study_id: Study identifier; defaults to the file stem.
        require_deidentified: When True (default) an identified study raises.
        source_dataset: Registry dataset id for provenance.

    Returns:
        A :class:`DicomLoadResult`.

    Raises:
        DeidentificationError: If identifiers are present and gating is on.
        DicomUnavailableError: If ``pydicom`` is not installed.
    """
    pydicom = _require_pydicom()
    path = Path(path)
    dataset = pydicom.dcmread(str(path))
    resolved_study = study_id or path.stem

    warnings: list[str] = []
    if require_deidentified:
        assert_deidentified(dataset, study_id=resolved_study)
        deidentified = True
    else:
        deidentified = is_deidentified(dataset)
        if not deidentified:
            warnings.append(
                "Study retains identifying DICOM tags; de-identification gate was bypassed."
            )

    array = _pixel_array(dataset)
    spacing = _spacing_from(dataset)
    if spacing is None:
        warnings.append("PixelSpacing absent: physical measurements will be withheld.")

    mode = _classify(array, dataset)
    if mode in ("single_frame", "multi_frame", "cine_loop"):
        warnings.append(
            f"Acquisition mode {mode}: this is a 2D acquisition, so no ovarian volume and no "
            "true per-ovary follicle count can be derived from it."
        )
    laterality = _laterality_from(dataset)
    if laterality == "unknown":
        warnings.append("Laterality unknown: left/right asymmetry cannot be computed.")
    route = _route_from(dataset)
    if route == "unknown":
        warnings.append("Acquisition route unknown: counts are not comparable across routes.")

    metadata = UltrasoundStudyMetadata(
        study_id=resolved_study,
        patient_id=patient_id or resolved_study,
        laterality=laterality,  # type: ignore[arg-type]
        route=route,  # type: ignore[arg-type]
        spacing_mm=spacing,
        shape=tuple(int(n) for n in array.shape),
        is_3d=mode == "volume_3d",
        deidentified=deidentified,
        source_dataset=source_dataset,
        warnings=[f"Acquisition mode: {mode}.", *warnings],
    )
    return DicomLoadResult(array=array, metadata=metadata, acquisition_mode=mode)


def read_dicom_series(
    directory: Path | str,
    *,
    patient_id: str | None = None,
    study_id: str | None = None,
    require_deidentified: bool = True,
    source_dataset: str | None = None,
) -> DicomLoadResult:
    """Stack a directory of single-frame DICOM slices into one 3D volume.

    Slices are ordered by ``ImagePositionPatient`` when available and by
    ``InstanceNumber`` otherwise; an unordered stack would scramble the
    through-plane axis and corrupt every volume measurement.
    """
    pydicom = _require_pydicom()
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No DICOM files in {directory}")

    slices: list[Any] = []
    for file in files:
        try:
            slices.append(pydicom.dcmread(str(file)))
        except Exception:  # noqa: BLE001 - non-DICOM sidecar files are common
            continue
    if not slices:
        raise FileNotFoundError(f"No readable DICOM files in {directory}")

    resolved_study = study_id or directory.name
    if require_deidentified:
        for dataset in slices:
            assert_deidentified(dataset, study_id=resolved_study)

    def _key(dataset: Any) -> float:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            return float(position[2])
        return float(getattr(dataset, "InstanceNumber", 0) or 0)

    slices.sort(key=_key)
    array = np.stack([_pixel_array(s) for s in slices], axis=0)

    first = slices[0]
    spacing = _spacing_from(first)
    if spacing is not None and len(slices) > 1:
        z_positions = [_key(s) for s in slices]
        deltas = np.abs(np.diff(z_positions))
        measured = float(np.median(deltas)) if deltas.size and np.median(deltas) > 0 else spacing[0]
        spacing = (measured, spacing[1], spacing[2])

    warnings: list[str] = ["Acquisition mode: volume_3d."]
    if spacing is None:
        warnings.append("PixelSpacing absent: physical measurements will be withheld.")

    # A stacked series of single-frame slices with recorded positions is a genuine
    # volume: the through-plane spacing was measured from ImagePositionPatient
    # above, so the geometry the 2D modes lack is present here.
    if len(slices) < 2:
        warnings.append(
            "Series contains a single slice; treated as one frame rather than a volume."
        )
        mode = "single_frame"
    else:
        mode = "volume_3d"

    metadata = UltrasoundStudyMetadata(
        study_id=resolved_study,
        patient_id=patient_id or resolved_study,
        laterality=_laterality_from(first),  # type: ignore[arg-type]
        route=_route_from(first),  # type: ignore[arg-type]
        spacing_mm=spacing,
        shape=tuple(int(n) for n in array.shape),
        is_3d=mode == "volume_3d",
        deidentified=True if require_deidentified else is_deidentified(first),
        source_dataset=source_dataset,
        warnings=warnings,
    )
    return DicomLoadResult(array=array, metadata=metadata, acquisition_mode=mode)
