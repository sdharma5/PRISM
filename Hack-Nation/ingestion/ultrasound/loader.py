"""Uniform entry point returning pixels, metadata and the **acquisition mode**.

Every downstream ultrasound component consumes exactly that triple, regardless of
whether the study arrived as DICOM, as a NumPy array with a sidecar, or as a
synthetic phantom. Keeping one shape of input means the quality gate and the
measurement code cannot accidentally be bypassed by a new file format.

The acquisition mode is not decoration — it is the field that decides which
measurements the study is permitted to claim downstream. A single frame may claim
a per-section follicle count; a cine loop may claim a tracked *estimate* of the
per-ovary count; only a genuine volume may claim a true per-ovary count and an
ovarian volume. Getting the mode wrong is therefore a correctness failure, not a
labelling nicety, so detection is explicit and its uncertainty is recorded as a
warning rather than resolved by a guess.

**The default for an ambiguous rank-3 array is a cine loop, not a volume.** 2D is
the routine clinical acquisition, and the two errors are not symmetric: calling a
real volume a cine loop merely forfeits measurements it could have supported,
while calling a freehand sweep a volume lets it claim an ovarian volume and a
true per-ovary count that nothing in the data supports.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ingestion.ultrasound.metadata import build_metadata, metadata_from_sidecar
from ingestion.ultrasound.validation import ValidationReport, validate_study
from schemas.imaging import UltrasoundStudyMetadata

LoadedStudy = tuple[np.ndarray, UltrasoundStudyMetadata]

DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}
ARRAY_SUFFIXES = {".npy", ".npz"}

#: Acquisition modes this loader can report, matching ``schemas.imaging``.
ACQUISITION_MODES = ("single_frame", "multi_frame", "cine_loop", "volume_3d", "unknown")

#: Below this many frames a stack is treated as separately captured stills
#: (``multi_frame``) rather than a continuous sweep (``cine_loop``). The
#: distinction matters for tracking: cross-frame follicle matching assumes small
#: inter-frame motion, which separately captured stills do not provide.
MIN_FRAMES_FOR_CINE = 5


@dataclass
class LoadedUltrasound:
    """Pixels, metadata and the acquisition mode that gates every measurement."""

    array: np.ndarray
    metadata: UltrasoundStudyMetadata
    acquisition_mode: str = "unknown"
    #: How the mode was determined, for provenance.
    mode_source: str = "inferred"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_2d_pathway(self) -> bool:
        """True when this study goes down the primary 2D path."""
        return self.acquisition_mode in ("single_frame", "multi_frame", "cine_loop")

    def as_tuple(self) -> LoadedStudy:
        """Backwards-compatible ``(array, metadata)`` view."""
        return self.array, self.metadata


def detect_acquisition_mode(
    array: np.ndarray,
    *,
    declared: str | None = None,
    is_3d_hint: bool = False,
    n_frames: int | None = None,
) -> tuple[str, str, list[str]]:
    """Determine how a study was acquired.

    Args:
        array: The pixel data.
        declared: An explicitly declared mode, which always wins. Declaring the
            mode is the only way to be certain, and callers who know should say so.
        is_3d_hint: Source-derived evidence that this is a true volume — DICOM
            slice geometry, or a caller asserting it.
        n_frames: Frame count from the source, when the array rank alone is
            ambiguous.

    Returns:
        ``(mode, source, warnings)``.
    """
    if declared is not None:
        if declared not in ACQUISITION_MODES:
            raise ValueError(
                f"Unknown acquisition_mode '{declared}'; expected one of {ACQUISITION_MODES}."
            )
        return declared, "declared", []

    data = np.asarray(array)
    if data.ndim == 2:
        return "single_frame", "array_rank", []
    if data.ndim < 2:
        return (
            "unknown",
            "array_rank",
            [f"Array of shape {data.shape} is not an image; acquisition mode is unknown."],
        )

    frames = int(n_frames if n_frames is not None else data.shape[0])
    if is_3d_hint:
        return "volume_3d", "source_geometry", []

    if frames < MIN_FRAMES_FOR_CINE:
        return (
            "multi_frame",
            "array_rank",
            [
                f"{frames} frames with no through-plane geometry: treated as separately captured "
                "stills. Cross-frame follicle tracking assumes small inter-frame motion, which "
                "separate stills do not guarantee, so the unique-count estimate is weaker."
            ],
        )
    return (
        "cine_loop",
        "array_rank",
        [
            f"{frames} frames with no through-plane geometry: treated as a 2D cine loop. "
            "No ovarian volume and no true per-ovary follicle count can be derived from it."
        ],
    )


def load_array(
    path: Path | str,
    *,
    patient_id: str | None = None,
    study_id: str | None = None,
    sidecar: dict[str, Any] | None = None,
    source_dataset: str | None = None,
    acquisition_mode: str | None = None,
) -> LoadedStudy:
    """Load a ``.npy``/``.npz`` array plus its JSON sidecar, if present.

    The sidecar is where spacing and the acquisition mode live for non-DICOM
    formats. If spacing is missing it stays ``None`` and every physical
    measurement downstream abstains rather than assuming isotropic 1 mm voxels.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as bundle:
            key = "volume" if "volume" in bundle else list(bundle.keys())[0]
            array = np.asarray(bundle[key], dtype=float)
    else:
        array = np.asarray(np.load(path), dtype=float)

    resolved_sidecar = dict(sidecar or {})
    sidecar_path = path.with_suffix(".json")
    if not resolved_sidecar and sidecar_path.exists():
        resolved_sidecar = json.loads(sidecar_path.read_text())

    mode, _source, mode_warnings = detect_acquisition_mode(
        array,
        declared=acquisition_mode or resolved_sidecar.get("acquisition_mode"),
        is_3d_hint=bool(resolved_sidecar.get("is_3d", False)),
    )
    metadata = metadata_from_sidecar(
        resolved_sidecar,
        study_id=study_id or path.stem,
        patient_id=patient_id or resolved_sidecar.get("patient_id", path.stem),
        shape=tuple(array.shape),
        source_dataset=source_dataset,
        acquisition_mode=mode,
        extra_warnings=mode_warnings,
    )
    return array, metadata


def load_ultrasound(
    source: Path | str | np.ndarray,
    *,
    patient_id: str | None = None,
    study_id: str | None = None,
    spacing_mm: tuple[float, float, float] | None = None,
    laterality: str = "unknown",
    route: str = "unknown",
    acquisition_mode: str | None = None,
    sidecar: dict[str, Any] | None = None,
    source_dataset: str | None = None,
    require_deidentified: bool = True,
) -> LoadedUltrasound:
    """Load any supported ultrasound source, reporting its acquisition mode.

    2D frames and cine loops are first-class inputs here, not a degraded case of
    a volume. A rank-2 array is a frame; a rank-3 array without through-plane
    geometry is a sweep; only explicit source geometry or an explicit declaration
    makes something a volume.

    Args:
        source: A DICOM file, a DICOM series directory, a ``.npy``/``.npz`` file,
            or an in-memory array (used by phantoms and tests).
        patient_id: Dataset-scoped pseudonymous identifier.
        study_id: Study identifier.
        spacing_mm: Spacing for in-memory arrays. ``None`` means genuinely unknown.
        laterality: Known laterality for in-memory arrays.
        route: Known acquisition route for in-memory arrays.
        acquisition_mode: Explicit mode declaration; always wins over inference.
        sidecar: Optional metadata dict for array sources.
        source_dataset: Registry dataset id.
        require_deidentified: Enforce the DICOM de-identification gate.

    Returns:
        A :class:`LoadedUltrasound`.
    """
    if isinstance(source, np.ndarray):
        array = np.asarray(source, dtype=float)
        declared = acquisition_mode or (sidecar or {}).get("acquisition_mode")
        mode, mode_source, mode_warnings = detect_acquisition_mode(
            array, declared=declared, is_3d_hint=False
        )
        if sidecar:
            metadata = metadata_from_sidecar(
                sidecar,
                study_id=study_id or "in_memory_study",
                patient_id=patient_id or "in_memory_patient",
                shape=tuple(array.shape),
                source_dataset=source_dataset,
                acquisition_mode=mode,
                extra_warnings=mode_warnings,
            )
        else:
            metadata = build_metadata(
                study_id=study_id or "in_memory_study",
                patient_id=patient_id or "in_memory_patient",
                shape=tuple(array.shape),
                spacing_mm=spacing_mm,
                laterality=laterality,
                route=route,
                is_3d=mode == "volume_3d",
                # In-memory arrays never carried DICOM headers, so there is
                # nothing identifying to strip.
                deidentified=True,
                source_dataset=source_dataset,
                acquisition_mode=mode,
                extra_warnings=mode_warnings,
            )
        return LoadedUltrasound(
            array=array,
            metadata=metadata,
            acquisition_mode=mode,
            mode_source=mode_source,
            warnings=mode_warnings,
        )

    path = Path(source)
    if path.is_dir():
        from ingestion.ultrasound.dicom import read_dicom_series  # noqa: PLC0415

        result = read_dicom_series(
            path,
            patient_id=patient_id,
            study_id=study_id,
            require_deidentified=require_deidentified,
            source_dataset=source_dataset,
        )
        return _from_dicom(result, acquisition_mode)

    if path.suffix.lower() in ARRAY_SUFFIXES:
        array, metadata = load_array(
            path,
            patient_id=patient_id,
            study_id=study_id,
            sidecar=sidecar,
            source_dataset=source_dataset,
            acquisition_mode=acquisition_mode,
        )
        mode, mode_source, mode_warnings = detect_acquisition_mode(
            array, declared=acquisition_mode, is_3d_hint=metadata.is_3d
        )
        return LoadedUltrasound(
            array=array,
            metadata=metadata,
            acquisition_mode=mode,
            mode_source=mode_source,
            warnings=mode_warnings,
        )

    if path.suffix.lower() in DICOM_SUFFIXES or path.suffix == "":
        from ingestion.ultrasound.dicom import read_dicom  # noqa: PLC0415

        result = read_dicom(
            path,
            patient_id=patient_id,
            study_id=study_id,
            require_deidentified=require_deidentified,
            source_dataset=source_dataset,
        )
        return _from_dicom(result, acquisition_mode)

    raise ValueError(f"Unsupported ultrasound source '{path}' (suffix {path.suffix!r}).")


def _from_dicom(result: Any, declared: str | None) -> LoadedUltrasound:
    """Wrap a DICOM read result, honouring an explicit mode declaration."""
    mode = declared or result.acquisition_mode
    return LoadedUltrasound(
        array=result.array,
        metadata=result.metadata,
        acquisition_mode=mode,
        mode_source="declared" if declared else "dicom_header",
        warnings=list(result.metadata.warnings),
    )


def load_study(
    source: Path | str | np.ndarray,
    **kwargs: Any,
) -> LoadedStudy:
    """Load a study as ``(array, metadata)``.

    A thin view over :func:`load_ultrasound` for callers that already know the
    acquisition mode or do not need it. Prefer :func:`load_ultrasound`: dropping
    the mode is what allows a downstream consumer to run the wrong measurement
    path on the right pixels.
    """
    return load_ultrasound(source, **kwargs).as_tuple()


def load_and_validate(
    source: Path | str | np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, UltrasoundStudyMetadata, ValidationReport]:
    """Load a study and immediately validate it, returning the report too."""
    require_spacing = bool(kwargs.pop("require_spacing", False))
    volume, metadata = load_study(source, **kwargs)
    report = validate_study(
        volume,
        metadata,
        require_deidentified=bool(kwargs.get("require_deidentified", True)),
        require_spacing=require_spacing,
    )
    return volume, metadata, report


def iter_directory(
    directory: Path | str,
    *,
    source_dataset: str | None = None,
    require_deidentified: bool = True,
) -> Iterator[LoadedUltrasound]:
    """Yield every loadable study in a directory, skipping unreadable files.

    Unreadable files are skipped rather than raising so that one corrupt study
    does not abort a cohort-scale preparation run; the caller counts them.
    """
    directory = Path(directory)
    if not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if path.suffix.lower() == ".json":
            continue
        try:
            yield load_ultrasound(
                path,
                source_dataset=source_dataset,
                require_deidentified=require_deidentified,
            )
        except Exception:  # noqa: BLE001 - one bad study must not stop the cohort
            continue
