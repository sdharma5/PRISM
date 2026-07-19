"""Structural validation of ultrasound inputs before any model touches them.

These checks are cheap and catch the failure modes that silently produce
plausible-looking but wrong measurements: implausible spacing (a decimal-point
error turns a 6 mm follicle into a 60 mm cyst), degenerate constant volumes,
non-finite voxels, and studies that are not de-identified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from schemas.imaging import UltrasoundStudyMetadata

#: Plausible ultrasound voxel spacing in mm. Outside this range a spacing value
#: is far more likely to be a unit error (cm vs mm) than a real acquisition.
MIN_PLAUSIBLE_SPACING_MM = 0.02
MAX_PLAUSIBLE_SPACING_MM = 5.0

#: Volumes smaller than this on any axis cannot contain a resolvable ovary.
MIN_AXIS_VOXELS = 8


@dataclass
class ValidationReport:
    """Outcome of validating one study."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        """Raise :class:`UltrasoundValidationError` when validation failed."""
        if not self.ok:
            raise UltrasoundValidationError("; ".join(self.errors))


class UltrasoundValidationError(ValueError):
    """Raised when a study cannot be safely processed."""


def validate_spacing(spacing: tuple[float, float, float] | None) -> tuple[list[str], list[str]]:
    """Check spacing plausibility. Returns ``(errors, warnings)``."""
    errors: list[str] = []
    warnings: list[str] = []
    if spacing is None:
        warnings.append("spacing_unknown: no physical measurement may be emitted.")
        return errors, warnings
    if len(spacing) != 3:
        errors.append(f"spacing must have 3 components, got {len(spacing)}.")
        return errors, warnings
    for axis, value in enumerate(spacing):
        if not np.isfinite(value) or value <= 0:
            errors.append(f"spacing[{axis}]={value} is not a positive finite number.")
        elif value < MIN_PLAUSIBLE_SPACING_MM or value > MAX_PLAUSIBLE_SPACING_MM:
            errors.append(
                f"spacing[{axis}]={value} mm is outside the plausible range "
                f"[{MIN_PLAUSIBLE_SPACING_MM}, {MAX_PLAUSIBLE_SPACING_MM}] mm; "
                "this is usually a cm/mm unit error."
            )
    if not errors:
        ratio = max(spacing) / min(spacing)
        if ratio > 8.0:
            warnings.append(
                f"Highly anisotropic voxels (ratio {ratio:.1f}); through-plane "
                "follicle sizing will be unreliable."
            )
    return errors, warnings


def validate_volume(volume: np.ndarray) -> tuple[list[str], list[str]]:
    """Check the pixel array itself. Returns ``(errors, warnings)``."""
    errors: list[str] = []
    warnings: list[str] = []
    array = np.asarray(volume)
    if array.size == 0:
        errors.append("Volume is empty.")
        return errors, warnings
    if not np.all(np.isfinite(array)):
        errors.append("Volume contains non-finite voxels (NaN or inf).")
    if array.ndim < 2:
        errors.append(f"Volume rank {array.ndim} is below 2.")
    elif any(n < MIN_AXIS_VOXELS for n in array.shape):
        warnings.append(
            f"Volume shape {tuple(array.shape)} has an axis below {MIN_AXIS_VOXELS} voxels; "
            "an ovary is unlikely to be resolvable."
        )
    if array.size and float(np.nanstd(array)) < 1e-8:
        errors.append("Volume is constant; no anatomy present.")
    return errors, warnings


def validate_study(
    volume: np.ndarray,
    metadata: UltrasoundStudyMetadata,
    *,
    require_deidentified: bool = True,
    require_spacing: bool = False,
) -> ValidationReport:
    """Validate a loaded study end to end.

    Args:
        volume: The pixel array.
        metadata: Study metadata.
        require_deidentified: Treat a non-de-identified study as a hard error.
        require_spacing: Treat unknown spacing as a hard error. Off by default,
            because an unknown-spacing study is still usable for qualitative
            quality assessment — it just may not emit millimetres.

    Returns:
        A :class:`ValidationReport`.
    """
    errors: list[str] = []
    warnings: list[str] = list(metadata.warnings)

    vol_errors, vol_warnings = validate_volume(volume)
    errors.extend(vol_errors)
    warnings.extend(vol_warnings)

    sp_errors, sp_warnings = validate_spacing(metadata.spacing_mm)
    errors.extend(sp_errors)
    warnings.extend(sp_warnings)

    if require_spacing and metadata.spacing_mm is None:
        errors.append("Spacing is required by configuration but is unknown.")

    if require_deidentified and not metadata.deidentified:
        errors.append(
            f"Study {metadata.study_id} is not marked de-identified; refusing to process."
        )

    if metadata.shape is not None and tuple(metadata.shape) != tuple(np.asarray(volume).shape):
        warnings.append(
            f"metadata.shape {tuple(metadata.shape)} disagrees with array shape "
            f"{tuple(np.asarray(volume).shape)}."
        )

    return ValidationReport(ok=not errors, errors=errors, warnings=sorted(set(warnings)))
