"""Ultrasound preprocessing that is safe for *counting* and *measuring*.

The governing constraint of this module is that ovarian ultrasound is used to
produce **counts** (antral follicle count) and **physical sizes** (follicle
diameter, ovarian volume). Most of the standard imaging-augmentation toolbox is
therefore unusable here, because those transforms change the very quantity being
measured.

FORBIDDEN TRANSFORMS (enforced by :func:`assert_transforms_allowed`)
--------------------------------------------------------------------
* ``elastic_deformation`` / ``random_warp`` / ``grid_distortion`` — locally
  stretch anatomy, changing follicle diameters and potentially splitting or
  merging follicles, i.e. changing the count.
* ``random_erasing`` / ``cutout`` / ``random_crop_of_anatomy`` — can delete an
  entire follicle, so the label (a count) is no longer the label of the image.
* ``mixup`` / ``cutmix`` / ``copy_paste`` — blend two ovaries; the resulting
  count is undefined and the resulting volume corresponds to no real person.
* ``anisotropic_scaling`` / ``random_zoom`` / ``random_resized_crop`` — change
  physical size unless spacing is updated in lockstep, which augmentation
  libraries do not do.
* ``shear`` / ``perspective`` — do not preserve volume or diameter.
* ``histogram_matching_across_scanners`` — can invert the stroma/lumen contrast
  that follicle detection depends on.
* ``super_resolution`` / ``denoise_hallucination`` — generative models can
  synthesise follicles that were never imaged.

ALLOWED: intensity normalization, symmetric crop/pad (padding never removes
anatomy), rigid flips/90-degree rotations (isometries preserve counts and
volumes), and spacing-aware resampling that *records* the original spacing so
measurement always happens in the original physical frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FORBIDDEN_TRANSFORMS: dict[str, str] = {
    "elastic_deformation": "Locally stretches anatomy; changes follicle diameter and count.",
    "random_warp": "Locally stretches anatomy; changes follicle diameter and count.",
    "grid_distortion": "Locally stretches anatomy; changes follicle diameter and count.",
    "random_erasing": "Can delete a follicle, invalidating the count label.",
    "cutout": "Can delete a follicle, invalidating the count label.",
    "random_crop_of_anatomy": "Can cut off part of the ovary, invalidating volume and count.",
    "mixup": "Blends two ovaries; the resulting count and volume are undefined.",
    "cutmix": "Blends two ovaries; the resulting count and volume are undefined.",
    "copy_paste": "Fabricates follicles that were never imaged.",
    "anisotropic_scaling": "Changes physical size without updating spacing.",
    "random_zoom": "Changes physical size without updating spacing.",
    "random_resized_crop": "Changes physical size and can remove anatomy.",
    "shear": "Not an isometry; does not preserve diameter or volume.",
    "perspective": "Not an isometry; does not preserve diameter or volume.",
    "histogram_matching_across_scanners": "Can invert stroma/lumen contrast.",
    "super_resolution": "Generative; can hallucinate follicles.",
    "denoise_hallucination": "Generative; can hallucinate follicles.",
}

ALLOWED_TRANSFORMS: tuple[str, ...] = (
    "intensity_normalize",
    "clip_percentile",
    "crop_or_pad",
    "flip",
    "rot90",
    "resample_to_spacing",
)


class ForbiddenTransformError(ValueError):
    """Raised when a configured transform would change anatomical counts."""


def assert_transforms_allowed(transforms: list[str] | tuple[str, ...]) -> None:
    """Fail loudly if a configured transform is count-destroying.

    This is called from the config-driven entry points so that a forbidden
    augmentation cannot be enabled by editing YAML.

    Args:
        transforms: Transform names from a configuration file.

    Raises:
        ForbiddenTransformError: On any forbidden or unrecognised transform.
    """
    for name in transforms:
        key = str(name).strip().lower()
        if key in FORBIDDEN_TRANSFORMS:
            raise ForbiddenTransformError(
                f"Transform '{key}' is forbidden for ovarian ultrasound: "
                f"{FORBIDDEN_TRANSFORMS[key]}"
            )
        if key not in ALLOWED_TRANSFORMS:
            raise ForbiddenTransformError(
                f"Transform '{key}' is not on the allow-list {ALLOWED_TRANSFORMS}. "
                "Unknown transforms are rejected because their effect on counts is unproven."
            )


@dataclass
class PreprocessResult:
    """A preprocessed volume plus everything measurement needs afterwards."""

    volume: np.ndarray
    #: Spacing of ``volume`` after any resampling.
    spacing_mm: tuple[float, float, float] | None
    #: Spacing of the volume as acquired. Measurement uses THIS.
    original_spacing_mm: tuple[float, float, float] | None
    applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def measurement_spacing_mm(self) -> tuple[float, float, float] | None:
        """Spacing that physical measurements must use: the original one."""
        return self.original_spacing_mm


def normalize_intensity(
    volume: np.ndarray,
    *,
    mode: str = "percentile",
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Scale intensities to roughly [0, 1] per volume (per study).

    Normalization is done *per volume* rather than with a dataset-wide constant
    because ultrasound gain, depth and TGC settings are operator-controlled, so
    absolute intensity carries no cross-study meaning. Percentile clipping is the
    default because a single bright specular reflection would otherwise compress
    the entire stroma/lumen contrast that follicle detection relies on.

    Args:
        volume: Input array of any shape.
        mode: ``"percentile"``, ``"minmax"``, or ``"zscore"``.
        low: Lower percentile for ``"percentile"`` mode.
        high: Upper percentile for ``"percentile"`` mode.

    Returns:
        Float array of the same shape.
    """
    array = np.asarray(volume, dtype=float)
    if mode == "zscore":
        sd = float(array.std())
        return (array - float(array.mean())) / (sd if sd > 1e-8 else 1.0)
    if mode == "minmax":
        lo, hi = float(array.min()), float(array.max())
    elif mode == "percentile":
        lo, hi = (float(v) for v in np.percentile(array, [low, high]))
    else:
        raise ValueError(f"Unknown normalization mode '{mode}'.")
    if hi - lo < 1e-8:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def crop_or_pad(
    volume: np.ndarray,
    target_shape: tuple[int, ...],
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Centre crop or pad to ``target_shape``, symmetrically on every axis.

    Padding is preferred over cropping wherever possible because padding cannot
    remove anatomy. When cropping is unavoidable it is centred, on the assumption
    (checked by the quality head, not here) that the ovary is roughly centred in
    the acquisition.

    Args:
        volume: Input array.
        target_shape: Desired shape; must have the same rank as ``volume``.
        pad_value: Fill value for padded regions.

    Returns:
        Array with exactly ``target_shape``.
    """
    array = np.asarray(volume, dtype=float)
    if len(target_shape) != array.ndim:
        raise ValueError(f"target_shape rank {len(target_shape)} != volume rank {array.ndim}.")

    slices: list[slice] = []
    for size, target in zip(array.shape, target_shape, strict=True):
        if size > target:
            start = (size - target) // 2
            slices.append(slice(start, start + target))
        else:
            slices.append(slice(0, size))
    array = array[tuple(slices)]

    pads: list[tuple[int, int]] = []
    for size, target in zip(array.shape, target_shape, strict=True):
        missing = max(target - size, 0)
        before = missing // 2
        pads.append((before, missing - before))
    if any(sum(p) for p in pads):
        array = np.pad(array, pads, mode="constant", constant_values=pad_value)
    return array


def _as_triple(spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalise a 3-spacing to plain floats without losing its fixed length."""
    z, y, x = spacing
    return (float(z), float(y), float(x))


def resample_to_spacing(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float] | None,
    target_spacing: tuple[float, float, float],
    *,
    order: int = 1,
) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Resample to isotropic-ish target spacing, ONLY when spacing is known.

    If ``source_spacing`` is ``None`` the volume is returned untouched. Resampling
    an unknown-spacing volume would invent a physical scale, and every downstream
    millimetre figure would then be fabricated.

    Args:
        volume: Input array.
        source_spacing: Acquired spacing in mm, or ``None`` if unknown.
        target_spacing: Desired spacing in mm.
        order: Interpolation order (1 = linear; use 0 for label masks).

    Returns:
        ``(resampled_volume, new_spacing)``. ``new_spacing`` is ``None`` when no
        resampling was performed because the source spacing was unknown.
    """
    array = np.asarray(volume, dtype=float)
    if source_spacing is None:
        return array, None
    zoom_factors = tuple(
        float(s) / float(t) for s, t in zip(source_spacing, target_spacing, strict=True)
    )
    if all(abs(f - 1.0) < 1e-6 for f in zoom_factors):
        return array, _as_triple(target_spacing)
    try:
        from scipy.ndimage import zoom as ndzoom  # noqa: PLC0415

        resampled = ndzoom(array, zoom_factors, order=order)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        resampled = array
        return resampled, _as_triple(source_spacing)
    return resampled, _as_triple(target_spacing)


def preprocess_volume(
    volume: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] | None,
    target_shape: tuple[int, ...] | None = None,
    target_spacing_mm: tuple[float, float, float] | None = None,
    normalization: str = "percentile",
    transforms: list[str] | tuple[str, ...] | None = None,
) -> PreprocessResult:
    """Run the allowed preprocessing chain and preserve the original spacing.

    The returned :class:`PreprocessResult` deliberately carries *two* spacings.
    Networks consume ``volume``/``spacing_mm``; measurement code consumes
    ``original_spacing_mm``. Conflating them is how resampling silently corrupts
    reported ovarian volumes.

    Args:
        volume: Raw array.
        spacing_mm: Acquired spacing in mm, or ``None`` when unknown.
        target_shape: Optional crop/pad target.
        target_spacing_mm: Optional resample target; ignored when spacing unknown.
        normalization: Mode passed to :func:`normalize_intensity`.
        transforms: Optional explicit transform list, validated against the
            allow-list before anything runs.

    Returns:
        A :class:`PreprocessResult`.
    """
    if transforms is not None:
        assert_transforms_allowed(transforms)

    applied: list[str] = []
    warnings: list[str] = []
    original_spacing: tuple[float, float, float] | None = None
    if spacing_mm is not None:
        if len(spacing_mm) != 3:
            raise ValueError(f"spacing_mm must have three entries, got {len(spacing_mm)}.")
        sz, sy, sx = spacing_mm
        original_spacing = (float(sz), float(sy), float(sx))
    array = np.asarray(volume, dtype=float)

    array = normalize_intensity(array, mode=normalization)
    applied.append(f"intensity_normalize:{normalization}")

    current_spacing = original_spacing
    if target_spacing_mm is not None:
        if original_spacing is None:
            warnings.append(
                "Resampling skipped: spacing unknown. Physical measurements will be withheld."
            )
        else:
            array, current_spacing = resample_to_spacing(array, original_spacing, target_spacing_mm)
            applied.append(f"resample_to_spacing:{target_spacing_mm}")

    if target_shape is not None:
        array = crop_or_pad(array, tuple(target_shape))
        applied.append(f"crop_or_pad:{tuple(target_shape)}")

    if original_spacing is None:
        warnings.append("Original spacing unknown: only unitless features are valid.")

    return PreprocessResult(
        volume=array,
        spacing_mm=current_spacing,
        original_spacing_mm=original_spacing,
        applied=applied,
        warnings=warnings,
    )
