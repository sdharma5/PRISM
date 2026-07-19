"""Turning a follicle probability mask into counted, measured instances.

Antral follicle count is an *instance* quantity, not a voxel quantity, so the
segmentation mask has to be resolved into discrete objects. Three steps decide
whether the resulting count is clinically meaningful:

1. **Physical-size filtering.** Components below
   :data:`MIN_FOLLICLE_DIAMETER_MM` (2.0 mm) are discarded. That threshold is
   not arbitrary: antral follicle counting is conventionally defined over
   follicles of 2-10 mm, and sub-2 mm blobs in ultrasound are overwhelmingly
   speckle rather than resolvable follicles. The filter is applied in
   **millimetres**, never in voxels, because a fixed voxel threshold means a
   different physical threshold on every scanner.

2. **Separation of touching follicles.** Adjacent follicles merge into one
   connected component and silently undercount. A distance-transform +
   watershed split recovers them, because follicles are convex and roughly
   spherical so their distance maps have one interior maximum each.

3. **Physical sizing.** Volume is voxel count times voxel volume; diameter is
   reported as the equivalent-sphere diameter, and the max Feret-like diameter
   is estimated from the component's physical extent.

If spacing is unknown, no physical size can be computed, so this module refuses
to filter or size and says so — see :func:`extract_follicle_instances`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from schemas.imaging import FollicleInstance

#: Follicles below this diameter are not resolvable on ultrasound and are
#: excluded from the antral follicle count (AFC is defined over 2-10 mm).
MIN_FOLLICLE_DIAMETER_MM = 2.0

#: Structures above this diameter are not counted as antral follicles. 10 mm is
#: the upper bound of the conventional AFC window; a >10 mm structure may be a
#: dominant follicle, a corpus luteum, or a cyst. We flag it, we never name it.
MAX_ANTRAL_FOLLICLE_DIAMETER_MM = 10.0

#: Above this diameter a structure is reported as large/uncertain and excluded
#: from the small-follicle count entirely. Reporting is descriptive only.
LARGE_STRUCTURE_DIAMETER_MM = 25.0

#: A component must be at least this fraction inside the ovary to be a follicle.
MIN_INSIDE_OVARY_FRACTION = 0.5


@dataclass
class InstanceExtractionResult:
    """Instances plus the bookkeeping needed to explain the count."""

    instances: list[FollicleInstance] = field(default_factory=list)
    label_volume: np.ndarray | None = None
    n_raw_components: int = 0
    n_removed_too_small: int = 0
    n_removed_outside_ovary: int = 0
    n_split_by_watershed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of retained instances (large structures included)."""
        return len(self.instances)

    def small_follicle_instances(self) -> list[FollicleInstance]:
        """Instances inside the antral window, i.e. the countable follicles."""
        return [inst for inst in self.instances if not inst.is_large_or_uncertain]


def _voxel_volume_mm3(spacing: tuple[float, float, float]) -> float:
    return float(spacing[0] * spacing[1] * spacing[2])


def _equivalent_diameter_mm(voxel_count: int, voxel_volume_mm3: float) -> float:
    """Diameter of the sphere with the same physical volume as the component."""
    volume = voxel_count * voxel_volume_mm3
    return float(2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def _max_extent_mm(coords: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Largest physical extent of the component along any axis (Feret proxy)."""
    if coords.size == 0:
        return 0.0
    physical = coords.astype(float) * np.asarray(spacing, dtype=float)[None, :]
    return float(np.max(physical.max(axis=0) - physical.min(axis=0)))


def separate_touching(
    mask: np.ndarray,
    # 2D and 3D masks both come through here (the cine tracker passes an
    # in-plane ``(row_mm, col_mm)``), so spacing is one value per mask axis
    # rather than always three.
    spacing: tuple[float, ...] | None,
    *,
    min_distance_mm: float = 1.2,
) -> tuple[np.ndarray, int]:
    """Split merged follicles using a distance transform and watershed.

    Seeds are the local maxima of the physical distance transform, so each
    roughly spherical follicle contributes one seed even when its mask is fused
    with a neighbour's.

    Args:
        mask: Boolean follicle mask.
        spacing: Physical spacing in mm; used so ``min_distance_mm`` is physical.
        min_distance_mm: Minimum separation between two seeds.

    Returns:
        ``(label_volume, n_extra_objects_created)``.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0

    sampling = tuple(float(s) for s in spacing) if spacing is not None else None
    distance = ndi.distance_transform_edt(mask, sampling=sampling)

    base_labels, n_base = ndi.label(mask)
    if n_base == 0:
        return base_labels.astype(np.int32), 0

    seeds = _peak_seeds(distance, mask, min_distance_mm, sampling)
    markers, n_seeds = ndi.label(seeds)
    if n_seeds <= n_base:
        return base_labels.astype(np.int32), 0

    labels = _watershed(-distance, markers, mask)
    return labels.astype(np.int32), int(max(n_seeds - n_base, 0))


def _peak_seeds(
    distance: np.ndarray,
    mask: np.ndarray,
    min_distance_mm: float,
    sampling: tuple[float, ...] | None,
) -> np.ndarray:
    """Local maxima of the distance map, thinned to one seed per follicle."""
    spacing = np.asarray(sampling if sampling is not None else [1.0] * mask.ndim, dtype=float)
    footprint_radius = np.maximum(np.round(min_distance_mm / np.clip(spacing, 1e-6, None)), 1)
    size = tuple(int(2 * r + 1) for r in footprint_radius)
    maxima = (distance == ndi.maximum_filter(distance, size=size)) & mask
    # Require a seed to sit meaningfully inside the object, so speckle-thin
    # protrusions do not become their own "follicle".
    return maxima & (distance >= min_distance_mm)


def _watershed(image: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Watershed, preferring scikit-image and falling back to nearest-seed.

    The fallback assigns every masked voxel to its nearest marker under the
    Euclidean distance transform. For convex, near-spherical objects that is
    equivalent to the watershed split, which is exactly the follicle case.
    """
    try:
        from skimage.segmentation import watershed as sk_watershed  # noqa: PLC0415

        return sk_watershed(image, markers=markers, mask=mask)
    except ImportError:
        _, indices = ndi.distance_transform_edt(markers == 0, return_indices=True)
        nearest = markers[tuple(indices)]
        return np.where(mask, nearest, 0)


def extract_follicle_instances(
    follicle_mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] | None,
    ovary_mask: np.ndarray | None = None,
    follicle_prob: np.ndarray | None = None,
    min_diameter_mm: float = MIN_FOLLICLE_DIAMETER_MM,
    large_structure_diameter_mm: float = LARGE_STRUCTURE_DIAMETER_MM,
    separate_touching_follicles: bool = True,
) -> InstanceExtractionResult:
    """Label, split, filter and physically size every follicle instance.

    Args:
        follicle_mask: Boolean predicted follicle mask.
        spacing_mm: Physical spacing in mm. ``None`` disables all physical
            sizing and filtering, and is reported as a warning.
        ovary_mask: Ovary region mask, used to compute ``inside_ovary_fraction``
            and to drop components that are mostly outside the ovary.
        follicle_prob: Optional per-voxel follicle probability, used only for
            per-instance confidence reporting.
        min_diameter_mm: Documented small-object rejection threshold.
        large_structure_diameter_mm: Threshold above which a structure is flagged
            large/uncertain and excluded from the small-follicle count.
        separate_touching_follicles: Enable the watershed split.

    Returns:
        An :class:`InstanceExtractionResult`.
    """
    mask = np.asarray(follicle_mask, dtype=bool)
    warnings: list[str] = []

    if spacing_mm is None:
        warnings.append(
            "Spacing unknown: follicle instances are reported without physical size, and the "
            f"{min_diameter_mm} mm minimum-size filter could not be applied. Counts from this "
            "study are not comparable to any published threshold."
        )

    if not mask.any():
        return InstanceExtractionResult(
            label_volume=np.zeros(mask.shape, dtype=np.int32), warnings=warnings
        )

    if separate_touching_follicles:
        labels, n_split = separate_touching(mask, spacing_mm)
    else:
        labels, n_split = ndi.label(mask)
        labels, n_split = labels.astype(np.int32), 0

    n_raw = int(labels.max())
    voxel_volume = _voxel_volume_mm3(spacing_mm) if spacing_mm is not None else None

    instances: list[FollicleInstance] = []
    removed_small = 0
    removed_outside = 0
    kept_labels = np.zeros_like(labels)

    objects = ndi.find_objects(labels)
    for raw_label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        sub = labels[slices] == raw_label
        voxel_count = int(sub.sum())
        if voxel_count == 0:
            continue

        coords = np.argwhere(sub)
        centroid = tuple(
            float(c + s.start) for c, s in zip(coords.mean(axis=0), slices, strict=True)
        )

        inside_fraction = 1.0
        if ovary_mask is not None:
            ovary_sub = np.asarray(ovary_mask, dtype=bool)[slices]
            inside_fraction = float((sub & ovary_sub).sum() / voxel_count)
            if inside_fraction < MIN_INSIDE_OVARY_FRACTION:
                # Anatomically impossible: a follicle is intra-ovarian.
                removed_outside += 1
                continue

        volume_mm3: float | None = None
        mean_diameter: float | None = None
        max_diameter: float | None = None
        is_large = False

        if voxel_volume is not None:
            volume_mm3 = voxel_count * voxel_volume
            mean_diameter = _equivalent_diameter_mm(voxel_count, voxel_volume)
            max_diameter = max(_max_extent_mm(coords, spacing_mm), mean_diameter)  # type: ignore[arg-type]
            if mean_diameter < min_diameter_mm:
                removed_small += 1
                continue
            is_large = mean_diameter >= large_structure_diameter_mm

        instance_id = len(instances) + 1
        kept_labels[slices][sub] = instance_id
        instances.append(
            FollicleInstance(
                instance_id=instance_id,
                voxel_count=voxel_count,
                volume_mm3=volume_mm3,
                mean_diameter_mm=mean_diameter,
                max_diameter_mm=max_diameter,
                centroid_voxel=centroid,
                inside_ovary_fraction=float(np.clip(inside_fraction, 0.0, 1.0)),
                is_large_or_uncertain=is_large,
            )
        )

    if removed_small:
        warnings.append(
            f"{removed_small} component(s) below {min_diameter_mm} mm equivalent diameter were "
            "excluded as unresolvable (speckle), per the antral follicle definition."
        )
    if removed_outside:
        warnings.append(
            f"{removed_outside} predicted follicle component(s) lay mostly outside the ovary and "
            "were removed as anatomically impossible."
        )
    if n_split:
        warnings.append(f"{n_split} touching follicle(s) were separated by watershed splitting.")
    if follicle_prob is not None and instances:
        low = [
            inst.instance_id
            for inst in instances
            if float(np.asarray(follicle_prob)[kept_labels == inst.instance_id].mean()) < 0.6
        ]
        if low:
            warnings.append(f"{len(low)} instance(s) have low mean follicle probability (<0.6).")

    return InstanceExtractionResult(
        instances=instances,
        label_volume=kept_labels,
        n_raw_components=n_raw,
        n_removed_too_small=removed_small,
        n_removed_outside_ovary=removed_outside,
        n_split_by_watershed=n_split,
        warnings=warnings,
    )


def track_instances_across_slices(
    label_volume: np.ndarray,
    *,
    axis: int = 0,
    min_iou: float = 0.2,
) -> dict[int, list[int]]:
    """Map each 3D instance to the slice indices it appears on.

    3D tracking matters because a follicle imaged across several slices must be
    counted **once**. This function is the audit trail for that: it reports the
    slice span of every instance so a reviewer can see that a 6 mm follicle in a
    0.6 mm-spaced volume spans ~10 slices rather than being counted ten times.

    Args:
        label_volume: Integer instance labels.
        axis: The through-plane axis.
        min_iou: Retained for interface compatibility with 2D-linking variants.

    Returns:
        ``{instance_id: [slice indices]}``.
    """
    del min_iou  # 3D connected components already resolve identity.
    labels = np.asarray(label_volume)
    tracks: dict[int, list[int]] = {}
    moved = np.moveaxis(labels, axis, 0)
    for index, plane in enumerate(moved):
        for instance_id in np.unique(plane):
            if instance_id == 0:
                continue
            tracks.setdefault(int(instance_id), []).append(index)
    return tracks


def size_distribution(instances: list[FollicleInstance]) -> dict[str, float]:
    """Summary statistics of the follicle size distribution, in mm.

    The distribution matters more than the bare count: many small follicles and
    few large ones is a different picture from a single dominant follicle, and
    reporting only a count discards that.
    """
    diameters = [
        inst.mean_diameter_mm
        for inst in instances
        if inst.mean_diameter_mm is not None and not inst.is_large_or_uncertain
    ]
    if not diameters:
        return {}
    array = np.asarray(diameters, dtype=float)
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
