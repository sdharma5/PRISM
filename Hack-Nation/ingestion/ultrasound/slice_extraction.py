"""Extract labelled 2D slices from 3D volumes, for 2D pretraining.

Why this exists
---------------
The clinical input is a 2D transvaginal frame, so the model that matters is a 2D
model. But labelled 2D ovarian ultrasound with *individual follicle* annotations
is almost impossible to obtain publicly, while USOVA3D — a 3D dataset — carries
exactly those labels: expert ovary boundaries **and** individual follicle
instances. USOVA3D is therefore valuable here as a **label resource**, not as a
model of the deployment input.

This module bridges the two: it cuts a labelled volume into labelled 2D slices,
each with its ovary and follicle masks, so a 2D segmenter can be pretrained on
real annotated ovarian appearance before being fine-tuned on real 2D scans.

**The evaluation caveat, stated plainly.** Slices extracted from 3D volumes are
legitimate for learning *appearance* — the stroma/lumen contrast, the speckle
statistics, the shape of a follicle cross-section are all genuine. They are **not**
a substitute for an independent 2D test set, for three reasons:

1. **Slices from one volume are not independent samples.** Adjacent slices share
   most of their anatomy. A train/test split that puts slice 40 in training and
   slice 41 in test is measuring memorisation, not generalisation. :func:`extract_slices`
   therefore never splits within a volume, and :func:`assign_slice_splits` splits
   by *subject*.
2. **A reconstructed 3D volume is not a native 2D frame.** Volume reconstruction
   applies interpolation and its own filtering; the speckle texture and the
   resolution anisotropy of a resliced plane differ systematically from a live
   B-mode frame. A model can exploit that difference.
3. **The acquisition geometry differs.** A 3D probe sweep and a freehand 2D sweep
   sample the ovary at different, differently distributed angles.

A test set should ideally come from an **independent 2D clinical source**.
Reporting a 2D result measured only on USOVA3D-derived slices, without that
caveat, would overstate what has been demonstrated. Every function here returns
provenance recording that the slices are volume-derived, so the caveat survives
into the artifacts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "VOLUME_DERIVED_PROVENANCE",
    "ExtractedSlice",
    "SliceExtractionSummary",
    "assign_slice_splits",
    "extract_slices",
    "iter_slices",
]

#: Stamped on every extracted slice so its origin cannot be lost downstream.
VOLUME_DERIVED_PROVENANCE = (
    "2D slice extracted from a 3D volume. Legitimate for learning appearance; NOT an "
    "independent 2D test sample. Evaluate on a native 2D clinical source."
)

#: A slice whose ovary covers less than this fraction of the frame carries too
#: little anatomy to be a useful training example — it is the very edge of the
#: organ, where the cross-section is a sliver.
MIN_OVARY_FRACTION_FOR_SLICE = 0.005


@dataclass
class ExtractedSlice:
    """One labelled 2D training example derived from a volume."""

    image: np.ndarray
    ovary_mask: np.ndarray
    follicle_mask: np.ndarray
    #: In-plane ``(row_mm, col_mm)`` for this slice's plane.
    pixel_spacing_mm: tuple[float, float] | None
    subject_id: str
    volume_id: str
    slice_index: int
    axis: int = 0
    n_follicles_in_section: int = 0
    ovary_fraction: float = 0.0
    provenance: str = VOLUME_DERIVED_PROVENANCE

    @property
    def follicle_number_per_section(self) -> int:
        """Ground-truth per-section count for this slice.

        Named per-section, never ``follicle_count``: a slice's count is a
        cross-sectional quantity and is not the volume's per-ovary count.
        """
        return self.n_follicles_in_section


@dataclass
class SliceExtractionSummary:
    """What an extraction run produced, and what it must not be used for."""

    n_volumes: int = 0
    n_slices_kept: int = 0
    n_slices_rejected_empty: int = 0
    subjects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _in_plane_spacing(
    spacing_mm: tuple[float, float, float] | None, axis: int
) -> tuple[float, float] | None:
    """The two spacings that remain in-plane after slicing along ``axis``."""
    if spacing_mm is None:
        return None
    remaining = [float(s) for i, s in enumerate(spacing_mm) if i != axis]
    return (remaining[0], remaining[1])


def iter_slices(
    volume: np.ndarray,
    ovary_mask: np.ndarray,
    follicle_labels: np.ndarray,
    *,
    subject_id: str,
    volume_id: str,
    spacing_mm: tuple[float, float, float] | None = None,
    axis: int = 0,
    step: int = 1,
    min_ovary_fraction: float = MIN_OVARY_FRACTION_FOR_SLICE,
) -> Iterator[ExtractedSlice]:
    """Yield labelled 2D slices through one annotated volume.

    Slices are taken along one axis only, by default the through-plane axis. That
    is the plane a transvaginal probe actually produces; reslicing along the other
    two axes would manufacture views no sonographer ever acquires and would teach
    the model an appearance it will never see.

    Slices containing no meaningful ovary are dropped rather than emitted as
    negatives. A sliver at the pole of the organ is a labelling edge case, not a
    representative background frame, and training on it teaches the segmenter that
    ovaries can be arbitrarily small.

    Args:
        volume: The intensity volume.
        ovary_mask: Boolean ovary-region mask (stroma plus follicles).
        follicle_labels: Integer follicle *instance* labels — the individual
            follicle annotation that makes this dataset worth using.
        subject_id: Subject identifier, used for leakage-free splitting.
        volume_id: Study identifier.
        spacing_mm: Voxel spacing in mm.
        axis: Slicing axis.
        step: Take every ``step``-th slice. Values above 1 reduce the redundancy
            between near-identical adjacent slices.
        min_ovary_fraction: Minimum ovary share of the slice to keep it.

    Yields:
        :class:`ExtractedSlice` objects in slice order.
    """
    data = np.moveaxis(np.asarray(volume, dtype=float), axis, 0)
    ovary = np.moveaxis(np.asarray(ovary_mask, dtype=bool), axis, 0)
    follicles = np.moveaxis(np.asarray(follicle_labels), axis, 0)
    spacing = _in_plane_spacing(spacing_mm, axis)

    for index in range(0, data.shape[0], max(int(step), 1)):
        ovary_slice = ovary[index]
        fraction = float(ovary_slice.sum() / ovary_slice.size) if ovary_slice.size else 0.0
        if fraction < min_ovary_fraction:
            continue
        follicle_slice = follicles[index]
        n_follicles = int(len({int(v) for v in np.unique(follicle_slice) if v != 0}))
        yield ExtractedSlice(
            image=data[index],
            ovary_mask=ovary_slice,
            follicle_mask=follicle_slice > 0,
            pixel_spacing_mm=spacing,
            subject_id=subject_id,
            volume_id=volume_id,
            slice_index=index,
            axis=axis,
            n_follicles_in_section=n_follicles,
            ovary_fraction=fraction,
        )


def extract_slices(
    studies: list[dict[str, object]],
    *,
    axis: int = 0,
    step: int = 1,
    min_ovary_fraction: float = MIN_OVARY_FRACTION_FOR_SLICE,
) -> tuple[list[ExtractedSlice], SliceExtractionSummary]:
    """Extract 2D slices from a list of annotated volumes.

    Args:
        studies: Dicts with keys ``volume``, ``ovary_mask``, ``follicle_labels``,
            ``subject_id``, ``volume_id`` and optionally ``spacing_mm``.
        axis: Slicing axis.
        step: Slice stride.
        min_ovary_fraction: Minimum ovary share of a slice to keep it.

    Returns:
        ``(slices, summary)``. The summary always carries the independence
        caveat in its warnings.
    """
    slices: list[ExtractedSlice] = []
    subjects: list[str] = []
    rejected = 0

    for study in studies:
        volume = np.asarray(study["volume"])
        ovary = np.asarray(study["ovary_mask"], dtype=bool)
        subject_id = str(study["subject_id"])
        n_before = len(slices)
        slices.extend(
            iter_slices(
                volume,
                ovary,
                np.asarray(study["follicle_labels"]),
                subject_id=subject_id,
                volume_id=str(study["volume_id"]),
                spacing_mm=study.get("spacing_mm"),  # type: ignore[arg-type]
                axis=axis,
                step=step,
                min_ovary_fraction=min_ovary_fraction,
            )
        )
        candidate_slices = len(range(0, volume.shape[axis], max(int(step), 1)))
        rejected += candidate_slices - (len(slices) - n_before)
        if subject_id not in subjects:
            subjects.append(subject_id)

    summary = SliceExtractionSummary(
        n_volumes=len(studies),
        n_slices_kept=len(slices),
        n_slices_rejected_empty=rejected,
        subjects=subjects,
        warnings=[
            VOLUME_DERIVED_PROVENANCE,
            "Adjacent slices from one volume share most of their anatomy and are NOT "
            "independent samples. Split by subject, never by slice.",
            "A per-section follicle count from a slice is not the volume's per-ovary count.",
        ],
    )
    return slices, summary


def assign_slice_splits(
    slices: list[ExtractedSlice],
    *,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split extracted slices **by subject**, never by slice.

    Splitting by slice would put slice 40 of a volume in training and slice 41 in
    test. Those two images differ by one voxel step and share nearly all their
    anatomy, so the resulting score measures memorisation. Subject-level splitting
    is the only defensible option, and it is enforced here rather than left to the
    caller.

    Even so, a test split produced by this function is an *internal* split of
    volume-derived slices. It is not the independent 2D clinical test set the
    module docstring calls for, and a result measured on it must say so.

    Args:
        slices: Extracted slices.
        test_fraction: Share of subjects held out for test.
        val_fraction: Share of subjects held out for validation.
        seed: RNG seed.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` of indices into ``slices``.
    """
    subjects = sorted({s.subject_id for s in slices})
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(subjects)))
    shuffled = [subjects[i] for i in order]

    n = len(shuffled)
    n_test = max(int(round(n * test_fraction)), 1 if n > 2 else 0)
    n_val = max(int(round(n * val_fraction)), 1 if n > 3 else 0)
    test_subjects = set(shuffled[:n_test])
    val_subjects = set(shuffled[n_test : n_test + n_val])

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, item in enumerate(slices):
        if item.subject_id in test_subjects:
            splits["test"].append(index)
        elif item.subject_id in val_subjects:
            splits["val"].append(index)
        else:
            splits["train"].append(index)
    return splits
