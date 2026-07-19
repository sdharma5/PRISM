"""Match follicles across adjacent cine frames so each is counted ONCE.

This is the module that makes a 2D cine loop worth more than a single frame, and
it is also the module where the resulting number stops being a census and becomes
an estimate. A follicle visible on frames 3, 4, 5, 6 and 7 is **one** follicle. A
pipeline that sums per-frame counts over a 60-frame sweep reports a number an
order of magnitude too large, and reports it confidently. Everything below exists
to prevent that.

The matching rule
-----------------
Frames are processed in acquisition order. Each follicle candidate on the current
frame is compared against the most recent observation of every *active* track — a
track whose last observation is within :data:`DEFAULT_MAX_FRAME_GAP` + 1 frames.
A candidate/track pair is admissible only if it passes two hard gates:

1. **Spatial continuity.** Either mask IoU >= ``min_iou``, or the centroid
   displacement is <= ``max_centroid_shift_mm``. The disjunction matters: a small
   follicle moving by its own diameter has IoU 0 but is obviously the same
   follicle, while a large follicle can have high IoU despite a displacement that
   would look large in millimetres. Requiring *both* would fragment small
   follicles; requiring neither would link anything to anything.
2. **Size plausibility.** The ratio of the smaller to the larger cross-sectional
   area must be >= ``min_size_ratio``. Between adjacent frames a follicle's
   cross-section changes gradually as the plane of section moves through the
   sphere; a two-fold jump means the match is to a different structure.

Admissible pairs are then scored

    ``score = w_iou * IoU + w_distance * (1 - d / max_shift) + w_size * size_ratio``

and assigned greedily in descending score, one candidate per track and one track
per candidate. Greedy rather than Hungarian assignment: follicles are well
separated once split, so the greedy and optimal assignments coincide in practice
while greedy stays auditable — you can read off exactly why two things were
linked.

Unmatched candidates start new tracks. Tracks shorter than ``min_track_frames``
are discarded before counting.

Failure modes — read these before trusting the number
-----------------------------------------------------
* **Re-imaging the same anatomy (over-count).** If the probe sweeps forward and
  then back over the same plane, the same follicle produces two temporally
  disjoint tracks and is counted twice. The tracker has no out-of-plane position
  and *cannot detect this*. This is the single largest reason the result is
  reported as ``estimated_follicle_number_per_ovary`` and never as a true count.
* **Fast probe motion (over-count via fragmentation).** If between-frame
  displacement exceeds ``max_centroid_shift_mm`` and IoU drops to zero, one
  follicle's track breaks in two and is counted twice. Raising the threshold to
  compensate trades this for merges.
* **Adjacent similar-sized follicles (under-count via merging).** Two follicles a
  few millimetres apart with similar cross-sections can swap identities or
  collapse into one track under greedy assignment, especially when one is
  transiently unsegmented.
* **Tangential planes of section (fragmentation).** A follicle the plane clips
  near its pole presents a cross-section that shrinks below the minimum-diameter
  filter, disappears, and reappears. ``max_frame_gap`` bridges brief dropouts but
  a long dropout still splits the track.
* **Incomplete sweeps (under-count).** ``tracking_coverage`` reports the fraction
  of *frames* that yielded a usable mask. It says nothing about whether the sweep
  traversed the whole ovary. A perfectly tracked loop that only imaged half the
  ovary reports coverage 1.0 and an unique count that is half the truth. No 2D
  acquisition can rule this out, which is the deeper reason a true per-ovary count
  requires a volume.
* **Single-frame follicles are dropped.** ``min_track_frames`` defaults to 2, so a
  follicle genuinely seen on only one frame at the edge of the sweep is not
  counted. This is a deliberate bias towards under-counting: a one-frame blob is
  far more often speckle than a follicle, and an inflated antral follicle count is
  the more consequential error.

The estimate's confidence is degraded whenever coverage is low, when a large
fraction of tracks are single-frame, or when the loop is short — see
:func:`tracking_confidence`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

__all__ = [
    "DEFAULT_MAX_CENTROID_SHIFT_MM",
    "DEFAULT_MAX_FRAME_GAP",
    "DEFAULT_MIN_IOU",
    "DEFAULT_MIN_SIZE_RATIO",
    "DEFAULT_MIN_TRACK_FRAMES",
    "FollicleObservation",
    "FollicleTrack",
    "TrackingResult",
    "match_score",
    "observations_from_labels",
    "track_follicles",
    "tracking_confidence",
]

#: Minimum mask IoU for two observations to be spatially continuous.
DEFAULT_MIN_IOU = 0.20

#: Maximum centroid displacement, in mm, that is still the same follicle.
#: Sized against realistic freehand transvaginal sweep speed: a few mm per frame.
DEFAULT_MAX_CENTROID_SHIFT_MM = 4.0

#: Minimum smaller/larger cross-sectional area ratio for a plausible match.
DEFAULT_MIN_SIZE_RATIO = 0.35

#: How many consecutive frames a follicle may vanish for and still be re-linked.
#: Bridges brief segmentation dropouts without bridging a genuine re-imaging.
DEFAULT_MAX_FRAME_GAP = 2

#: Tracks shorter than this are discarded as speckle. See the failure-modes note.
DEFAULT_MIN_TRACK_FRAMES = 2


@dataclass
class FollicleObservation:
    """One follicle candidate on one frame."""

    frame_index: int
    label_id: int
    mask: np.ndarray
    centroid_px: tuple[float, float]
    area_px: int
    centroid_mm: tuple[float, float] | None = None
    area_mm2: float | None = None
    diameter_mm: float | None = None


@dataclass
class FollicleTrack:
    """One follicle followed across frames — the unit of the unique count."""

    track_id: int
    observations: list[FollicleObservation] = field(default_factory=list)

    @property
    def frames(self) -> list[int]:
        """Frame indices this follicle was observed on."""
        return [o.frame_index for o in self.observations]

    @property
    def n_frames(self) -> int:
        return len(self.observations)

    @property
    def frame_span(self) -> tuple[int, int]:
        frames = self.frames
        return (min(frames), max(frames)) if frames else (-1, -1)

    @property
    def max_diameter_mm(self) -> float | None:
        """Largest cross-sectional diameter seen, in mm.

        The maximum across frames is the right estimator for a sphere sampled by
        moving planes: the plane through the follicle's centre gives its true
        diameter, and every other plane under-reports. The mean would
        systematically under-size every follicle.
        """
        values = [o.diameter_mm for o in self.observations if o.diameter_mm is not None]
        return float(max(values)) if values else None

    @property
    def max_area_mm2(self) -> float | None:
        values = [o.area_mm2 for o in self.observations if o.area_mm2 is not None]
        return float(max(values)) if values else None

    @property
    def last(self) -> FollicleObservation:
        return self.observations[-1]


@dataclass
class TrackingResult:
    """Tracks, the unique-count estimate, and how much to believe it."""

    tracks: list[FollicleTrack] = field(default_factory=list)
    discarded_short_tracks: int = 0
    frames_total: int = 0
    frames_analyzed: int = 0
    #: Fraction of frames that contributed a usable mask. Frame coverage only.
    tracking_coverage: float = 0.0
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def estimated_unique_count(self) -> int:
        """Number of retained tracks: the estimated unique follicle count."""
        return len(self.tracks)

    @property
    def diameters_mm(self) -> list[float]:
        """Per-track maximum diameters, sorted ascending."""
        return sorted(
            d
            for d in (t.max_diameter_mm for t in self.tracks)
            if d is not None  # noqa: PLR1714
        )

    def track_for_frame(self, frame_index: int) -> dict[int, int]:
        """``{label_id: track_id}`` for one frame — the tracking audit trail."""
        return {
            o.label_id: t.track_id
            for t in self.tracks
            for o in t.observations
            if o.frame_index == frame_index
        }


def observations_from_labels(
    label_images: dict[int, np.ndarray],
    *,
    pixel_spacing_mm: tuple[float, float] | None,
) -> dict[int, list[FollicleObservation]]:
    """Turn per-frame instance-label images into observations.

    Args:
        label_images: ``{frame_index: integer label image}``. Only frames that
            passed the quality gate should be present — an unusable frame must
            not contribute candidates, or the tracker will chase artefacts.
        pixel_spacing_mm: In-plane ``(row_mm, col_mm)``. ``None`` leaves all
            physical fields unset, and the caller must then abstain.

    Returns:
        ``{frame_index: [FollicleObservation, ...]}``.
    """
    per_frame: dict[int, list[FollicleObservation]] = {}
    for frame_index in sorted(label_images):
        labels = np.asarray(label_images[frame_index])
        observations: list[FollicleObservation] = []
        for label_id in (int(v) for v in np.unique(labels) if v != 0):
            mask = labels == label_id
            area_px = int(mask.sum())
            if area_px == 0:
                continue
            coords = np.argwhere(mask)
            centroid_px = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
            centroid_mm: tuple[float, float] | None = None
            area_mm2: float | None = None
            diameter_mm: float | None = None
            if pixel_spacing_mm is not None:
                row_mm, col_mm = (float(s) for s in pixel_spacing_mm)
                centroid_mm = (centroid_px[0] * row_mm, centroid_px[1] * col_mm)
                area_mm2 = float(area_px * row_mm * col_mm)
                # Equivalent-circle diameter: the follicle cross-section is a
                # disc, so this is the diameter a sonographer would caliper.
                diameter_mm = float(2.0 * np.sqrt(area_mm2 / np.pi))
            observations.append(
                FollicleObservation(
                    frame_index=frame_index,
                    label_id=label_id,
                    mask=mask,
                    centroid_px=centroid_px,
                    area_px=area_px,
                    centroid_mm=centroid_mm,
                    area_mm2=area_mm2,
                    diameter_mm=diameter_mm,
                )
            )
        per_frame[frame_index] = observations
    return per_frame


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = float((a | b).sum())
    return float((a & b).sum() / union) if union > 0 else 0.0


def _centroid_distance(
    a: FollicleObservation, b: FollicleObservation, pixel_spacing_mm: tuple[float, float] | None
) -> float:
    """Centroid separation in mm, or in pixels when spacing is unknown."""
    if a.centroid_mm is not None and b.centroid_mm is not None:
        return float(
            np.hypot(a.centroid_mm[0] - b.centroid_mm[0], a.centroid_mm[1] - b.centroid_mm[1])
        )
    row_mm, col_mm = pixel_spacing_mm if pixel_spacing_mm is not None else (1.0, 1.0)
    return float(
        np.hypot(
            (a.centroid_px[0] - b.centroid_px[0]) * row_mm,
            (a.centroid_px[1] - b.centroid_px[1]) * col_mm,
        )
    )


def match_score(
    a: FollicleObservation,
    b: FollicleObservation,
    *,
    pixel_spacing_mm: tuple[float, float] | None = None,
    min_iou: float = DEFAULT_MIN_IOU,
    max_centroid_shift_mm: float = DEFAULT_MAX_CENTROID_SHIFT_MM,
    min_size_ratio: float = DEFAULT_MIN_SIZE_RATIO,
    w_iou: float = 0.5,
    w_distance: float = 0.3,
    w_size: float = 0.2,
) -> float | None:
    """Score a candidate match, or return ``None`` if it fails a hard gate.

    See the module docstring for the two hard gates and why they are a
    disjunction on spatial continuity but a conjunction with size plausibility.

    Returns:
        A score in [0, 1], or ``None`` when the pair is inadmissible.
    """
    size_ratio = float(min(a.area_px, b.area_px) / max(max(a.area_px, b.area_px), 1))
    if size_ratio < min_size_ratio:
        return None

    iou = _iou(a.mask, b.mask)
    distance = _centroid_distance(a, b, pixel_spacing_mm)
    if iou < min_iou and distance > max_centroid_shift_mm:
        return None

    proximity = float(np.clip(1.0 - distance / max(max_centroid_shift_mm, 1e-6), 0.0, 1.0))
    return float(w_iou * iou + w_distance * proximity + w_size * size_ratio)


def track_follicles(
    observations: dict[int, list[FollicleObservation]],
    *,
    frames_total: int | None = None,
    pixel_spacing_mm: tuple[float, float] | None = None,
    min_iou: float = DEFAULT_MIN_IOU,
    max_centroid_shift_mm: float = DEFAULT_MAX_CENTROID_SHIFT_MM,
    min_size_ratio: float = DEFAULT_MIN_SIZE_RATIO,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
) -> TrackingResult:
    """Link follicle observations across frames into unique tracks.

    Args:
        observations: ``{frame_index: [FollicleObservation, ...]}``, typically
            from :func:`observations_from_labels`, containing only quality-passing
            frames.
        frames_total: Total frames in the loop *including* unusable ones. Needed
            to compute honest coverage; defaults to the number of keys supplied,
            which would report coverage 1.0 and overstate the estimate.
        pixel_spacing_mm: In-plane ``(row_mm, col_mm)``.
        min_iou: Spatial-continuity IoU gate.
        max_centroid_shift_mm: Spatial-continuity displacement gate.
        min_size_ratio: Size-plausibility gate.
        max_frame_gap: Consecutive frames a follicle may vanish for.
        min_track_frames: Tracks shorter than this are discarded.

    Returns:
        A :class:`TrackingResult`. ``estimated_unique_count`` is the number of
        retained tracks, and it is an **estimate** — see the module's failure
        modes.
    """
    analyzed_frames = sorted(observations)
    total = int(frames_total) if frames_total is not None else len(analyzed_frames)
    warnings: list[str] = []

    tracks: list[FollicleTrack] = []
    next_track_id = 1

    for frame_index in analyzed_frames:
        candidates = list(observations[frame_index])
        active = [t for t in tracks if 0 < frame_index - t.last.frame_index <= max_frame_gap + 1]

        pairs: list[tuple[float, int, int]] = []
        for track_pos, track in enumerate(active):
            for cand_pos, candidate in enumerate(candidates):
                score = match_score(
                    track.last,
                    candidate,
                    pixel_spacing_mm=pixel_spacing_mm,
                    min_iou=min_iou,
                    max_centroid_shift_mm=max_centroid_shift_mm,
                    min_size_ratio=min_size_ratio,
                )
                if score is not None:
                    pairs.append((score, track_pos, cand_pos))

        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _score, track_pos, cand_pos in pairs:
            if track_pos in used_tracks or cand_pos in used_candidates:
                continue
            used_tracks.add(track_pos)
            used_candidates.add(cand_pos)
            active[track_pos].observations.append(candidates[cand_pos])

        for cand_pos, candidate in enumerate(candidates):
            if cand_pos in used_candidates:
                continue
            tracks.append(FollicleTrack(track_id=next_track_id, observations=[candidate]))
            next_track_id += 1

    retained = [t for t in tracks if t.n_frames >= min_track_frames]
    discarded = len(tracks) - len(retained)
    # Renumber so track ids are contiguous in the reported result.
    for new_id, track in enumerate(retained, start=1):
        track.track_id = new_id

    if discarded:
        warnings.append(
            f"{discarded} candidate track(s) appeared on fewer than {min_track_frames} frames and "
            "were discarded as probable speckle. This biases the estimate towards under-counting "
            "genuinely brief follicles at the edges of the sweep."
        )

    coverage = float(len(analyzed_frames) / total) if total else 0.0
    if coverage < 1.0:
        warnings.append(
            f"Only {len(analyzed_frames)}/{total} frames ({coverage:.0%}) contributed usable "
            "masks; the unique-follicle estimate is correspondingly less reliable."
        )
    warnings.append(
        "Unique follicle count is an ESTIMATE from 2D tracking, not a per-ovary census. "
        "A probe sweep that re-images the same plane counts a follicle twice, and a sweep that "
        "does not traverse the whole ovary misses follicles entirely; neither is detectable "
        "from the frames alone."
    )

    confidence = tracking_confidence(
        coverage=coverage,
        n_frames_analyzed=len(analyzed_frames),
        n_tracks=len(retained),
        n_discarded=discarded,
    )

    return TrackingResult(
        tracks=retained,
        discarded_short_tracks=discarded,
        frames_total=total,
        frames_analyzed=len(analyzed_frames),
        tracking_coverage=coverage,
        confidence=confidence,
        warnings=warnings,
    )


#: A loop shorter than this cannot establish continuity well enough for the
#: unique-count estimate to be meaningfully better than a per-section count.
MIN_FRAMES_FOR_FULL_CONFIDENCE = 10


def tracking_confidence(
    *,
    coverage: float,
    n_frames_analyzed: int,
    n_tracks: int,
    n_discarded: int,
) -> float:
    """How much to believe the unique-count estimate, in [0, 1].

    Three independent degradations, multiplied because they compound:

    * **Coverage.** Applied as ``coverage ** 1.5`` rather than linearly, because
      missing frames do not merely remove information — they create gaps that
      break tracks, so the harm grows faster than the fraction lost.
    * **Loop length.** A handful of frames cannot establish continuity; below
      :data:`MIN_FRAMES_FOR_FULL_CONFIDENCE` the confidence is scaled down
      proportionally.
    * **Fragmentation pressure.** A large share of discarded single-frame tracks
      means the segmentation was unstable, so the surviving tracks are likelier to
      be fragments too.

    Args:
        coverage: Fraction of frames that contributed usable masks.
        n_frames_analyzed: Number of frames that contributed masks.
        n_tracks: Retained tracks.
        n_discarded: Tracks discarded for being too short.

    Returns:
        A confidence in [0, 1].
    """
    coverage_term = float(np.clip(coverage, 0.0, 1.0) ** 1.5)
    length_term = float(np.clip(n_frames_analyzed / MIN_FRAMES_FOR_FULL_CONFIDENCE, 0.0, 1.0))
    total_candidate_tracks = n_tracks + n_discarded
    stability_term = float(n_tracks / total_candidate_tracks) if total_candidate_tracks > 0 else 0.0
    return float(np.clip(coverage_term * length_term * stability_term, 0.0, 1.0))


def follicle_label_image(
    follicle_mask: np.ndarray,
    *,
    pixel_spacing_mm: tuple[float, float] | None,
    min_diameter_mm: float = 2.0,
    separate_touching_follicles: bool = True,
) -> np.ndarray:
    """Label one frame's follicle mask, splitting touching follicles first.

    Two filters are applied here, per frame, rather than after tracking, because
    both errors compound once an observation reaches the tracker.

    * **Touching follicles are separated by watershed.** Two adjacent follicles
      whose cross-sections abut form one connected component and would enter the
      tracker as a single object — under-counting on that frame and, worse,
      producing one track where there should be two. This is the single most
      common cause of undercounting in antral follicle counting, so the 2D path
      applies the same distance-transform split the 3D path does.
    * **Sub-resolution components are dropped.** A speckle blob that survives into
      the tracker can seed a spurious track, and a spurious track is a spurious
      follicle in the final unique count.

    Args:
        follicle_mask: Boolean follicle mask for one frame.
        pixel_spacing_mm: In-plane ``(row_mm, col_mm)``. When ``None`` no physical
            filter is applied and every component is kept.
        min_diameter_mm: Equivalent-circle diameter below which a component is
            discarded as speckle.
        separate_touching_follicles: Enable the watershed split.

    Returns:
        An integer label image with contiguous labels starting at 1.
    """
    from models.ultrasound.follicle_instances import separate_touching  # noqa: PLC0415

    mask = np.asarray(follicle_mask, dtype=bool)
    if separate_touching_follicles and mask.any():
        labels, _ = separate_touching(mask, pixel_spacing_mm)
        n = int(labels.max())
    else:
        labels, n = ndi.label(mask)
    if n == 0 or pixel_spacing_mm is None:
        return labels.astype(np.int32)

    row_mm, col_mm = (float(s) for s in pixel_spacing_mm)
    pixel_area = row_mm * col_mm
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for label_id in (int(v) for v in np.unique(labels) if v != 0):
        component = labels == label_id
        area_mm2 = float(component.sum() * pixel_area)
        diameter = float(2.0 * np.sqrt(area_mm2 / np.pi))
        if diameter < min_diameter_mm:
            continue
        out[component] = next_id
        next_id += 1
    return out
