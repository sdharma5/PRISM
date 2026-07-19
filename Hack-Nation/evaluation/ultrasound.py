"""Metrics for ovarian ultrasound segmentation, counting and quality gating.

Voxel Dice alone is a misleading summary for this task: a model can score well on
Dice while merging two follicles into one, which changes the *count* — the
quantity a clinician actually uses. This module therefore reports voxel overlap,
**instance**-level precision/recall under greedy IoU matching, count error,
volume error, an anatomical-impossibility count, and two safety metrics for the
quality gate.

**Counts are scored per method, never pooled.** ``follicle_number_per_section``
and the cine-tracked ``estimated_follicle_number_per_ovary`` are different
physical quantities on different supports, so this module reports
:func:`per_section_count_mae` and :func:`unique_track_count_mae` as separate
metrics. Averaging them would produce a number describing nothing, and would hide
the most informative case: a model that reads individual frames well but tracks
badly. Tracking is scored further by :func:`tracking_fragmentation_and_merge`,
which separates the two directional failures — one follicle split into several
tracks (inflates the count) versus several follicles merged into one (deflates
it) — because a count error alone cannot tell them apart, and they cancel.

The two gate metrics answer different questions, and only one of them is a
safety metric:

* ``quality_gate_sensitivity`` — of the studies that truly *are* measurable, how
  many did the gate allow? Low values mean the gate is wastefully conservative.
* ``quality_gate_unsafe_acceptance_rate`` — of the studies that truly are *not*
  measurable, how many did the gate wrongly allow? Every such study yields a
  confidently wrong number, so this is the metric that must be driven to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from schemas.imaging import FollicleInstance, ImageQualityAssessment

EPS = 1e-9


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice overlap of two boolean masks. 1.0 when both are empty."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(target, dtype=bool)
    denom = float(p.sum() + g.sum())
    if denom == 0.0:
        return 1.0
    return float(2.0 * (p & g).sum() / denom)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(target, dtype=bool)
    union = float((p | g).sum())
    if union == 0.0:
        return 1.0
    return float((p & g).sum() / union)


def ovary_dice(pred_region: np.ndarray, true_region: np.ndarray) -> float:
    """Dice of the full ovarian region (stroma plus follicles)."""
    return dice_score(pred_region, true_region)


def follicle_dice(pred_follicle: np.ndarray, true_follicle: np.ndarray) -> float:
    """Dice of the follicle class."""
    return dice_score(pred_follicle, true_follicle)


@dataclass
class InstanceMatchResult:
    """Instance-level matching outcome under greedy IoU assignment."""

    n_pred: int
    n_true: int
    n_matched: int
    matched_ious: list[float] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_true: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Fraction of predicted follicles that correspond to a real one."""
        return float(self.n_matched / self.n_pred) if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        """Fraction of real follicles that were found."""
        return float(self.n_matched / self.n_true) if self.n_true else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def mean_matched_iou(self) -> float:
        return float(np.mean(self.matched_ious)) if self.matched_ious else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "instance_precision": self.precision,
            "instance_recall": self.recall,
            "instance_f1": self.f1,
            "mean_matched_iou": self.mean_matched_iou,
            "n_pred_instances": float(self.n_pred),
            "n_true_instances": float(self.n_true),
        }


def match_instances(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    *,
    iou_threshold: float = 0.3,
) -> InstanceMatchResult:
    """Greedily match predicted to true instances by descending IoU.

    Greedy matching (rather than Hungarian assignment) is used because follicles
    are well separated once split, so the greedy and optimal assignments coincide
    in practice while greedy stays trivially auditable.

    Args:
        pred_labels: Integer instance label volume from the model.
        true_labels: Integer instance label volume of the ground truth.
        iou_threshold: Minimum IoU for a pair to count as a match. 0.3 is a
            detection-oriented threshold: the question is whether the follicle
            was *found*, with sizing accuracy reported separately.

    Returns:
        An :class:`InstanceMatchResult`.
    """
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)
    pred_ids = [int(i) for i in np.unique(pred) if i != 0]
    true_ids = [int(i) for i in np.unique(true) if i != 0]

    pairs: list[tuple[float, int, int]] = []
    for pid in pred_ids:
        pmask = pred == pid
        for tid in true_ids:
            tmask = true == tid
            intersection = float((pmask & tmask).sum())
            if intersection == 0.0:
                continue
            union = float((pmask | tmask).sum())
            iou = intersection / max(union, EPS)
            if iou >= iou_threshold:
                pairs.append((iou, pid, tid))

    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_true: set[int] = set()
    matched_ious: list[float] = []
    for iou, pid, tid in pairs:
        if pid in used_pred or tid in used_true:
            continue
        used_pred.add(pid)
        used_true.add(tid)
        matched_ious.append(float(iou))

    return InstanceMatchResult(
        n_pred=len(pred_ids),
        n_true=len(true_ids),
        n_matched=len(matched_ious),
        matched_ious=matched_ious,
        unmatched_pred=sorted(set(pred_ids) - used_pred),
        unmatched_true=sorted(set(true_ids) - used_true),
    )


def follicle_count_mae(predicted: list[int | None], truth: list[int]) -> float:
    """Mean absolute follicle-count error over studies where a count was emitted.

    Abstained studies (``None``) are excluded from the mean rather than scored as
    zero: refusing to count is not the same as counting zero, and scoring it as
    zero would punish exactly the behaviour the quality gate exists to produce.

    **This function is method-agnostic and must never be applied across mixed
    counting methods.** Use :func:`per_section_count_mae` and
    :func:`unique_track_count_mae` instead; they exist precisely so the two
    quantities cannot be pooled.
    """
    errors = [
        abs(float(p) - float(t)) for p, t in zip(predicted, truth, strict=True) if p is not None
    ]
    return float(np.mean(errors)) if errors else float("nan")


def per_section_count_mae(predicted: list[int | None], truth: list[int]) -> dict[str, float]:
    """MAE of the **per-section** follicle count: follicles in one cross-section.

    Reported separately from :func:`unique_track_count_mae` and never averaged
    with it. A per-section count and a per-ovary count are different physical
    quantities measured on different supports; their mean is a number that
    describes nothing. A model with an excellent per-section MAE and a terrible
    unique-count MAE is a model that reads frames well and tracks badly, and
    pooling the two would hide exactly that.
    """
    errors = [
        abs(float(p) - float(t)) for p, t in zip(predicted, truth, strict=True) if p is not None
    ]
    n_abstained = sum(1 for p in predicted if p is None)
    return {
        "per_section_count_mae": float(np.mean(errors)) if errors else float("nan"),
        "per_section_count_exact_match": (
            float(np.mean([e == 0.0 for e in errors])) if errors else float("nan")
        ),
        "per_section_n_scored": float(len(errors)),
        "per_section_n_abstained": float(n_abstained),
    }


def unique_track_count_mae(predicted: list[int | None], truth: list[int]) -> dict[str, float]:
    """MAE of the **unique-track** count: distinct follicles across a cine loop.

    The quantity a cine loop estimates by tracking. Reported separately from
    :func:`per_section_count_mae`; see that function for why they must never be
    pooled.

    ``unique_track_count_bias`` is signed and is the more diagnostic number of the
    two: tracking failures are directional. Fragmentation inflates the count and
    merging deflates it, so a near-zero MAE built from large opposite-signed errors
    is a very different model from one that is genuinely accurate.
    """
    pairs = [(float(p), float(t)) for p, t in zip(predicted, truth, strict=True) if p is not None]
    n_abstained = sum(1 for p in predicted if p is None)
    if not pairs:
        return {
            "unique_track_count_mae": float("nan"),
            "unique_track_count_bias": float("nan"),
            "unique_track_count_exact_match": float("nan"),
            "unique_track_n_scored": 0.0,
            "unique_track_n_abstained": float(n_abstained),
        }
    errors = [abs(p - t) for p, t in pairs]
    return {
        "unique_track_count_mae": float(np.mean(errors)),
        "unique_track_count_bias": float(np.mean([p - t for p, t in pairs])),
        "unique_track_count_exact_match": float(np.mean([e == 0.0 for e in errors])),
        "unique_track_n_scored": float(len(pairs)),
        "unique_track_n_abstained": float(n_abstained),
    }


def tracking_fragmentation_and_merge(
    assignments: list[tuple[int, int]],
    *,
    n_true_follicles: int | None = None,
    n_predicted_tracks: int | None = None,
) -> dict[str, float]:
    """Fragmentation and merge rates for cine follicle tracking.

    Count error alone cannot distinguish a tracker that is right from one whose
    errors happen to cancel. These two metrics separate the two directional
    failures:

    * **Fragmentation** — one true follicle split across several predicted tracks,
      so it is counted more than once. Caused by fast probe motion or by a
      follicle dropping out of segmentation mid-span. Inflates the count.
    * **Merging** — two true follicles collapsed into one predicted track, so they
      are counted once. Caused by adjacent similar-sized follicles under greedy
      assignment. Deflates the count.

    Both are reported as excess-track and excess-follicle rates rather than as
    binary "was anything wrong" flags, because a follicle split into four tracks
    is three times the error of one split into two.

    Args:
        assignments: ``(predicted_track_id, true_follicle_id)`` pairs, one per
            matched observation. Repeated pairs are fine and are deduplicated.
        n_true_follicles: Total true follicles, including any that were never
            matched. Defaults to the number appearing in ``assignments``.
        n_predicted_tracks: Total predicted tracks, including unmatched ones.
            Defaults to the number appearing in ``assignments``.

    Returns:
        Fragmentation and merge rates plus the raw counts behind them.
    """
    pairs = set(assignments)
    tracks_per_true: dict[int, set[int]] = {}
    trues_per_track: dict[int, set[int]] = {}
    for track_id, true_id in pairs:
        tracks_per_true.setdefault(true_id, set()).add(track_id)
        trues_per_track.setdefault(track_id, set()).add(true_id)

    n_true = int(n_true_follicles if n_true_follicles is not None else len(tracks_per_true))
    n_tracks = int(n_predicted_tracks if n_predicted_tracks is not None else len(trues_per_track))

    excess_tracks = sum(max(len(v) - 1, 0) for v in tracks_per_true.values())
    excess_trues = sum(max(len(v) - 1, 0) for v in trues_per_track.values())
    n_fragmented = sum(1 for v in tracks_per_true.values() if len(v) > 1)
    n_merged = sum(1 for v in trues_per_track.values() if len(v) > 1)

    return {
        # Excess tracks per true follicle: how much the count is inflated.
        "tracking_fragmentation_rate": float(excess_tracks / n_true) if n_true else float("nan"),
        # Excess true follicles per predicted track: how much it is deflated.
        "tracking_merge_rate": float(excess_trues / n_tracks) if n_tracks else float("nan"),
        "fraction_true_follicles_fragmented": (
            float(n_fragmented / n_true) if n_true else float("nan")
        ),
        "fraction_tracks_merged": float(n_merged / n_tracks) if n_tracks else float("nan"),
        "n_true_follicles": float(n_true),
        "n_predicted_tracks": float(n_tracks),
        "n_excess_tracks": float(excess_tracks),
        "n_excess_merged_follicles": float(excess_trues),
    }


def match_tracks_to_truth(
    predicted_observations: dict[int, dict[int, tuple[float, float]]],
    true_observations: dict[int, dict[int, tuple[float, float]]],
    *,
    max_distance_mm: float = 5.0,
) -> list[tuple[int, int]]:
    """Build the ``(track, true follicle)`` assignments from per-frame positions.

    Matching is done **per observation**, on the same frame, by nearest centroid.
    Frame-span overlap alone is not sufficient evidence of identity: several true
    follicles are visible simultaneously in any real sweep, so two different
    follicles routinely share frames. Matching on shared frames would then declare
    every track a match for every concurrent follicle and report fragmentation
    that did not occur.

    Matching per observation is also what lets both directional failures be seen:
    a true follicle whose observations land in two different tracks is
    fragmentation, and a track whose observations land on two different true
    follicles is a merge.

    Args:
        predicted_observations: ``{track_id: {frame index: (row_mm, col_mm)}}``.
        true_observations: ``{true follicle id: {frame index: (row_mm, col_mm)}}``.
        max_distance_mm: Beyond this separation an observation is unmatched rather
            than forced onto the nearest follicle.

    Returns:
        Deduplicated ``(track_id, true_follicle_id)`` pairs, suitable for
        :func:`tracking_fragmentation_and_merge`.
    """
    assignments: set[tuple[int, int]] = set()
    for track_id, frames in predicted_observations.items():
        for frame_index, position in frames.items():
            best_id: int | None = None
            best_distance = float(max_distance_mm)
            for true_id, true_frames in true_observations.items():
                truth = true_frames.get(frame_index)
                if truth is None:
                    continue
                distance = float(np.hypot(position[0] - truth[0], position[1] - truth[1]))
                if distance <= best_distance:
                    best_distance, best_id = distance, int(true_id)
            if best_id is not None:
                assignments.add((int(track_id), best_id))
    return sorted(assignments)


def ovarian_volume_absolute_error(
    predicted_ml: list[float | None], truth_ml: list[float]
) -> dict[str, float]:
    """Absolute and relative ovarian-volume error over non-abstained studies."""
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    for p, t in zip(predicted_ml, truth_ml, strict=True):
        if p is None:
            continue
        abs_errors.append(abs(float(p) - float(t)))
        if t > 0:
            rel_errors.append(abs(float(p) - float(t)) / float(t))
    if not abs_errors:
        return {"volume_mae_ml": float("nan"), "volume_mape": float("nan"), "n_measured": 0.0}
    return {
        "volume_mae_ml": float(np.mean(abs_errors)),
        "volume_mape": float(np.mean(rel_errors)) if rel_errors else float("nan"),
        "n_measured": float(len(abs_errors)),
    }


def false_follicle_voxels_outside_ovary(
    pred_follicle: np.ndarray, true_ovary_region: np.ndarray
) -> int:
    """Predicted follicle voxels outside the TRUE ovary: anatomically impossible.

    This is the evaluation-time twin of the ``L_outside`` training penalty, and
    it is a count rather than a rate because a single misplaced follicle is one
    spurious object in the antral follicle count.
    """
    follicle = np.asarray(pred_follicle, dtype=bool)
    ovary = np.asarray(true_ovary_region, dtype=bool)
    return int((follicle & ~ovary).sum())


def quality_gate_sensitivity(
    assessments: list[ImageQualityAssessment], truly_measurable: list[bool]
) -> float:
    """Fraction of genuinely measurable studies the gate allowed through."""
    positives = [
        a.measurement_feasible for a, ok in zip(assessments, truly_measurable, strict=True) if ok
    ]
    return float(np.mean(positives)) if positives else float("nan")


def quality_gate_specificity(
    assessments: list[ImageQualityAssessment], truly_measurable: list[bool]
) -> float:
    """Fraction of unmeasurable studies the gate correctly refused."""
    negatives = [
        not a.measurement_feasible
        for a, ok in zip(assessments, truly_measurable, strict=True)
        if not ok
    ]
    return float(np.mean(negatives)) if negatives else float("nan")


def quality_gate_unsafe_acceptance_rate(
    assessments: list[ImageQualityAssessment], truly_measurable: list[bool]
) -> float:
    """Fraction of unmeasurable studies the gate WRONGLY allowed.

    The safety metric of this module. Each such study produces a confident
    measurement on an image that cannot support one, which is the failure mode
    most likely to mislead a reader.
    """
    specificity = quality_gate_specificity(assessments, truly_measurable)
    return float("nan") if np.isnan(specificity) else float(1.0 - specificity)


def evaluate_segmentation(
    *,
    pred_ovary_region: np.ndarray,
    pred_follicle: np.ndarray,
    true_ovary_region: np.ndarray,
    true_follicle: np.ndarray,
    pred_instance_labels: np.ndarray | None = None,
    true_instance_labels: np.ndarray | None = None,
    iou_threshold: float = 0.3,
) -> dict[str, float]:
    """Full per-study segmentation metric bundle."""
    metrics: dict[str, float] = {
        "ovary_dice": ovary_dice(pred_ovary_region, true_ovary_region),
        "follicle_dice": follicle_dice(pred_follicle, true_follicle),
        "ovary_iou": iou_score(pred_ovary_region, true_ovary_region),
        "follicle_voxels_outside_ovary": float(
            false_follicle_voxels_outside_ovary(pred_follicle, true_ovary_region)
        ),
    }
    if pred_instance_labels is not None and true_instance_labels is not None:
        metrics.update(
            match_instances(
                pred_instance_labels, true_instance_labels, iou_threshold=iou_threshold
            ).as_dict()
        )
    return metrics


def diameter_error(
    predicted_mm: list[float], truth_mm: list[float], *, tolerance_mm: float = 1.0
) -> dict[str, float]:
    """Sorted-diameter error, plus the fraction within a stated tolerance.

    Diameters are compared after sorting because instance identity is not
    meaningful; what matters clinically is whether the *size distribution* is
    recovered.
    """
    p = np.sort(np.asarray(predicted_mm, dtype=float))
    t = np.sort(np.asarray(truth_mm, dtype=float))
    n = min(p.size, t.size)
    if n == 0:
        return {"diameter_mae_mm": float("nan"), "diameter_within_tolerance": float("nan")}
    errors = np.abs(p[:n] - t[:n])
    return {
        "diameter_mae_mm": float(errors.mean()),
        "diameter_max_error_mm": float(errors.max()),
        "diameter_within_tolerance": float((errors <= tolerance_mm).mean()),
    }


def summarize_instances(instances: list[FollicleInstance]) -> dict[str, float]:
    """Counts of retained, large-flagged and partially extra-ovarian instances."""
    return {
        "n_instances": float(len(instances)),
        "n_large_flagged": float(sum(1 for i in instances if i.is_large_or_uncertain)),
        "n_partially_outside_ovary": float(
            sum(1 for i in instances if i.inside_ovary_fraction < 0.95)
        ),
    }
