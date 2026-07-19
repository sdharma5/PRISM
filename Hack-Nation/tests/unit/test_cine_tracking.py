"""A follicle seen on many frames is ONE follicle.

This is the test that falsifies the counting logic. A pipeline that summed
per-frame follicle counts over a sweep would report roughly six times the truth on
the default phantom and would look entirely plausible doing it, because every
individual per-section count would be correct. Only a test that knows which frames
belong to which follicle can catch that, which is what
:func:`make_cine_phantom` provides.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.ultrasound import (
    match_tracks_to_truth,
    tracking_fragmentation_and_merge,
    unique_track_count_mae,
)
from ingestion.ultrasound.loader import load_ultrasound
from models.ultrasound.cine_tracking import tracking_confidence
from models.ultrasound.encoder import UltrasoundEncoder
from tests.fixtures.synthetic_ultrasound import FollicleSpec2D, make_cine_phantom

SPACING = (0.35, 0.35, 0.35)


def _encode(phantom, **kwargs):
    loaded = load_ultrasound(
        phantom.frames,
        patient_id="P",
        study_id="CINE",
        spacing_mm=SPACING,
        laterality="left",
        route="transvaginal",
        **kwargs,
    )
    encoder = UltrasoundEncoder(segmenter_kind="threshold")
    return encoder.encode(
        loaded.array, loaded.metadata, acquisition_mode=loaded.acquisition_mode
    ), loaded


def test_a_frame_stack_is_detected_as_a_cine_loop():
    """A rank-3 array with no through-plane geometry is a sweep, not a volume."""
    _, loaded = _encode(make_cine_phantom())
    assert loaded.acquisition_mode == "cine_loop"
    assert loaded.is_2d_pathway is True
    assert loaded.metadata.is_3d is False


def test_follicle_spanning_frames_3_to_7_is_counted_once():
    """The canonical case: one follicle on five consecutive frames counts ONCE."""
    phantom = make_cine_phantom(
        specs=(FollicleSpec2D(6.0, 3, 7, -5.0, -4.0),),
        n_frames=12,
    )
    encoding, _ = _encode(phantom)
    tracking = encoding.tracking
    assert tracking is not None

    # Exactly one follicle exists, so exactly one track must survive.
    assert tracking.estimated_unique_count == 1
    assert encoding.morphology.estimated_follicle_number_per_ovary == 1

    # And it must genuinely have been seen on the whole span, not on one frame.
    track = tracking.tracks[0]
    assert set(track.frames) == set(range(3, 8)), track.frames
    assert track.n_frames == 5

    # The per-section counts on those frames are each 1 — five ones that must not
    # become a five.
    counts = [f.follicle_number_per_section for f in encoding.cine_morphology.per_frame]
    assert [counts[k] for k in range(3, 8)] == [1, 1, 1, 1, 1]
    assert sum(c for c in counts if c) == 5
    assert tracking.estimated_unique_count == 1


def test_estimated_unique_count_matches_the_phantom_truth():
    """The tracked estimate matches the true unique count within tolerance."""
    phantom = make_cine_phantom()
    encoding, _ = _encode(phantom)
    estimated = encoding.morphology.estimated_follicle_number_per_ovary

    assert estimated is not None
    assert abs(estimated - phantom.true_unique_follicle_count) <= 1, (
        f"estimated {estimated} vs true {phantom.true_unique_follicle_count}"
    )
    # Nothing like what naive per-frame summation would produce.
    assert estimated < phantom.naive_summed_count / 2


def _predicted_observations(tracking):
    """``{track_id: {frame: (row_mm, col_mm)}}`` for the evaluation matcher."""
    return {
        t.track_id: {o.frame_index: o.centroid_mm for o in t.observations} for t in tracking.tracks
    }


def test_track_spans_and_diameters_recover_the_truth():
    """Each track's frame span and peak diameter match the follicle it followed."""
    phantom = make_cine_phantom()
    encoding, _ = _encode(phantom)
    tracking = encoding.tracking

    assignments = match_tracks_to_truth(
        _predicted_observations(tracking), phantom.true_observations_mm()
    )
    matched_true = {true_id for _, true_id in assignments}
    assert len(matched_true) == phantom.true_unique_follicle_count

    # Peak cross-section is the follicle's true diameter; other planes under-cut it.
    predicted = sorted(t.max_diameter_mm for t in tracking.tracks)
    assert np.allclose(predicted, phantom.true_diameters_mm, atol=1.0), predicted


def test_no_fragmentation_or_merging_on_a_clean_sweep():
    """A clean sweep produces one track per follicle: no splits, no merges."""
    phantom = make_cine_phantom()
    encoding, _ = _encode(phantom)
    tracking = encoding.tracking

    assignments = match_tracks_to_truth(
        _predicted_observations(tracking), phantom.true_observations_mm()
    )
    metrics = tracking_fragmentation_and_merge(
        assignments,
        n_true_follicles=phantom.true_unique_follicle_count,
        n_predicted_tracks=tracking.estimated_unique_count,
    )
    assert metrics["tracking_fragmentation_rate"] == 0.0
    assert metrics["tracking_merge_rate"] == 0.0


def test_fragmentation_is_detected_when_a_track_breaks():
    """A follicle split across two tracks must register as fragmentation.

    Constructed directly rather than provoked through the segmenter, so the metric
    is tested on a case whose ground truth is unambiguous.
    """
    predicted = {
        1: {3: (10.0, 10.0), 4: (10.0, 10.0)},
        2: {6: (10.0, 10.0), 7: (10.0, 10.0)},  # same follicle, track broke
    }
    truth = {0: {k: (10.0, 10.0) for k in range(3, 8)}}
    metrics = tracking_fragmentation_and_merge(
        match_tracks_to_truth(predicted, truth), n_true_follicles=1, n_predicted_tracks=2
    )
    assert metrics["tracking_fragmentation_rate"] == pytest.approx(1.0)
    assert metrics["tracking_merge_rate"] == 0.0


def test_merging_is_detected_when_two_follicles_share_a_track():
    """Two follicles collapsed into one track must register as a merge."""
    predicted = {1: {3: (10.0, 10.0), 4: (20.0, 20.0)}}
    truth = {0: {3: (10.0, 10.0)}, 1: {4: (20.0, 20.0)}}
    metrics = tracking_fragmentation_and_merge(
        match_tracks_to_truth(predicted, truth), n_true_follicles=2, n_predicted_tracks=1
    )
    assert metrics["tracking_merge_rate"] == pytest.approx(1.0)
    assert metrics["tracking_fragmentation_rate"] == 0.0


def test_low_tracking_coverage_lowers_confidence():
    """Dropping frames must reduce both coverage and the reported confidence."""
    clean = make_cine_phantom()
    degraded = make_cine_phantom(unusable_frames=(2, 4, 6, 8, 10, 12))

    clean_encoding, _ = _encode(clean)
    degraded_encoding, _ = _encode(degraded)

    clean_track = clean_encoding.tracking
    degraded_track = degraded_encoding.tracking

    assert degraded_track.tracking_coverage < clean_track.tracking_coverage
    assert degraded_track.confidence < clean_track.confidence

    # The degradation must reach the schema output, not stop at the tracker.
    assert degraded_encoding.morphology.tracking_coverage < 1.0
    assert (
        degraded_encoding.morphology.follicle_detection_confidence
        < clean_encoding.morphology.follicle_detection_confidence
    )
    assert any("usable" in w.lower() for w in degraded_encoding.morphology.warnings)


def test_confidence_degrades_monotonically_with_coverage():
    """``tracking_confidence`` is monotone in coverage, all else equal."""
    scores = [
        tracking_confidence(coverage=c, n_frames_analyzed=20, n_tracks=5, n_discarded=0)
        for c in (0.2, 0.5, 0.8, 1.0)
    ]
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)
    assert scores[0] < 0.2


def test_a_loop_with_no_usable_frames_abstains():
    """Every frame unusable means no count at all, not a count of zero."""
    phantom = make_cine_phantom(n_frames=8, unusable_frames=tuple(range(8)))
    encoding, _ = _encode(phantom)
    morphology = encoding.morphology

    assert morphology.measurement_feasible is False
    assert morphology.estimated_follicle_number_per_ovary is None
    assert morphology.follicle_number_per_section is None
    assert morphology.follicle_count_method == "not_assessed"
    assert morphology.reportable_follicle_count == (None, "not_assessed")


def test_a_broken_track_never_reports_zero_follicles():
    """When tracking collapses, withhold the estimate — do not claim zero.

    A reported count of zero is a strong clinical claim: *this ovary has no
    follicles*. When the probe moves faster than the tracker can follow, zero
    surviving tracks means "we could not follow anything", which is a completely
    different statement. The pipeline must fall back to the per-section count it
    can still support rather than emitting a confident zero.
    """
    # Drift far exceeding the 4 mm centroid gate: tracking cannot survive it.
    phantom = make_cine_phantom(drift_mm_per_frame=(8.0, 0.0))
    encoding, _ = _encode(phantom)
    morphology = encoding.morphology

    assert encoding.tracking.estimated_unique_count == 0
    assert morphology.estimated_follicle_number_per_ovary is None
    assert morphology.follicle_count_method != "estimated_per_ovary"
    assert any(
        "withheld" in w.lower() and ("zero" in w.lower() or "coverage" in w.lower())
        for w in morphology.warnings
    ), morphology.warnings


def test_moderate_drift_degrades_the_estimate_rather_than_breaking_it():
    """Drift within the centroid gate must still track; beyond it, degrade."""
    ok, _ = _encode(make_cine_phantom(drift_mm_per_frame=(2.0, 0.0)))
    assert ok.morphology.estimated_follicle_number_per_ovary == 4
    assert ok.tracking.confidence > 0.9


def test_tracking_warns_that_the_count_is_an_estimate():
    """The estimate must never travel without the caveat that it is one."""
    encoding, _ = _encode(make_cine_phantom())
    joined = " ".join(encoding.morphology.warnings).lower()
    assert "estimate" in joined
    assert "census" in joined or "true per-ovary" in joined
    assert encoding.morphology.follicle_count_method == "estimated_per_ovary"


def test_unique_track_mae_is_reported_separately_from_per_section():
    """The two count MAEs are distinct metrics with distinct keys."""
    phantom = make_cine_phantom()
    encoding, _ = _encode(phantom)
    metrics = unique_track_count_mae(
        [encoding.morphology.estimated_follicle_number_per_ovary],
        [phantom.true_unique_follicle_count],
    )
    assert "unique_track_count_mae" in metrics
    assert "per_section_count_mae" not in metrics
    assert metrics["unique_track_count_mae"] <= 1.0
