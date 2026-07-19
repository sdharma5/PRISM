"""Missing must never silently become zero.

Every time-varying channel is carried as the triple
``(value, is_observed, time_since_last_observed)``. The reason is that the
missingness in longitudinal hormone data is non-ignorable: people test around
expected ovulation and skip during menses, so *whether* a value exists is itself
informative. A model fed zero-filled values learns the testing schedule and
reports it as physiology.

A zero LH is also a real, meaningful measurement. Any encoding that cannot
distinguish "LH was 0" from "LH was not measured" is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.temporal import ablate_days
from models.temporal.gru import build_feature_matrix, fit_feature_spec, make_windows
from schemas.temporal import TEMPORAL_FEATURE_SUFFIXES, ParticipantDay
from tests.fixtures.synthetic_cycles import ALL_CHANNELS, generate_cohort


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n_participants=4, n_days=45, seed=0)


def test_every_day_carries_the_full_triple(cohort):
    """value / is_observed / time_since_last_observed present for every channel."""
    assert TEMPORAL_FEATURE_SUFFIXES == ("value", "is_observed", "time_since_last_observed")
    for day in cohort.days:
        for channel in ALL_CHANNELS:
            assert channel in day.values
            assert channel in day.is_observed
            assert channel in day.time_since_last_observed


def test_unobserved_values_are_none_not_zero(cohort):
    """The fixture itself must never zero-fill."""
    unobserved = [
        day.values[channel]
        for day in cohort.days
        for channel in ALL_CHANNELS
        if not day.is_observed[channel]
    ]
    assert unobserved, "the fixture must actually contain missing values"
    assert all(v is None for v in unobserved)


def test_observed_values_are_never_none(cohort):
    """A flag saying observed must be backed by an actual number."""
    for day in cohort.days:
        for channel in ALL_CHANNELS:
            if day.is_observed[channel]:
                assert day.values[channel] is not None


def test_staleness_grows_across_a_missing_run(cohort):
    """time_since_last_observed must increase while a channel goes unobserved."""
    series = cohort.for_participant(cohort.participant_ids[0])
    channel = "pdg"
    run: list[float] = []
    for day in series:
        if day.is_observed[channel]:
            if len(run) > 1:
                assert run == sorted(run), "staleness must be non-decreasing in a gap"
            run = []
        else:
            run.append(day.time_since_last_observed[channel])
    assert True


def test_feature_matrix_flags_observation_separately_from_value(cohort):
    """A zero value and a missing value must produce different feature rows."""
    spec = fit_feature_spec(cohort.days, lookback_days=14)
    channel = spec.channels[0]
    index = spec.channels.index(channel)

    observed_zero = ParticipantDay(
        participant_id="X",
        study_day=0,
        cycle_day=5,
        values={c: (0.0 if c == channel else None) for c in spec.channels},
        is_observed={c: (c == channel) for c in spec.channels},
        time_since_last_observed=dict.fromkeys(spec.channels, 0.0),
    )
    truly_missing = ParticipantDay(
        participant_id="X",
        study_day=0,
        cycle_day=5,
        values=dict.fromkeys(spec.channels),
        is_observed=dict.fromkeys(spec.channels, False),
        time_since_last_observed=dict.fromkeys(spec.channels, 3.0),
    )

    row_zero = build_feature_matrix([observed_zero], spec)[0]
    row_missing = build_feature_matrix([truly_missing], spec)[0]

    assert row_zero[3 * index + 1] == 1.0, "observed flag must be set"
    assert row_missing[3 * index + 1] == 0.0, "missing flag must be clear"
    assert not np.allclose(row_zero, row_missing), "a zero value must not look missing"


def test_observed_flag_without_a_value_is_rejected():
    """Refusing to guess is better than fabricating a number."""
    spec = fit_feature_spec(
        [
            ParticipantDay(
                participant_id="X",
                study_day=0,
                values={"lh": 5.0},
                is_observed={"lh": True},
                time_since_last_observed={"lh": 0.0},
            )
        ]
    )
    broken = ParticipantDay(
        participant_id="X",
        study_day=1,
        values={"lh": None},
        is_observed={"lh": True},  # inconsistent
        time_since_last_observed={"lh": 0.0},
    )
    with pytest.raises(ValueError, match="observed but carries no value"):
        build_feature_matrix([broken], spec)


def test_feature_layout_has_three_columns_per_channel(cohort):
    """The triple is structural, not conventional."""
    spec = fit_feature_spec(cohort.days, lookback_days=14)
    names = spec.feature_names()
    assert len(names) == spec.n_features
    for channel in spec.channels:
        assert f"{channel}__value" in names
        assert f"{channel}__is_observed" in names
        assert f"{channel}__staleness" in names


def test_normalisation_statistics_use_observed_values_only(cohort):
    """Carried-forward values must not leak into the normalisation constants."""
    spec = fit_feature_spec(cohort.days)
    for channel in ("lh", "e3g", "pdg"):
        observed = [
            float(day.values[channel])
            for day in cohort.days
            if day.is_observed[channel] and day.values[channel] is not None
        ]
        assert spec.channel_means[channel] == pytest.approx(float(np.mean(observed)))


def test_decay_shrinks_a_stale_value_toward_the_mean(cohort):
    """A week-old measurement must count for less than today's."""
    spec = fit_feature_spec(cohort.days, use_decay=True)
    channel = spec.channels[0]
    index = spec.channels.index(channel)
    high = spec.channel_means[channel] + 3 * spec.channel_scales[channel]

    def day(study_day: int, observed: bool, staleness: float) -> ParticipantDay:
        return ParticipantDay(
            participant_id="X",
            study_day=study_day,
            cycle_day=10,
            values={c: (high if (observed and c == channel) else None) for c in spec.channels},
            is_observed={c: (observed and c == channel) for c in spec.channels},
            time_since_last_observed=dict.fromkeys(spec.channels, staleness),
        )

    window = [day(0, True, 0.0), day(1, False, 1.0), day(2, False, 7.0)]
    matrix = build_feature_matrix(window, spec)
    fresh = abs(matrix[0, 3 * index])
    one_day = abs(matrix[1, 3 * index])
    week = abs(matrix[2, 3 * index])
    assert fresh > one_day > week
    assert week < 0.2 * fresh


def test_ablation_marks_channels_missing_rather_than_zero(cohort):
    """Removing the wearable must look missing, not look like a zero heart rate."""
    ablated = ablate_days(cohort.days, "no_wearable")
    for day in ablated:
        for channel in ("resting_heart_rate", "wrist_temperature", "hrv_rmssd"):
            assert day.is_observed[channel] is False
            assert day.values[channel] is None
            assert day.time_since_last_observed[channel] > 0


def test_coverage_reflects_the_ablation(cohort):
    """input_coverage must drop when a modality is removed."""
    spec = fit_feature_spec(cohort.days, lookback_days=14)
    full, _ = make_windows(cohort.for_participant(cohort.participant_ids[0]), spec)
    ablated_days = ablate_days(cohort.days, "no_wearable")
    ablated = [d for d in ablated_days if d.participant_id == cohort.participant_ids[0]]
    reduced, _ = make_windows(ablated, spec)

    indicators = [3 * i + 1 for i in range(len(spec.channels))]
    assert reduced[:, :, indicators].mean() < full[:, :, indicators].mean()
