"""Cycle-day alignment and recency bookkeeping for participant-days.

Two jobs:

1. Map a calendar date onto a menstrual cycle day, given the participant's
   recorded menses onset dates.
2. For each feature, record how long ago it was last actually observed.

The second is what lets a downstream model use a stale value without being
misled by it. PRISM never carries a value forward silently: a day that had no
reading keeps ``value=None``, ``is_observed=False``, and a growing
``time_since_last_observed``. Forward-filling would make an unobserved day
indistinguishable from a measured one, which is the single easiest way to
manufacture a confident wrong prediction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, timedelta

from schemas.temporal import CyclePhase, ParticipantDay

__all__ = [
    "UNOBSERVED_SENTINEL",
    "assign_cycle_day",
    "assign_cycle_phase",
    "build_participant_days",
    "derive_time_since_last_observed",
]

#: Recency for a feature never yet observed. Deliberately large and finite so it
#: sorts last, and distinguishable from any plausible real gap.
UNOBSERVED_SENTINEL = 9999.0


def assign_cycle_day(day: date, menses_onsets: Sequence[date]) -> int | None:
    """Cycle day for ``day``, where the onset date itself is day 1.

    Args:
        day: The calendar date to place.
        menses_onsets: Recorded first-day-of-menses dates.

    Returns:
        1-based cycle day, or None when no onset precedes ``day`` — the cycle
        day is genuinely unknown before the first recorded onset.
    """
    prior = [d for d in sorted(menses_onsets) if d <= day]
    if not prior:
        return None
    return (day - prior[-1]).days + 1


def assign_cycle_phase(cycle_day: int | None, cycle_length: int | None = None) -> CyclePhase:
    """Coarse cycle phase from cycle day.

    Uses a fixed 14-day luteal phase counted backwards from the next expected
    menses, because luteal length is the more stable of the two phases;
    follicular length absorbs cycle-length variation. Returns ``unknown``
    whenever the inputs cannot support a defensible assignment.
    """
    if cycle_day is None or cycle_day < 1:
        return "unknown"
    if cycle_day <= 5:
        return "menstrual"
    if cycle_length is None or cycle_length < 15:
        return "unknown"
    ovulation_day = cycle_length - 14
    if cycle_day < ovulation_day - 2:
        return "follicular"
    if cycle_day <= ovulation_day + 2:
        return "peri_ovulatory"
    if cycle_day <= cycle_length:
        return "luteal"
    return "unknown"


def derive_time_since_last_observed(
    observations: Sequence[bool], step_days: float = 1.0
) -> list[float]:
    """Days since each feature was last observed, walking forward in time.

    Args:
        observations: Per-day observation flags for one feature, in date order.
        step_days: Spacing between consecutive entries.

    Returns:
        One float per day. On a day the feature *was* observed the value is 0.0;
        otherwise it grows by ``step_days`` per day. Days before the first ever
        observation get :data:`UNOBSERVED_SENTINEL`.
    """
    result: list[float] = []
    elapsed: float | None = None
    for observed in observations:
        if observed:
            elapsed = 0.0
        elif elapsed is not None:
            elapsed += step_days
        result.append(UNOBSERVED_SENTINEL if elapsed is None else elapsed)
    return result


def build_participant_days(
    participant_id: str,
    dates: Sequence[date],
    features_by_date: dict[date, dict[str, float | None]],
    *,
    menses_onsets: Sequence[date] = (),
    cycle_length: int | None = None,
    symptoms_by_date: dict[date, dict[str, bool]] | None = None,
    source_dataset: str = "mcphases",
) -> list[ParticipantDay]:
    """Build a contiguous ParticipantDay series.

    Days with no data are still emitted — the gap itself is information — but
    carry ``value=None`` and ``is_observed=False`` for every feature.

    Args:
        participant_id: Dataset-scoped identifier.
        dates: Calendar dates to cover, in ascending order.
        features_by_date: Date -> {feature name: value or None}.
        menses_onsets: Recorded menses onset dates for cycle alignment.
        cycle_length: Typical cycle length, used for phase assignment.
        symptoms_by_date: Date -> {symptom: present}.
        source_dataset: Recorded on each row.

    Returns:
        One :class:`ParticipantDay` per date, in order.
    """
    ordered = sorted(dates)
    feature_names = sorted({name for row in features_by_date.values() for name in row})

    observed_matrix: dict[str, list[bool]] = {}
    for name in feature_names:
        observed_matrix[name] = [
            features_by_date.get(day, {}).get(name) is not None for day in ordered
        ]
    recency = {
        name: derive_time_since_last_observed(flags) for name, flags in observed_matrix.items()
    }

    first_day = ordered[0] if ordered else None
    days: list[ParticipantDay] = []
    for i, day in enumerate(ordered):
        row = features_by_date.get(day, {})
        cycle_day = assign_cycle_day(day, menses_onsets) if menses_onsets else None
        days.append(
            ParticipantDay(
                participant_id=participant_id,
                study_day=(day - first_day).days if first_day else i,
                calendar_date=day.isoformat(),
                cycle_day=cycle_day,
                cycle_phase=assign_cycle_phase(cycle_day, cycle_length),
                values={name: row.get(name) for name in feature_names},
                is_observed={name: observed_matrix[name][i] for name in feature_names},
                time_since_last_observed={name: recency[name][i] for name in feature_names},
                daily_symptoms=(symptoms_by_date or {}).get(day, {}),
                source_dataset=source_dataset,
            )
        )
    return days


def date_range(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end``."""
    if end < start:
        raise ValueError("end date precedes start date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def observed_fractions(days: Iterable[ParticipantDay]) -> dict[str, float]:
    """Per-feature fraction of days actually observed across a series."""
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for day in days:
        for name, flag in day.is_observed.items():
            totals[name] = totals.get(name, 0) + 1
            hits[name] = hits.get(name, 0) + int(flag)
    return {name: hits[name] / totals[name] for name in totals}
