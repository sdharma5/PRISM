"""Validation for mcPHASES participant-day construction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ingestion.mcphases.alignment import UNOBSERVED_SENTINEL
from registry.loader import load_dataset_registry
from schemas.temporal import ParticipantDay

__all__ = [
    "assert_use_permitted",
    "validate_no_fabricated_values",
    "validate_participant_days",
    "validate_stream_columns",
]


def assert_use_permitted(use: str, dataset_id: str = "mcphases") -> None:
    """Fail closed on a use the registry does not allow for mcPHASES."""
    load_dataset_registry().require(dataset_id, use)


def validate_stream_columns(
    columns: Sequence[str], required: Sequence[str] = ("participant_id", "timestamp")
) -> list[str]:
    """Check that a high-frequency stream table has the columns we depend on."""
    return [f"Stream is missing required column '{c}'." for c in required if c not in columns]


def validate_no_fabricated_values(days: Sequence[ParticipantDay]) -> list[str]:
    """Assert the core invariant: an unobserved feature carries no value.

    A value present where ``is_observed`` is False means something forward-filled
    or imputed upstream, which would let a model treat a guess as a measurement.
    """
    errors: list[str] = []
    for day in days:
        for name, observed in day.is_observed.items():
            value = day.values.get(name)
            if not observed and value is not None:
                errors.append(
                    f"{day.participant_id} day {day.study_day}: '{name}' is not observed "
                    f"but carries value {value!r} — fabricated data."
                )
            if observed and value is None:
                errors.append(
                    f"{day.participant_id} day {day.study_day}: '{name}' claims to be observed "
                    "but has no value."
                )
    return errors


def validate_participant_days(days: Sequence[ParticipantDay]) -> list[str]:
    """Full structural check of a participant-day series."""
    errors = list(validate_no_fabricated_values(days))

    if not days:
        return errors

    participants = {d.participant_id for d in days}
    if len(participants) > 1:
        errors.append(f"Series mixes participants: {sorted(participants)}")

    seen_dates: set[str] = set()
    previous: date | None = None
    for day in days:
        if day.calendar_date is None:
            continue
        if day.calendar_date in seen_dates:
            errors.append(f"Duplicate calendar_date {day.calendar_date}.")
        seen_dates.add(day.calendar_date)
        current = date.fromisoformat(day.calendar_date)
        if previous is not None and current <= previous:
            errors.append(f"calendar_date {day.calendar_date} is not strictly increasing.")
        previous = current

    for day in days:
        for name, elapsed in day.time_since_last_observed.items():
            if elapsed < 0:
                errors.append(f"Negative time_since_last_observed for '{name}'.")
            if day.is_observed.get(name) and elapsed not in (0.0, UNOBSERVED_SENTINEL):
                errors.append(
                    f"{name} is observed on day {day.study_day} but recency is {elapsed}, not 0."
                )
        if day.cycle_day is not None and day.cycle_day < 1:
            errors.append(f"Non-positive cycle_day {day.cycle_day} on day {day.study_day}.")

    return errors
