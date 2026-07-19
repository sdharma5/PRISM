"""Read-only query helpers over a collection of events.

These are pure filters. They return new lists and never mutate or reorder the
underlying store, which is what makes it safe to expose the store's internal
list to them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from schemas.event import HormonalHealthEvent

__all__ = [
    "by_confirmation_status",
    "by_modality",
    "by_patient",
    "by_time_window",
    "by_variable_code",
    "latest_per_code",
    "model_ready",
]


def by_patient(events: Iterable[HormonalHealthEvent], patient_id: str) -> list[HormonalHealthEvent]:
    """All events for one patient. Ids are dataset-scoped and never merged."""
    return [e for e in events if e.patient_id == patient_id]


def by_time_window(
    events: Iterable[HormonalHealthEvent],
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    include_undated: bool = False,
) -> list[HormonalHealthEvent]:
    """Events observed within ``[start, end]``.

    Args:
        events: Source events.
        start: Inclusive lower bound, or None for unbounded.
        end: Inclusive upper bound, or None for unbounded.
        include_undated: Whether events without ``observed_at`` are kept.
            Defaults to False so an undated event is never assumed to be recent.
    """
    selected: list[HormonalHealthEvent] = []
    for event in events:
        if event.observed_at is None:
            if include_undated:
                selected.append(event)
            continue
        stamp = _naive(event.observed_at)
        if start is not None and stamp < _naive(start):
            continue
        if end is not None and stamp > _naive(end):
            continue
        selected.append(event)
    return selected


def by_modality(
    events: Iterable[HormonalHealthEvent], modalities: str | Sequence[str]
) -> list[HormonalHealthEvent]:
    """Events from one modality or any of several."""
    wanted = {modalities} if isinstance(modalities, str) else set(modalities)
    return [e for e in events if e.modality in wanted]


def by_variable_code(
    events: Iterable[HormonalHealthEvent], codes: str | Sequence[str]
) -> list[HormonalHealthEvent]:
    """Events for one canonical variable code or any of several."""
    wanted = {codes} if isinstance(codes, str) else set(codes)
    return [e for e in events if e.canonical_variable_code in wanted]


def by_confirmation_status(
    events: Iterable[HormonalHealthEvent], statuses: str | Sequence[str]
) -> list[HormonalHealthEvent]:
    """Events in one confirmation status or any of several."""
    wanted = {statuses} if isinstance(statuses, str) else set(statuses)
    return [e for e in events if e.confirmation_status in wanted]


def model_ready(events: Iterable[HormonalHealthEvent]) -> list[HormonalHealthEvent]:
    """Only events that may enter a model-ready snapshot."""
    return [e for e in events if e.is_model_ready]


def latest_per_code(
    events: Iterable[HormonalHealthEvent],
) -> dict[str, HormonalHealthEvent]:
    """Most recently observed event per canonical code, ignoring undated ones."""
    latest: dict[str, HormonalHealthEvent] = {}
    for event in events:
        if event.observed_at is None:
            latest.setdefault(event.canonical_variable_code, event)
            continue
        current = latest.get(event.canonical_variable_code)
        if (
            current is None
            or current.observed_at is None
            or _naive(event.observed_at) > _naive(current.observed_at)
        ):
            latest[event.canonical_variable_code] = event
    return latest


def _naive(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=None) if stamp.tzinfo is not None else stamp
