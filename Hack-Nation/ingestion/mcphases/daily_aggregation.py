"""Daily aggregation of high-frequency wearable and CGM streams.

Every aggregation choice below is a modelling decision, so each is stated
explicitly rather than left to a library default:

* **mean / median** — both are kept. The mean is the natural summary for
  glucose exposure; the median is robust to the sensor dropouts and compression
  artefacts that are routine in wearable data. Reporting only one hides which
  regime a day was in.
* **std** — sample standard deviation (ddof=1) because a day's readings are a
  sample of that day's underlying process, not the whole population. A day with
  a single reading has undefined variability and returns ``None``, never 0;
  reporting 0 would falsely assert perfect stability.
* **min / max** — raw extremes, not winsorized. Hypo- and hyperglycaemic
  excursions are clinically the most interesting part of a CGM day, and
  trimming them is exactly the wrong reduction.
* **day-night difference** — mean over 06:00-22:00 local minus mean over
  22:00-06:00. This split is used rather than a fitted circadian model because
  it needs no per-participant calibration and degrades gracefully with missing
  data. Undefined (``None``) unless both windows have observations.
* **time in range** — fraction of readings inside the variable's registry
  ``valid_range`` by default, or an explicit range for CGM (70-180 mg/dL, the
  consensus target). It is a fraction of *observed* readings, so it must always
  be read alongside ``missing_fraction``.
* **rate of change** — median absolute first difference per hour, using actual
  timestamp gaps rather than assuming a fixed sampling interval, so irregular
  sampling does not inflate it.
* **missing fraction** — 1 minus (observed readings / expected readings), where
  expected comes from the declared sampling interval. This is the number that
  tells a consumer whether any of the above means anything.

No aggregate is ever computed by filling gaps. A day with no readings yields
``None`` values and ``is_observed=False``, because a fabricated day is
indistinguishable to a model from a real one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time

import numpy as np

from registry.loader import load_variable_registry

__all__ = [
    "CGM_TARGET_RANGE_MGDL",
    "DAY_WINDOW",
    "DailyAggregate",
    "aggregate_day",
    "day_night_difference",
    "missing_fraction",
    "rate_of_change_per_hour",
    "time_in_range",
]

#: Daytime window used for the day-night contrast (local clock time).
DAY_WINDOW: tuple[time, time] = (time(6, 0), time(22, 0))

#: Consensus CGM target range in mg/dL.
CGM_TARGET_RANGE_MGDL: tuple[float, float] = (70.0, 180.0)


@dataclass(frozen=True)
class DailyAggregate:
    """Per-variable summary of one participant-day.

    ``is_observed`` is False whenever the day contained no usable readings; in
    that case every statistic is ``None``.
    """

    code: str
    n_observations: int
    is_observed: bool
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    day_night_difference: float | None = None
    time_in_range: float | None = None
    rate_of_change_per_hour: float | None = None
    missing_fraction: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_feature_dict(self) -> dict[str, float | None]:
        """Flatten to ``{code}_{statistic}`` keys for a ParticipantDay row."""
        return {
            f"{self.code}_mean": self.mean,
            f"{self.code}_median": self.median,
            f"{self.code}_std": self.std,
            f"{self.code}_min": self.minimum,
            f"{self.code}_max": self.maximum,
            f"{self.code}_day_night_diff": self.day_night_difference,
            f"{self.code}_time_in_range": self.time_in_range,
            f"{self.code}_roc_per_hour": self.rate_of_change_per_hour,
            f"{self.code}_missing_fraction": self.missing_fraction,
        }


def _finite(
    timestamps: Sequence[datetime], values: Sequence[float] | np.ndarray
) -> tuple[list[datetime], np.ndarray]:
    array = np.asarray(values, dtype=float)
    if len(timestamps) != array.size:
        raise ValueError("timestamps and values must be the same length.")
    keep = np.isfinite(array)
    return [t for t, k in zip(timestamps, keep, strict=True) if k], array[keep]


def day_night_difference(
    timestamps: Sequence[datetime],
    values: Sequence[float] | np.ndarray,
    window: tuple[time, time] = DAY_WINDOW,
) -> float | None:
    """Mean over the daytime window minus mean over the night window.

    Returns None unless both windows contain at least one reading — a one-sided
    difference would be a fabricated contrast.
    """
    stamps, array = _finite(timestamps, values)
    if array.size == 0:
        return None
    start, end = window
    is_day = np.array([start <= t.time() < end for t in stamps])
    if not is_day.any() or is_day.all():
        return None
    return float(array[is_day].mean() - array[~is_day].mean())


def time_in_range(
    values: Sequence[float] | np.ndarray,
    target: tuple[float, float] | None = None,
    code: str | None = None,
) -> float | None:
    """Fraction of observed readings inside the target range.

    Args:
        values: Readings for the day.
        target: Explicit ``(low, high)`` bounds.
        code: Canonical code whose registry ``valid_range`` supplies the bounds
            when ``target`` is not given.

    Returns:
        A fraction in [0, 1], or None when there are no readings or no bounds.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    if target is None:
        if code == "cgm_mean_glucose":
            target = CGM_TARGET_RANGE_MGDL
        else:
            spec = load_variable_registry().variables.get(code or "")
            if spec is None or spec.valid_range is None:
                return None
            low = spec.valid_range.min if spec.valid_range.min is not None else -np.inf
            high = spec.valid_range.max if spec.valid_range.max is not None else np.inf
            target = (float(low), float(high))
    low, high = target
    return float(np.mean((array >= low) & (array <= high)))


def rate_of_change_per_hour(
    timestamps: Sequence[datetime], values: Sequence[float] | np.ndarray
) -> float | None:
    """Median absolute first difference per hour, using real timestamp gaps."""
    stamps, array = _finite(timestamps, values)
    if array.size < 2:
        return None
    order = np.argsort([t.timestamp() for t in stamps])
    sorted_stamps = [stamps[i] for i in order]
    sorted_values = array[order]
    hours = np.array(
        [
            (sorted_stamps[i + 1] - sorted_stamps[i]).total_seconds() / 3600.0
            for i in range(len(sorted_stamps) - 1)
        ]
    )
    deltas = np.abs(np.diff(sorted_values))
    usable = hours > 0
    if not usable.any():
        return None
    return float(np.median(deltas[usable] / hours[usable]))


def missing_fraction(n_observed: int, expected_per_day: int | None) -> float | None:
    """Fraction of the day's expected readings that never arrived."""
    if not expected_per_day or expected_per_day <= 0:
        return None
    return float(max(0.0, 1.0 - n_observed / expected_per_day))


def aggregate_day(
    code: str,
    timestamps: Sequence[datetime],
    values: Sequence[float],
    *,
    expected_per_day: int | None = None,
    target_range: tuple[float, float] | None = None,
) -> DailyAggregate:
    """Summarize one variable over one participant-day.

    Args:
        code: Canonical variable code.
        timestamps: Reading timestamps.
        values: Reading values, aligned with ``timestamps``.
        expected_per_day: Expected reading count, used for ``missing_fraction``.
        target_range: Explicit time-in-range bounds.

    Returns:
        A :class:`DailyAggregate`. A day with no usable readings comes back with
        ``is_observed=False`` and all statistics None — never zero-filled.
    """
    stamps, array = _finite(timestamps, values)
    if array.size == 0:
        return DailyAggregate(
            code=code,
            n_observations=0,
            is_observed=False,
            missing_fraction=missing_fraction(0, expected_per_day),
            warnings=["no usable readings; day left unobserved rather than imputed"],
        )

    warnings: list[str] = []
    # ddof=1 is undefined for a single reading: report None, not a false 0.
    std = float(np.std(array, ddof=1)) if array.size > 1 else None
    if std is None:
        warnings.append("single reading: standard deviation undefined, not zero")

    return DailyAggregate(
        code=code,
        n_observations=int(array.size),
        is_observed=True,
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=std,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        day_night_difference=day_night_difference(stamps, array),
        time_in_range=time_in_range(array, target_range, code),
        rate_of_change_per_hour=rate_of_change_per_hour(stamps, array),
        missing_fraction=missing_fraction(int(array.size), expected_per_day),
        warnings=warnings,
    )
