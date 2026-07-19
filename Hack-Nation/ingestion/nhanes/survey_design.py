"""Survey-weighted estimators for NHANES.

NHANES is not a simple random sample: participants are selected with unequal
probability and some subgroups are deliberately oversampled. An unweighted mean
of an NHANES column therefore estimates nothing about the US population. Every
population-level number PRISM derives from NHANES must come through one of
these functions, with the appropriate weight column
(``WTMEC2YR`` for MEC-examination variables, ``WTINT2YR`` for interview-only
variables, and the fasting subsample weight for fasting labs).

Variance estimation with the design strata (``SDMVSTRA``) and PSUs
(``SDMVPSU``) is intentionally *not* implemented here: PRISM uses NHANES only
for reference ranges and unit harmonization, and a half-correct standard error
would invite exactly the population inference the dataset registry prohibits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np

__all__ = [
    "EXAM_WEIGHT_COLUMN",
    "FASTING_WEIGHT_COLUMN",
    "INTERVIEW_WEIGHT_COLUMN",
    "weighted_mean",
    "weighted_quantile",
    "weighted_reference_range",
    "weighted_std",
]

#: Two-year MEC examination weight; use for anything measured in the mobile clinic.
EXAM_WEIGHT_COLUMN = "WTMEC2YR"
#: Two-year interview weight; use for questionnaire-only variables.
INTERVIEW_WEIGHT_COLUMN = "WTINT2YR"
#: Fasting subsample weight; required for fasting glucose/insulin.
FASTING_WEIGHT_COLUMN = "WTSAF2YR"


def _clean(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite values and non-positive weights, keeping the pair aligned."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError(f"values and weights must align: {v.shape} vs {w.shape}")
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        raise ValueError("No usable (finite value, positive weight) pairs.")
    return v[keep], w[keep]


def weighted_mean(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray
) -> float:
    """Survey-weighted mean.

    Args:
        values: Observed values.
        weights: Sampling weights, one per value.

    Returns:
        The weighted mean.
    """
    v, w = _clean(values, weights)
    return float(np.sum(v * w) / np.sum(w))


def weighted_std(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray
) -> float:
    """Survey-weighted standard deviation about the weighted mean.

    Uses the reliability-weight correction, which is the right one when weights
    represent how many population members each row stands for.
    """
    v, w = _clean(values, weights)
    mean = float(np.sum(v * w) / np.sum(w))
    total = float(np.sum(w))
    denominator = total - float(np.sum(w**2)) / total
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(np.sum(w * (v - mean) ** 2) / denominator))


@overload
def weighted_quantile(
    values: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray, q: float
) -> float: ...


@overload
def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    q: Sequence[float],
) -> list[float]: ...


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    q: float | Sequence[float],
) -> float | list[float]:
    """Survey-weighted quantile(s) via the cumulative-weight step function.

    Args:
        values: Observed values.
        weights: Sampling weights, one per value.
        q: A quantile in [0, 1], or a sequence of them.

    Returns:
        A float for scalar ``q``, otherwise a list of floats.
    """
    v, w = _clean(values, weights)
    order = np.argsort(v)
    v, w = v[order], w[order]
    # Midpoint of each weight block: the standard plotting-position convention
    # for weighted quantiles, and unbiased for equal weights.
    cumulative = (np.cumsum(w) - 0.5 * w) / np.sum(w)

    scalar = isinstance(q, (int, float))
    qs = [float(q)] if isinstance(q, (int, float)) else [float(x) for x in q]
    for value in qs:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Quantile {value} outside [0, 1].")
    result = [float(np.interp(value, cumulative, v)) for value in qs]
    return result[0] if scalar else result


def weighted_reference_range(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    lower: float = 0.025,
    upper: float = 0.975,
) -> dict[str, float]:
    """Weighted central reference interval plus summary statistics.

    Defaults to the conventional 2.5th-97.5th percentile clinical reference
    interval rather than a mean +/- 2 SD interval, because hormone
    distributions are strongly right-skewed.
    """
    low, high = weighted_quantile(values, weights, [lower, upper])
    median = weighted_quantile(values, weights, 0.5)
    v, w = _clean(values, weights)
    return {
        "lower": float(low),
        "median": float(median),
        "upper": float(high),
        "weighted_mean": weighted_mean(v, w),
        "weighted_std": weighted_std(v, w),
        "n_unweighted": int(v.size),
        "sum_weights": float(np.sum(w)),
        "lower_quantile": lower,
        "upper_quantile": upper,
    }
