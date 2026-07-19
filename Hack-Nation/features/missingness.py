"""Explicit missingness handling.

A missing hormone value is information, not a zero. This module keeps the six
``MissingnessStatus`` values distinct all the way into the feature matrix, emits
indicator columns so a model can *learn* from absence, and refuses to let anyone
quietly fill an unobserved value with 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import get_args

import numpy as np
import pandas as pd

from schemas.event import MissingnessStatus

#: The six statuses, in the order the schema declares them.
MISSINGNESS_STATUSES: tuple[str, ...] = tuple(get_args(MissingnessStatus))

#: Suffix for the per-variable status column a caller may supply alongside a value.
STATUS_SUFFIX = "__missingness_status"

#: Suffix for the generated binary indicator column.
INDICATOR_SUFFIX = "__is_missing"

#: Statuses that mean "no value exists for this patient, and that is expected".
#: They are excluded from coverage denominators: a variable that cannot apply to
#: a patient should not be counted against that patient's data completeness.
NOT_APPLICABLE_STATUSES: frozenset[str] = frozenset({"not_applicable"})

#: The status assumed for a bare ``NaN`` with no accompanying status column.
DEFAULT_MISSING_STATUS = "not_collected"


class SilentZeroFillError(ValueError):
    """Raised when unobserved values were replaced by zeros without an indicator."""


def status_column(code: str) -> str:
    """Name of the companion status column for ``code``."""
    return f"{code}{STATUS_SUFFIX}"


def indicator_column(code: str) -> str:
    """Name of the generated missingness indicator for ``code``."""
    return f"{code}{INDICATOR_SUFFIX}"


def resolve_status(df: pd.DataFrame, code: str) -> pd.Series:
    """Return the per-row ``MissingnessStatus`` for one canonical code.

    An explicit companion status column always wins; otherwise presence of a
    value implies ``observed`` and absence implies ``not_collected``. Guessing a
    more specific reason would fabricate provenance we do not have.
    """
    n = len(df)
    if code not in df.columns:
        return pd.Series([DEFAULT_MISSING_STATUS] * n, index=df.index, dtype=object)

    values = df[code]
    inferred = np.where(values.isna(), DEFAULT_MISSING_STATUS, "observed")
    status = pd.Series(inferred, index=df.index, dtype=object)

    explicit_col = status_column(code)
    if explicit_col in df.columns:
        explicit = df[explicit_col]
        unknown = set(explicit.dropna().unique()) - set(MISSINGNESS_STATUSES)
        if unknown:
            raise ValueError(f"{explicit_col}: unknown missingness status(es) {sorted(unknown)}.")
        status = status.mask(explicit.notna(), explicit)

    # A declared non-observed status must not sit on top of a present value.
    conflict = (status != "observed") & values.notna()
    if conflict.any():
        raise ValueError(
            f"{code}: {int(conflict.sum())} row(s) declare a non-observed status but carry a "
            "value. Resolve the contradiction upstream rather than picking a winner here."
        )
    return status


def status_frame(df: pd.DataFrame, codes: Sequence[str]) -> pd.DataFrame:
    """A DataFrame of missingness statuses, one column per canonical code."""
    return pd.DataFrame({code: resolve_status(df, code) for code in codes}, index=df.index)


def observed_mask(df: pd.DataFrame, codes: Sequence[str]) -> pd.DataFrame:
    """Boolean mask: True where the value is genuinely observed."""
    return status_frame(df, codes) == "observed"


def applicable_mask(df: pd.DataFrame, codes: Sequence[str]) -> pd.DataFrame:
    """Boolean mask: True where the variable could in principle have a value."""
    return ~status_frame(df, codes).isin(NOT_APPLICABLE_STATUSES)


def build_indicator_columns(
    df: pd.DataFrame,
    codes: Sequence[str],
    *,
    per_status: bool = False,
) -> pd.DataFrame:
    """Binary indicators marking *which* values were unobserved.

    With ``per_status`` the indicators are broken out by reason, so a model can
    distinguish "the clinic never ordered this assay" from "extraction failed".
    """
    statuses = status_frame(df, codes)
    out: dict[str, pd.Series] = {}
    for code in codes:
        out[indicator_column(code)] = (statuses[code] != "observed").astype(float)
        if per_status:
            for status in MISSINGNESS_STATUSES:
                if status == "observed":
                    continue
                out[f"{code}__missing_{status}"] = (statuses[code] == status).astype(float)
    return pd.DataFrame(out, index=df.index)


def coverage_by_row(df: pd.DataFrame, codes: Sequence[str]) -> pd.Series:
    """Fraction of *applicable* codes observed for each row."""
    observed = observed_mask(df, codes)
    applicable = applicable_mask(df, codes)
    denominator = applicable.sum(axis=1)
    numerator = (observed & applicable).sum(axis=1)
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0)


def missingness_summary(df: pd.DataFrame, codes: Sequence[str]) -> dict[str, dict[str, int]]:
    """Per-code counts of each missingness status, for the feature manifest."""
    statuses = status_frame(df, codes)
    summary: dict[str, dict[str, int]] = {}
    for code in codes:
        counts = statuses[code].value_counts().to_dict()
        summary[code] = {status: int(counts.get(status, 0)) for status in MISSINGNESS_STATUSES}
    return summary


def assert_no_silent_zero_fill(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    codes: Iterable[str],
) -> None:
    """Fail if an unobserved value became exactly 0 with no indicator to explain it.

    Zero is a legitimate value for several canonical variables (symptom flags,
    follicle counts), so a silent 0-fill is indistinguishable from a real reading
    unless an indicator column accompanies it.
    """
    offenders: list[str] = []
    for code in codes:
        if code not in original.columns or code not in transformed.columns:
            continue
        was_missing = original[code].isna()
        if not was_missing.any():
            continue
        became_zero = pd.to_numeric(transformed[code], errors="coerce").fillna(np.nan) == 0
        if (was_missing & became_zero).any() and indicator_column(code) not in transformed.columns:
            offenders.append(code)
    if offenders:
        raise SilentZeroFillError(
            f"Unobserved values were zero-filled without a missingness indicator: {offenders}. "
            "Missing must never silently become 0."
        )
