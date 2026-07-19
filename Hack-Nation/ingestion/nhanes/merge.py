"""Merging NHANES component tables on SEQN.

NHANES ships one file per component (demographics, biochemistry, reproductive
health questionnaire, ...) joined by the respondent sequence number ``SEQN``.
Merges are always outer joins on SEQN: an inner join would silently define the
analysis population as "people who happened to take every component", which is
itself a selection effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

__all__ = [
    "SEQN",
    "SURVEY_DESIGN_COLUMNS",
    "collect_survey_metadata",
    "merge_components",
]

SEQN = "SEQN"

#: Columns that describe the sample design rather than the person. They are
#: carried as event metadata, never emitted as clinical variables.
SURVEY_DESIGN_COLUMNS: tuple[str, ...] = (
    "WTINT2YR",
    "WTMEC2YR",
    "WTSAF2YR",
    "WTSAFPRP",
    "SDMVPSU",
    "SDMVSTRA",
    "RIDSTATR",
)


def merge_components(
    components: Mapping[str, pd.DataFrame],
    *,
    required: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Outer-join NHANES component tables on SEQN.

    Args:
        components: Component name -> DataFrame, each containing a SEQN column.
        required: Components a respondent must appear in to be kept. Defaults to
            keeping everyone, so missingness stays visible rather than becoming
            silent exclusion.

    Returns:
        One row per SEQN. Overlapping non-SEQN columns are suffixed with the
        component name so no column is silently overwritten by another file.

    Raises:
        ValueError: If a component lacks SEQN or has duplicate SEQN values.
    """
    if not components:
        raise ValueError("No NHANES components supplied.")

    frames: dict[str, pd.DataFrame] = {}
    for name, frame in components.items():
        if SEQN not in frame.columns:
            raise ValueError(f"Component '{name}' has no {SEQN} column.")
        duplicated = frame[SEQN].duplicated().sum()
        if duplicated:
            raise ValueError(
                f"Component '{name}' has {duplicated} duplicate {SEQN} value(s); "
                "one row per respondent is required."
            )
        frames[name] = frame.copy()

    merged: pd.DataFrame | None = None
    seen: set[str] = set()
    for name, frame in frames.items():
        overlap = (set(frame.columns) - {SEQN}) & seen
        if overlap:
            frame = frame.rename(columns={c: f"{c}__{name}" for c in overlap})
        seen |= set(frame.columns) - {SEQN}
        merged = frame if merged is None else merged.merge(frame, on=SEQN, how="outer")

    assert merged is not None
    if required:
        for name in required:
            present = set(frames[name][SEQN])
            merged = merged[merged[SEQN].isin(present)]

    return merged.sort_values(SEQN).reset_index(drop=True)


def collect_survey_metadata(row: pd.Series) -> dict[str, float | int | str]:
    """Extract the survey-design fields for one respondent.

    These travel with every emitted event so that any later population estimate
    can be weighted correctly without re-opening the source files.
    """
    metadata: dict[str, float | int | str] = {}
    for column in SURVEY_DESIGN_COLUMNS:
        if column in row.index and pd.notna(row[column]):
            value = row[column]
            metadata[column] = float(value) if isinstance(value, (int, float)) else str(value)
    return metadata
