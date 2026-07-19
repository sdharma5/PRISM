"""Validation for NHANES ingestion, including registry use enforcement."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ingestion.nhanes.merge import SEQN, SURVEY_DESIGN_COLUMNS
from registry.loader import load_dataset_registry, load_variable_registry

__all__ = [
    "absent_mapped_columns",
    "assert_use_permitted",
    "validate_merged_frame",
    "validate_weight_availability",
]


def absent_mapped_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> list[str]:
    """Mapped NHANES columns not present in this merge.

    A warning rather than an error: which components an analysis loads is a
    legitimate choice, and demanding all of them would force every caller to
    download files they do not use.
    """
    return [
        f"Mapped NHANES column '{c}' is absent from the merged frame; no events emitted for it."
        for c in mapping
        if c not in frame.columns
    ]


def assert_use_permitted(dataset_id: str, use: str) -> None:
    """Fail closed when the registry does not allow ``use`` for ``dataset_id``.

    Raises:
        PermissionError: If the use is prohibited or simply not listed.
    """
    load_dataset_registry().require(dataset_id, use)


def validate_merged_frame(frame: pd.DataFrame, mapping: dict[str, str]) -> list[str]:
    """Structural checks on a merged NHANES frame.

    Args:
        frame: The merged component table.
        mapping: NHANES column -> canonical variable code.

    Returns:
        Human-readable validation errors; empty means valid.
    """
    errors: list[str] = []

    if SEQN not in frame.columns:
        errors.append(f"Merged frame is missing {SEQN}.")
    elif frame[SEQN].isna().any():
        errors.append(f"{int(frame[SEQN].isna().sum())} row(s) have a null {SEQN}.")
    elif frame[SEQN].duplicated().any():
        errors.append("Merged frame has duplicate SEQN values; components merged incorrectly.")

    registry = load_variable_registry().variables
    for column, code in mapping.items():
        if code not in registry:
            errors.append(f"NHANES column '{column}' maps to unknown canonical code '{code}'.")

    for column in mapping:
        if column in SURVEY_DESIGN_COLUMNS:
            errors.append(
                f"'{column}' is a survey-design column and must be carried as metadata, "
                "not emitted as a clinical variable."
            )

    return errors


def validate_weight_availability(frame: pd.DataFrame, weight_columns: Sequence[str]) -> list[str]:
    """Warn when a survey weight needed for population estimates is missing.

    Missing weights are a warning rather than an error because reference-range
    and unit-harmonization uses are legitimate without them; only population
    estimates require them.
    """
    warnings: list[str] = []
    for column in weight_columns:
        if column not in frame.columns:
            warnings.append(
                f"Survey weight '{column}' absent; population-weighted estimates are unavailable."
            )
            continue
        missing = int(frame[column].isna().sum())
        if missing:
            warnings.append(f"Survey weight '{column}' is null for {missing} respondent(s).")
    return warnings
